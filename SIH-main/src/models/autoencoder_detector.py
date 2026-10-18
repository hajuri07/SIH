"""
Deep Autoencoder Anomaly Detection Module for Turbofan Prognostics.
Reconstruction-error based unsupervised anomaly detection:
Trains an encoder-decoder neural network strictly on nominal early-life cycles (t <= 30).
High reconstruction error on degraded cycles drives the continuous Health Index (HI)
and flags WARNING / CRITICAL states.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from .evaluator import evaluate_anomaly_detector


class AutoencoderNet(nn.Module):
    """
    Symmetric multi-layer compression Autoencoder with BatchNorm and LeakyReLU.
    Compresses high-dimensional feature vectors to a low-dimensional latent bottleneck.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim),
            nn.LeakyReLU(0.1)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderAnomalyDetector:
    """
    Autoencoder anomaly detector trained on pristine baseline telemetry.
    Reconstruction MSE is mapped to a continuous Health Index in [0.0, 1.0].
    """

    def __init__(
        self,
        input_dim: int = 140,
        latent_dim: int = 16,
        lr: float = 0.001,
        warning_hi_threshold: float = 0.65,
        critical_hi_threshold: float = 0.35,
        device: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.warning_hi_threshold = warning_hi_threshold
        self.critical_hi_threshold = critical_hi_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = AutoencoderNet(input_dim, latent_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss(reduction="none")

        self.feature_names: List[str] = []
        self.reconstruction_threshold: float = 0.05
        self.mse_min: float = 0.0
        self.mse_max: float = 0.5
        self.is_fitted: bool = False

    def fit(self, X_baseline: pd.DataFrame, epochs: int = 25, batch_size: int = 64) -> Dict[str, float]:
        """
        Trains the Autoencoder strictly on nominal baseline cycles (e.g. cycles <= 30).
        """
        self.feature_names = list(X_baseline.columns)
        self.input_dim = len(self.feature_names)
        self.net = AutoencoderNet(self.input_dim, self.latent_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        x_tensor = torch.tensor(X_baseline.values, dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.net.train()
        for epoch in range(epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                reconstructed = self.net(batch_x)
                loss = nn.functional.mse_loss(reconstructed, batch_x)
                loss.backward()
                self.optimizer.step()

        # Compute baseline reconstruction errors to calibrate Health Index scale
        self.net.eval()
        with torch.no_grad():
            x_b = x_tensor.to(self.device)
            rec_b = self.net(x_b)
            # Per-sample MSE
            baseline_mses = torch.mean((rec_b - x_b) ** 2, dim=1).cpu().numpy()

        self.mse_min = float(np.percentile(baseline_mses, 5))
        # Set anomaly threshold at 95th percentile of healthy baseline
        self.reconstruction_threshold = float(np.percentile(baseline_mses, 95))
        # Maximum expected MSE under severe degradation
        self.mse_max = float(self.reconstruction_threshold * 4.0)
        self.is_fitted = True

        return {
            "baseline_mean_mse": round(float(np.mean(baseline_mses)), 6),
            "reconstruction_threshold": round(self.reconstruction_threshold, 6),
            "baseline_samples": len(X_baseline)
        }

    def compute_health_index(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates continuous Health Index HI in [0.0, 1.0].
        Pristine baseline (low reconstruction error) = 1.0; severe anomaly = 0.0.
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder anomaly detector is not fitted yet.")

        x_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32).to(self.device)
        self.net.eval()
        with torch.no_grad():
            rec = self.net(x_tensor)
            sample_mses = torch.mean((rec - x_tensor) ** 2, dim=1).cpu().numpy()

        # Linear mapping: HI = 1 - (mse - mse_min) / (mse_max - mse_min)
        denom = max(1e-6, (self.mse_max - self.mse_min))
        normalized_error = (sample_mses - self.mse_min) / denom
        hi = 1.0 - normalized_error
        return np.clip(hi, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns:
            is_anomaly: 1 if anomalous (MSE > threshold), 0 if nominal
            health_indices: float array in [0.0, 1.0]
            status_labels: "HEALTHY", "WARNING", or "CRITICAL"
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder anomaly detector is not fitted yet.")

        x_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32).to(self.device)
        self.net.eval()
        with torch.no_grad():
            rec = self.net(x_tensor)
            sample_mses = torch.mean((rec - x_tensor) ** 2, dim=1).cpu().numpy()

        is_anomaly = (sample_mses > self.reconstruction_threshold).astype(int)
        health_indices = self.compute_health_index(X)

        status_labels = []
        for hi, anom in zip(health_indices, is_anomaly):
            if hi <= self.critical_hi_threshold:
                status_labels.append("CRITICAL")
            elif hi <= self.warning_hi_threshold or anom == 1:
                status_labels.append("WARNING")
            else:
                status_labels.append("HEALTHY")

        return is_anomaly, health_indices, status_labels

    def evaluate_on_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        baseline_cycle_limit: int = 30
    ) -> Dict[str, float]:
        """
        Evaluates False-Alert Rate, Detection Lead Time, and Missed Failures.
        """
        is_anom, hi, status = self.predict(df[feature_cols])
        df_eval = df.copy()
        df_eval["is_anomaly"] = is_anom
        df_eval["health_index"] = hi
        df_eval["status"] = status

        metrics = evaluate_anomaly_detector(
            df_eval,
            is_anomaly_col="is_anomaly",
            baseline_cycle_limit=baseline_cycle_limit
        )
        return metrics

    def save(self, filepath: Path) -> None:
        """Saves Autoencoder model weights and parameters."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "feature_names": self.feature_names,
            "reconstruction_threshold": self.reconstruction_threshold,
            "mse_min": self.mse_min,
            "mse_max": self.mse_max,
            "warning_hi_threshold": self.warning_hi_threshold,
            "critical_hi_threshold": self.critical_hi_threshold,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: Path, device: Optional[str] = None) -> "AutoencoderAnomalyDetector":
        """Loads Autoencoder model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu", weights_only=False)
        instance = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            warning_hi_threshold=checkpoint["warning_hi_threshold"],
            critical_hi_threshold=checkpoint["critical_hi_threshold"],
            device=device
        )
        instance.net.load_state_dict(checkpoint["state_dict"])
        instance.feature_names = checkpoint["feature_names"]
        instance.reconstruction_threshold = checkpoint["reconstruction_threshold"]
        instance.mse_min = checkpoint["mse_min"]
        instance.mse_max = checkpoint["mse_max"]
        instance.is_fitted = checkpoint["is_fitted"]
        instance.net.eval()
        return instance
