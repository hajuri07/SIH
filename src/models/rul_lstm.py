"""
PyTorch Stacked LSTM Architecture for Sequential RUL Prediction.
Captures non-linear temporal degradation trends across sliding windows of sensor readings.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from .evaluator import evaluate_rul_predictions


class LSTMRULNet(nn.Module):
    """
    Stacked LSTM network with Dropout, LayerNorm, and multi-layer regression head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Take representation from the final time step
        last_step = out[:, -1, :]
        normed = self.layer_norm(last_step)
        rul = self.regressor(normed)
        return rul.squeeze(-1)


class RULLSTMModel:
    """
    High-level training and inference wrapper for the PyTorch LSTM RUL model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 0.001,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LSTMRULNet(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20
    ) -> Dict[str, float]:
        """
        Trains the LSTM model with MSE loss and AdamW optimizer.
        """
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.net(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

        # Compute evaluation metrics
        train_metrics = self.evaluate(train_loader)
        metrics = {f"train_{k}": v for k, v in train_metrics.items()}

        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        return metrics

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Generates predictions across a DataLoader.
        """
        self.net.eval()
        preds_list = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                preds = self.net(X_batch).cpu().numpy()
                preds_list.extend(preds)
        return np.clip(np.array(preds_list), a_min=0.0, a_max=None)

    def predict_single_sequence(self, seq_array: np.ndarray) -> float:
        """
        Predicts RUL for a single sequence window (shape: [seq_len, input_dim]).
        """
        self.net.eval()
        with torch.no_grad():
            tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.net(tensor).item()
        return max(0.0, float(pred))

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates predictions against true targets.
        """
        self.net.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                preds = self.net(X_batch).cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(y_batch.numpy())

        return evaluate_rul_predictions(np.array(all_trues), np.clip(np.array(all_preds), 0, None))

    def save(self, filepath: Path) -> None:
        """Saves model weights and architecture state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr
        }, filepath)

    @classmethod
    def load(cls, filepath: Path, device: Optional[str] = None) -> "RULLSTMModel":
        """Loads model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            lr=checkpoint["lr"],
            device=device
        )
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        return model
