"""
Temporal Convolutional Network (TCN) for Turbofan RUL Prognostics.
Employs 1D dilated causal convolutions, residual connections, and dropout
to capture long-range sensor degradation dynamics with high gradient stability.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from .evaluator import evaluate_rul_predictions


class Chomp1d(nn.Module):
    """Trims trailing padding to ensure strict causal temporal convolution."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Single residual block with two dilated causal convolutions, batch norm, and dropout."""
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNDeepNet(nn.Module):
    """Full TCN network with dilated residual stack and linear regression head."""
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, num_inputs)
        # Conv1d expects: (batch_size, num_inputs, sequence_length)
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        # Final time step feature vector
        last_step = y[:, :, -1]
        rul = self.fc(last_step)
        return rul.squeeze(-1)


class RULTCNModel:
    """Wrapper for training, evaluating, and checkpointing the TCN model."""
    def __init__(
        self,
        input_dim: int,
        num_channels: List[int] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
        lr: float = 0.001,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr

        self.net = TCNDeepNet(input_dim, self.num_channels, kernel_size, dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20
    ) -> Dict[str, float]:
        self.net.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.net(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

        train_metrics = self.evaluate(train_loader)
        metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        return metrics

    def predict_single_sequence(self, seq_array: np.ndarray) -> float:
        self.net.eval()
        with torch.no_grad():
            tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.net(tensor).item()
        return max(0.0, float(pred))

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
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
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "lr": self.lr
        }, filepath)

    @classmethod
    def load(cls, filepath: Path, device: Optional[str] = None) -> "RULTCNModel":
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        model = cls(
            input_dim=checkpoint["input_dim"],
            num_channels=checkpoint["num_channels"],
            kernel_size=checkpoint["kernel_size"],
            dropout=checkpoint["dropout"],
            lr=checkpoint["lr"],
            device=device
        )
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        return model
