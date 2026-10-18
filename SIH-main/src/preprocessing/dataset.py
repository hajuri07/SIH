"""
PyTorch Dataset Builder for Sequence-based RUL Prognostics (LSTM / TCN).
Handles sliding window sequence extraction, zero-padding for short sequences,
and batch collation.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CMAPSSSequenceDataset(Dataset):
    """
    Sliding window PyTorch dataset for sequential time-series modeling.
    Outputs:
        X: Tensor of shape (sequence_length, n_features)
        y: Target RUL at the end of the sequence window (scalar float)
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        unit_indices: Optional[List[Tuple[int, int]]] = None,
        sequence_length: int = 30
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        self.sequence_length = sequence_length
        self.windows = []

        if unit_indices is not None:
            # Generate sliding windows within each engine unit independently
            for start_idx, end_idx in unit_indices:
                unit_len = end_idx - start_idx
                if unit_len < sequence_length:
                    # Single padded window for short runs
                    self.windows.append((start_idx, end_idx, end_idx - 1))
                else:
                    for i in range(start_idx + sequence_length, end_idx + 1):
                        self.windows.append((i - sequence_length, i, i - 1))
        else:
            # Single sequence or direct sequence array
            for i in range(len(features)):
                self.windows.append((max(0, i - sequence_length + 1), i + 1, i))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end, target_idx = self.windows[idx]
        seq = self.features[start:end]

        # Left-pad with zeros if sequence length < required sequence_length
        if seq.shape[0] < self.sequence_length:
            pad_len = self.sequence_length - seq.shape[0]
            padding = torch.zeros((pad_len, seq.shape[1]), dtype=torch.float32)
            seq = torch.cat([padding, seq], dim=0)

        target = self.targets[target_idx] if self.targets is not None else torch.tensor(0.0)
        return seq, target


def build_sequence_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
    is_train: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Extracts contiguous arrays and engine unit boundaries from a DataFrame.
    """
    df_sorted = df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)
    features = df_sorted[feature_cols].values
    targets = df_sorted["rul"].values if "rul" in df_sorted.columns else np.zeros(len(df_sorted))

    # Identify unit boundaries [start_idx, end_idx)
    unit_indices = []
    current_unit = None
    start_idx = 0

    for idx, unit in enumerate(df_sorted["unit_nr"].values):
        if current_unit is None:
            current_unit = unit
            start_idx = idx
        elif unit != current_unit:
            unit_indices.append((start_idx, idx))
            current_unit = unit
            start_idx = idx
    unit_indices.append((start_idx, len(df_sorted)))

    return features, targets, unit_indices


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for training and validation sequences.
    """
    train_feat, train_targ, train_units = build_sequence_data(train_df, feature_cols, sequence_length)
    val_feat, val_targ, val_units = build_sequence_data(val_df, feature_cols, sequence_length)

    train_ds = CMAPSSSequenceDataset(train_feat, train_targ, train_units, sequence_length)
    val_ds = CMAPSSSequenceDataset(val_feat, val_targ, val_units, sequence_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
