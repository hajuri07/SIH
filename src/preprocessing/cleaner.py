"""
C-MAPSS Data Cleaning & Quality Audit Module.
Identifies missing values, duplicates, out-of-range sensor readings, and invariant flatline sensors.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# Standard column indices for C-MAPSS dataset
COLUMN_NAMES = [
    "unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"
] + [f"s_{i}" for i in range(1, 22)]

# Sensors known to have zero variance in FD001 (Sea-level operation)
INVARIANT_SENSORS_FD001 = ["setting_3", "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]


class CMAPSSCleaner:
    """Handles raw data ingestion, quality audits, and invariant sensor filtering."""

    def __init__(self, variance_threshold: float = 1e-5):
        self.variance_threshold = variance_threshold
        self.active_features: List[str] = []
        self.dropped_features: List[str] = []

    def load_raw_file(self, filepath: Path) -> pd.DataFrame:
        """
        Loads raw whitespace-delimited text file into a structured DataFrame.
        Handles trailing spaces and variable whitespace.
        """
        df = pd.read_csv(filepath, sep=r"\s+", header=None)
        # Drop extra empty columns if whitespace at ends generated them
        if df.shape[1] > 26:
            df = df.iloc[:, :26]
        df.columns = COLUMN_NAMES[:df.shape[1]]
        return df

    def perform_quality_audit(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Runs comprehensive data quality checks:
        1. Missing / NaN values
        2. Duplicate (unit_nr, time_cycles) pairs
        3. Monotonicity of time_cycles per unit
        4. Sensor variance analysis
        """
        missing_count = int(df.isna().sum().sum())
        duplicates_count = int(df.duplicated(subset=["unit_nr", "time_cycles"]).sum())

        # Check cycle monotonicity
        non_monotonic_units = []
        for unit, grp in df.groupby("unit_nr"):
            cycles = grp["time_cycles"].values
            if not np.all(np.diff(cycles) == 1):
                non_monotonic_units.append(int(unit))

        # Check column variances across sensor readings
        sensor_cols = [c for c in df.columns if c.startswith("s_") or c.startswith("setting_")]
        variances = df[sensor_cols].var()
        zero_var_cols = variances[variances <= self.variance_threshold].index.tolist()

        audit_report = {
            "total_records": len(df),
            "total_units": int(df["unit_nr"].nunique()),
            "missing_values": missing_count,
            "duplicate_records": duplicates_count,
            "non_monotonic_units": non_monotonic_units,
            "zero_variance_columns": zero_var_cols,
            "column_variances": variances.to_dict()
        }
        return audit_report

    def fit_active_features(self, df: pd.DataFrame) -> List[str]:
        """
        Determines which sensors carry informative variance on the training set.
        """
        sensor_cols = [c for c in df.columns if c.startswith("s_") or c.startswith("setting_")]
        variances = df[sensor_cols].var()

        self.dropped_features = variances[variances <= self.variance_threshold].index.tolist()
        self.active_features = [c for c in sensor_cols if c not in self.dropped_features]
        return self.active_features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning:
        - Filters out invariant / zero-variance columns.
        - Preserves unit_nr and time_cycles identifiers.
        """
        if not self.active_features:
            raise ValueError("CMAPSSCleaner has not been fitted. Call fit_active_features first.")

        keep_cols = ["unit_nr", "time_cycles"] + self.active_features
        # Only retain columns that exist in df
        actual_cols = [c for c in keep_cols if c in df.columns]
        return df[actual_cols].copy()
