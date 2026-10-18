"""
Feature Engineering Module for Turbofan Engine Prognostics.
Generates piecewise linear RUL targets, multi-scale rolling temporal statistics,
cumulative extrema, and training-fitted normalization.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path


class CMAPSSFeatureEngineer:
    """
    Transforms clean C-MAPSS telemetry into rich predictive feature vectors.
    """

    def __init__(
        self,
        max_rul: int = 125,
        window_sizes: List[int] = (5, 10, 20),
        scaler: Optional[MinMaxScaler] = None
    ):
        self.max_rul = max_rul
        self.window_sizes = list(window_sizes)
        self.scaler = scaler or MinMaxScaler(feature_range=(0, 1))
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

    def add_rul_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes piecewise linear Remaining Useful Life (RUL) target for run-to-failure engines.
        Clipped at max_rul (default 125 cycles).
        """
        df = df.copy()
        # Compute maximum cycle reached per unit
        max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
        # True linear RUL
        df["rul_raw"] = max_cycles - df["time_cycles"]
        # Piecewise linear clipping
        df["rul"] = df["rul_raw"].clip(upper=self.max_rul)
        return df

    def compute_rolling_features(
        self, df: pd.DataFrame, sensor_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calculates multi-scale rolling statistics per engine unit:
        - Rolling mean (short, medium, long term)
        - Rolling standard deviation (vibration / turbulence increase)
        - Differential from rolling mean (instantaneous drift)
        """
        df = df.copy().sort_values(["unit_nr", "time_cycles"])
        grouped = df.groupby("unit_nr")[sensor_cols]

        engineered_dfs = [df]

        for window in self.window_sizes:
            # Rolling mean
            roll_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(drop=True)
            roll_mean.columns = [f"{col}_roll_mean_{window}" for col in sensor_cols]

            # Rolling std
            roll_std = grouped.rolling(window=window, min_periods=1).std().fillna(0).reset_index(drop=True)
            roll_std.columns = [f"{col}_roll_std_{window}" for col in sensor_cols]

            # Sensor deviation from its rolling mean
            diff = df[sensor_cols].values - roll_mean.values
            diff_df = pd.DataFrame(diff, columns=[f"{col}_diff_{window}" for col in sensor_cols])

            engineered_dfs.extend([roll_mean, roll_std, diff_df])

        result_df = pd.concat(engineered_dfs, axis=1)
        return result_df

    def fit_transform(
        self, df: pd.DataFrame, sensor_cols: List[str], is_train: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Executes full feature engineering pipeline:
        1. Adds RUL target if training data
        2. Computes temporal rolling features
        3. Fits MinMaxScaler on training features
        4. Returns normalized feature dataframe and feature column names
        """
        processed_df = self.add_rul_target(df) if is_train else df.copy()

        # Generate rolling temporal features
        featured_df = self.compute_rolling_features(processed_df, sensor_cols)

        # Non-feature identifier and target columns
        non_feature_cols = {"unit_nr", "time_cycles", "rul", "rul_raw"}
        self.feature_columns = [c for c in featured_df.columns if c not in non_feature_cols]

        # Fit and transform scaler
        scaled_features = self.scaler.fit_transform(featured_df[self.feature_columns])
        scaled_df = featured_df.copy()
        scaled_df[self.feature_columns] = scaled_features

        self.is_fitted = True
        return scaled_df, self.feature_columns

    def transform(
        self, df: pd.DataFrame, sensor_cols: List[str]
    ) -> pd.DataFrame:
        """
        Transforms testing or streaming telemetry using frozen scaler parameters.
        """
        if not self.is_fitted:
            raise ValueError("CMAPSSFeatureEngineer must be fitted before calling transform.")

        featured_df = self.compute_rolling_features(df, sensor_cols)

        scaled_features = self.scaler.transform(featured_df[self.feature_columns])
        scaled_df = featured_df.copy()
        scaled_df[self.feature_columns] = scaled_features
        return scaled_df

    def save(self, filepath: Path) -> None:
        """Persists fitted scaler and feature configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "max_rul": self.max_rul,
            "window_sizes": self.window_sizes,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "CMAPSSFeatureEngineer":
        """Loads a persisted feature engineer."""
        state = joblib.load(filepath)
        fe = cls(max_rul=state["max_rul"], window_sizes=state["window_sizes"], scaler=state["scaler"])
        fe.feature_columns = state["feature_columns"]
        fe.is_fitted = state["is_fitted"]
        return fe
