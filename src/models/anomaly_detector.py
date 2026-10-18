"""
Unsupervised Anomaly Detection Module using Isolation Forest.
Calculates continuous Health Index (HI), flags health states (HEALTHY, WARNING, CRITICAL),
and computes alert reliability metrics (False-Alert Rate, Detection Lead Time, Missed Failures).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

from .evaluator import evaluate_anomaly_detector


class EngineAnomalyDetector:
    """
    Isolation Forest anomaly detector fitted on nominal early-life operational cycles.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
        warning_hi_threshold: float = 0.65,
        critical_hi_threshold: float = 0.35
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.warning_hi_threshold = warning_hi_threshold
        self.critical_hi_threshold = critical_hi_threshold

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names: List[str] = []
        self.score_min: float = -1.0
        self.score_max: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X_baseline: pd.DataFrame) -> None:
        """
        Fits Isolation Forest strictly on healthy baseline telemetry (e.g. cycles <= 30).
        """
        self.feature_names = list(X_baseline.columns)
        self.model.fit(X_baseline)

        # Baseline decision function scores to normalize Health Index to [0, 1]
        scores = self.model.decision_function(X_baseline)
        self.score_min = float(scores.min() - 0.1)
        self.score_max = float(scores.max())
        self.is_fitted = True

    def compute_health_index(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes continuous Health Index HI in [0.0, 1.0].
        Pristine baseline = 1.0, severe anomaly = 0.0.
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet.")

        raw_scores = self.model.decision_function(X[self.feature_names])
        # Map raw anomaly score to [0, 1]
        denom = max(1e-6, (self.score_max - self.score_min))
        hi = (raw_scores - self.score_min) / denom
        return np.clip(hi, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns:
            is_anomaly: 1 if anomalous, 0 if nominal
            health_indices: float in [0.0, 1.0]
            status_labels: "HEALTHY", "WARNING", or "CRITICAL"
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet.")

        # Isolation forest: -1 for anomaly, 1 for inlier
        raw_preds = self.model.predict(X[self.feature_names])
        is_anomaly = (raw_preds == -1).astype(int)
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
        """Saves model state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "warning_hi_threshold": self.warning_hi_threshold,
            "critical_hi_threshold": self.critical_hi_threshold,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "EngineAnomalyDetector":
        """Loads model from disk."""
        data = joblib.load(filepath)
        detector = cls(
            contamination=data["contamination"],
            n_estimators=data["n_estimators"],
            warning_hi_threshold=data["warning_hi_threshold"],
            critical_hi_threshold=data["critical_hi_threshold"]
        )
        detector.model = data["model"]
        detector.feature_names = data["feature_names"]
        detector.score_min = data["score_min"]
        detector.score_max = data["score_max"]
        detector.is_fitted = data["is_fitted"]
        return detector
