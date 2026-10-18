"""
XGBoost Baseline Model for Remaining Useful Life (RUL) Prediction.
Fast, interpretable, gradient-boosted decision trees trained on multi-scale rolling temporal features.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

from .evaluator import evaluate_rul_predictions


class RULXGBoostModel:
    """
    XGBoost Regressor for RUL prediction with hyperparameter tuning and model persistence.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1,
            "objective": "reg:squarederror"
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Trains the XGBoost regressor and calculates validation performance.
        """
        self.feature_names = list(X_train.columns)
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True

        # Training metrics
        train_preds = self.predict(X_train)
        metrics = evaluate_rul_predictions(y_train.values, train_preds)

        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            val_metrics = evaluate_rul_predictions(y_val.values, val_preds)
            metrics = {f"train_{k}": v for k, v in metrics.items()}
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts RUL for given feature matrix. Clips negative predictions to 0.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.model.predict(X[self.feature_names])
        return np.clip(preds, a_min=0.0, a_max=None)

    def get_feature_importances(self, top_n: int = 15) -> List[Tuple[str, float]]:
        """
        Extracts feature importances (gain-based).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        top_features = [
            (self.feature_names[i], float(importances[i]))
            for i in sorted_indices[:top_n]
        ]
        return top_features

    def save(self, filepath: Path) -> None:
        """Saves model weights and feature configuration to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "RULXGBoostModel":
        """Loads model from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.params = data["params"]
        instance.is_fitted = data["is_fitted"]
        return instance
