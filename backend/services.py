"""
Model Registry Service.
Caches trained machine learning models in memory for fast microsecond API inference.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from .config import settings
from src.models.rul_xgboost import RULXGBoostModel
from src.models.rul_lstm import RULLSTMModel
from src.models.rul_tcn import RULTCNModel
from src.models.anomaly_detector import EngineAnomalyDetector
from src.preprocessing.feature_engineering import CMAPSSFeatureEngineer
from src.explainability.shap_explainer import CMAPSSSHAPExplainer


class ModelRegistry:
    def __init__(self):
        self.xgboost_model: Optional[RULXGBoostModel] = None
        self.lstm_model: Optional[RULLSTMModel] = None
        self.tcn_model: Optional[RULTCNModel] = None
        self.anomaly_detector: Optional[EngineAnomalyDetector] = None
        self.feature_engineer: Optional[CMAPSSFeatureEngineer] = None
        self.shap_explainer: Optional[CMAPSSSHAPExplainer] = None
        self.is_loaded: bool = False

    def load_models(self):
        artifacts = settings.ARTIFACTS_DIR
        try:
            fe_path = artifacts / "feature_engineer.joblib"
            if fe_path.exists():
                self.feature_engineer = CMAPSSFeatureEngineer.load(fe_path)

            xgb_path = artifacts / "xgboost_rul.joblib"
            if xgb_path.exists():
                self.xgboost_model = RULXGBoostModel.load(xgb_path)

            lstm_path = artifacts / "lstm_rul.pt"
            if lstm_path.exists():
                self.lstm_model = RULLSTMModel.load(lstm_path, device="cpu")

            tcn_path = artifacts / "tcn_rul.pt"
            if tcn_path.exists():
                self.tcn_model = RULTCNModel.load(tcn_path, device="cpu")

            anom_path = artifacts / "isolation_forest.joblib"
            if anom_path.exists():
                self.anomaly_detector = EngineAnomalyDetector.load(anom_path)

            if self.xgboost_model and self.feature_engineer:
                self.shap_explainer = CMAPSSSHAPExplainer(
                    self.xgboost_model, self.feature_engineer.feature_columns
                )

            self.is_loaded = True
            print("[INFO] ModelRegistry: All available models loaded into memory.")
        except Exception as e:
            print(f"[WARNING] ModelRegistry load warning: {e}")

    def predict_rul(self, model_type: str, recent_df: pd.DataFrame) -> float:
        """
        Runs RUL prediction on a sequence or tabular feature window.
        """
        if not self.is_loaded or not self.feature_engineer:
            raise RuntimeError("Models not loaded. Train pipeline first.")

        # Identify active raw sensors from feature engineer
        sensor_cols = [
            c for c in recent_df.columns
            if (c.startswith("s_") or c.startswith("setting_"))
            and not ("_roll_" in c or "_diff_" in c)
        ]

        # Feature engineering transformation
        transformed_df = self.feature_engineer.transform(recent_df, sensor_cols)
        feature_cols = self.feature_engineer.feature_columns

        model_type_lower = model_type.lower()
        if model_type_lower == "xgboost" and self.xgboost_model:
            latest_row = transformed_df.iloc[[-1]]
            preds = self.xgboost_model.predict(latest_row)
            return float(preds[0])

        elif model_type_lower == "lstm" and self.lstm_model:
            seq = transformed_df[feature_cols].values[-30:]
            if len(seq) < 30:
                pad = np.zeros((30 - len(seq), len(feature_cols)))
                seq = np.vstack([pad, seq])
            return float(self.lstm_model.predict_single_sequence(seq))

        elif model_type_lower == "tcn" and self.tcn_model:
            seq = transformed_df[feature_cols].values[-30:]
            if len(seq) < 30:
                pad = np.zeros((30 - len(seq), len(feature_cols)))
                seq = np.vstack([pad, seq])
            return float(self.tcn_model.predict_single_sequence(seq))

        else:
            # Fallback to XGBoost if chosen DL model isn't active
            latest_row = transformed_df.iloc[[-1]]
            preds = self.xgboost_model.predict(latest_row)
            return float(preds[0])

    def detect_anomaly(self, recent_df: pd.DataFrame):
        """
        Calculates Health Index and Anomaly status.
        """
        if not self.anomaly_detector or not self.feature_engineer:
            raise RuntimeError("Anomaly detector not initialized.")

        sensor_cols = [
            c for c in recent_df.columns
            if (c.startswith("s_") or c.startswith("setting_"))
            and not ("_roll_" in c or "_diff_" in c)
        ]
        transformed_df = self.feature_engineer.transform(recent_df, sensor_cols)
        feature_cols = self.feature_engineer.feature_columns

        latest_row = transformed_df.iloc[[-1]][feature_cols]
        is_anom, hi, status = self.anomaly_detector.predict(latest_row)
        return bool(is_anom[0]), float(hi[0]), status[0]


registry = ModelRegistry()
