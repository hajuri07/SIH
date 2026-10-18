"""
Model Registry Service.
Caches trained machine learning models in memory for fast microsecond API inference.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from .config import settings
from src.models.rul_xgboost import RULXGBoostModel
from src.models.rul_lstm import RULLSTMModel
from src.models.rul_tcn import RULTCNModel
from src.models.anomaly_detector import EngineAnomalyDetector
from src.models.autoencoder_detector import AutoencoderAnomalyDetector
from src.preprocessing.feature_engineering import CMAPSSFeatureEngineer
from src.explainability.shap_explainer import CMAPSSSHAPExplainer


class ModelRegistry:
    def __init__(self):
        self.xgboost_model: Optional[RULXGBoostModel] = None
        self.lstm_model: Optional[RULLSTMModel] = None
        self.tcn_model: Optional[RULTCNModel] = None
        self.anomaly_detector: Optional[EngineAnomalyDetector] = None
        self.autoencoder_detector: Optional[AutoencoderAnomalyDetector] = None
        self.feature_engineer: Optional[CMAPSSFeatureEngineer] = None
        self.shap_explainer: Optional[CMAPSSSHAPExplainer] = None
        self.uncertainty_config: Dict[str, Any] = {
            "method": "empirical_residual_conformal",
            "nominal_confidence": 0.90,
            "half_width_cycles": 12.85,
            "residual_std": 10.97
        }
        self.selected_best_model: str = "lstm"
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

            ae_path = artifacts / "autoencoder.pt"
            if ae_path.exists():
                self.autoencoder_detector = AutoencoderAnomalyDetector.load(ae_path, device="cpu")

            unc_path = artifacts / "uncertainty_config.json"
            if unc_path.exists():
                import json
                with open(unc_path, "r") as f:
                    self.uncertainty_config = json.load(f)

            meta_path = artifacts / "metadata.json"
            if meta_path.exists():
                import json
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.selected_best_model = meta.get("selected_best_model", "lstm")

            if self.xgboost_model and self.feature_engineer:
                self.shap_explainer = CMAPSSSHAPExplainer(
                    self.xgboost_model, self.feature_engineer.feature_columns
                )

            self.is_loaded = True
            print(f"[INFO] ModelRegistry: Models loaded. Selected best architecture: {self.selected_best_model.upper()}")
        except Exception as e:
            print(f"[WARNING] ModelRegistry load warning: {e}")

    def predict_rul(self, model_type: str, recent_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Runs RUL prediction on a sequence or tabular feature window.
        Returns:
            predicted_rul: float
            confidence_interval: dict with lower_bound, upper_bound, confidence_level
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
        if model_type_lower in ("auto", "best", "default"):
            model_type_lower = self.selected_best_model.lower()

        pred_rul = 0.0
        if model_type_lower == "xgboost" and self.xgboost_model:
            latest_row = transformed_df.iloc[[-1]]
            preds = self.xgboost_model.predict(latest_row)
            pred_rul = float(preds[0])

        elif model_type_lower == "lstm" and self.lstm_model:
            seq = transformed_df[feature_cols].values[-30:]
            if len(seq) < 30:
                pad = np.zeros((30 - len(seq), len(feature_cols)))
                seq = np.vstack([pad, seq])
            pred_rul = float(self.lstm_model.predict_single_sequence(seq))

        elif model_type_lower == "tcn" and self.tcn_model:
            seq = transformed_df[feature_cols].values[-30:]
            if len(seq) < 30:
                pad = np.zeros((30 - len(seq), len(feature_cols)))
                seq = np.vstack([pad, seq])
            pred_rul = float(self.tcn_model.predict_single_sequence(seq))

        else:
            # Fallback to whatever model is available
            if self.lstm_model:
                seq = transformed_df[feature_cols].values[-30:]
                if len(seq) < 30:
                    pad = np.zeros((30 - len(seq), len(feature_cols)))
                    seq = np.vstack([pad, seq])
                pred_rul = float(self.lstm_model.predict_single_sequence(seq))
            elif self.tcn_model:
                seq = transformed_df[feature_cols].values[-30:]
                if len(seq) < 30:
                    pad = np.zeros((30 - len(seq), len(feature_cols)))
                    seq = np.vstack([pad, seq])
                pred_rul = float(self.tcn_model.predict_single_sequence(seq))
            elif self.xgboost_model:
                latest_row = transformed_df.iloc[[-1]]
                pred_rul = float(self.xgboost_model.predict(latest_row)[0])
            else:
                raise RuntimeError("No trained prognostic models available.")

        # Conformal uncertainty prediction bounds calibrated for specific model
        model_unc = self.uncertainty_config.get("models", {}).get(model_type_lower)
        if not model_unc or not isinstance(model_unc, dict):
            model_unc = self.uncertainty_config

        half_width = float(model_unc.get("half_width_cycles", self.uncertainty_config.get("half_width_cycles", 12.85)))
        confidence_level = float(model_unc.get("nominal_confidence", self.uncertainty_config.get("nominal_confidence", 0.90)))
        ci = {
            "lower_bound": round(max(0.0, pred_rul - half_width), 2),
            "upper_bound": round(pred_rul + half_width, 2),
            "confidence_level": confidence_level,
            "margin_error_cycles": round(half_width, 2),
            "calibrated_model": model_type_lower
        }
        return float(pred_rul), ci

    def get_readiness(self) -> Dict[str, Any]:
        """Reports actual model loading and prognostic readiness."""
        if not self.is_loaded:
            self.load_models()
        models_status = {
            "xgboost": self.xgboost_model is not None,
            "lstm": self.lstm_model is not None,
            "tcn": self.tcn_model is not None,
            "anomaly_detector": self.anomaly_detector is not None,
            "autoencoder": self.autoencoder_detector is not None,
            "shap_explainer": self.shap_explainer is not None,
            "feature_engineer": self.feature_engineer is not None
        }
        is_ready = self.is_loaded and (self.feature_engineer is not None) and any([
            self.xgboost_model is not None,
            self.lstm_model is not None,
            self.tcn_model is not None
        ])
        return {
            "ready": is_ready,
            "selected_best_model": self.selected_best_model,
            "models_loaded": models_status
        }

    def detect_anomaly(
        self, recent_df: pd.DataFrame, method: str = "auto"
    ) -> Tuple[bool, float, str, Optional[float]]:
        """
        Calculates Health Index, Anomaly status, and dynamic detection lead time.
        Supports both Isolation Forest and Deep Autoencoder.
        Returns:
            is_anomaly: bool
            health_index: float in [0.0, 1.0]
            status: "HEALTHY", "WARNING", or "CRITICAL"
            lead_time_cycles: calculated dynamic lead time or None
        """
        # Select detector
        detector = None
        if method.lower() in ("autoencoder", "ae") and self.autoencoder_detector:
            detector = self.autoencoder_detector
        else:
            detector = self.anomaly_detector or self.autoencoder_detector

        if not detector or not self.feature_engineer:
            raise RuntimeError("Anomaly detector not initialized.")

        sensor_cols = [
            c for c in recent_df.columns
            if (c.startswith("s_") or c.startswith("setting_"))
            and not ("_roll_" in c or "_diff_" in c)
        ]
        transformed_df = self.feature_engineer.transform(recent_df, sensor_cols)
        feature_cols = self.feature_engineer.feature_columns

        latest_row = transformed_df.iloc[[-1]][feature_cols]
        is_anom, hi, status = detector.predict(latest_row)

        is_anom_val = bool(is_anom[0])
        hi_val = float(hi[0])
        status_val = status[0]

        # Dynamic Lead Time estimation based on degradation trajectory
        lead_time = None
        if status_val in ("WARNING", "CRITICAL"):
            # Estimate remaining cycles before critical threshold (HI = 0.35) or failure
            # If historical window available, calculate local degradation velocity
            if len(transformed_df) >= 3:
                all_his = detector.compute_health_index(transformed_df[feature_cols])
                delta_hi = all_his[-1] - all_his[0]
                delta_t = max(1, len(all_his) - 1)
                deg_rate = delta_hi / delta_t  # HI drop per cycle
                if deg_rate < -1e-4:
                    cycles_to_zero = (hi_val - 0.0) / abs(deg_rate)
                    lead_time = round(max(1.0, float(cycles_to_zero)), 1)
                else:
                    # Fallback to empirical validation lead time scaled by HI
                    lead_time = round(max(5.0, hi_val * 60.0), 1)
            else:
                lead_time = round(max(5.0, hi_val * 60.0), 1)

        return is_anom_val, hi_val, status_val, lead_time


registry = ModelRegistry()
