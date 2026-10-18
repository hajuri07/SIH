"""
SHAP Explainability Module for Turbofan Prognostics.
Generates local sensor attributions and global feature rankings using TreeSHAP on XGBoost models.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path


class CMAPSSSHAPExplainer:
    """
    Computes Shapley additive explanations for RUL predictions.
    Explains which physical sensors are driving early warnings or low RUL estimates.
    """

    def __init__(self, model_obj, feature_names: List[str]):
        self.model_obj = model_obj
        self.feature_names = feature_names
        # Use underlying XGBoost booster
        raw_model = getattr(model_obj, "model", model_obj)
        self.explainer = shap.TreeExplainer(raw_model)
        if hasattr(self.explainer, "expected_value"):
            exp_val = self.explainer.expected_value
            self.base_value = float(np.atleast_1d(exp_val)[0])
        else:
            self.base_value = 0.0

    def explain_instance(self, feature_row: pd.Series, top_k: int = 8) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single engine cycle.
        Returns top positive (extending RUL) and negative (reducing RUL) features.
        """
        df_row = pd.DataFrame([feature_row[self.feature_names]])
        shap_values = self.explainer(df_row)
        values = shap_values.values[0]

        contributions = []
        for feat, val, shap_val in zip(self.feature_names, df_row.values[0], values):
            contributions.append({
                "feature": feat,
                "feature_value": round(float(val), 4),
                "shap_value": round(float(shap_val), 4),
                "impact": "DEGRADATION_DRIVER" if shap_val < 0 else "HEALTH_EXTENDER"
            })

        # Sort by absolute magnitude of impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": round(self.base_value, 2),
            "predicted_offset": round(float(np.sum(values)), 2),
            "top_contributions": contributions[:top_k]
        }

    def explain_global(self, sample_df: pd.DataFrame, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Computes mean absolute SHAP values across a representative sample of cycles.
        """
        shap_values = self.explainer.shap_values(sample_df[self.feature_names])
        mean_abs = np.mean(np.abs(shap_values), axis=0)

        ranking = []
        for feat, score in zip(self.feature_names, mean_abs):
            ranking.append({
                "feature": feat,
                "mean_abs_shap": round(float(score), 4)
            })

        ranking.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        return ranking[:top_k]

    def save(self, filepath: Path) -> None:
        """Saves explainer state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_names": self.feature_names,
            "base_value": self.base_value
        }, filepath)
