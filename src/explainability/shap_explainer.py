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


SENSOR_DESCRIPTIONS = {
    "s_2": "LPC Outlet Temp (T24)",
    "s_3": "HPC Outlet Temp (T30)",
    "s_4": "LPT Outlet Temp (T50)",
    "s_7": "HPC Outlet Pressure (P30)",
    "s_8": "Physical Fan Speed (Nf)",
    "s_9": "Physical Core Speed (Nc)",
    "s_11": "Static HPC Pressure (Ps30)",
    "s_12": "Fuel Flow Ratio (phi)",
    "s_13": "Corrected Fan Speed (NRf)",
    "s_14": "Corrected Core Speed (NRc)",
    "s_15": "Bypass Ratio (BPR)",
    "s_17": "Bleed Enthalpy (htBleed)",
    "s_20": "HPT Coolant Bleed (W31)",
    "s_21": "LPT Coolant Bleed (W32)"
}


def format_feature_name(feat: str) -> str:
    """Provides human-readable physical descriptions for engineered feature names."""
    for s_id, s_desc in SENSOR_DESCRIPTIONS.items():
        if feat.startswith(s_id):
            suffix = feat[len(s_id):]
            if "_roll_mean_" in suffix:
                win = suffix.split("_roll_mean_")[-1]
                return f"{s_desc} [{win}-cycle rolling mean]"
            elif "_roll_std_" in suffix:
                win = suffix.split("_roll_std_")[-1]
                return f"{s_desc} [{win}-cycle rolling std]"
            elif "_diff_" in suffix:
                win = suffix.split("_diff_")[-1]
                return f"{s_desc} [{win}-cycle drift]"
            return s_desc
    return feat


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
        exp_val = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(exp_val, (np.ndarray, list)):
            self.base_value = float(np.asarray(exp_val).ravel()[0])
        else:
            self.base_value = float(exp_val)

    def explain_instance(self, feature_row: pd.Series, top_k: int = 8) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single engine cycle.
        Returns top positive (extending RUL) and negative (reducing RUL) features.
        """
        df_row = pd.DataFrame([feature_row[self.feature_names]])
        shap_values = self.explainer(df_row)
        if hasattr(shap_values, "values"):
            values = np.asarray(shap_values.values)[0]
        else:
            values = np.asarray(shap_values)[0]
        if values.ndim > 1:
            values = values.ravel()

        contributions = []
        for feat, val, shap_val in zip(self.feature_names, df_row.values[0], values):
            contributions.append({
                "feature": feat,
                "feature_name": format_feature_name(feat),
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
