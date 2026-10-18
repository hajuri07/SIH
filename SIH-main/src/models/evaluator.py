"""
Evaluation Metrics for Turbofan Engine Prognostics & Health Management.
Includes standard regression metrics (RMSE, MAE, R2), official NASA C-MAPSS asymmetric score,
and Anomaly Alert Lead-Time & Accuracy metrics.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA C-MAPSS competition asymmetric scoring function.
    Late predictions (predicting engine will last longer than reality) are penalized
    exponentially harder than early predictions due to catastrophic flight risk.

    d_i = y_pred_i - y_true_i
    s_i = exp(-d_i / 13) - 1  if d_i < 0  (early alert)
    s_i = exp(d_i / 10) - 1   if d_i >= 0 (late alert)
    Score = sum(s_i)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true

    score = 0.0
    for d in diff:
        if d < 0:
            score += np.exp(-d / 13.0) - 1.0
        else:
            score += np.exp(d / 10.0) - 1.0
    return float(score)


def evaluate_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes a comprehensive benchmark report for RUL regressors:
    - RMSE
    - MAE
    - R2 Score
    - NASA Asymmetric Score
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    nasa_score = compute_nasa_score(y_true, y_pred)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "nasa_score": round(nasa_score, 2)
    }


def evaluate_anomaly_detector(
    df: pd.DataFrame,
    is_anomaly_col: str = "is_anomaly",
    baseline_cycle_limit: int = 30,
    critical_rul_threshold: int = 25
) -> Dict[str, float]:
    """
    Evaluates unsupervised anomaly detection reliability:
    1. False-Alert Rate (FAR): % of baseline healthy cycles (cycles <= 30) flagged.
    2. Detection Lead Time (DLT): Average cycles remaining before failure when first alert triggers.
    3. Missed Failure Rate (MFR): % of engines that breached critical threshold with no alert.
    """
    # 1. False Alert Rate on healthy baseline cycles
    baseline_records = df[df["time_cycles"] <= baseline_cycle_limit]
    if len(baseline_records) > 0:
        false_alerts = baseline_records[is_anomaly_col].sum()
        far = float(false_alerts / len(baseline_records)) * 100.0
    else:
        far = 0.0

    # 2. Lead Time & Missed Failures per unit
    lead_times = []
    missed_count = 0
    total_units = df["unit_nr"].nunique()

    for unit, grp in df.groupby("unit_nr"):
        grp = grp.sort_values("time_cycles")
        max_cycle = grp["time_cycles"].max()
        anomalies = grp[grp[is_anomaly_col] == 1]

        if len(anomalies) > 0:
            first_alert_cycle = anomalies["time_cycles"].min()
            lead_time = max_cycle - first_alert_cycle
            lead_times.append(lead_time)
        else:
            missed_count += 1

    avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    mfr = float(missed_count / total_units) * 100.0 if total_units > 0 else 0.0

    return {
        "false_alert_rate_pct": round(far, 2),
        "avg_detection_lead_time_cycles": round(avg_lead_time, 2),
        "missed_failure_rate_pct": round(mfr, 2),
        "total_units_evaluated": total_units
    }
