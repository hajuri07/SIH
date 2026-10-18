"""
RUL Prediction Router.
Supports XGBoost, PyTorch LSTM, and Temporal Convolutional Network (TCN).
"""

from fastapi import APIRouter, HTTPException
import pandas as pd

from ..schemas import PredictionRequest, PredictionResponse
from ..services import registry
from ..db import get_recent_telemetry, save_prediction

router = APIRouter(prefix="/predict", tags=["Prognostics & RUL Prediction"])


@router.post("/rul", response_model=PredictionResponse)
async def predict_engine_rul(req: PredictionRequest):
    """
    Predicts Remaining Useful Life (RUL) in operational flight cycles for a given engine.
    Accepts explicit recent telemetry or queries the engine's latest historical window from DB.
    """
    if not registry.is_loaded:
        registry.load_models()

    if not registry.is_loaded or not registry.feature_engineer:
        raise HTTPException(
            status_code=503,
            detail="Models are not yet loaded. Ensure the training pipeline has executed."
        )

    # 1. Fetch recent telemetry records
    if req.recent_telemetry and len(req.recent_telemetry) > 0:
        telemetry_rows = req.recent_telemetry
    else:
        telemetry_rows = await get_recent_telemetry(req.unit_nr, limit=30)

    if not telemetry_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry found for Engine Unit {req.unit_nr}. Please ingest telemetry first."
        )

    df_recent = pd.DataFrame(telemetry_rows)

    try:
        rul_pred = registry.predict_rul(req.model_type, df_recent)
        is_anom, hi, status = registry.detect_anomaly(df_recent)

        # Record prediction to database
        current_cycle = int(df_recent["time_cycles"].iloc[-1])
        await save_prediction(req.unit_nr, current_cycle, req.model_type, rul_pred, hi, status)

        alert_triggered = status in ("WARNING", "CRITICAL") or rul_pred <= 25.0

        return PredictionResponse(
            unit_nr=req.unit_nr,
            model_used=req.model_type.upper(),
            predicted_rul=round(rul_pred, 2),
            health_index=round(hi, 3),
            status=status,
            alert_triggered=alert_triggered,
            confidence_interval={
                "lower_bound": round(max(0.0, rul_pred - 12.5), 2),
                "upper_bound": round(rul_pred + 12.5, 2)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
