"""
RUL Prediction Router.
Supports XGBoost, PyTorch LSTM, and Temporal Convolutional Network (TCN).
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
import pandas as pd

from ..schemas import PredictionRequest, PredictionResponse
from ..services import registry
from ..db import get_recent_telemetry, save_prediction, save_alert, get_prediction_history
from ..kafka_client import kafka_manager

router = APIRouter(prefix="/predict", tags=["Prognostics & RUL Prediction"])


@router.post("/rul", response_model=PredictionResponse)
async def predict_engine_rul(req: PredictionRequest):
    """
    Predicts Remaining Useful Life (RUL) in operational flight cycles for a given engine.
    Accepts explicit recent telemetry or queries the engine's latest historical window from DB.
    Provides calibrated empirical prediction intervals based on held-out residual distributions.
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
        rul_pred, ci = registry.predict_rul(req.model_type, df_recent)
        is_anom, hi, status, lead_time = registry.detect_anomaly(df_recent)

        # Record prediction to database
        current_cycle = int(df_recent["time_cycles"].iloc[-1])
        model_name = req.model_type.upper() if req.model_type.lower() != "auto" else registry.selected_best_model.upper()
        await save_prediction(req.unit_nr, current_cycle, model_name, rul_pred, hi, status)

        alert_triggered = status in ("WARNING", "CRITICAL") or rul_pred <= 25.0

        # Persist alert if triggered
        if alert_triggered:
            alert_level = "CRITICAL" if (status == "CRITICAL" or rul_pred <= 25.0) else "WARNING"
            alert_msg = (
                f"Engine {req.unit_nr} alert ({alert_level}): Predicted RUL is {round(rul_pred, 1)} cycles "
                f"with Health Index {round(hi, 3)}."
            )
            await save_alert(req.unit_nr, current_cycle, alert_level, hi, alert_msg)
            await kafka_manager.send_alert({
                "unit_nr": req.unit_nr,
                "time_cycles": current_cycle,
                "alert_level": alert_level,
                "health_index": hi,
                "predicted_rul": rul_pred,
                "message": alert_msg
            })

        return PredictionResponse(
            unit_nr=req.unit_nr,
            model_used=model_name,
            predicted_rul=round(rul_pred, 2),
            health_index=round(hi, 3),
            status=status,
            alert_triggered=alert_triggered,
            confidence_interval=ci
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/history/{unit_nr}")
async def get_engine_prediction_history(unit_nr: int, limit: int = Query(default=50, ge=1, le=200)):
    """
    Retrieves historical RUL predictions and health index records for an engine unit.
    """
    history = await get_prediction_history(unit_nr, limit=limit)
    return {
        "unit_nr": unit_nr,
        "count": len(history),
        "predictions": history
    }
