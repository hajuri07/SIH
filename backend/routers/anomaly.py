"""
Anomaly Detection & Health Index Router (Arena 3).
"""

from fastapi import APIRouter, HTTPException
import pandas as pd

from ..schemas import AnomalyDetectRequest, AnomalyDetectResponse
from ..services import registry
from ..db import get_recent_telemetry

router = APIRouter(prefix="/detect", tags=["Anomaly Detection"])


@router.post("/anomaly", response_model=AnomalyDetectResponse)
async def detect_engine_anomaly(req: AnomalyDetectRequest):
    """
    Evaluates current engine sensor readings with the Isolation Forest model.
    Returns anomaly indicator, continuous Health Index, and severity level.
    """
    if not registry.is_loaded:
        registry.load_models()

    if not registry.anomaly_detector or not registry.feature_engineer:
        raise HTTPException(
            status_code=503,
            detail="Anomaly detector model not loaded. Run training pipeline first."
        )

    # Reconstruct single cycle dataframe
    cycle_data = {
        "unit_nr": req.unit_nr,
        "time_cycles": 1,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
        **req.telemetry
    }
    df = pd.DataFrame([cycle_data])

    try:
        is_anom, hi, status = registry.detect_anomaly(df)

        if status == "CRITICAL":
            msg = f"CRITICAL ANOMALY: Severe sensor pattern deviation detected on Engine {req.unit_nr}."
            lead_time = 15.0
        elif status == "WARNING":
            msg = f"WARNING: Early thermodynamic drift observed on Engine {req.unit_nr}."
            lead_time = 45.0
        else:
            msg = f"NOMINAL: Engine {req.unit_nr} operating within healthy baseline envelope."
            lead_time = None

        return AnomalyDetectResponse(
            unit_nr=req.unit_nr,
            is_anomaly=is_anom,
            health_index=round(hi, 3),
            status=status,
            lead_time_cycles=lead_time,
            alert_message=msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")
