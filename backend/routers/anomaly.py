from fastapi import APIRouter, HTTPException
import pandas as pd

from ..schemas import AnomalyDetectRequest, AnomalyDetectResponse
from ..services import registry
from ..db import get_recent_telemetry, save_alert
from ..kafka_client import kafka_manager

router = APIRouter(prefix="/detect", tags=["Anomaly Detection"])


@router.post("/anomaly", response_model=AnomalyDetectResponse)
async def detect_engine_anomaly(req: AnomalyDetectRequest):
    """
    Evaluates current engine sensor readings with the Isolation Forest model.
    Returns anomaly indicator, continuous Health Index, severity level, and dynamic lead time.
    Persists alert records to the database when abnormal health states are flagged.
    """
    if not registry.is_loaded:
        registry.load_models()

    if not registry.anomaly_detector or not registry.feature_engineer:
        raise HTTPException(
            status_code=503,
            detail="Anomaly detector model not loaded. Run training pipeline first."
        )

    # Reconstruct telemetry window (query DB for recent context or use current cycle)
    recent_history = await get_recent_telemetry(req.unit_nr, limit=20)
    current_cycle = (recent_history[-1]["time_cycles"] + 1) if recent_history else 1

    current_row = {
        "unit_nr": req.unit_nr,
        "time_cycles": current_cycle,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
        **req.telemetry
    }

    if recent_history:
        combined_rows = recent_history + [current_row]
        df = pd.DataFrame(combined_rows)
    else:
        df = pd.DataFrame([current_row])

    try:
        is_anom, hi, status, lead_time = registry.detect_anomaly(df, method=req.method)
        detector_name = "AUTOENCODER" if req.method.lower() in ("autoencoder", "ae") else "ISOLATION_FOREST"

        if status == "CRITICAL":
            msg = f"CRITICAL ANOMALY: Severe sensor pattern deviation detected on Engine {req.unit_nr}."
        elif status == "WARNING":
            msg = f"WARNING: Early thermodynamic drift observed on Engine {req.unit_nr}."
        else:
            msg = f"NOMINAL: Engine {req.unit_nr} operating within healthy baseline envelope."

        # Persist alert if abnormal
        if status in ("WARNING", "CRITICAL"):
            await save_alert(req.unit_nr, current_cycle, status, hi, msg)
            await kafka_manager.send_alert({
                "unit_nr": req.unit_nr,
                "time_cycles": current_cycle,
                "alert_level": status,
                "health_index": hi,
                "lead_time_cycles": lead_time,
                "message": msg
            })

        return AnomalyDetectResponse(
            unit_nr=req.unit_nr,
            is_anomaly=is_anom,
            health_index=round(hi, 3),
            status=status,
            detector_used=detector_name,
            lead_time_cycles=lead_time,
            alert_message=msg
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")
