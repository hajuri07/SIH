"""
FastAPI Microservice Entrypoint for SIH Turbofan Predictive Maintenance & RUL System.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

from .config import settings
from .db import init_db
from .services import registry
from .kafka_client import kafka_manager
from .ws_manager import ws_manager
from .routers import (
    telemetry_router,
    prediction_router,
    anomaly_router,
    explainability_router,
    simulation_router,
    health_router,
    alerts_router,
    maintenance_router
)


async def handle_telemetry_inference(packet: dict):
    """
    Message handler invoked when telemetry arrives via Kafka or direct stream.
    Triggers automated RUL inference and anomaly detection, persists predictions,
    and publishes alerts when health thresholds are breached.
    """
    try:
        unit_nr = packet.get("unit_nr")
        time_cycles = packet.get("time_cycles")
        if unit_nr is None or time_cycles is None:
            return

        import pandas as pd
        from .db import get_recent_telemetry, save_prediction, save_alert, save_telemetry

        recent = await get_recent_telemetry(unit_nr, limit=30)
        # If not already recorded in DB, persist it
        if not recent or recent[-1].get("time_cycles") != time_cycles:
            settings_dict = packet.get("settings", {})
            sensors = packet.get("sensors", {})
            if sensors:
                await save_telemetry(unit_nr, time_cycles, settings_dict, sensors)
                recent = await get_recent_telemetry(unit_nr, limit=30)

        if not recent:
            return

        df_recent = pd.DataFrame(recent)
        if registry.is_loaded and registry.feature_engineer:
            rul_pred, ci = registry.predict_rul("auto", df_recent)
            is_anom, hi, status, lead_time = registry.detect_anomaly(df_recent)
            model_name = registry.selected_best_model.upper()
            await save_prediction(unit_nr, time_cycles, model_name, rul_pred, hi, status)

            # Broadcast live health update over WebSocket
            update_payload = {
                "event": "HEALTH_UPDATE",
                "unit_nr": unit_nr,
                "time_cycles": time_cycles,
                "predicted_rul": round(rul_pred, 2),
                "health_index": round(hi, 3),
                "status": status,
                "confidence_interval": ci,
                "lead_time_cycles": lead_time
            }
            await ws_manager.broadcast_to_unit(unit_nr, update_payload)

            if status in ("WARNING", "CRITICAL") or rul_pred <= 25.0:
                alert_level = "CRITICAL" if (status == "CRITICAL" or rul_pred <= 25.0) else "WARNING"
                alert_msg = (
                    f"Automated Alert on Engine {unit_nr} at cycle {time_cycles}: "
                    f"Status {alert_level}, Predicted RUL {round(rul_pred, 1)} cycles, Health Index {round(hi, 3)}."
                )
                await save_alert(unit_nr, time_cycles, alert_level, hi, alert_msg)
                await kafka_manager.send_alert({
                    "unit_nr": unit_nr,
                    "time_cycles": time_cycles,
                    "alert_level": alert_level,
                    "health_index": hi,
                    "predicted_rul": rul_pred,
                    "message": alert_msg
                })
                await ws_manager.broadcast_to_fleet({
                    "event": "ALERT_TRIGGERED",
                    "unit_nr": unit_nr,
                    "time_cycles": time_cycles,
                    "alert_level": alert_level,
                    "health_index": round(hi, 3),
                    "predicted_rul": round(rul_pred, 1),
                    "message": alert_msg
                })
    except Exception as e:
        import logging
        logging.getLogger("vayupulse.consumer").error(f"Error in telemetry inference trigger: {e}")


# Register telemetry inference handler so direct streaming & tests can dispatch
kafka_manager.register_handler(handle_telemetry_inference)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("[INFO] Initializing Database...")
    await init_db()
    print("[INFO] Loading ML/DL Prognostic Models into memory...")
    registry.load_models()
    print("[INFO] Starting Kafka Streaming Manager...")
    await kafka_manager.start()
    print("[INFO] Starting Kafka Consumer & Telemetry Inference Trigger...")
    await kafka_manager.start_consumer(handle_telemetry_inference)
    yield
    # Shutdown actions
    print("[INFO] Stopping Kafka Streaming Manager...")
    await kafka_manager.stop()
    print("[INFO] Shutting down application services...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production Predictive Maintenance & RUL Prognostics Microservice for NASA C-MAPSS Engines (Smart India Hackathon).",
    lifespan=lifespan
)

# Enable CORS for Streamlit / Frontend / External Integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(telemetry_router, prefix=settings.API_V1_STR)
app.include_router(prediction_router, prefix=settings.API_V1_STR)
app.include_router(anomaly_router, prefix=settings.API_V1_STR)
app.include_router(explainability_router, prefix=settings.API_V1_STR)
app.include_router(simulation_router, prefix=settings.API_V1_STR)
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(alerts_router, prefix=settings.API_V1_STR)
app.include_router(maintenance_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["System Health"])
async def health_check():
    """
    Returns actual readiness status for prognostic models, database, and Kafka streaming bus.
    """
    from .db import check_db_readiness
    db_readiness = await check_db_readiness()
    model_readiness = registry.get_readiness()
    kafka_readiness = kafka_manager.get_readiness()

    # Determine composite health status
    if not db_readiness.get("ready"):
        overall_status = "UNHEALTHY"
    elif not model_readiness.get("ready"):
        overall_status = "DEGRADED"
    else:
        overall_status = "HEALTHY"

    benchmark_path = settings.ARTIFACTS_DIR / "benchmark_metrics.json"
    benchmark = {}
    if benchmark_path.exists():
        with open(benchmark_path, "r") as f:
            benchmark = json.load(f)

    return {
        "status": overall_status,
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "readiness": {
            "database": db_readiness,
            "models": model_readiness,
            "kafka": kafka_readiness
        },
        "models_loaded": model_readiness["models_loaded"],
        "latest_benchmarks": benchmark.get("models", {})
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to SIH Turbofan Predictive Maintenance API",
        "docs_url": "/docs",
        "health_url": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
