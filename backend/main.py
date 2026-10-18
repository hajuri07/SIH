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
from .routers import (
    telemetry_router,
    prediction_router,
    anomaly_router,
    explainability_router,
    simulation_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("[INFO] Initializing Database...")
    await init_db()
    print("[INFO] Loading ML/DL Prognostic Models into memory...")
    registry.load_models()
    yield
    # Shutdown actions
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


@app.get("/health", tags=["System Health"])
async def health_check():
    """
    Returns system health status and loaded model inventory.
    """
    benchmark_path = settings.ARTIFACTS_DIR / "benchmark_metrics.json"
    benchmark = {}
    if benchmark_path.exists():
        with open(benchmark_path, "r") as f:
            benchmark = json.load(f)

    return {
        "status": "HEALTHY",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "models_loaded": {
            "xgboost": registry.xgboost_model is not None,
            "lstm": registry.lstm_model is not None,
            "tcn": registry.tcn_model is not None,
            "anomaly_detector": registry.anomaly_detector is not None,
            "shap_explainer": registry.shap_explainer is not None
        },
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
