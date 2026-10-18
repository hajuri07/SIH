from .telemetry import router as telemetry_router
from .prediction import router as prediction_router
from .anomaly import router as anomaly_router
from .explainability import router as explainability_router
from .simulation import router as simulation_router
from .health import router as health_router
from .alerts import router as alerts_router
from .maintenance import router as maintenance_router

__all__ = [
    "telemetry_router",
    "prediction_router",
    "anomaly_router",
    "explainability_router",
    "simulation_router",
    "health_router",
    "alerts_router",
    "maintenance_router"
]

