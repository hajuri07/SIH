"""
Pydantic Request and Response Schemas for FastAPI Endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TelemetryPacket(BaseModel):
    """Single cycle telemetry payload from an engine unit."""
    unit_nr: int = Field(..., description="Engine Unit Identifier", json_schema_extra={"example": 1})
    time_cycles: int = Field(..., description="Flight Cycle Count", json_schema_extra={"example": 45})
    setting_1: float = Field(default=0.0, description="Altitude / Mach command 1")
    setting_2: float = Field(default=0.0, description="Mach command 2")
    setting_3: float = Field(default=100.0, description="Throttle Resolver Angle (TRA)")
    sensors: Dict[str, float] = Field(
        ...,
        description="Sensor readings dictionary (e.g. {'s_2': 642.5, 's_3': 1588.2, ...})",
        json_schema_extra={"example": {"s_2": 642.3, "s_3": 1589.5, "s_4": 1402.1, "s_7": 553.8, "s_8": 2388.1}}
    )


class TelemetryIngestResponse(BaseModel):
    status: str
    message: str
    unit_nr: int
    time_cycles: int
    recorded_at: str


class PredictionRequest(BaseModel):
    unit_nr: int = Field(..., json_schema_extra={"example": 1})
    model_type: str = Field(default="tcn", description="Model architecture: 'xgboost', 'lstm', or 'tcn'")
    recent_telemetry: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of recent cycle readings. If omitted, fetched from database."
    )


class PredictionResponse(BaseModel):
    unit_nr: int
    model_used: str
    predicted_rul: float
    health_index: float
    status: str  # HEALTHY, WARNING, CRITICAL
    alert_triggered: bool
    confidence_interval: Optional[Dict[str, float]] = None


class AnomalyDetectRequest(BaseModel):
    unit_nr: int = Field(..., json_schema_extra={"example": 1})
    telemetry: Dict[str, float] = Field(..., description="Sensor readings for current cycle")


class AnomalyDetectResponse(BaseModel):
    unit_nr: int
    is_anomaly: bool
    health_index: float
    status: str  # HEALTHY, WARNING, CRITICAL
    lead_time_cycles: Optional[float] = None
    alert_message: str


class SHAPContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    impact: str


class ExplainResponse(BaseModel):
    unit_nr: int
    cycle: int
    predicted_rul: float
    base_value: float
    top_contributions: List[SHAPContribution]
    summary: str


class StreamSimulationControl(BaseModel):
    action: str = Field(..., description="'start' or 'stop'")
    speed_seconds: float = Field(default=1.0, description="Delay between cycle emissions")
    unit_id: Optional[int] = Field(default=1, description="Engine unit to stream")


class StreamSimulationStatus(BaseModel):
    is_active: bool
    current_unit: Optional[int]
    current_cycle: int
    total_cycles_emitted: int
    speed_seconds: float
