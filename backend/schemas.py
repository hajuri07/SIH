"""
Pydantic Request and Response Schemas for FastAPI Endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


# Physical limits based on NASA C-MAPSS FD001 Data Dictionary
PHYSICAL_SENSOR_RANGES = {
    "s_2": (400.0, 900.0, "LPC Outlet Temp (T24)"),
    "s_3": (1200.0, 1900.0, "HPC Outlet Temp (T30)"),
    "s_4": (1100.0, 1700.0, "LPT Outlet Temp (T50)"),
    "s_7": (450.0, 650.0, "HPC Outlet Pressure (P30)"),
    "s_8": (2000.0, 2600.0, "Physical Fan Speed (Nf)"),
    "s_9": (8500.0, 9800.0, "Physical Core Speed (Nc)"),
    "s_11": (35.0, 60.0, "Static HPC Pressure (Ps30)"),
    "s_12": (450.0, 600.0, "Fuel Flow Ratio (phi)"),
    "s_13": (2200.0, 2500.0, "Corrected Fan Speed (NRf)"),
    "s_14": (7800.0, 8500.0, "Corrected Core Speed (NRc)"),
    "s_15": (7.0, 10.0, "Bypass Ratio (BPR)"),
    "s_17": (350.0, 450.0, "Bleed Enthalpy (htBleed)"),
    "s_20": (30.0, 50.0, "HPT Coolant Bleed (W31)"),
    "s_21": (15.0, 30.0, "LPT Coolant Bleed (W32)")
}


class TelemetryPacket(BaseModel):
    """Single cycle telemetry payload from an engine unit."""
    unit_nr: int = Field(..., ge=1, description="Engine Unit Identifier", json_schema_extra={"example": 1})
    time_cycles: int = Field(..., ge=1, description="Flight Cycle Count", json_schema_extra={"example": 45})
    setting_1: float = Field(default=0.0, description="Altitude / Mach command 1")
    setting_2: float = Field(default=0.0, description="Mach command 2")
    setting_3: float = Field(default=100.0, description="Throttle Resolver Angle (TRA)")
    sensors: Dict[str, float] = Field(
        ...,
        description="Sensor readings dictionary (e.g. {'s_2': 642.5, 's_3': 1588.2, ...})",
        json_schema_extra={"example": {"s_2": 642.3, "s_3": 1589.5, "s_4": 1402.1, "s_7": 553.8, "s_8": 2388.1}}
    )

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Telemetry sensor readings dictionary cannot be empty.")

        for k, val in v.items():
            if not k.startswith("s_"):
                raise ValueError(f"Invalid sensor channel '{k}'. Sensor channels must start with 's_'.")

            # Check known physical bounds
            if k in PHYSICAL_SENSOR_RANGES:
                min_val, max_val, name = PHYSICAL_SENSOR_RANGES[k]
                if val < min_val or val > max_val:
                    raise ValueError(
                        f"Sensor reading out of physical range: {k} ({name}) = {val}. Expected range: [{min_val}, {max_val}]."
                    )
        return v


class TelemetryIngestResponse(BaseModel):
    status: str
    message: str
    unit_nr: int
    time_cycles: int
    recorded_at: str
    kafka_published: bool = False


class PredictionRequest(BaseModel):
    unit_nr: int = Field(..., ge=1, json_schema_extra={"example": 1})
    model_type: str = Field(default="auto", description="Model architecture: 'auto', 'xgboost', 'lstm', or 'tcn'")
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
    unit_nr: int = Field(..., ge=1, json_schema_extra={"example": 1})
    method: str = Field(default="auto", description="Detection method: 'auto', 'isolation_forest', or 'autoencoder'")
    telemetry: Dict[str, float] = Field(..., description="Sensor readings for current cycle")

    @field_validator("telemetry")
    @classmethod
    def validate_telemetry(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Telemetry readings dictionary cannot be empty.")
        for k in v.keys():
            if not k.startswith("s_"):
                raise ValueError(f"Invalid sensor identifier '{k}'.")
        return v


class AnomalyDetectResponse(BaseModel):
    unit_nr: int
    is_anomaly: bool
    health_index: float
    status: str  # HEALTHY, WARNING, CRITICAL
    detector_used: str = "ISOLATION_FOREST"
    lead_time_cycles: Optional[float] = None
    alert_message: str


class AnomalyAlertItem(BaseModel):
    id: int
    unit_nr: int
    time_cycles: int
    alert_level: str
    health_index: float
    message: str
    created_at: Optional[str] = None


class FleetHealthSummary(BaseModel):
    total_engines: int
    tier_counts: Dict[str, int]
    engines: List[Dict[str, Any]]


class EngineHealthResponse(BaseModel):
    unit_nr: int
    latest_cycle: int
    health_index: float
    status: str
    predicted_rul: float
    confidence_interval: Optional[Dict[str, float]] = None
    recent_alerts: List[AnomalyAlertItem] = []


class SHAPContribution(BaseModel):
    feature: str
    feature_name: Optional[str] = None
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
    speed_seconds: float = Field(default=0.5, description="Delay between cycle emissions")
    unit_id: Optional[int] = Field(default=1, description="Engine unit to stream")


class StreamSimulationStatus(BaseModel):
    is_active: bool
    current_unit: Optional[int]
    current_cycle: int
    total_cycles_emitted: int
    speed_seconds: float
    kafka_streaming: bool = False


class MaintenanceScheduleResponse(BaseModel):
    unit_nr: int
    urgency: str  # IMMEDIATE_AOG, EXPEDITED, ROUTINE
    predicted_rul: float
    health_index: float
    recommended_action: str
    target_hangar_slot: str
    estimated_turnaround_hours: float
    required_parts: List[str]
