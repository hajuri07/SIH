"""
Telemetry Ingestion & Query Router.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any

from ..schemas import TelemetryPacket, TelemetryIngestResponse
from ..db import save_telemetry, get_recent_telemetry, get_latest_cycle_for_unit
from ..kafka_client import kafka_manager

router = APIRouter(prefix="/telemetry", tags=["Telemetry Ingestion"])


@router.post("", response_model=TelemetryIngestResponse)
async def ingest_telemetry(packet: TelemetryPacket):
    """
    Ingests a single operational flight cycle for an engine unit.
    Validates cycle monotonicity and saves sensor readings to time-series store and Kafka bus.
    """
    # 1. Enforce cycle monotonicity per engine
    latest_cycle = await get_latest_cycle_for_unit(packet.unit_nr)
    if latest_cycle is not None and packet.time_cycles <= latest_cycle:
        raise HTTPException(
            status_code=400,
            detail=f"Cycle monotonicity violation: cycle {packet.time_cycles} must be strictly greater than latest recorded cycle {latest_cycle} for Engine Unit {packet.unit_nr}."
        )

    try:
        settings_dict = {
            "setting_1": packet.setting_1,
            "setting_2": packet.setting_2,
            "setting_3": packet.setting_3
        }
        await save_telemetry(packet.unit_nr, packet.time_cycles, settings_dict, packet.sensors)

        # 2. Publish to Kafka telemetry topic if Kafka is enabled
        kafka_published = await kafka_manager.send_telemetry({
            "unit_nr": packet.unit_nr,
            "time_cycles": packet.time_cycles,
            "settings": settings_dict,
            "sensors": packet.sensors,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return TelemetryIngestResponse(
            status="SUCCESS",
            message=f"Cycle {packet.time_cycles} recorded for Engine {packet.unit_nr}.",
            unit_nr=packet.unit_nr,
            time_cycles=packet.time_cycles,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            kafka_published=kafka_published
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{unit_nr}/history")
async def get_engine_history(unit_nr: int, limit: int = Query(default=30, ge=1, le=200)):
    """
    Retrieves the most recent telemetry cycles for a given engine.
    """
    records = await get_recent_telemetry(unit_nr, limit=limit)
    return {"unit_nr": unit_nr, "count": len(records), "data": records}
