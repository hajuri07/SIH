"""
Telemetry Ingestion & Query Router.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any

from ..schemas import TelemetryPacket, TelemetryIngestResponse
from ..db import save_telemetry, get_recent_telemetry

router = APIRouter(prefix="/telemetry", tags=["Telemetry Ingestion"])


@router.post("", response_model=TelemetryIngestResponse)
async def ingest_telemetry(packet: TelemetryPacket):
    """
    Ingests a single operational flight cycle for an engine unit.
    Saves sensor readings directly to the time-series store.
    """
    try:
        settings_dict = {
            "setting_1": packet.setting_1,
            "setting_2": packet.setting_2,
            "setting_3": packet.setting_3
        }
        await save_telemetry(packet.unit_nr, packet.time_cycles, settings_dict, packet.sensors)

        return TelemetryIngestResponse(
            status="SUCCESS",
            message=f"Cycle {packet.time_cycles} recorded for Engine {packet.unit_nr}.",
            unit_nr=packet.unit_nr,
            time_cycles=packet.time_cycles,
            recorded_at=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{unit_nr}/history")
async def get_engine_history(unit_nr: int, limit: int = Query(default=30, ge=1, le=200)):
    """
    Retrieves the most recent telemetry cycles for a given engine.
    """
    records = await get_recent_telemetry(unit_nr, limit=limit)
    return {"unit_nr": unit_nr, "count": len(records), "data": records}
