"""
Fleet and Aircraft Engine Health Status Router.
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any

from ..schemas import FleetHealthSummary, EngineHealthResponse, AnomalyAlertItem
from ..db import get_fleet_health_summary, get_recent_telemetry, get_alerts, get_prediction_history
from ..services import registry
from ..ws_manager import ws_manager

router = APIRouter(prefix="/health", tags=["Fleet & Aircraft Health"])


@router.get("/fleet", response_model=FleetHealthSummary)
async def get_fleet_health():
    """
    Returns aggregate health status across all active aircraft engines in the fleet.
    Triage engines into HEALTHY, WARNING, and CRITICAL categories.
    """
    summary = await get_fleet_health_summary()
    return FleetHealthSummary(**summary)


@router.get("/{unit_nr}", response_model=EngineHealthResponse)
async def get_single_engine_health(unit_nr: int):
    """
    Returns comprehensive health overview for an individual engine:
    current flight cycle, predicted RUL, health index, triage status, and recent alerts.
    """
    telemetry = await get_recent_telemetry(unit_nr, limit=30)
    if not telemetry:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry records found for Engine Unit {unit_nr}."
        )

    latest_cycle = telemetry[-1]["time_cycles"]

    # Get latest prediction or calculate
    predictions = await get_prediction_history(unit_nr, limit=1)
    if predictions:
        pred_rul = predictions[0]["predicted_rul"]
        hi = predictions[0]["health_index"]
        status = predictions[0]["status"]
    else:
        pred_rul = 125.0
        hi = 1.0
        status = "HEALTHY"

    # Get confidence interval
    half_width = float(registry.uncertainty_config.get("half_width_cycles", 12.85))
    ci = {
        "lower_bound": round(max(0.0, pred_rul - half_width), 2),
        "upper_bound": round(pred_rul + half_width, 2)
    }

    raw_alerts = await get_alerts(unit_nr=unit_nr, limit=5)
    alerts = [AnomalyAlertItem(**a) for a in raw_alerts]

    return EngineHealthResponse(
        unit_nr=unit_nr,
        latest_cycle=latest_cycle,
        health_index=round(hi, 3),
        status=status,
        predicted_rul=round(pred_rul, 2),
        confidence_interval=ci,
        recent_alerts=alerts
    )


@router.websocket("/ws/fleet")
async def websocket_fleet_health(websocket: WebSocket):
    """
    WebSocket endpoint streaming fleet-wide health status and critical alerts in real-time.
    """
    await ws_manager.connect_fleet(websocket)
    try:
        # Send initial fleet summary upon connection
        try:
            summary = await get_fleet_health_summary()
            await websocket.send_json({"event": "FLEET_SNAPSHOT", "data": summary})
        except Exception:
            pass

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect_fleet(websocket)
    except Exception:
        ws_manager.disconnect_fleet(websocket)


@router.websocket("/ws/{unit_nr}")
async def websocket_single_engine_health(websocket: WebSocket, unit_nr: int):
    """
    WebSocket endpoint streaming live real-time telemetry, RUL predictions,
    and anomaly alerts for a specific aircraft engine.
    """
    await ws_manager.connect_unit(websocket, unit_nr)
    try:
        # Send initial snapshot upon connection
        try:
            telemetry = await get_recent_telemetry(unit_nr, limit=1)
            predictions = await get_prediction_history(unit_nr, limit=1)
            snapshot = {
                "event": "INITIAL_SNAPSHOT",
                "unit_nr": unit_nr,
                "latest_cycle": telemetry[-1]["time_cycles"] if telemetry else 0,
                "predicted_rul": predictions[0]["predicted_rul"] if predictions else 125.0,
                "health_index": predictions[0]["health_index"] if predictions else 1.0,
                "status": predictions[0]["status"] if predictions else "HEALTHY"
            }
            await websocket.send_json(snapshot)
        except Exception:
            pass

        # Keep connection open and receive optional client messages/pings
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect_unit(websocket, unit_nr)
    except Exception:
        ws_manager.disconnect_unit(websocket, unit_nr)


