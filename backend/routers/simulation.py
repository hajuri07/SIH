"""
Telemetry Stream Simulation Router.
Controls background streaming simulation for live demos.
"""

from fastapi import APIRouter
from ..schemas import StreamSimulationControl, StreamSimulationStatus
from ..stream_simulator import simulator

router = APIRouter(prefix="/stream", tags=["Simulation Control"])


@router.post("/control", response_model=StreamSimulationStatus)
async def control_telemetry_stream(ctrl: StreamSimulationControl):
    """
    Starts or stops background streaming of engine cycles from the test dataset.
    """
    if ctrl.action.lower() == "start":
        simulator.start(unit_id=ctrl.unit_id or 1, speed=ctrl.speed_seconds)
    elif ctrl.action.lower() == "stop":
        simulator.stop()

    status = simulator.get_status()
    return StreamSimulationStatus(**status)


@router.get("/status", response_model=StreamSimulationStatus)
async def get_stream_status():
    """
    Returns current streaming status and metrics.
    """
    return StreamSimulationStatus(**simulator.get_status())
