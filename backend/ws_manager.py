"""
WebSocket Connection Manager for Real-Time Fleet & Aircraft Health Streaming.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import WebSocket

logger = logging.getLogger("vayupulse.websocket")


class HealthWebSocketManager:
    """Manages active WebSocket client connections for live health and alert updates."""

    def __init__(self):
        # Specific engine listeners: unit_nr -> List[WebSocket]
        self.unit_listeners: Dict[int, List[WebSocket]] = {}
        # Fleet-wide listeners: List[WebSocket]
        self.fleet_listeners: List[WebSocket] = []

    async def connect_unit(self, websocket: WebSocket, unit_nr: int):
        await websocket.accept()
        if unit_nr not in self.unit_listeners:
            self.unit_listeners[unit_nr] = []
        self.unit_listeners[unit_nr].append(websocket)
        logger.info(f"WebSocket client connected to Engine {unit_nr} stream.")

    def disconnect_unit(self, websocket: WebSocket, unit_nr: int):
        if unit_nr in self.unit_listeners:
            if websocket in self.unit_listeners[unit_nr]:
                self.unit_listeners[unit_nr].remove(websocket)
            if not self.unit_listeners[unit_nr]:
                del self.unit_listeners[unit_nr]
        logger.info(f"WebSocket client disconnected from Engine {unit_nr} stream.")

    async def connect_fleet(self, websocket: WebSocket):
        await websocket.accept()
        self.fleet_listeners.append(websocket)
        logger.info("WebSocket client connected to Fleet-wide stream.")

    def disconnect_fleet(self, websocket: WebSocket):
        if websocket in self.fleet_listeners:
            self.fleet_listeners.remove(websocket)
        logger.info("WebSocket client disconnected from Fleet-wide stream.")

    async def broadcast_to_unit(self, unit_nr: int, message: Dict[str, Any]):
        """Broadcasts live packet/RUL/alert to all clients subscribed to a specific engine."""
        listeners = self.unit_listeners.get(unit_nr, [])
        if not listeners:
            return
        dead_sockets = []
        msg_str = json.dumps(message)
        for ws in listeners:
            try:
                await ws.send_text(msg_str)
            except Exception:
                dead_sockets.append(ws)
        for dead in dead_sockets:
            self.disconnect_unit(dead, unit_nr)

    async def broadcast_to_fleet(self, message: Dict[str, Any]):
        """Broadcasts fleet-wide updates or critical alerts to all fleet subscribers."""
        if not self.fleet_listeners:
            return
        dead_sockets = []
        msg_str = json.dumps(message)
        for ws in self.fleet_listeners:
            try:
                await ws.send_text(msg_str)
            except Exception:
                dead_sockets.append(ws)
        for dead in dead_sockets:
            self.disconnect_fleet(dead)


ws_manager = HealthWebSocketManager()
