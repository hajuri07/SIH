"""
API Integration Tests for FastAPI Microservice.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert "models_loaded" in data


@pytest.mark.asyncio
async def test_telemetry_ingest_endpoint():
    payload = {
        "unit_nr": 99,
        "time_cycles": 1,
        "setting_1": 0.002,
        "setting_2": 0.0001,
        "setting_3": 100.0,
        "sensors": {
            "s_2": 642.1,
            "s_3": 1585.0,
            "s_4": 1401.5,
            "s_7": 553.9,
            "s_8": 2388.05
        }
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/telemetry", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["unit_nr"] == 99


@pytest.mark.asyncio
async def test_stream_simulation_control():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Start
        start_res = await ac.post("/api/v1/stream/control", json={"action": "start", "speed_seconds": 10.0, "unit_id": 1})
        assert start_res.status_code == 200
        # Check status
        status_res = await ac.get("/api/v1/stream/status")
        assert status_res.status_code == 200
        # Stop
        stop_res = await ac.post("/api/v1/stream/control", json={"action": "stop"})
        assert stop_res.status_code == 200
        assert stop_res.json()["is_active"] is False
