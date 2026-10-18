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
        assert "readiness" in data
        assert "database" in data["readiness"]
        assert "models" in data["readiness"]
        assert "kafka" in data["readiness"]


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


def test_websocket_single_engine_stream():
    """Verifies WebSocket connection and live bidirectional ping-pong for a single engine."""
    from starlette.testclient import TestClient
    client = TestClient(app)
    with client.websocket_connect("/api/v1/health/ws/1") as ws:
        init_data = ws.receive_json()
        assert "event" in init_data
        assert init_data["unit_nr"] == 1
        ws.send_text("ping")
        resp = ws.receive_text()
        assert resp == "pong"


def test_websocket_fleet_stream():
    """Verifies fleet-wide WebSocket connection and initial snapshot."""
    from starlette.testclient import TestClient
    client = TestClient(app)
    with client.websocket_connect("/api/v1/health/ws/fleet") as ws:
        init_data = ws.receive_json()
        assert "event" in init_data
        assert init_data["event"] == "FLEET_SNAPSHOT"
        ws.send_text("ping")
        resp = ws.receive_text()
        assert resp == "pong"


@pytest.mark.asyncio
async def test_async_retry_mechanism():
    """Tests exponential backoff retry utility."""
    from backend.retry import async_retry

    call_count = 0

    @async_retry(max_retries=3, initial_delay=0.01, backoff_factor=1.5)
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Transient error")
        return "success"

    res = await flaky_function()
    assert res == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_maintenance_endpoints():
    """Verifies maintenance scheduling, hangar bays, and spare parts endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        sched_res = await ac.get("/api/v1/maintenance/schedule")
        assert sched_res.status_code == 200
        assert isinstance(sched_res.json(), list)

        slots_res = await ac.get("/api/v1/maintenance/slots")
        assert slots_res.status_code == 200
        slots_data = slots_res.json()
        assert "available_bays" in slots_data

        parts_res = await ac.get("/api/v1/maintenance/parts")
        assert parts_res.status_code == 200
        parts_data = parts_res.json()
        assert "inventory" in parts_data
        assert len(parts_data["inventory"]) > 0


@pytest.mark.asyncio
async def test_anomaly_detection_with_autoencoder():
    """Verifies that the /detect/anomaly endpoint supports the Autoencoder method."""
    payload = {
        "unit_nr": 1,
        "method": "autoencoder",
        "telemetry": {
            "s_2": 642.5,
            "s_3": 1588.0,
            "s_4": 1400.0,
            "s_7": 553.8,
            "s_8": 2388.1,
            "s_9": 9050.0,
            "s_11": 47.3,
            "s_12": 521.5,
            "s_13": 2388.1,
            "s_14": 8135.0,
            "s_15": 8.41,
            "s_17": 392.0,
            "s_20": 38.8,
            "s_21": 23.3
        }
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/detect/anomaly", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["unit_nr"] == 1
        assert "is_anomaly" in data
        assert "health_index" in data
        assert data["detector_used"] == "AUTOENCODER"

