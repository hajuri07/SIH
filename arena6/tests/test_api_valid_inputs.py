"""
Arena 6 - API tests with VALID inputs (happy paths and the end-to-end flow).

Needs the backend requirements installed (requirements.txt). Tests that need trained models
skip themselves when models/artifacts/ is empty.

Run from the repository root:   pytest arena6/tests/test_api_valid_inputs.py -v
"""

import math

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pytest_asyncio")

from arena6 import payloads  # noqa: E402

pytestmark = pytest.mark.asyncio

API = "/api/v1"
STATUSES = {"HEALTHY", "WARNING", "CRITICAL"}


def _check_prediction(body, model):
    assert body["model_used"] == model.upper()
    assert isinstance(body["predicted_rul"], (int, float)) and math.isfinite(body["predicted_rul"])
    assert body["predicted_rul"] <= 400, "RUL far beyond anything in C-MAPSS"
    assert 0.0 <= body["health_index"] <= 1.0
    assert body["status"] in STATUSES
    assert isinstance(body["alert_triggered"], bool)
    ci = body["confidence_interval"]
    assert ci["lower_bound"] <= body["predicted_rul"] <= ci["upper_bound"] or ci["lower_bound"] == 0.0
    assert ci["lower_bound"] <= ci["upper_bound"]


async def _ingest_engine(client, unit, n=30, source_unit=1):
    for row in payloads.load_engine_rows(source_unit, n):
        r = await client.post(f"{API}/telemetry", json=payloads.row_to_packet(row, unit_nr=unit))
        assert r.status_code == 200, r.text


# ------------------------------------------------------------------ system

async def test_health_endpoint_reports_status_and_models(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "HEALTHY"
    assert set(body["models_loaded"]) == {"xgboost", "lstm", "tcn", "anomaly_detector", "shap_explainer"}


async def test_root_returns_links(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert r.json()["docs_url"] == "/docs"


async def test_openapi_lists_every_documented_endpoint(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"])
    expected = {
        "/health", f"{API}/telemetry", f"{API}/telemetry/{{unit_nr}}/history", f"{API}/predict/rul",
        f"{API}/detect/anomaly", f"{API}/explain/{{unit_nr}}", f"{API}/stream/control", f"{API}/stream/status",
    }
    assert expected <= paths, f"missing endpoints: {expected - paths}"


# ------------------------------------------------------------------ telemetry

async def test_ingest_valid_packet(client, fresh_unit):
    unit = fresh_unit()
    r = await client.post(f"{API}/telemetry", json=payloads.valid_telemetry_packet(unit, 1))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "SUCCESS"
    assert body["unit_nr"] == unit and body["time_cycles"] == 1
    assert body["recorded_at"]


async def test_ingest_then_history_returns_the_same_cycles_in_order(client, fresh_unit):
    unit = fresh_unit()
    for cycle in (1, 2, 3):
        pkt = payloads.valid_telemetry_packet(unit, cycle)
        pkt["sensors"]["s_3"] = 1580.0 + cycle
        assert (await client.post(f"{API}/telemetry", json=pkt)).status_code == 200

    r = await client.get(f"{API}/telemetry/{unit}/history")
    assert r.status_code == 200
    body = r.json()
    assert body["unit_nr"] == unit and body["count"] == 3
    assert [row["time_cycles"] for row in body["data"]] == [1, 2, 3]  # oldest -> newest
    assert body["data"][2]["s_3"] == 1583.0
    assert body["data"][0]["setting_3"] == 100.0


async def test_history_limit_returns_the_latest_cycles(client, fresh_unit):
    unit = fresh_unit()
    for cycle in range(1, 6):
        await client.post(f"{API}/telemetry", json=payloads.valid_telemetry_packet(unit, cycle))
    r = await client.get(f"{API}/telemetry/{unit}/history", params={"limit": 2})
    assert [row["time_cycles"] for row in r.json()["data"]] == [4, 5]


async def test_history_for_unknown_engine_is_an_empty_list(client, fresh_unit):
    r = await client.get(f"{API}/telemetry/{fresh_unit()}/history")
    assert r.status_code == 200
    assert r.json()["count"] == 0 and r.json()["data"] == []


# ------------------------------------------------------------------ RUL prediction

@pytest.mark.parametrize("model", ["xgboost", "lstm", "tcn"])
async def test_predict_rul_with_supplied_telemetry(client, models_ready, model):
    rows = payloads.load_engine_rows(1, 30)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "model_type": model, "recent_telemetry": rows})
    assert r.status_code == 200, r.text
    _check_prediction(r.json(), model)


async def test_predict_rul_is_repeatable(client, models_ready):
    rows = payloads.load_engine_rows(1, 30)
    body = {"unit_nr": 1, "model_type": "xgboost", "recent_telemetry": rows}
    a = (await client.post(f"{API}/predict/rul", json=body)).json()
    b = (await client.post(f"{API}/predict/rul", json=body)).json()
    assert a["predicted_rul"] == b["predicted_rul"]


async def test_predict_rul_default_model_is_tcn(client, models_ready):
    rows = payloads.load_engine_rows(1, 30)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "recent_telemetry": rows})
    assert r.status_code == 200
    assert r.json()["model_used"] == "TCN"


# ------------------------------------------------------------------ anomaly detection

async def test_anomaly_detection_on_a_real_cycle(client, models_ready):
    row = payloads.load_engine_rows(1, 1)[0]
    r = await client.post(f"{API}/detect/anomaly", json={"unit_nr": 1, "telemetry": payloads.sensors_only(row)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["is_anomaly"], bool)
    assert 0.0 <= body["health_index"] <= 1.0
    assert body["status"] in STATUSES
    assert body["alert_message"]
    if body["status"] == "HEALTHY":
        assert body["lead_time_cycles"] is None


# ------------------------------------------------------------------ stream control

async def test_stream_start_status_stop(client):
    try:
        start = await client.post(f"{API}/stream/control", json={"action": "start", "speed_seconds": 10.0, "unit_id": 1})
        assert start.status_code == 200
        status = await client.get(f"{API}/stream/status")
        assert status.status_code == 200
        assert {"is_active", "current_unit", "current_cycle", "total_cycles_emitted", "speed_seconds"} <= set(status.json())
    finally:
        stop = await client.post(f"{API}/stream/control", json={"action": "stop"})
    assert stop.status_code == 200
    assert stop.json()["is_active"] is False


# ------------------------------------------------------------------ end to end

async def test_end_to_end_ingest_predict_anomaly_explain(client, models_ready, fresh_unit):
    """The demo flow: stream cycles in -> history -> RUL from DB -> anomaly -> SHAP explanation."""
    unit = fresh_unit()
    await _ingest_engine(client, unit, n=30)

    hist = (await client.get(f"{API}/telemetry/{unit}/history")).json()
    assert hist["count"] == 30

    pred = await client.post(f"{API}/predict/rul", json={"unit_nr": unit, "model_type": "xgboost"})  # reads from DB
    assert pred.status_code == 200, pred.text
    _check_prediction(pred.json(), "xgboost")

    last = hist["data"][-1]
    anomaly = await client.post(f"{API}/detect/anomaly", json={"unit_nr": unit, "telemetry": payloads.sensors_only(last)})
    assert anomaly.status_code == 200, anomaly.text

    explain = await client.get(f"{API}/explain/{unit}")
    assert explain.status_code == 200, explain.text
    body = explain.json()
    assert body["unit_nr"] == unit and body["cycle"] == 30
    assert 1 <= len(body["top_contributions"]) <= 8
    first = body["top_contributions"][0]
    assert {"feature", "feature_value", "shap_value", "impact"} <= set(first)
    assert str(unit) in body["summary"]
