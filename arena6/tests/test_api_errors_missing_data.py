"""
Arena 6 - API tests for MISSING DATA and ERROR HANDLING.

Section A: unknown engines / no data in the database (expected 404, not a crash).
Section B: incomplete telemetry rows (expected a clear 4xx; xfail where the code suggests HTTP 500).
Section C: simulated internal failures (database down, model crash, models not loaded) via monkeypatch.

The API must fail gracefully: a clear JSON error, never a stack-trace crash.

Run from the repository root:   pytest arena6/tests/test_api_errors_missing_data.py -v
"""

import copy

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pytest_asyncio")

from arena6 import payloads  # noqa: E402

pytestmark = pytest.mark.asyncio

API = "/api/v1"
CLIENT_ERRORS = (400, 404, 409, 422)


# ------------------------------------------------------------------ A. no data for this engine

async def test_predict_for_engine_without_telemetry_is_404(client, models_ready, fresh_unit):
    unit = fresh_unit()
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": unit, "model_type": "xgboost"})
    assert r.status_code == 404
    assert str(unit) in r.json()["detail"]


async def test_predict_with_empty_telemetry_list_falls_back_to_db_and_404s(client, models_ready, fresh_unit):
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": fresh_unit(), "model_type": "xgboost", "recent_telemetry": []})
    assert r.status_code == 404


async def test_explain_for_engine_without_telemetry_is_404(client, models_ready, fresh_unit):
    unit = fresh_unit()
    r = await client.get(f"{API}/explain/{unit}")
    assert r.status_code == 404
    assert str(unit) in r.json()["detail"]


@pytest.mark.parametrize("model", ["xgboost", "tcn"])
async def test_predict_works_with_only_one_cycle_of_history(client, models_ready, model):
    """Brand-new engine: fewer cycles than the 30-cycle window. Must still answer (padding), not crash."""
    rows = payloads.load_engine_rows(1, 1)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "model_type": model, "recent_telemetry": rows})
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ B. incomplete rows

@pytest.mark.xfail(reason="BUG-005: rows missing time_cycles/unit_nr cause HTTP 500 instead of a 4xx", strict=False)
@pytest.mark.parametrize("missing", ["time_cycles", "unit_nr"])
async def test_predict_should_reject_rows_missing_required_columns(client, models_ready, missing):
    rows = copy.deepcopy(payloads.load_engine_rows(1, 5))
    for r_ in rows:
        r_.pop(missing)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "model_type": "xgboost", "recent_telemetry": rows})
    assert r.status_code in CLIENT_ERRORS, f"got HTTP {r.status_code}"


@pytest.mark.xfail(reason="BUG-005: rows with missing sensor columns cause HTTP 500 instead of a 4xx", strict=False)
async def test_predict_should_reject_rows_missing_sensors(client, models_ready):
    rows = copy.deepcopy(payloads.load_engine_rows(1, 5))
    for r_ in rows:
        for s in ("s_2", "s_3", "s_4"):
            r_.pop(s)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "model_type": "xgboost", "recent_telemetry": rows})
    assert r.status_code in CLIENT_ERRORS, f"got HTTP {r.status_code}"


@pytest.mark.xfail(reason="BUG-005: anomaly check with a partial sensor set returns 500/garbage instead of a 4xx", strict=False)
async def test_anomaly_should_reject_partial_sensor_set(client, models_ready):
    r = await client.post(f"{API}/detect/anomaly", json={"unit_nr": 1, "telemetry": {"s_2": 642.1, "s_3": 1585.0}})
    assert r.status_code in CLIENT_ERRORS, f"got HTTP {r.status_code}"


@pytest.mark.xfail(reason="BUG-005: empty sensor dict for anomaly check is not rejected cleanly", strict=False)
async def test_anomaly_should_reject_empty_sensor_dict(client, models_ready):
    r = await client.post(f"{API}/detect/anomaly", json={"unit_nr": 1, "telemetry": {}})
    assert r.status_code in CLIENT_ERRORS, f"got HTTP {r.status_code}"


@pytest.mark.xfail(reason="BUG-004: the same (unit, cycle) can be stored twice - no duplicate protection", strict=False)
async def test_duplicate_cycle_should_not_create_duplicate_rows(client, fresh_unit):
    unit = fresh_unit()
    pkt = payloads.valid_telemetry_packet(unit, 7)
    await client.post(f"{API}/telemetry", json=pkt)
    await client.post(f"{API}/telemetry", json=pkt)
    hist = (await client.get(f"{API}/telemetry/{unit}/history")).json()
    assert hist["count"] == 1, f"cycle 7 stored {hist['count']} times"


async def test_stop_stream_when_nothing_is_running_is_harmless(client):
    r = await client.post(f"{API}/stream/control", json={"action": "stop"})
    assert r.status_code == 200
    assert r.json()["is_active"] is False


# ------------------------------------------------------------------ C. simulated internal failures

async def _boom(*args, **kwargs):
    raise RuntimeError("simulated database outage")


def _boom_sync(*args, **kwargs):
    raise RuntimeError("simulated model crash")


async def test_ingest_returns_500_with_message_when_database_write_fails(client, monkeypatch, fresh_unit):
    monkeypatch.setattr("backend.routers.telemetry.save_telemetry", _boom)
    r = await client.post(f"{API}/telemetry", json=payloads.valid_telemetry_packet(fresh_unit(), 1))
    assert r.status_code == 500
    assert "simulated database outage" in r.json()["detail"]


async def test_history_database_failure_should_return_500_not_crash(client, monkeypatch, fresh_unit):
    monkeypatch.setattr("backend.routers.telemetry.get_recent_telemetry", _boom)
    try:
        r = await client.get(f"{API}/telemetry/{fresh_unit()}/history")
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(f"BUG-006: unhandled {type(exc).__name__} escapes the endpoint instead of HTTP 500")
    assert r.status_code == 500


async def test_predict_database_failure_should_return_500_not_crash(client, models_ready, monkeypatch, fresh_unit):
    monkeypatch.setattr("backend.routers.prediction.get_recent_telemetry", _boom)
    try:
        r = await client.post(f"{API}/predict/rul", json={"unit_nr": fresh_unit(), "model_type": "xgboost"})
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(f"BUG-006: unhandled {type(exc).__name__} escapes the endpoint instead of HTTP 500")
    assert r.status_code == 500


async def test_predict_model_crash_returns_500_with_message(client, models_ready, monkeypatch):
    monkeypatch.setattr(models_ready, "predict_rul", _boom_sync)
    rows = payloads.load_engine_rows(1, 30)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "model_type": "xgboost", "recent_telemetry": rows})
    assert r.status_code == 500
    assert "Prediction failed" in r.json()["detail"]


async def test_anomaly_model_crash_returns_500_with_message(client, models_ready, monkeypatch):
    monkeypatch.setattr(models_ready, "detect_anomaly", _boom_sync)
    row = payloads.load_engine_rows(1, 1)[0]
    r = await client.post(f"{API}/detect/anomaly", json={"unit_nr": 1, "telemetry": payloads.sensors_only(row)})
    assert r.status_code == 500
    assert "Anomaly detection failed" in r.json()["detail"]


async def test_predict_returns_503_when_models_are_not_loaded(client, monkeypatch):
    from backend.services import registry
    monkeypatch.setattr(registry, "is_loaded", True)
    monkeypatch.setattr(registry, "feature_engineer", None)
    rows = payloads.load_engine_rows(1, 5)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "model_type": "xgboost", "recent_telemetry": rows})
    assert r.status_code == 503


async def test_anomaly_returns_503_when_detector_is_missing(client, monkeypatch):
    from backend.services import registry
    monkeypatch.setattr(registry, "is_loaded", True)
    monkeypatch.setattr(registry, "anomaly_detector", None)
    row = payloads.load_engine_rows(1, 1)[0]
    r = await client.post(f"{API}/detect/anomaly", json={"unit_nr": 1, "telemetry": payloads.sensors_only(row)})
    assert r.status_code == 503


async def test_explain_returns_503_when_shap_engine_is_missing(client, monkeypatch, fresh_unit):
    from backend.services import registry
    monkeypatch.setattr(registry, "is_loaded", True)
    monkeypatch.setattr(registry, "shap_explainer", None)
    r = await client.get(f"{API}/explain/{fresh_unit()}")
    assert r.status_code == 503
