"""
Arena 6 - API tests with INVALID inputs.

Part 1: inputs the API must reject with HTTP 422/404/405 (regular tests).
Part 2: inputs a robust API *should* reject but that the code appears to accept
        (marked xfail - they turn into "XPASS" once Arena 2 adds the validation).
        Each is described in arena6/docs/BUG_REPORTS.md.

Run from the repository root:   pytest arena6/tests/test_api_invalid_inputs.py -v
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pytest_asyncio")

from arena6 import payloads  # noqa: E402

pytestmark = pytest.mark.asyncio

API = "/api/v1"
REJECTED = (400, 409, 422)


# ------------------------------------------------------------------ Part 1: must be rejected

@pytest.mark.parametrize("case", sorted(payloads.invalid_telemetry_payloads()))
async def test_telemetry_rejects_malformed_packets(client, case):
    r = await client.post(f"{API}/telemetry", json=payloads.invalid_telemetry_payloads()[case])
    assert r.status_code == 422, f"{case}: {r.status_code} {r.text[:200]}"
    assert isinstance(r.json()["detail"], list) and r.json()["detail"]


async def test_telemetry_rejects_body_that_is_not_json(client):
    r = await client.post(f"{API}/telemetry", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


async def test_telemetry_rejects_form_data_instead_of_json(client):
    r = await client.post(f"{API}/telemetry", data={"unit_nr": "1", "time_cycles": "1"})
    assert r.status_code == 422


@pytest.mark.parametrize("case", sorted(payloads.invalid_prediction_payloads()))
async def test_predict_rejects_malformed_requests(client, case):
    r = await client.post(f"{API}/predict/rul", json=payloads.invalid_prediction_payloads()[case])
    assert r.status_code == 422, f"{case}: {r.status_code} {r.text[:200]}"


@pytest.mark.parametrize("case", sorted(payloads.invalid_anomaly_payloads()))
async def test_anomaly_rejects_malformed_requests(client, case):
    r = await client.post(f"{API}/detect/anomaly", json=payloads.invalid_anomaly_payloads()[case])
    assert r.status_code == 422, f"{case}: {r.status_code} {r.text[:200]}"


@pytest.mark.parametrize("limit", [0, -1, 201, 100000, "abc", 2.5])
async def test_history_rejects_out_of_range_limit(client, fresh_unit, limit):
    r = await client.get(f"{API}/telemetry/{fresh_unit()}/history", params={"limit": limit})
    assert r.status_code == 422


@pytest.mark.parametrize("path", [f"{API}/telemetry/abc/history", f"{API}/explain/abc", f"{API}/explain/1.5"])
async def test_non_numeric_engine_id_in_url_is_rejected(client, path):
    r = await client.get(path)
    assert r.status_code == 422


@pytest.mark.parametrize("body", [{}, {"speed_seconds": 1.0}, {"action": "start", "speed_seconds": "fast"},
                                  {"action": "start", "unit_id": "one"}, {"action": 5}])
async def test_stream_control_rejects_malformed_requests(client, body):
    r = await client.post(f"{API}/stream/control", json=body)
    assert r.status_code == 422


@pytest.mark.parametrize("method,path", [
    ("GET", f"{API}/telemetry"), ("DELETE", f"{API}/telemetry"), ("GET", f"{API}/predict/rul"),
    ("PUT", f"{API}/detect/anomaly"), ("POST", "/health"), ("POST", f"{API}/explain/1"),
])
async def test_wrong_http_method_is_405(client, method, path):
    r = await client.request(method, path)
    assert r.status_code == 405


@pytest.mark.parametrize("path", ["/api/v1/nothing", "/api/v2/telemetry", "/predict/rul", "/telemetry"])
async def test_unknown_routes_are_404(client, path):
    r = await client.get(path)
    assert r.status_code == 404


# ------------------------------------------------------------------ Part 2: should be rejected, currently accepted

@pytest.mark.xfail(reason="BUG-001/003: no range or content validation on telemetry packets (backend/schemas.py)", strict=False)
@pytest.mark.parametrize("case", sorted(payloads.should_be_rejected_telemetry()))
async def test_telemetry_should_reject_impossible_values(client, case):
    r = await client.post(f"{API}/telemetry", json=payloads.should_be_rejected_telemetry()[case])
    assert r.status_code in REJECTED, f"{case} was accepted with HTTP {r.status_code}"


@pytest.mark.xfail(reason="BUG-002: unknown model_type silently falls back to XGBoost (backend/services.py)", strict=False)
async def test_predict_should_reject_unknown_model_type(client, models_ready):
    rows = payloads.load_engine_rows(1, 30)
    r = await client.post(f"{API}/predict/rul", json={"unit_nr": 1, "model_type": "quantum", "recent_telemetry": rows})
    assert r.status_code in REJECTED


@pytest.mark.xfail(reason="BUG-007: stream control ignores unknown actions and returns 200", strict=False)
async def test_stream_control_should_reject_unknown_action(client):
    r = await client.post(f"{API}/stream/control", json={"action": "explode"})
    assert r.status_code in REJECTED
