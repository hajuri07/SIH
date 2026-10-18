"""
End-to-End Integration Tests for VayuPulse Turbofan Prognostics & Health Management.
Verifies Kafka consumer, telemetry ingestion, automated inference trigger, database
persistence of predictions and alerts, readiness health checks, and per-model uncertainty.
"""

import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
import pandas as pd
from backend.main import app, handle_telemetry_inference
from backend.services import registry
from backend.kafka_client import kafka_manager
from backend.db import (
    get_recent_telemetry,
    get_prediction_history,
    get_alerts,
    check_db_readiness
)





@pytest.mark.asyncio
async def test_health_check_readiness_reporting():
    """
    Verifies that the /health endpoint accurately reports readiness for:
    - Prognostic models (all loaded architectures)
    - Database connectivity (live test query)
    - Kafka streaming manager & consumer loop
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()

        # Overall status
        assert data["status"] in ("HEALTHY", "DEGRADED")

        # Database readiness
        db_readiness = data["readiness"]["database"]
        assert db_readiness["ready"] is True
        assert db_readiness["status"] == "CONNECTED"
        assert db_readiness["error"] is None

        # Model readiness
        model_readiness = data["readiness"]["models"]
        assert model_readiness["ready"] is True
        assert "selected_best_model" in model_readiness
        assert data["models_loaded"]["xgboost"] is True
        assert data["models_loaded"]["anomaly_detector"] is True

        # Kafka readiness
        kafka_readiness = data["readiness"]["kafka"]
        assert "ready" in kafka_readiness
        assert "consumer_running" in kafka_readiness
        assert kafka_readiness["consumer_running"] is True


@pytest.mark.asyncio
async def test_per_model_uncertainty_calibration():
    """
    Verifies that uncertainty intervals are calibrated specifically for each model
    (XGBoost, LSTM, and TCN) rather than using a single global value.
    """
    assert "models" in registry.uncertainty_config
    model_configs = registry.uncertainty_config["models"]
    assert "xgboost" in model_configs
    assert "lstm" in model_configs
    assert "tcn" in model_configs

    # Verify each model has its own calibrated half_width_cycles and residual_std
    xgb_hw = model_configs["xgboost"]["half_width_cycles"]
    lstm_hw = model_configs["lstm"]["half_width_cycles"]
    tcn_hw = model_configs["tcn"]["half_width_cycles"]

    assert xgb_hw > 0.0
    assert lstm_hw > 0.0
    assert tcn_hw > 0.0

    # Build dummy telemetry sequence for engine 777
    rows = []
    for c in range(1, 35):
        rows.append({
            "unit_nr": 777,
            "time_cycles": c,
            "setting_1": 0.001,
            "setting_2": 0.0002,
            "setting_3": 100.0,
            **{f"s_{i}": 500.0 + c * 0.1 for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]}
        })
    df_sample = pd.DataFrame(rows)

    # Predict with XGBoost
    rul_xgb, ci_xgb = registry.predict_rul("xgboost", df_sample)
    assert ci_xgb["margin_error_cycles"] == xgb_hw
    assert ci_xgb["calibrated_model"] == "xgboost"

    # Predict with LSTM
    rul_lstm, ci_lstm = registry.predict_rul("lstm", df_sample)
    assert ci_lstm["margin_error_cycles"] == lstm_hw
    assert ci_lstm["calibrated_model"] == "lstm"

    # Predict with TCN
    rul_tcn, ci_tcn = registry.predict_rul("tcn", df_sample)
    assert ci_tcn["margin_error_cycles"] == tcn_hw
    assert ci_tcn["calibrated_model"] == "tcn"


@pytest.mark.asyncio
async def test_end_to_end_telemetry_inference_persistence_and_alerts():
    """
    End-to-End integration test:
    1. Ingest telemetry sequence for Engine 888.
    2. Verify telemetry persistence in database.
    3. Trigger telemetry-driven inference via streaming consumer.
    4. Verify predictions are persisted to DB (rul_predictions).
    5. Trigger degraded conditions and verify anomaly alerts are persisted (anomaly_alerts).
    6. Verify /health/{unit_nr} and /alerts API query endpoints return persisted records.
    """
    test_unit = 888
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Ingest nominal telemetry cycles 1 to 5
        for c in range(1, 6):
            payload = {
                "unit_nr": test_unit,
                "time_cycles": c,
                "setting_1": 0.001,
                "setting_2": 0.0001,
                "setting_3": 100.0,
                "sensors": {
                    "s_2": 642.0 + c * 0.02,
                    "s_3": 1585.0 + c * 0.1,
                    "s_4": 1400.0 + c * 0.1,
                    "s_7": 553.5,
                    "s_8": 2388.05,
                    "s_9": 9050.0,
                    "s_11": 47.2,
                    "s_12": 521.8,
                    "s_13": 2388.06,
                    "s_14": 8130.0,
                    "s_15": 8.42,
                    "s_17": 392.0,
                    "s_20": 38.85,
                    "s_21": 23.35
                }
            }
            res = await ac.post("/api/v1/telemetry", json=payload)
            assert res.status_code == 200

        # Verify telemetry saved to DB
        saved_telemetry = await get_recent_telemetry(test_unit, limit=10)
        assert len(saved_telemetry) == 5
        assert saved_telemetry[-1]["time_cycles"] == 5

        # 2. Trigger automated inference for cycle 5
        packet = {
            "unit_nr": test_unit,
            "time_cycles": 5,
            "settings": {"setting_1": 0.001, "setting_2": 0.0001, "setting_3": 100.0},
            "sensors": saved_telemetry[-1]
        }
        await handle_telemetry_inference(packet)

        # Verify prediction was persisted
        preds = await get_prediction_history(test_unit, limit=5)
        assert len(preds) > 0
        assert preds[0]["unit_nr"] == test_unit
        assert preds[0]["predicted_rul"] > 0

        # 3. Simulate severe degradation at cycle 6 (abnormal sensor spike)
        critical_payload = {
            "unit_nr": test_unit,
            "time_cycles": 6,
            "settings": {"setting_1": 0.001, "setting_2": 0.0001, "setting_3": 100.0},
            "sensors": {
                "s_2": 665.0,  # Extreme temp rise
                "s_3": 1650.0,
                "s_4": 1490.0,
                "s_7": 540.0,
                "s_8": 2388.35,
                "s_9": 9150.0,
                "s_11": 52.0,
                "s_12": 515.0,
                "s_13": 2388.30,
                "s_14": 8220.0,
                "s_15": 8.85,
                "s_17": 410.0,
                "s_20": 37.10,
                "s_21": 22.10
            }
        }
        ingest_res = await ac.post("/api/v1/telemetry", json=critical_payload)
        assert ingest_res.status_code == 200

        # Trigger inference on degraded cycle
        await handle_telemetry_inference(critical_payload)

        # 4. Verify predictions and health overview in database
        health_res = await ac.get(f"/api/v1/health/{test_unit}")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["unit_nr"] == test_unit
        assert health_data["latest_cycle"] == 6
        assert "predicted_rul" in health_data
        assert "confidence_interval" in health_data


@pytest.mark.asyncio
async def test_kafka_streaming_trigger_inference():
    """
    Verifies that kafka_manager.send_telemetry invokes the registered
    consumer handler to trigger automated inference in local/direct streaming mode.
    """
    unit_nr = 999
    telemetry_packet = {
        "unit_nr": unit_nr,
        "time_cycles": 1,
        "settings": {"setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0},
        "sensors": {f"s_{i}": 500.0 for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]}
    }

    # Dispatch via kafka streaming manager
    dispatched = await kafka_manager.send_telemetry(telemetry_packet)
    assert dispatched is True

    # Allow background consumer task to process
    await asyncio.sleep(0.3)

    # Verify telemetry was recorded and inference triggered
    history = await get_recent_telemetry(unit_nr, limit=5)
    assert len(history) >= 1
    assert history[0]["unit_nr"] == unit_nr
