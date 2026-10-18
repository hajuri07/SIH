-- TimescaleDB Initialization Script for Turbofan Prognostics & Predictive Maintenance

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. Engine Telemetry Hypertable
CREATE TABLE IF NOT EXISTS telemetry_records (
    id SERIAL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    unit_nr INT NOT NULL,
    time_cycles INT NOT NULL,
    setting_1 DOUBLE PRECISION DEFAULT 0.0,
    setting_2 DOUBLE PRECISION DEFAULT 0.0,
    setting_3 DOUBLE PRECISION DEFAULT 100.0,
    sensor_data JSONB NOT NULL,
    PRIMARY KEY (recorded_at, unit_nr, time_cycles)
);

-- Convert to hypertable partitioned on recorded_at
SELECT create_hypertable('telemetry_records', 'recorded_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_telemetry_unit_cycle ON telemetry_records (unit_nr, time_cycles DESC);

-- 2. Predictions Table
CREATE TABLE IF NOT EXISTS rul_predictions (
    id SERIAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    unit_nr INT NOT NULL,
    time_cycles INT NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    predicted_rul DOUBLE PRECISION NOT NULL,
    health_index DOUBLE PRECISION NOT NULL,
    status VARCHAR(20) NOT NULL,
    PRIMARY KEY (created_at, unit_nr, time_cycles)
);

SELECT create_hypertable('rul_predictions', 'created_at', if_not_exists => TRUE);

-- 3. Anomaly Alerts Table
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id SERIAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    unit_nr INT NOT NULL,
    time_cycles INT NOT NULL,
    alert_level VARCHAR(20) NOT NULL,
    health_index DOUBLE PRECISION NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY (created_at, unit_nr, time_cycles)
);

SELECT create_hypertable('anomaly_alerts', 'created_at', if_not_exists => TRUE);
