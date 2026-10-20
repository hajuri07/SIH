-- Arena 6 maintenance tables (PostgreSQL / TimescaleDB). Safe to run next to the backend's own tables
-- (telemetry_records, rul_predictions, anomaly_alerts) - it never touches them.

CREATE TABLE IF NOT EXISTS aircraft (
    aircraft_id        TEXT PRIMARY KEY,
    tail_number        TEXT NOT NULL UNIQUE,
    aircraft_type      TEXT NOT NULL,
    manufacture_year   INTEGER NOT NULL CHECK (manufacture_year BETWEEN 1980 AND 2100),
    total_flight_hours DOUBLE PRECISION NOT NULL CHECK (total_flight_hours >= 0),
    total_cycles       INTEGER NOT NULL CHECK (total_cycles >= 0),
    base_code          TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('ACTIVE','IN_MAINTENANCE','AOG')),
    last_check_date    DATE
);

CREATE TABLE IF NOT EXISTS engines (
    engine_id        TEXT PRIMARY KEY,
    aircraft_id      TEXT NOT NULL REFERENCES aircraft(aircraft_id),
    position         INTEGER NOT NULL CHECK (position IN (1, 2)),
    unit_nr          INTEGER NOT NULL UNIQUE,          -- = unit_nr in telemetry_records / C-MAPSS
    engine_model     TEXT NOT NULL,
    true_rul_cycles  INTEGER CHECK (true_rul_cycles >= 0),
    health_index     DOUBLE PRECISION CHECK (health_index BETWEEN 0 AND 1),
    health_status    TEXT CHECK (health_status IN ('HEALTHY','WARNING','CRITICAL')),
    UNIQUE (aircraft_id, position)
);

CREATE TABLE IF NOT EXISTS spare_parts (
    part_id           TEXT PRIMARY KEY,
    part_number       TEXT NOT NULL UNIQUE,
    name              TEXT NOT NULL,
    category          TEXT NOT NULL,
    related_sensor    TEXT,
    variant           TEXT,
    warehouse_code    TEXT NOT NULL,
    quantity_in_stock INTEGER NOT NULL CHECK (quantity_in_stock >= 0),
    reorder_level     INTEGER NOT NULL CHECK (reorder_level >= 0),
    unit_cost_inr     DOUBLE PRECISION NOT NULL CHECK (unit_cost_inr >= 0),
    lead_time_days    INTEGER NOT NULL CHECK (lead_time_days >= 0),
    is_critical       INTEGER NOT NULL CHECK (is_critical IN (0, 1))
);

CREATE TABLE IF NOT EXISTS repair_slots (
    slot_id      TEXT PRIMARY KEY,
    hangar_name  TEXT NOT NULL,
    base_code    TEXT NOT NULL,
    bay          INTEGER NOT NULL CHECK (bay >= 1),
    start_time   TIMESTAMP NOT NULL,
    end_time     TIMESTAMP NOT NULL,
    is_available INTEGER NOT NULL CHECK (is_available IN (0, 1)),
    CHECK (end_time > start_time)
);

CREATE TABLE IF NOT EXISTS defect_logs (
    defect_id    TEXT PRIMARY KEY,
    aircraft_id  TEXT NOT NULL REFERENCES aircraft(aircraft_id),
    engine_id    TEXT REFERENCES engines(engine_id),   -- NULL for airframe-level defects
    reported_at  TIMESTAMP NOT NULL,
    source       TEXT NOT NULL CHECK (source IN ('PILOT_REPORT','ROUTINE_INSPECTION','AI_ALERT','BOROSCOPE')),
    reported_by  TEXT NOT NULL,
    ata_chapter  TEXT,
    sensor       TEXT,                                 -- e.g. s_3, NULL when not sensor related
    severity     TEXT NOT NULL CHECK (severity IN ('MINOR','MAJOR','CRITICAL')),
    description  TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('OPEN','IN_REVIEW','WORK_ORDER_CREATED','CLOSED'))
);

CREATE TABLE IF NOT EXISTS work_orders (
    work_order_id   TEXT PRIMARY KEY,
    defect_id       TEXT NOT NULL REFERENCES defect_logs(defect_id),
    aircraft_id     TEXT NOT NULL REFERENCES aircraft(aircraft_id),
    engine_id       TEXT REFERENCES engines(engine_id),
    task_type       TEXT NOT NULL CHECK (task_type IN ('INSPECTION','SENSOR_REPLACEMENT','PART_REPLACEMENT','OVERHAUL')),
    priority        TEXT NOT NULL CHECK (priority IN ('P1','P2','P3','P4')),
    status          TEXT NOT NULL CHECK (status IN ('PLANNED','SCHEDULED','IN_PROGRESS','WAITING_PARTS','COMPLETED','CANCELLED')),
    repair_slot_id  TEXT REFERENCES repair_slots(slot_id),
    estimated_hours DOUBLE PRECISION NOT NULL CHECK (estimated_hours > 0),
    assigned_team   TEXT,
    created_at      TIMESTAMP NOT NULL,
    due_date        DATE
);

CREATE TABLE IF NOT EXISTS work_order_parts (
    work_order_id TEXT NOT NULL REFERENCES work_orders(work_order_id),
    part_id       TEXT NOT NULL REFERENCES spare_parts(part_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (work_order_id, part_id)
);

CREATE INDEX IF NOT EXISTS idx_engines_aircraft  ON engines(aircraft_id);
CREATE INDEX IF NOT EXISTS idx_defects_aircraft  ON defect_logs(aircraft_id);
CREATE INDEX IF NOT EXISTS idx_defects_engine    ON defect_logs(engine_id);
CREATE INDEX IF NOT EXISTS idx_wo_status         ON work_orders(status, priority);
CREATE INDEX IF NOT EXISTS idx_parts_warehouse   ON spare_parts(warehouse_code, name);
CREATE INDEX IF NOT EXISTS idx_slots_base_avail  ON repair_slots(base_code, is_available, start_time);
