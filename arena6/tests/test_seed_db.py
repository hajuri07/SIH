"""Arena 6 - tests for the database seeding script (uses a temporary SQLite file, no API needed)."""

import json
import sqlite3

import pytest

from arena6 import seed_db
from arena6.generate_data import build_all, TABLE_ORDER

DATA = build_all()
UNITS = seed_db.pick_demo_units(DATA["engines"], DATA["demo_scenarios"], 10)
TELEMETRY = seed_db.telemetry_to_records(seed_db.read_telemetry(seed_db.TEST_FILE, UNITS, 50)) \
    if seed_db.TEST_FILE.exists() else []


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _db(tmp_path, name="seed.db"):
    return tmp_path / name


# ------------------------------------------------------------------ seeding

def test_seed_creates_all_tables_with_expected_counts(tmp_path):
    counts = seed_db.seed_sqlite(_db(tmp_path), DATA, [])
    assert counts["aircraft"] == 50
    assert counts["engines"] == 100
    assert counts["spare_parts"] == 200
    assert counts["repair_slots"] == 30
    assert counts["defect_logs"] == 60
    assert counts["work_orders"] == 40
    assert counts["work_order_parts"] == len(DATA["work_order_parts"])


def test_seed_leaves_no_foreign_key_violations(tmp_path):
    seed_db.seed_sqlite(_db(tmp_path), DATA, [])
    conn = sqlite3.connect(_db(tmp_path))
    conn.execute("PRAGMA foreign_keys = ON")
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_empty_strings_become_null(tmp_path):
    seed_db.seed_sqlite(_db(tmp_path), DATA, [])
    conn = sqlite3.connect(_db(tmp_path))
    assert conn.execute("SELECT COUNT(*) FROM defect_logs WHERE engine_id = ''").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM defect_logs WHERE engine_id IS NULL").fetchone()[0] > 0
    conn.close()


def test_seeding_twice_does_not_duplicate_anything(tmp_path):
    first = seed_db.seed_sqlite(_db(tmp_path), DATA, TELEMETRY)
    second = seed_db.seed_sqlite(_db(tmp_path), DATA, TELEMETRY)
    for t in TABLE_ORDER:
        assert first[t] == second[t]
    assert second["telemetry_rows_added"] == 0
    conn = sqlite3.connect(_db(tmp_path))
    assert _count(conn, "telemetry_records") == first["telemetry_rows_added"]
    conn.close()


def test_telemetry_rows_match_what_the_api_reads(tmp_path):
    if not TELEMETRY:
        return  # raw C-MAPSS file not present on this machine
    seed_db.seed_sqlite(_db(tmp_path), DATA, TELEMETRY)
    conn = sqlite3.connect(_db(tmp_path))
    unit = UNITS[0]
    rows = conn.execute(
        "SELECT unit_nr, time_cycles, setting_3, sensor_data FROM telemetry_records WHERE unit_nr=? ORDER BY time_cycles",
        (unit,)).fetchall()
    assert 1 <= len(rows) <= 50
    cycles = [r[1] for r in rows]
    assert cycles == sorted(cycles) and len(set(cycles)) == len(cycles)
    sensors = json.loads(rows[0][3])
    assert set(sensors) == {f"s_{i}" for i in range(1, 22)}
    assert all(isinstance(v, float) for v in sensors.values())
    conn.close()


def test_existing_backend_data_is_preserved(tmp_path):
    path = _db(tmp_path)
    conn = sqlite3.connect(path)
    conn.executescript(seed_db.TELEMETRY_DDL_SQLITE)
    conn.execute("INSERT INTO telemetry_records (unit_nr, time_cycles, sensor_data) VALUES (999999, 1, '{}')")
    conn.execute("CREATE TABLE rul_predictions (id INTEGER PRIMARY KEY, unit_nr INTEGER)")
    conn.execute("INSERT INTO rul_predictions (unit_nr) VALUES (999999)")
    conn.execute("CREATE TABLE anomaly_alerts (id INTEGER PRIMARY KEY, unit_nr INTEGER)")
    conn.execute("INSERT INTO anomaly_alerts (unit_nr) VALUES (999999)")
    conn.commit()
    conn.close()

    seed_db.seed_sqlite(path, DATA, TELEMETRY)

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM telemetry_records WHERE unit_nr=999999").fetchone()[0] == 1
    assert _count(conn, "rul_predictions") == 1
    assert _count(conn, "anomaly_alerts") == 1
    conn.close()


def test_without_telemetry_no_telemetry_table_is_created(tmp_path):
    seed_db.seed_sqlite(_db(tmp_path), DATA, [])
    conn = sqlite3.connect(_db(tmp_path))
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "telemetry_records" not in names
    conn.close()


# ------------------------------------------------------------------ constraints reject bad data

def _seeded(tmp_path):
    seed_db.seed_sqlite(_db(tmp_path), DATA, [])
    conn = sqlite3.connect(_db(tmp_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def test_invalid_aircraft_status_is_rejected(tmp_path):
    conn = _seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO aircraft VALUES ('AC-999','VT-ZZZ','Airbus A320',2015,100,50,'DEL','FLYING',NULL)")
    conn.close()


def test_negative_stock_is_rejected(tmp_path):
    conn = _seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE spare_parts SET quantity_in_stock = -1 WHERE part_id = 'SP-0001'")
    conn.close()


def test_work_order_with_unknown_defect_is_rejected(tmp_path):
    conn = _seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO work_orders (work_order_id, defect_id, aircraft_id, task_type, priority, status, estimated_hours, created_at) "
                     "VALUES ('WO-X','DEF-9999','AC-001','INSPECTION','P3','PLANNED',2,'2026-11-01 10:00:00')")
    conn.close()


def test_slot_ending_before_it_starts_is_rejected(tmp_path):
    conn = _seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO repair_slots VALUES ('SLOT-X','Hangar A - Delhi','DEL',1,'2026-11-02 10:00:00','2026-11-02 08:00:00',1)")
    conn.close()


def test_duplicate_engine_unit_is_rejected(tmp_path):
    conn = _seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO engines VALUES ('ENG-X','AC-001',1,1,'x',10,0.1,'CRITICAL')")
    conn.close()


# ------------------------------------------------------------------ demo engine selection

def test_pick_demo_units_includes_scenario_engines_and_is_unique():
    units = seed_db.pick_demo_units(DATA["engines"], DATA["demo_scenarios"], 10)
    assert len(units) == 10 and len(set(units)) == 10
    for key in ("critical_engine", "healthy_engine"):
        assert DATA["demo_scenarios"][key]["unit_nr"] in units


def test_pick_demo_units_edge_sizes():
    assert seed_db.pick_demo_units(DATA["engines"], DATA["demo_scenarios"], 0) == []
    assert len(seed_db.pick_demo_units(DATA["engines"], DATA["demo_scenarios"], 2)) == 2
    assert len(seed_db.pick_demo_units(DATA["engines"], DATA["demo_scenarios"], 100)) == 100


def test_read_telemetry_returns_only_last_n_cycles_sorted():
    if not seed_db.TEST_FILE.exists():
        return
    rows = seed_db.read_telemetry(seed_db.TEST_FILE, [1], 5)[1]
    assert len(rows) == 5
    assert [r["time_cycles"] for r in rows] == sorted(r["time_cycles"] for r in rows)


# ------------------------------------------------------------------ PostgreSQL export

def test_postgres_export_contains_schema_and_all_rows(tmp_path):
    out = seed_db.export_postgres_sql(tmp_path / "seed.sql", DATA, TELEMETRY)
    text = out.read_text()
    assert text.startswith("-- Arena 6 seed data")
    assert "BEGIN;" in text and text.rstrip().endswith("COMMIT;")
    assert text.count("CREATE TABLE IF NOT EXISTS") == 7
    expected = sum(len(DATA[t]) for t in TABLE_ORDER)
    assert text.count("INSERT INTO ") - text.count("INSERT INTO telemetry_records") == expected
    assert "TIMESTAMP" in text and "DOUBLE PRECISION" in text


def test_sql_literal_escaping():
    assert seed_db._lit("it's") == "'it''s'"
    assert seed_db._lit(None) == "NULL"
    assert seed_db._lit("") == "NULL"
    assert seed_db._lit(5) == "5"
    assert seed_db._lit(2.5) == "2.5"


# ------------------------------------------------------------------ command line

def test_cli_seeds_a_database(tmp_path):
    code = seed_db.main(["--db", str(_db(tmp_path)), "--no-telemetry"])
    assert code == 0
    conn = sqlite3.connect(_db(tmp_path))
    assert _count(conn, "aircraft") == 50
    conn.close()


def test_cli_fails_cleanly_when_folder_is_missing(tmp_path):
    assert seed_db.main(["--db", str(tmp_path / "no_such_folder" / "x.db"), "--no-telemetry"]) == 2


def test_cli_custom_volumes(tmp_path):
    code = seed_db.main(["--db", str(_db(tmp_path)), "--no-telemetry", "--aircraft", "10", "--parts", "20",
                         "--slots", "5", "--defects", "15", "--work-orders", "8"])
    assert code == 0
    conn = sqlite3.connect(_db(tmp_path))
    assert (_count(conn, "aircraft"), _count(conn, "engines"), _count(conn, "spare_parts"), _count(conn, "repair_slots")) == (10, 20, 20, 5)
    conn.close()


# ------------------------------------------------------------------ demo views

def test_demo_queries_run_and_return_rows(tmp_path):
    from arena6 import show_demo_data
    seed_db.seed_sqlite(_db(tmp_path), DATA, [])
    conn = sqlite3.connect(_db(tmp_path))
    results = dict((t, rows) for t, _, rows in show_demo_data.run_all(conn))
    conn.close()
    assert len(results) == 6
    assert all(rows for rows in results.values()), [t for t, r in results.items() if not r]


def test_demo_views_report_clear_error_on_unseeded_db(tmp_path):
    from arena6 import show_demo_data
    sqlite3.connect(_db(tmp_path, "empty.db")).close()
    assert show_demo_data.main(["--db", str(_db(tmp_path, "empty.db"))]) == 2
    assert show_demo_data.main(["--db", str(tmp_path / "missing.db")]) == 2
