"""
Arena 6 - Populate the development database with demo data (one command).

    python -m arena6.seed_db                       # seeds ./telemetry.db (the DB the API uses by default)
    python -m arena6.seed_db --db my_demo.db       # seeds a separate file (point the API at it with DATABASE_URL)
    python -m arena6.seed_db --export-sql arena6/sample_data/seed_postgres.sql   # SQL file for TimescaleDB/PostgreSQL

What it does
  1. (Re)creates ONLY the 7 Arena 6 tables: aircraft, engines, spare_parts, repair_slots,
     defect_logs, work_orders, work_order_parts. Re-running is safe.
  2. Loads real C-MAPSS cycles for a handful of demo engines into `telemetry_records`
     so /predict, /detect and /explain work immediately. Existing rows are never deleted;
     (unit_nr, time_cycles) pairs that already exist are skipped.

It never modifies backend code, and never touches `rul_predictions` or `anomaly_alerts`.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .generate_data import (
    DEFAULT_SEED, DEFAULT_START_DATE, MAX_AIRCRAFT, TABLE_ORDER, build_all,
)

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = Path(__file__).resolve().parent / "sql"
DEFAULT_DB = ROOT / "telemetry.db"
TEST_FILE = ROOT / "data" / "raw" / "test_FD001.txt"

COLS = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]

# Same shape SQLAlchemy creates for backend/db.py::TelemetryRecord (used only if the table is missing).
TELEMETRY_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS telemetry_records (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    unit_nr INTEGER NOT NULL,
    time_cycles INTEGER NOT NULL,
    recorded_at DATETIME,
    setting_1 FLOAT,
    setting_2 FLOAT,
    setting_3 FLOAT,
    sensor_data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_telemetry_records_unit_nr ON telemetry_records (unit_nr);
CREATE INDEX IF NOT EXISTS ix_telemetry_records_time_cycles ON telemetry_records (time_cycles);
"""


# --------------------------------------------------------------------------- telemetry

def pick_demo_units(engines: List[dict], scenarios: dict, n_units: int) -> List[int]:
    """Scenario engines first (critical / warning / healthy / AOG), then evenly spaced by RUL."""
    if n_units <= 0:
        return []
    units: List[int] = []
    for key in ("critical_engine", "aog_engine", "warning_engine", "healthy_engine"):
        sc = scenarios.get(key)
        if sc and sc["unit_nr"] not in units:
            units.append(sc["unit_nr"])
    units = units[:n_units]
    ordered = [e["unit_nr"] for e in sorted(engines, key=lambda e: e["true_rul_cycles"])]
    if len(units) < n_units and ordered:
        step = max(1, len(ordered) // n_units)
        for u in ordered[::step]:
            if len(units) >= n_units:
                break
            if u not in units:
                units.append(u)
        for u in ordered:  # top up if spacing left gaps
            if len(units) >= n_units:
                break
            if u not in units:
                units.append(u)
    return sorted(units)


def read_telemetry(test_file: Path, units: List[int], last_n: int) -> Dict[int, List[dict]]:
    """Return the last `last_n` cycles of each requested unit from the raw C-MAPSS test file."""
    wanted = set(units)
    per_unit: Dict[int, List[dict]] = {u: [] for u in units}
    with open(test_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 26:
                continue
            unit = int(float(parts[0]))
            if unit not in wanted:
                continue
            vals = parts[:26]
            row = {c: float(v) for c, v in zip(COLS, vals)}
            row["unit_nr"], row["time_cycles"] = unit, int(float(vals[1]))
            per_unit[unit].append(row)
    return {u: sorted(rows, key=lambda r: r["time_cycles"])[-last_n:] for u, rows in per_unit.items()}


def telemetry_to_records(rows_by_unit: Dict[int, List[dict]]) -> List[dict]:
    """Flatten to DB-shaped records, with recent timestamps (one minute apart per engine)."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC, like backend/db.py
    out = []
    for unit, rows in rows_by_unit.items():
        for i, r in enumerate(rows):
            out.append({
                "unit_nr": unit,
                "time_cycles": r["time_cycles"],
                "recorded_at": (now - timedelta(minutes=len(rows) - i)).strftime("%Y-%m-%d %H:%M:%S.%f"),
                "setting_1": r["setting_1"], "setting_2": r["setting_2"], "setting_3": r["setting_3"],
                "sensor_data": json.dumps({f"s_{k}": r[f"s_{k}"] for k in range(1, 22)}),
            })
    return out


# --------------------------------------------------------------------------- SQLite seeding

def seed_sqlite(db_path: Path, data: Dict[str, list], telemetry: List[dict]) -> Dict[str, int]:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for table in reversed(TABLE_ORDER):  # only Arena 6 tables are dropped
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript((SQL_DIR / "schema_sqlite.sql").read_text())

        for table in TABLE_ORDER:
            rows = data[table]
            if not rows:
                continue
            cols = list(rows[0].keys())
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
            conn.executemany(sql, [tuple(_nullable(r[c]) for c in cols) for r in rows])

        added = 0
        if telemetry:
            conn.executescript(TELEMETRY_DDL_SQLITE)
            existing = {(u, c) for u, c in conn.execute("SELECT unit_nr, time_cycles FROM telemetry_records")}
            for rec in telemetry:
                if (rec["unit_nr"], rec["time_cycles"]) in existing:
                    continue
                conn.execute(
                    "INSERT INTO telemetry_records (unit_nr, time_cycles, recorded_at, setting_1, setting_2, setting_3, sensor_data) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (rec["unit_nr"], rec["time_cycles"], rec["recorded_at"], rec["setting_1"],
                     rec["setting_2"], rec["setting_3"], rec["sensor_data"]),
                )
                added += 1
        conn.commit()

        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise RuntimeError(f"Foreign key violations after seeding: {bad[:5]}")

        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLE_ORDER}
        counts["telemetry_rows_added"] = added
        return counts
    finally:
        conn.close()


def _nullable(v):
    """CSV-style empty strings become NULL so optional foreign keys stay valid."""
    return None if v == "" else v


# --------------------------------------------------------------------------- PostgreSQL export

def _lit(v) -> str:
    if v is None or v == "":
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def export_postgres_sql(path: Path, data: Dict[str, list], telemetry: List[dict]) -> Path:
    lines = [
        "-- Arena 6 seed data for PostgreSQL / TimescaleDB. Generated by arena6/seed_db.py",
        "-- Run:  psql \"$DATABASE_URL\" -f " + Path(path).name,
        "BEGIN;",
    ]
    for table in reversed(TABLE_ORDER):
        lines.append(f"DROP TABLE IF EXISTS {table} CASCADE;")
    lines.append((SQL_DIR / "schema_postgres.sql").read_text())
    for table in TABLE_ORDER:
        for r in data[table]:
            cols = list(r.keys())
            lines.append(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(_lit(r[c]) for c in cols)});")
    if telemetry:
        lines.append("-- Demo telemetry (requires telemetry_records from deployment/init_timescaledb.sql)")
        for r in telemetry:
            lines.append(
                "INSERT INTO telemetry_records (recorded_at, unit_nr, time_cycles, setting_1, setting_2, setting_3, sensor_data) "
                f"VALUES ({_lit(r['recorded_at'])}, {r['unit_nr']}, {r['time_cycles']}, {r['setting_1']!r}, "
                f"{r['setting_2']!r}, {r['setting_3']!r}, {_lit(r['sensor_data'])}::jsonb) ON CONFLICT DO NOTHING;"
            )
    lines.append("COMMIT;")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Seed the dev database with Arena 6 demo data.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite file (default: {DEFAULT_DB.name} in repo root)")
    p.add_argument("--export-sql", type=Path, default=None, help="Write a PostgreSQL/TimescaleDB .sql file instead of touching a DB")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--aircraft", type=int, default=MAX_AIRCRAFT)
    p.add_argument("--parts", type=int, default=200)
    p.add_argument("--slots", type=int, default=30)
    p.add_argument("--defects", type=int, default=60)
    p.add_argument("--work-orders", type=int, default=40)
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD or 'today'")
    p.add_argument("--telemetry-units", type=int, default=10, help="How many demo engines get telemetry (default 10)")
    p.add_argument("--telemetry-cycles", type=int, default=50, help="Cycles per engine (default 50; API history max is 200)")
    p.add_argument("--no-telemetry", action="store_true", help="Skip loading C-MAPSS telemetry")
    args = p.parse_args(argv)

    data = build_all(args.seed, args.aircraft, args.parts, args.slots, args.defects, args.work_orders, args.start_date)

    telemetry: List[dict] = []
    if not args.no_telemetry and args.telemetry_units > 0:
        if TEST_FILE.exists():
            units = pick_demo_units(data["engines"], data["demo_scenarios"], args.telemetry_units)
            telemetry = telemetry_to_records(read_telemetry(TEST_FILE, units, args.telemetry_cycles))
            print(f"Telemetry: units {units}, up to {args.telemetry_cycles} cycles each")
        else:
            print(f"WARNING: {TEST_FILE} not found - skipping telemetry (run data/download_data.py first).")

    if args.export_sql:
        out = export_postgres_sql(args.export_sql, data, telemetry)
        print(f"Wrote PostgreSQL seed file: {out}")
        return 0

    if not args.db.parent.exists():
        print(f"ERROR: folder does not exist: {args.db.parent}")
        return 2
    counts = seed_sqlite(args.db, data, telemetry)
    print(f"Seeded {args.db}:")
    for k, v in counts.items():
        print(f"  {k:<22} {v}")
    print("Done. Start the API with:  uvicorn backend.main:app --port 8000")
    if args.db.resolve() != DEFAULT_DB.resolve():
        print(f"NOTE: the API reads {DEFAULT_DB.name} unless you set DATABASE_URL=sqlite+aiosqlite:///{args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
