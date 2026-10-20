"""
Arena 6 - Synthetic maintenance data generator.

Generates fake-but-realistic fleet, spare-parts, repair-slot, defect-log and
work-order data that is *linked* to the C-MAPSS engines already in this repo:

    aircraft (50) --2 engines each--> engines (100, unit_nr 1..100 = FD001 test units)
    engines with low true RUL  --> AI_ALERT defect logs --> work orders --> repair slots + spare parts

Only the Python standard library is used. Output is deterministic for a given --seed.

Usage (from the repository root):
    python -m arena6.generate_data                      # writes arena6/sample_data/*.csv + .json
    python -m arena6.generate_data --seed 7 --aircraft 30
    python -m arena6.generate_data --start-date today   # slots start today instead of the fixed demo date
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUL_FILE = ROOT / "data" / "raw" / "RUL_FD001.txt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "sample_data"
DEFAULT_START_DATE = "2026-11-02"
DEFAULT_SEED = 42

MAX_AIRCRAFT = 50  # 2 engines each -> 100 C-MAPSS FD001 test engines

# Base (airport code) -> hangar name
BASES: Dict[str, str] = {
    "DEL": "Hangar A - Delhi",
    "BOM": "Hangar B - Mumbai",
    "BLR": "Hangar C - Bengaluru",
    "HYD": "Hangar D - Hyderabad",
}

AIRCRAFT_TYPES = ["Airbus A320", "Airbus A321", "Boeing 737-800"]
ENGINE_MODEL = "Simulated turbofan (NASA C-MAPSS FD001)"

# Health thresholds mirror dashboard/app.py: HI = min(1, RUL/125)
HI_HEALTHY = 0.65
HI_WARNING = 0.35
RUL_CAP = 125.0

# Sensor -> (label used in defect text, phrase). Names follow data/DATA_DICTIONARY.md
SENSOR_ISSUES = {
    "s_2": ("LPC outlet temperature (T24)", "drifting {v:.1f} degR above baseline"),
    "s_3": ("HPC outlet temperature (T30)", "running {v:.1f} degR above baseline"),
    "s_4": ("LPT outlet temperature (T50)", "running {v:.1f} degR above baseline"),
    "s_7": ("HPC outlet pressure (P30)", "dropped {v:.1f} psia below baseline"),
    "s_8": ("Fan speed (Nf)", "deviating {v:.1f} rpm from expected"),
    "s_9": ("Core speed (Nc)", "deviating {v:.1f} rpm from expected"),
    "s_11": ("HPC outlet static pressure (Ps30)", "dropped {v:.2f} psia below baseline"),
    "s_12": ("Fuel flow ratio (Phi)", "shifted {v:.2f} pps/psi from baseline"),
    "s_15": ("Bypass ratio (BPR)", "drifting {v:.3f} below baseline"),
    "s_17": ("Bleed enthalpy (htBleed)", "elevated by {v:.1f} BTU/lbm"),
    "s_20": ("HPT coolant bleed (W31)", "reduced by {v:.2f} lbm/s"),
    "s_21": ("LPT coolant bleed (W32)", "reduced by {v:.2f} lbm/s"),
}

# (part name, category, related sensor or None)
PART_TEMPLATES = [
    ("HPC Rotor Blade Set", "Compressor", "s_3"),
    ("HPC Stator Vane Assembly", "Compressor", "s_7"),
    ("LPC Blade Set", "Compressor", "s_2"),
    ("Compressor Seal Kit", "Compressor", "s_11"),
    ("HPT Blade Set", "Turbine", "s_4"),
    ("LPT Nozzle Guide Vane", "Turbine", "s_4"),
    ("Turbine Shroud Segment", "Turbine", None),
    ("HPT Cooling Air Valve", "Turbine", "s_20"),
    ("LPT Coolant Bleed Valve", "Turbine", "s_21"),
    ("Fuel Nozzle", "Combustor", "s_12"),
    ("Igniter Plug", "Combustor", None),
    ("Combustor Liner", "Combustor", None),
    ("Fuel Metering Unit", "Fuel System", "s_12"),
    ("Fuel Pump", "Fuel System", None),
    ("Main Bearing No.3", "Lubrication", "s_9"),
    ("Oil Pump", "Lubrication", None),
    ("Oil Filter", "Lubrication", None),
    ("T24 Temperature Sensor", "Sensors", "s_2"),
    ("T30 Temperature Sensor", "Sensors", "s_3"),
    ("T50 EGT Probe", "Sensors", "s_4"),
    ("P30 Pressure Transducer", "Sensors", "s_7"),
    ("Fan Speed Sensor (Nf)", "Sensors", "s_8"),
    ("Core Speed Sensor (Nc)", "Sensors", "s_9"),
    ("Customer Bleed Valve", "Bleed System", "s_17"),
    ("Variable Bleed Valve", "Bleed System", "s_15"),
]

# ATA chapter -> part names usable for non-sensor (routine) defects
ATA_PARTS = {
    "71": ["Customer Bleed Valve", "Igniter Plug"],
    "72": ["Turbine Shroud Segment", "Combustor Liner", "Igniter Plug"],
    "73": ["Fuel Pump", "Fuel Metering Unit", "Fuel Nozzle"],
    "79": ["Oil Filter", "Oil Pump"],
}

ROUTINE_DEFECTS = [
    ("Oil filter differential pressure indication intermittent", "79"),
    ("Minor fuel weep at fuel nozzle B-line fitting", "73"),
    ("Borescope: light thermal-barrier coating loss on HPT blade (within limits)", "72"),
    ("Igniter plug erosion found during scheduled inspection", "72"),
    ("Bleed valve actuator slow to respond during ground run", "71"),
    ("Oil quantity trending low over last 5 flights", "79"),
    ("Fuel pump outlet pressure fluctuation reported by crew", "73"),
]

AIRFRAME_DEFECTS = [  # not engine-specific -> engine_id stays empty
    ("Cabin pressure controller fault message (reset cleared)", "21"),
    ("Left main gear door seal worn", "52"),
    ("Cockpit display flicker on ground", "31"),
]

REPORTERS = ["Captain", "First Officer", "Line Engineer", "Maintenance Controller"]

TASK_HOURS = {"INSPECTION": 4, "SENSOR_REPLACEMENT": 6, "PART_REPLACEMENT": 12, "OVERHAUL": 48}
PRIORITY_DUE_DAYS = {"P1": 3, "P2": 10, "P3": 30, "P4": 60}

TABLE_ORDER = [
    "aircraft", "engines", "spare_parts", "repair_slots",
    "defect_logs", "work_orders", "work_order_parts",
]


def health_from_rul(rul: float):
    """Return (health_index, status) using the same thresholds as dashboard/app.py."""
    hi = max(0.0, min(1.0, rul / RUL_CAP))
    if hi > HI_HEALTHY:
        return round(hi, 3), "HEALTHY"
    if hi > HI_WARNING:
        return round(hi, 3), "WARNING"
    return round(hi, 3), "CRITICAL"


def load_true_ruls(path: Path, n_units: int, rng: random.Random) -> List[int]:
    """True RUL per FD001 test engine. Falls back to random values if the file is missing."""
    if path.exists():
        values = [int(float(x)) for x in path.read_text().split()]
        if len(values) >= n_units:
            return values[:n_units]
    return [rng.randint(10, 200) for _ in range(n_units)]


def _tail_numbers(rng: random.Random, n: int) -> List[str]:
    seen, out = set(), []
    while len(out) < n:
        t = "VT-" + "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_aircraft_and_engines(rng, n_aircraft, ruls):
    tails = _tail_numbers(rng, n_aircraft)
    base_codes = list(BASES)
    unit_order = list(range(1, n_aircraft * 2 + 1))
    rng.shuffle(unit_order)  # engine unit -> aircraft mapping is random but deterministic

    aircraft, engines = [], []
    for i in range(n_aircraft):
        aid = f"AC-{i + 1:03d}"
        hours = round(rng.uniform(5000, 45000), 1)
        aircraft.append({
            "aircraft_id": aid,
            "tail_number": tails[i],
            "aircraft_type": rng.choice(AIRCRAFT_TYPES),
            "manufacture_year": rng.randint(2006, 2022),
            "total_flight_hours": hours,
            "total_cycles": int(hours / rng.uniform(1.5, 2.2)),
            "base_code": base_codes[i % len(base_codes)],
            "status": "ACTIVE",  # refined after work orders exist
            "last_check_date": "",  # filled by build_all (needs start date)
        })
        for pos in (1, 2):
            unit = unit_order[i * 2 + (pos - 1)]
            rul = ruls[unit - 1]
            hi, status = health_from_rul(rul)
            engines.append({
                "engine_id": f"ENG-{unit:03d}",
                "aircraft_id": aid,
                "position": pos,
                "unit_nr": unit,
                "engine_model": ENGINE_MODEL,
                "true_rul_cycles": rul,
                "health_index": hi,
                "health_status": status,
            })
    engines.sort(key=lambda e: e["unit_nr"])
    return aircraft, engines


def build_spare_parts(rng, n_parts):
    parts, idx = [], 1
    revision = 0
    bases = list(BASES)
    while len(parts) < n_parts:
        rev_label = string.ascii_uppercase[revision % 26]
        for name, category, sensor in PART_TEMPLATES:
            for base in bases:
                if len(parts) >= n_parts:
                    break
                roll = rng.random()
                stock = 0 if roll < 0.08 else rng.randint(1, 12)
                parts.append({
                    "part_id": f"SP-{idx:04d}",
                    "part_number": f"PN-{10000 + idx * 7}",
                    "name": name,
                    "category": category,
                    "related_sensor": sensor or "",
                    "variant": f"Rev {rev_label}",
                    "warehouse_code": base,
                    "quantity_in_stock": stock,
                    "reorder_level": rng.randint(1, 4),
                    "unit_cost_inr": round(rng.uniform(4_000, 2_500_000), 2),
                    "lead_time_days": rng.randint(2, 45),
                    "is_critical": 1 if category in ("Compressor", "Turbine", "Combustor") else 0,
                })
                idx += 1
        revision += 1
    return parts


def build_repair_slots(rng, n_slots, start: date):
    slots = []
    bases = list(BASES)
    per_base_day = {b: 0 for b in bases}
    for i in range(n_slots):
        base = bases[i % len(bases)]
        day = per_base_day[base]
        per_base_day[base] += 1
        begin = datetime(start.year, start.month, start.day, 8, 0) + timedelta(days=day * 2 + (i % 3))
        hours = rng.choice([8, 12, 24, 48])
        slots.append({
            "slot_id": f"SLOT-{i + 1:03d}",
            "hangar_name": BASES[base],
            "base_code": base,
            "bay": rng.randint(1, 3),
            "start_time": _iso(begin),
            "end_time": _iso(begin + timedelta(hours=hours)),
            "is_available": 1,
        })
    slots.sort(key=lambda s: (s["base_code"], s["start_time"]))
    # first slot of every base starts exactly on the reference day (used for IN_PROGRESS work)
    return slots


def build_defects(rng, aircraft, engines, n_defects, start: date):
    ac_by_id = {a["aircraft_id"]: a for a in aircraft}
    now = datetime(start.year, start.month, start.day, 6, 0)
    defects: List[dict] = []

    def add(**kw):
        defects.append({"defect_id": "", **kw})

    # 1) AI alerts for engines in poor health - the "story" of the project
    for e in engines:
        if e["health_status"] == "CRITICAL":
            send = True
        elif e["health_status"] == "WARNING":
            send = rng.random() < 0.7
        else:
            send = False
        if not send:
            continue
        sensor = rng.choice(list(SENSOR_ISSUES))
        label, phrase = SENSOR_ISSUES[sensor]
        scale = 0.6 if e["health_status"] == "WARNING" else 1.0
        v = rng.uniform(2, 15) * scale
        pred = max(0, int(e["true_rul_cycles"] + rng.randint(-6, 6)))
        add(
            aircraft_id=e["aircraft_id"], engine_id=e["engine_id"],
            reported_at=_iso(now - timedelta(days=rng.randint(0, 10), hours=rng.randint(0, 20))),
            source="AI_ALERT", reported_by="AI Monitoring System", ata_chapter="72", sensor=sensor,
            severity="CRITICAL" if e["health_status"] == "CRITICAL" else "MAJOR",
            description=f"{label} {phrase.format(v=v)}. Predicted RUL about {pred} cycles (TCN).",
            status="OPEN",
        )

    # 2) Routine findings on random engines
    engine_ids = [e["engine_id"] for e in engines]
    eng_ac = {e["engine_id"]: e["aircraft_id"] for e in engines}
    while len(defects) < n_defects:
        if rng.random() < 0.12:
            text, ata = rng.choice(AIRFRAME_DEFECTS)
            ac = rng.choice(aircraft)["aircraft_id"]
            eng_id, sensor = "", ""
        else:
            text, ata = rng.choice(ROUTINE_DEFECTS)
            eng_id = rng.choice(engine_ids)
            ac, sensor = eng_ac[eng_id], ""
        add(
            aircraft_id=ac, engine_id=eng_id,
            reported_at=_iso(now - timedelta(days=rng.randint(0, 55), hours=rng.randint(0, 20))),
            source=rng.choice(["PILOT_REPORT", "ROUTINE_INSPECTION", "BOROSCOPE"]),
            reported_by=rng.choice(REPORTERS), ata_chapter=ata, sensor=sensor,
            severity=rng.choices(["MINOR", "MAJOR"], weights=[75, 25])[0],
            description=text, status="OPEN",
        )

    defects = defects[:max(n_defects, 0)] if n_defects < len(defects) else defects
    defects.sort(key=lambda d: d["reported_at"])
    for i, d in enumerate(defects, 1):
        d["defect_id"] = f"DEF-{i:04d}"
    return defects


def _candidate_parts(parts, base, defect):
    in_base = [p for p in parts if p["warehouse_code"] == base]
    if defect["sensor"]:
        c = [p for p in in_base if p["related_sensor"] == defect["sensor"]]
    else:
        names = ATA_PARTS.get(defect["ata_chapter"], [])
        c = [p for p in in_base if p["name"] in names]
    return c or [p for p in in_base if p["category"] == "Sensors"]


def build_work_orders(rng, aircraft, engines, parts, slots, defects, n_work_orders, start: date):
    ac_by_id = {a["aircraft_id"]: a for a in aircraft}
    parts_by_id = {p["part_id"]: p for p in parts}
    slot_pool = {b: [s for s in slots if s["base_code"] == b] for b in BASES}

    # Which defects get a work order: all CRITICAL, most MAJOR, some MINOR
    chosen = []
    for d in defects:
        p = {"CRITICAL": 1.0, "MAJOR": 0.85, "MINOR": 0.35}[d["severity"]]
        if rng.random() < p:
            chosen.append(d)
    chosen.sort(key=lambda d: {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}[d["severity"]])
    chosen = chosen[:n_work_orders]
    chosen.sort(key=lambda d: d["reported_at"])

    work_orders, wo_parts = [], []
    for i, d in enumerate(chosen, 1):
        base = ac_by_id[d["aircraft_id"]]["base_code"]
        cands = _candidate_parts(parts, base, d)
        part = None
        if cands:
            zero = [p for p in cands if p["quantity_in_stock"] == 0]
            if d["severity"] == "CRITICAL" and zero and rng.random() < 0.25:
                part = rng.choice(zero)
            else:
                part = rng.choice(cands)

        sev = d["severity"]
        if sev == "CRITICAL":
            priority = "P1"
            task = rng.choice(["PART_REPLACEMENT", "OVERHAUL"])
            status = rng.choices(["IN_PROGRESS", "SCHEDULED", "PLANNED"], weights=[25, 60, 15])[0]
        elif sev == "MAJOR":
            priority = "P2"
            task = rng.choice(["PART_REPLACEMENT", "SENSOR_REPLACEMENT", "INSPECTION"])
            status = rng.choices(["IN_PROGRESS", "SCHEDULED", "PLANNED"], weights=[10, 55, 35])[0]
        else:
            priority = rng.choice(["P3", "P4"])
            task = rng.choice(["INSPECTION", "SENSOR_REPLACEMENT"])
            status = rng.choices(["COMPLETED", "PLANNED", "CANCELLED"], weights=[50, 30, 20])[0]

        if part is not None and part["category"] == "Sensors":
            task = "SENSOR_REPLACEMENT"  # a sensor part means a sensor job, not an overhaul

        if part is not None and part["quantity_in_stock"] == 0 and status in ("IN_PROGRESS", "SCHEDULED", "PLANNED"):
            status = "WAITING_PARTS"

        slot_id = ""
        if status in ("IN_PROGRESS", "SCHEDULED"):
            free = [s for s in slot_pool[base] if s["is_available"] == 1]
            if free:
                slot = free[0]  # earliest free slot at the aircraft's base
                slot["is_available"] = 0
                slot_id = slot["slot_id"]
            else:
                status = "PLANNED"

        created = datetime.strptime(d["reported_at"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=rng.randint(1, 24))
        wo_id = f"WO-2026-{i:04d}"
        if status in ("COMPLETED", "CANCELLED"):
            due = created + timedelta(days=PRIORITY_DUE_DAYS[priority])
        else:  # open work is due in the future, counted from the reference date (no RNG: keeps data stable)
            due = datetime(start.year, start.month, start.day) + timedelta(days=PRIORITY_DUE_DAYS[priority] + i % 3)
        work_orders.append({
            "work_order_id": wo_id,
            "defect_id": d["defect_id"],
            "aircraft_id": d["aircraft_id"],
            "engine_id": d["engine_id"],
            "task_type": task,
            "priority": priority,
            "status": status,
            "repair_slot_id": slot_id,
            "estimated_hours": TASK_HOURS[task],
            "assigned_team": f"Team-{base}-{rng.randint(1, 3)}",
            "created_at": _iso(created),
            "due_date": due.strftime("%Y-%m-%d"),
        })
        if part is not None:
            wo_parts.append({"work_order_id": wo_id, "part_id": part["part_id"], "quantity": rng.randint(1, 2)})

    # Guarantee at least one "critical engine blocked by missing part" scenario for the demo
    if not any(w["status"] == "WAITING_PARTS" for w in work_orders):
        for w in work_orders:
            if w["priority"] == "P1" and w["status"] in ("SCHEDULED", "PLANNED", "IN_PROGRESS"):
                link = next((x for x in wo_parts if x["work_order_id"] == w["work_order_id"]), None)
                if link:
                    parts_by_id[link["part_id"]]["quantity_in_stock"] = 0
                    if w["repair_slot_id"]:
                        for s in slots:
                            if s["slot_id"] == w["repair_slot_id"]:
                                s["is_available"] = 1
                        w["repair_slot_id"] = ""
                    w["status"] = "WAITING_PARTS"
                    break

    # Defect status follows its work order
    wo_by_defect = {w["defect_id"]: w for w in work_orders}
    for d in defects:
        w = wo_by_defect.get(d["defect_id"])
        if w is None:
            d["status"] = rng.choice(["OPEN", "IN_REVIEW"])
        elif w["status"] == "COMPLETED":
            d["status"] = "CLOSED"
        else:
            d["status"] = "WORK_ORDER_CREATED"

    # Aircraft status follows engine work
    for a in aircraft:
        wos = [w for w in work_orders if w["aircraft_id"] == a["aircraft_id"]]
        if any(w["status"] == "WAITING_PARTS" and w["priority"] == "P1" for w in wos):
            a["status"] = "AOG"
        elif any(w["status"] == "IN_PROGRESS" for w in wos):
            a["status"] = "IN_MAINTENANCE"
        else:
            a["status"] = "ACTIVE"
    return work_orders, wo_parts


def build_demo_scenarios(aircraft, engines, work_orders, wo_parts, parts):
    ac_by_id = {a["aircraft_id"]: a for a in aircraft}
    wo_by_engine = {}
    for w in work_orders:
        wo_by_engine.setdefault(w["engine_id"], []).append(w)

    def describe(e, label):
        a = ac_by_id[e["aircraft_id"]]
        return {
            "scenario": label,
            "unit_nr": e["unit_nr"], "engine_id": e["engine_id"],
            "aircraft_id": a["aircraft_id"], "tail_number": a["tail_number"], "base_code": a["base_code"],
            "true_rul_cycles": e["true_rul_cycles"], "health_status": e["health_status"],
            "work_orders": [
                {k: w[k] for k in ("work_order_id", "priority", "status", "task_type", "repair_slot_id")}
                for w in wo_by_engine.get(e["engine_id"], [])
            ],
        }

    by_rul = sorted(engines, key=lambda e: e["true_rul_cycles"])
    out = {"critical_engine": describe(by_rul[0], "Lowest RUL in the fleet")}
    warnings = [e for e in by_rul if e["health_status"] == "WARNING"]
    if warnings:
        out["warning_engine"] = describe(warnings[len(warnings) // 2], "Mid-life engine showing early drift")
    out["healthy_engine"] = describe(by_rul[-1], "Highest RUL in the fleet")

    parts_by_id = {p["part_id"]: p for p in parts}
    for w in work_orders:
        if w["status"] == "WAITING_PARTS" and w["priority"] == "P1":
            link = next((x for x in wo_parts if x["work_order_id"] == w["work_order_id"]), None)
            e = next(x for x in engines if x["engine_id"] == w["engine_id"])
            d = describe(e, "P1 work order blocked by a part that is out of stock (AOG risk)")
            if link:
                p = parts_by_id[link["part_id"]]
                d["missing_part"] = {"part_id": p["part_id"], "name": p["name"], "warehouse": p["warehouse_code"],
                                     "lead_time_days": p["lead_time_days"]}
            out["aog_engine"] = d
            break
    return out


def build_all(seed: int = DEFAULT_SEED, n_aircraft: int = MAX_AIRCRAFT, n_parts: int = 200,
              n_slots: int = 30, n_defects: int = 60, n_work_orders: int = 40,
              start_date: str = DEFAULT_START_DATE, rul_file: Optional[Path] = None) -> Dict[str, list]:
    """Build every table. Returns {table_name: [row dicts]} plus a 'demo_scenarios' dict."""
    if not 1 <= n_aircraft <= MAX_AIRCRAFT:
        raise ValueError(f"n_aircraft must be between 1 and {MAX_AIRCRAFT} (2 C-MAPSS engines per aircraft).")
    for name, val in (("n_parts", n_parts), ("n_slots", n_slots), ("n_defects", n_defects), ("n_work_orders", n_work_orders)):
        if val < 0:
            raise ValueError(f"{name} must be >= 0")

    start = date.today() if start_date == "today" else date.fromisoformat(start_date)
    rng = random.Random(seed)
    ruls = load_true_ruls(rul_file or DEFAULT_RUL_FILE, MAX_AIRCRAFT * 2, rng)

    aircraft, engines = build_aircraft_and_engines(rng, n_aircraft, ruls)
    for a in aircraft:
        a["last_check_date"] = (start - timedelta(days=rng.randint(10, 180))).isoformat()
    parts = build_spare_parts(rng, n_parts)
    slots = build_repair_slots(rng, n_slots, start)
    defects = build_defects(rng, aircraft, engines, n_defects, start)
    work_orders, wo_parts = build_work_orders(rng, aircraft, engines, parts, slots, defects, n_work_orders, start)
    scenarios = build_demo_scenarios(aircraft, engines, work_orders, wo_parts, parts)

    return {
        "aircraft": aircraft, "engines": engines, "spare_parts": parts, "repair_slots": slots,
        "defect_logs": defects, "work_orders": work_orders, "work_order_parts": wo_parts,
        "demo_scenarios": scenarios,
    }


def write_outputs(data: Dict[str, list], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table in TABLE_ORDER:
        rows = data[table]
        path = out_dir / f"{table}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        written.append(path)
    json_path = out_dir / "maintenance_data.json"
    json_path.write_text(json.dumps({t: data[t] for t in TABLE_ORDER}, indent=2), encoding="utf-8")
    scen_path = out_dir / "demo_scenarios.json"
    scen_path.write_text(json.dumps(data["demo_scenarios"], indent=2), encoding="utf-8")
    return written + [json_path, scen_path]


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate synthetic aircraft / parts / slots / defects / work-order data.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output folder (default: arena6/sample_data)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--aircraft", type=int, default=MAX_AIRCRAFT, help=f"1-{MAX_AIRCRAFT} (default 50)")
    p.add_argument("--parts", type=int, default=200)
    p.add_argument("--slots", type=int, default=30)
    p.add_argument("--defects", type=int, default=60)
    p.add_argument("--work-orders", type=int, default=40)
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD or 'today'")
    args = p.parse_args(argv)

    data = build_all(args.seed, args.aircraft, args.parts, args.slots, args.defects, args.work_orders, args.start_date)
    files = write_outputs(data, args.out)
    print(f"Generated (seed={args.seed}):")
    for t in TABLE_ORDER:
        print(f"  {t:<18} {len(data[t]):>4} rows")
    print(f"Wrote {len(files)} files to {args.out}")


if __name__ == "__main__":
    main()
