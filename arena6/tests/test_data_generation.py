"""Arena 6 - tests for the synthetic data generator (no API, no database needed)."""

from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

from arena6.generate_data import build_all, health_from_rul, write_outputs, TABLE_ORDER

DATA = build_all()  # defaults: 50 aircraft, 200 parts, 30 slots, 60 defects, 40 work orders
FMT = "%Y-%m-%d %H:%M:%S"


def _ids(table, key):
    return [r[key] for r in DATA[table]]


# ------------------------------------------------------------------ volumes & determinism

def test_default_volumes():
    assert len(DATA["aircraft"]) == 50
    assert len(DATA["engines"]) == 100
    assert len(DATA["spare_parts"]) == 200
    assert len(DATA["repair_slots"]) == 30
    assert len(DATA["defect_logs"]) == 60
    assert len(DATA["work_orders"]) == 40


def test_same_seed_gives_identical_data():
    again = build_all(seed=42)
    for t in TABLE_ORDER:
        assert again[t] == DATA[t], f"{t} differs between runs with the same seed"


def test_different_seed_gives_different_data():
    other = build_all(seed=43)
    assert other["aircraft"] != DATA["aircraft"]


def test_csv_and_json_files_are_written(tmp_path):
    files = write_outputs(DATA, tmp_path)
    names = {Path(f).name for f in files}
    assert {f"{t}.csv" for t in TABLE_ORDER} <= names
    assert "maintenance_data.json" in names and "demo_scenarios.json" in names


# ------------------------------------------------------------------ uniqueness

def test_primary_keys_are_unique():
    for table, key in [("aircraft", "aircraft_id"), ("engines", "engine_id"), ("spare_parts", "part_id"),
                       ("repair_slots", "slot_id"), ("defect_logs", "defect_id"), ("work_orders", "work_order_id")]:
        ids = _ids(table, key)
        assert len(ids) == len(set(ids)), f"duplicate {key} in {table}"


def test_tail_numbers_and_part_numbers_are_unique():
    tails = _ids("aircraft", "tail_number")
    pns = _ids("spare_parts", "part_number")
    assert len(tails) == len(set(tails))
    assert len(pns) == len(set(pns))


def test_no_duplicate_rows_in_any_table():
    for t in TABLE_ORDER:
        rows = [tuple(sorted(r.items())) for r in DATA[t]]
        assert len(rows) == len(set(rows)), f"duplicate rows in {t}"


# ------------------------------------------------------------------ engines <-> C-MAPSS

def test_every_aircraft_has_exactly_two_engines_at_positions_1_and_2():
    per_ac = {}
    for e in DATA["engines"]:
        per_ac.setdefault(e["aircraft_id"], []).append(e["position"])
    assert len(per_ac) == 50
    assert all(sorted(v) == [1, 2] for v in per_ac.values())


def test_engine_units_cover_cmapss_1_to_100_once():
    units = sorted(e["unit_nr"] for e in DATA["engines"])
    assert units == list(range(1, 101))


def test_engine_health_matches_rul_thresholds():
    for e in DATA["engines"]:
        hi, status = health_from_rul(e["true_rul_cycles"])
        assert e["health_status"] == status
        assert abs(e["health_index"] - hi) < 1e-9
        assert 0.0 <= e["health_index"] <= 1.0


def test_health_thresholds():
    assert health_from_rul(125)[1] == "HEALTHY"
    assert health_from_rul(300)[0] == 1.0            # index is capped at 1
    assert health_from_rul(60)[1] == "WARNING"
    assert health_from_rul(10)[1] == "CRITICAL"
    assert health_from_rul(0)[1] == "CRITICAL"


# ------------------------------------------------------------------ value ranges

def test_aircraft_values_are_sane():
    for a in DATA["aircraft"]:
        assert a["total_flight_hours"] > 0 and a["total_cycles"] > 0
        assert 2006 <= a["manufacture_year"] <= 2022
        assert a["status"] in {"ACTIVE", "IN_MAINTENANCE", "AOG"}
        assert a["tail_number"].startswith("VT-") and len(a["tail_number"]) == 6


def test_spare_part_values_are_sane():
    for p in DATA["spare_parts"]:
        assert p["quantity_in_stock"] >= 0
        assert p["unit_cost_inr"] > 0
        assert p["lead_time_days"] >= 0
        assert p["warehouse_code"] in {"DEL", "BOM", "BLR", "HYD"}


def test_some_parts_are_out_of_stock_for_edge_case_demos():
    assert any(p["quantity_in_stock"] == 0 for p in DATA["spare_parts"])


def test_repair_slots_have_valid_time_ranges():
    for s in DATA["repair_slots"]:
        assert datetime.strptime(s["end_time"], FMT) > datetime.strptime(s["start_time"], FMT)
        assert s["is_available"] in (0, 1)


def test_work_order_values_are_sane():
    for w in DATA["work_orders"]:
        assert w["priority"] in {"P1", "P2", "P3", "P4"}
        assert w["estimated_hours"] > 0
        created = datetime.strptime(w["created_at"], FMT)
        assert datetime.strptime(w["due_date"], "%Y-%m-%d").date() >= created.date()


# ------------------------------------------------------------------ referential integrity

def test_foreign_keys_resolve():
    ac = set(_ids("aircraft", "aircraft_id"))
    eng = set(_ids("engines", "engine_id"))
    defects = set(_ids("defect_logs", "defect_id"))
    slots = set(_ids("repair_slots", "slot_id"))
    parts = set(_ids("spare_parts", "part_id"))
    wos = set(_ids("work_orders", "work_order_id"))

    assert all(e["aircraft_id"] in ac for e in DATA["engines"])
    for d in DATA["defect_logs"]:
        assert d["aircraft_id"] in ac
        assert d["engine_id"] == "" or d["engine_id"] in eng
    for w in DATA["work_orders"]:
        assert w["defect_id"] in defects and w["aircraft_id"] in ac
        assert w["engine_id"] == "" or w["engine_id"] in eng
        assert w["repair_slot_id"] == "" or w["repair_slot_id"] in slots
    for link in DATA["work_order_parts"]:
        assert link["work_order_id"] in wos and link["part_id"] in parts
        assert link["quantity"] > 0


def test_each_defect_has_at_most_one_work_order():
    counts = Counter(w["defect_id"] for w in DATA["work_orders"])
    assert max(counts.values()) == 1


def test_work_order_matches_its_defect():
    by_id = {d["defect_id"]: d for d in DATA["defect_logs"]}
    for w in DATA["work_orders"]:
        d = by_id[w["defect_id"]]
        assert w["aircraft_id"] == d["aircraft_id"]
        assert w["engine_id"] == d["engine_id"]


def test_defect_belongs_to_the_aircraft_that_owns_the_engine():
    owner = {e["engine_id"]: e["aircraft_id"] for e in DATA["engines"]}
    for d in DATA["defect_logs"]:
        if d["engine_id"]:
            assert owner[d["engine_id"]] == d["aircraft_id"]


# ------------------------------------------------------------------ business rules (the demo story)

def test_every_critical_engine_has_a_critical_ai_alert_defect():
    alerted = {d["engine_id"] for d in DATA["defect_logs"] if d["source"] == "AI_ALERT" and d["severity"] == "CRITICAL"}
    critical = {e["engine_id"] for e in DATA["engines"] if e["health_status"] == "CRITICAL"}
    assert critical <= alerted


def test_healthy_engines_get_no_ai_alerts():
    healthy = {e["engine_id"] for e in DATA["engines"] if e["health_status"] == "HEALTHY"}
    assert not [d for d in DATA["defect_logs"] if d["source"] == "AI_ALERT" and d["engine_id"] in healthy]


def test_slots_are_never_double_booked_and_match_the_aircraft_base():
    ac_base = {a["aircraft_id"]: a["base_code"] for a in DATA["aircraft"]}
    slot_base = {s["slot_id"]: s["base_code"] for s in DATA["repair_slots"]}
    used = [w["repair_slot_id"] for w in DATA["work_orders"] if w["repair_slot_id"]]
    assert len(used) == len(set(used)), "one slot assigned to two work orders"
    for w in DATA["work_orders"]:
        if w["repair_slot_id"]:
            assert slot_base[w["repair_slot_id"]] == ac_base[w["aircraft_id"]]


def test_slot_availability_matches_assignments():
    used = {w["repair_slot_id"] for w in DATA["work_orders"] if w["repair_slot_id"]}
    for s in DATA["repair_slots"]:
        assert (s["is_available"] == 0) == (s["slot_id"] in used)


def test_scheduled_or_in_progress_work_orders_have_a_slot():
    for w in DATA["work_orders"]:
        if w["status"] in ("SCHEDULED", "IN_PROGRESS"):
            assert w["repair_slot_id"], f"{w['work_order_id']} has no slot"


def test_waiting_parts_work_orders_have_an_out_of_stock_part_and_no_slot():
    stock = {p["part_id"]: p["quantity_in_stock"] for p in DATA["spare_parts"]}
    waiting = [w for w in DATA["work_orders"] if w["status"] == "WAITING_PARTS"]
    assert waiting, "demo needs at least one WAITING_PARTS work order"
    for w in waiting:
        parts = [l["part_id"] for l in DATA["work_order_parts"] if l["work_order_id"] == w["work_order_id"]]
        assert parts and all(stock[p] == 0 for p in parts)
        assert w["repair_slot_id"] == ""


def test_active_work_has_parts_in_stock():
    stock = {p["part_id"]: p["quantity_in_stock"] for p in DATA["spare_parts"]}
    for w in DATA["work_orders"]:
        if w["status"] in ("PLANNED", "SCHEDULED", "IN_PROGRESS"):
            for l in DATA["work_order_parts"]:
                if l["work_order_id"] == w["work_order_id"]:
                    assert stock[l["part_id"]] > 0


def test_defect_status_follows_work_order_status():
    wo = {w["defect_id"]: w for w in DATA["work_orders"]}
    for d in DATA["defect_logs"]:
        w = wo.get(d["defect_id"])
        if w is None:
            assert d["status"] in ("OPEN", "IN_REVIEW")
        elif w["status"] == "COMPLETED":
            assert d["status"] == "CLOSED"
        else:
            assert d["status"] == "WORK_ORDER_CREATED"


def test_aircraft_status_follows_work_orders():
    for a in DATA["aircraft"]:
        wos = [w for w in DATA["work_orders"] if w["aircraft_id"] == a["aircraft_id"]]
        if any(w["priority"] == "P1" and w["status"] == "WAITING_PARTS" for w in wos):
            assert a["status"] == "AOG"
        elif any(w["status"] == "IN_PROGRESS" for w in wos):
            assert a["status"] == "IN_MAINTENANCE"
        else:
            assert a["status"] == "ACTIVE"


def test_data_contains_edge_cases_for_missing_data_tests():
    assert any(d["engine_id"] == "" for d in DATA["defect_logs"]), "need airframe-level defects with no engine"
    assert any(d["status"] in ("OPEN", "IN_REVIEW") for d in DATA["defect_logs"]), "need defects without work orders"


def test_demo_scenarios():
    sc = DATA["demo_scenarios"]
    assert {"critical_engine", "healthy_engine"} <= set(sc)
    worst = min(DATA["engines"], key=lambda e: e["true_rul_cycles"])
    assert sc["critical_engine"]["unit_nr"] == worst["unit_nr"]
    assert sc["critical_engine"]["health_status"] == "CRITICAL"
    assert sc["healthy_engine"]["health_status"] == "HEALTHY"
    assert "aog_engine" in sc and sc["aog_engine"]["work_orders"][0]["status"] == "WAITING_PARTS"


# ------------------------------------------------------------------ bad arguments & small configs

@pytest.mark.parametrize("bad", [0, -1, 51, 1000])
def test_invalid_aircraft_count_is_rejected(bad):
    with pytest.raises(ValueError):
        build_all(n_aircraft=bad)


@pytest.mark.parametrize("field", ["n_parts", "n_slots", "n_defects", "n_work_orders"])
def test_negative_volumes_are_rejected(field):
    with pytest.raises(ValueError):
        build_all(**{field: -1})


def test_bad_start_date_is_rejected():
    with pytest.raises(ValueError):
        build_all(start_date="31-12-2026")


def test_tiny_configuration_still_consistent():
    d = build_all(n_aircraft=1, n_parts=4, n_slots=1, n_defects=2, n_work_orders=1)
    assert len(d["aircraft"]) == 1 and len(d["engines"]) == 2
    eng = {e["engine_id"] for e in d["engines"]}
    assert all(x["engine_id"] in eng or x["engine_id"] == "" for x in d["defect_logs"])


def test_zero_parts_slots_and_work_orders_does_not_crash():
    d = build_all(n_parts=0, n_slots=0, n_work_orders=0)
    assert d["spare_parts"] == [] and d["repair_slots"] == [] and d["work_orders"] == []


def test_works_without_the_rul_file():
    d = build_all(rul_file=Path("/definitely/not/here.txt"))
    assert len(d["engines"]) == 100
