"""
Arena 6 - Reusable request payloads for API tests and manual demos.

Valid payloads use real C-MAPSS test cycles (data/raw/test_FD001.txt) when available,
otherwise nominal healthy values. Invalid payloads are grouped by what is wrong with them.
Standard library only.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "raw" / "test_FD001.txt"
COLS = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]

# Nominal healthy FD001 values (first cycles of engine 1) - fallback when the raw file is missing
NOMINAL_SENSORS: Dict[str, float] = {
    "s_1": 518.67, "s_2": 641.94, "s_3": 1586.28, "s_4": 1399.92, "s_5": 14.62, "s_6": 21.61,
    "s_7": 553.72, "s_8": 2388.05, "s_9": 9053.72, "s_10": 1.3, "s_11": 47.29, "s_12": 521.66,
    "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03, "s_17": 392.0, "s_18": 2388.0,
    "s_19": 100.0, "s_20": 39.06, "s_21": 23.42,
}


def load_engine_rows(unit_nr: int = 1, n: int = 30) -> List[dict]:
    """First `n` cycles of a C-MAPSS test engine as API-style rows (unit_nr, time_cycles, settings, s_1..s_21)."""
    rows: List[dict] = []
    if TEST_FILE.exists():
        with open(TEST_FILE) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 26 and int(float(parts[0])) == unit_nr:
                    row = {c: float(v) for c, v in zip(COLS, parts[:26])}
                    row["unit_nr"], row["time_cycles"] = unit_nr, int(float(parts[1]))
                    rows.append(row)
    if not rows:  # fallback: nominal values with a tiny deterministic drift
        for cycle in range(1, n + 1):
            row = {"unit_nr": unit_nr, "time_cycles": cycle, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
            row.update({k: v + (0.001 * cycle if k not in ("s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19") else 0.0)
                        for k, v in NOMINAL_SENSORS.items()})
            rows.append(row)
    return sorted(rows, key=lambda r: r["time_cycles"])[:n]


def row_to_packet(row: dict, unit_nr: int | None = None) -> dict:
    """Convert a flat row into the POST /api/v1/telemetry body (TelemetryPacket)."""
    return {
        "unit_nr": unit_nr if unit_nr is not None else int(row["unit_nr"]),
        "time_cycles": int(row["time_cycles"]),
        "setting_1": row["setting_1"], "setting_2": row["setting_2"], "setting_3": row["setting_3"],
        "sensors": {f"s_{i}": row[f"s_{i}"] for i in range(1, 22)},
    }


def valid_telemetry_packet(unit_nr: int = 9001, cycle: int = 1) -> dict:
    return {
        "unit_nr": unit_nr, "time_cycles": cycle,
        "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
        "sensors": dict(NOMINAL_SENSORS),
    }


def sensors_only(row: dict) -> Dict[str, float]:
    """Body for POST /api/v1/detect/anomaly -> field `telemetry`."""
    return {f"s_{i}": row[f"s_{i}"] for i in range(1, 22)}


# --------------------------------------------------------------------------- invalid payload catalogue
# Each case: id -> (payload, why it is invalid). Pydantic should answer HTTP 422 for all of these.

def invalid_telemetry_payloads() -> Dict[str, dict]:
    base = valid_telemetry_packet()
    cases: Dict[str, dict] = {}

    def without(key):
        d = copy.deepcopy(base)
        d.pop(key)
        return d

    def with_(**kw):
        d = copy.deepcopy(base)
        d.update(kw)
        return d

    cases["missing_unit_nr"] = without("unit_nr")
    cases["missing_time_cycles"] = without("time_cycles")
    cases["missing_sensors"] = without("sensors")
    cases["empty_body"] = {}
    cases["unit_nr_is_text"] = with_(unit_nr="engine-one")
    cases["unit_nr_is_fraction"] = with_(unit_nr=1.5)
    cases["unit_nr_is_null"] = with_(unit_nr=None)
    cases["time_cycles_is_text"] = with_(time_cycles="forty")
    cases["time_cycles_is_null"] = with_(time_cycles=None)
    cases["sensors_is_list"] = with_(sensors=[642.1, 1585.0])
    cases["sensors_is_text"] = with_(sensors="s_2=642.1")
    cases["sensor_value_is_text"] = with_(sensors={**base["sensors"], "s_3": "hot"})
    cases["sensor_value_is_null"] = with_(sensors={**base["sensors"], "s_3": None})
    cases["setting_is_text"] = with_(setting_1="high")
    return cases


def invalid_prediction_payloads() -> Dict[str, dict]:
    return {
        "missing_unit_nr": {"model_type": "xgboost"},
        "unit_nr_is_text": {"unit_nr": "abc", "model_type": "xgboost"},
        "model_type_is_number": {"unit_nr": 1, "model_type": 123},
        "recent_telemetry_is_text": {"unit_nr": 1, "recent_telemetry": "not-a-list"},
        "recent_telemetry_has_non_objects": {"unit_nr": 1, "recent_telemetry": [1, 2, 3]},
        "empty_body": {},
    }


def invalid_anomaly_payloads() -> Dict[str, dict]:
    good = sensors_only(load_engine_rows(1, 1)[0])
    return {
        "missing_unit_nr": {"telemetry": good},
        "missing_telemetry": {"unit_nr": 1},
        "telemetry_is_list": {"unit_nr": 1, "telemetry": [1.0, 2.0]},
        "telemetry_value_is_text": {"unit_nr": 1, "telemetry": {**good, "s_3": "hot"}},
        "telemetry_value_is_null": {"unit_nr": 1, "telemetry": {**good, "s_3": None}},
        "empty_body": {},
    }


# Inputs the API *should* reject but currently accepts (found by reading backend/schemas.py).
# Used by xfail tests so they flip to "XPASS" once Arena 2 adds validation.
def should_be_rejected_telemetry() -> Dict[str, dict]:
    base = valid_telemetry_packet()
    return {
        "negative_time_cycles": {**base, "time_cycles": -5},
        "zero_time_cycles": {**base, "time_cycles": 0},
        "negative_unit_nr": {**base, "unit_nr": -1},
        "empty_sensors": {**base, "sensors": {}},
        "impossible_temperature": {**base, "sensors": {**base["sensors"], "s_3": -99999.0}},
        "absurdly_large_sensor_value": {**base, "sensors": {**base["sensors"], "s_7": 1e15}},
    }
