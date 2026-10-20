# Arena 6 Test Plan

Run everything from the repository root: `pytest arena6/tests -v`

| File | Needs | Tests | What it proves |
| :--- | :--- | :---: | :--- |
| `test_data_generation.py` | pytest only | 43 | Volumes, determinism, unique ids, foreign keys, value ranges, business rules (slots, parts, statuses), bad arguments, tiny configs |
| `test_seed_db.py` | pytest only | 22 | Tables created, re-run safe, backend tables preserved, constraints reject bad rows, SQL export, CLI errors, demo queries |
| `test_api_valid_inputs.py` | backend requirements + model files | 15 | Health, endpoint list, ingest, history, predict (3 models), anomaly, stream, full ingest -> predict -> anomaly -> explain flow |
| `test_api_invalid_inputs.py` | backend requirements | 60 | 422 for malformed bodies/params, 405 wrong method, 404 unknown route; `xfail` for inputs that should be rejected but are accepted |
| `test_api_errors_missing_data.py` | backend requirements + model files | 20 | 404 for unknown engines, short history, incomplete rows, simulated DB outage / model crash / models not loaded |

Counts for the three API files were derived by counting test functions and parameters (they were not executed where they were written); the pytest summary shows the exact numbers.

## Result legend

* **passed**: behaviour is correct.
* **xfailed**: a known problem (listed in `BUG_REPORTS.md`), the test documents it. Not a failure of the test run.
* **xpassed**: the problem was fixed - remove the `xfail` mark.
* **skipped**: a dependency (package or trained models) is missing; the reason is printed.
* **failed**: new problem - write it up in `BUG_REPORTS.md`.

## Test cases for missing data and errors

| ID | Situation | Expected |
| :--- | :--- | :--- |
| E-01 | Predict for an engine with no stored telemetry | 404 naming the engine |
| E-02 | Predict with `recent_telemetry: []` and nothing in DB | 404 |
| E-03 | Explain for unknown engine | 404 |
| E-04 | History for unknown engine | 200 with empty list |
| E-05 | Only 1 cycle of history (window is 30) | 200 (padding) for XGBoost and TCN |
| E-06 | Rows missing `time_cycles` / `unit_nr` / sensors | clear 4xx (currently suspected 500 - BUG-005) |
| E-07 | Anomaly check with partial or empty sensor dict | clear 4xx (BUG-005) |
| E-08 | Database write fails during ingest | 500 with message (passes) |
| E-09 | Database read fails in history / predict | 500 with message (currently crashes - BUG-006) |
| E-10 | Model raises during predict / anomaly | 500 `Prediction failed` / `Anomaly detection failed` (passes) |
| E-11 | Model files not loaded | 503 on predict, anomaly, explain |
| E-12 | Same cycle ingested twice | one stored row (BUG-004) |
| E-13 | Stop stream when none is running | 200, `is_active: false` |

## Manual checks (need the running system)

1. `uvicorn backend.main:app --port 8000`, open `/docs`, send one valid and one invalid request per endpoint.
2. `streamlit run dashboard/app.py`, select units 73, 76, 16, 64 and check each tab renders.
3. Start the stream for unit 73 from `/docs` (`POST /api/v1/stream/control`), watch `GET /api/v1/stream/status` count up, then stop it.
