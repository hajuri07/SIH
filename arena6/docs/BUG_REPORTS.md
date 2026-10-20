# Bug Reports

## How to use this file

Copy the template, give the bug the next number, attach a screenshot into `arena6/screenshots/` named `BUG-0XX_short-name.png`.

```
### BUG-0XX  Short title
Status:      Open | Fixed | Won't fix        Severity: High | Medium | Low        Owner: Arena N
Found by:    test name or manual step
Steps:       1. ...  2. ...
Expected:    ...
Actual:      ...
Screenshot:  arena6/screenshots/BUG-0XX_....png
```

## Findings from code review - NOT yet confirmed by running

These were found by reading the code and writing tests. The API tests could not be run where they were written
(no network, packages not installable), so every item below is **Suspected**. Run `pytest arena6/tests -v`, then mark each as
Confirmed or Not reproduced. Tests marked `xfail` cover items 1-7.

| ID | Severity | Owner | Problem (from code) | Expected | Test |
| :--- | :---: | :---: | :--- | :--- | :--- |
| BUG-001 | Medium | Arena 2 | `TelemetryPacket` accepts `time_cycles` of 0 or negative and negative `unit_nr` (`backend/schemas.py`) | 422 | `test_telemetry_should_reject_impossible_values` |
| BUG-002 | Medium | Arena 2 | Unknown `model_type` (e.g. `"quantum"`) silently runs XGBoost, the response says `QUANTUM`, and that name is saved in `rul_predictions` (`backend/services.py`) | 422 | `test_predict_should_reject_unknown_model_type` |
| BUG-003 | Medium | Arena 2 | Empty `sensors: {}` and impossible values (`s_3 = -99999`, `s_7 = 1e15`) are stored without complaint | 422 | `test_telemetry_should_reject_impossible_values` |
| BUG-004 | Low | Arena 2 | Same `(unit_nr, time_cycles)` can be stored twice; no unique constraint | one row | `test_duplicate_cycle_should_not_create_duplicate_rows` |
| BUG-005 | Medium | Arena 2 | Rows with missing columns / partial sensor sets reach feature engineering and likely end as HTTP 500 | 400 or 422 | `test_predict_should_reject_rows_missing_*`, `test_anomaly_should_reject_*` |
| BUG-006 | High | Arena 2 | `GET /telemetry/{id}/history` and the DB lookup in `/predict/rul` have no error handling; a DB outage crashes the request | 500 JSON error | `test_history_database_failure_*`, `test_predict_database_failure_*` |
| BUG-007 | Low | Arena 2 | `POST /stream/control` with an unknown `action` returns 200 and does nothing | 422 | `test_stream_control_should_reject_unknown_action` |
| BUG-008 | Medium | Arena 3 | `/detect/anomaly` returns a fixed `lead_time_cycles` (15 for CRITICAL, 45 for WARNING) and always analyses one row with `time_cycles = 1` and default settings; the number is not computed from data | computed lead time or field omitted | observation |
| BUG-009 | **High for demo** | Arena 5 / 1 | Dashboard KPI "Predicted RUL" is `true RUL + random noise` (`dashboard/app.py`), not a model output and not an API call | real model prediction | observation |
| BUG-010 | Low | Arena 2 | `confidence_interval` is always prediction +/- 12.5 cycles, not a statistical interval | real interval or rename | observation |
| BUG-011 | Medium | Arena 2 / 3 | Table `anomaly_alerts` exists but nothing writes to it; alerts are never stored | alerts persisted | observation |
| BUG-012 | Medium | Arena 2 | README says "Kafka-compatible", but the backend has settings only: no Kafka producer or consumer code | match docs to code | observation |
| BUG-013 | Low | Arena 2 | Stream simulator swallows prediction errors (`except Exception: pass`), so a broken model looks like a working stream | log the error | observation |
| BUG-014 | Low | All | Repo contains `__pycache__` folders and a `telemetry.db` with test rows (engine 99); `tests/test_api.py` writes into that file; README shows an author's local path (`c:\Ibrahim\...`) | clean repo, `.gitignore` | observation |
| BUG-015 | Medium | Arena 3 | `benchmark_metrics.json` reports average detection lead time of 237.8 cycles, longer than many FD001 engine lives (about 128-362 cycles); judges may question how it is measured | explain definition | observation |

## Confirmed bugs

_None yet. Move items here after running the tests, with a screenshot._
