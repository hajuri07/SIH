# Arena 6 - Test Data, Maintenance Records & Demo Preparation

Everything in this folder is **new and self-contained**. No existing file of the project was changed.

## What is here

| Item | Purpose |
| :--- | :--- |
| `generate_data.py` | Creates synthetic aircraft, engines, spare parts, repair slots, defect logs and work orders (CSV + JSON). Standard library only. |
| `seed_db.py` | One command that loads that data (plus real C-MAPSS telemetry for demo engines) into the dev database. Can also export a PostgreSQL/TimescaleDB `.sql` file. |
| `show_demo_data.py` | Prints the demo views (fleet health, critical engines, open work orders, AOG, free slots, parts shortage). |
| `payloads.py` | Valid and invalid request bodies used by the API tests. |
| `sql/` | `schema_sqlite.sql`, `schema_postgres.sql`, `demo_queries.sql`. |
| `sample_data/` | Ready-made output of `generate_data.py` (seed 42). |
| `tests/` | 4 test files: data generator, database seeding, API valid inputs, API invalid/missing/error cases. |
| `docs/` | `MAINTENANCE_DATA_CONTRACT.md`, `TEST_PLAN.md`, `BUG_REPORTS.md`, `DEMO_SCRIPT.md`. |
| `screenshots/` | Shot list for the screenshots only a running system can produce. |

## Quick start (run from the repository root)

```powershell
# 1. Generate sample files (optional - already in arena6/sample_data)
python -m arena6.generate_data

# 2. Load the dev database (re-runnable; only touches the 7 Arena 6 tables)
python -m arena6.seed_db

# 3. See the demo views
python -m arena6.show_demo_data

# 4. Run the tests
pytest arena6/tests -v
```

Data and seeding tests need only `pytest`. API tests need `pip install -r requirements.txt`
and skip themselves if a package or the trained model files (`models/artifacts/`) are missing.

### Useful options

```powershell
python -m arena6.seed_db --db arena6_demo.db        # separate DB file (then: set DATABASE_URL=sqlite+aiosqlite:///arena6_demo.db)
python -m arena6.seed_db --no-telemetry              # maintenance tables only
python -m arena6.seed_db --telemetry-units 20        # more engines with live-looking history
python -m arena6.seed_db --start-date today          # slots/due dates counted from today
python -m arena6.seed_db --export-sql seed.sql       # TimescaleDB: psql "$DATABASE_URL" -f seed.sql
python -m arena6.generate_data --seed 7 --aircraft 20 --work-orders 25
```

> Seeding the default target writes into `telemetry.db` in the repo root (the file the API reads).
> Use `--db` if you want to keep that file untouched.

## How the data fits the project

* 50 aircraft x 2 engines = **100 engines = the 100 C-MAPSS FD001 test engines** (`engines.unit_nr` = `telemetry_records.unit_nr`).
* Engine health uses the same thresholds as `dashboard/app.py` (HI = RUL/125: >0.65 healthy, >0.35 warning, else critical) and the true RUL from `data/raw/RUL_FD001.txt`.
* Every CRITICAL engine gets an AI-alert defect log; defects become work orders; work orders get a repair slot at the aircraft's home hangar and spare parts from that hangar's warehouse.
* Deliberate edge cases for demos and tests: parts with zero stock, a P1 work order blocked by a missing part (aircraft status `AOG`), defects with no engine (airframe issues), defects with no work order yet.

## What was checked and what was not

Checked in the build environment: generator, seeding (including re-run safety, constraints, SQL export) and the 65 data/seeding tests all pass.

**Not run there:** the API tests and the demo itself. The build environment had no network, so FastAPI, PyTorch, XGBoost and pytest could not be installed. The API tests were written against your code by reading it; run `pytest arena6/tests -v` once on your machine. Tests marked `xfail` describe problems found by code review (see `docs/BUG_REPORTS.md`); an `xfail` is expected, an `XPASS` means someone fixed it.
