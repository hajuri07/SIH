# Arena 6 Maintenance Data Contract

Tables created by `arena6/seed_db.py` (DDL in `arena6/sql/`). They live **next to** the backend's own tables
(`telemetry_records`, `rul_predictions`, `anomaly_alerts`) and join to them through `engines.unit_nr`.

```
aircraft 1──2 engines ──(unit_nr)── telemetry_records / rul_predictions
   │              │
   └── defect_logs ──1:0..1── work_orders ──0..1── repair_slots
                                   └── work_order_parts ── spare_parts
```

## Tables

| Table | Key columns | Notes |
| :--- | :--- | :--- |
| `aircraft` | `aircraft_id` (AC-001), `tail_number` (VT-ABC), `aircraft_type`, `base_code`, `status` | status: `ACTIVE`, `IN_MAINTENANCE`, `AOG` |
| `engines` | `engine_id` (ENG-073), `aircraft_id`, `position` (1/2), `unit_nr` (1-100), `true_rul_cycles`, `health_index`, `health_status` | `unit_nr` = C-MAPSS engine id. Health fields come from the RUL file, not from a model |
| `spare_parts` | `part_id`, `part_number`, `name`, `category`, `related_sensor`, `warehouse_code`, `quantity_in_stock`, `reorder_level`, `unit_cost_inr`, `lead_time_days`, `is_critical` | `related_sensor` links a part to a C-MAPSS sensor (e.g. `s_3` -> HPC rotor blades / T30 sensor) |
| `repair_slots` | `slot_id`, `hangar_name`, `base_code`, `bay`, `start_time`, `end_time`, `is_available` | 0 = booked by a work order |
| `defect_logs` | `defect_id`, `aircraft_id`, `engine_id` (nullable), `source`, `severity`, `sensor` (nullable), `description`, `status` | source: `PILOT_REPORT`, `ROUTINE_INSPECTION`, `AI_ALERT`, `BOROSCOPE`; severity: `MINOR`, `MAJOR`, `CRITICAL` |
| `work_orders` | `work_order_id`, `defect_id`, `aircraft_id`, `engine_id`, `task_type`, `priority` (P1-P4), `status`, `repair_slot_id`, `estimated_hours`, `due_date` | status: `PLANNED`, `SCHEDULED`, `IN_PROGRESS`, `WAITING_PARTS`, `COMPLETED`, `CANCELLED` |
| `work_order_parts` | `work_order_id`, `part_id`, `quantity` | |

Rules the data follows (all covered by `tests/test_data_generation.py`): unique ids; every foreign key resolves;
a slot is used by one work order and sits at the aircraft's home base; `SCHEDULED`/`IN_PROGRESS` work has a slot;
`WAITING_PARTS` work has a zero-stock part and no slot; defect status follows work-order status.

Empty optional values are `NULL` in the database and empty strings in the CSV files.

## Proposed endpoints (NOT implemented - for Arena 2 and Arena 5 to agree on)

The current backend has no routes for maintenance data. If the frontend should show aircraft lists, status cards and work orders,
these read-only routes cover it. Shapes mirror the table columns above.

| Method and path | Returns |
| :--- | :--- |
| `GET /api/v1/aircraft?status=&base=` | list of aircraft with their two engines and worst `health_status` |
| `GET /api/v1/aircraft/{aircraft_id}` | aircraft detail + engines + open defects + open work orders |
| `GET /api/v1/work-orders?status=&priority=` | work orders with slot and part summary |
| `GET /api/v1/spare-parts?warehouse=&low_stock=true` | parts list |
| `GET /api/v1/repair-slots?base=&available=true` | slots |

Until then, teammates can read `arena6/sample_data/maintenance_data.json` (same shapes) to build screens.
