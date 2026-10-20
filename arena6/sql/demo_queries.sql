-- Arena 6 demo queries (SQLite). Each block starts with "-- name: <title>" and is run by arena6/show_demo_data.py

-- name: Fleet health summary (engines per status)
SELECT health_status AS status, COUNT(*) AS engines, MIN(true_rul_cycles) AS min_rul, MAX(true_rul_cycles) AS max_rul
FROM engines GROUP BY health_status ORDER BY MIN(true_rul_cycles);

-- name: Critical engines and their maintenance status
SELECT e.engine_id, e.unit_nr, a.tail_number, a.base_code, e.true_rul_cycles AS rul,
       COALESCE(w.work_order_id, '-') AS work_order, COALESCE(w.status, 'NO WORK ORDER') AS wo_status
FROM engines e
JOIN aircraft a ON a.aircraft_id = e.aircraft_id
LEFT JOIN work_orders w ON w.engine_id = e.engine_id
WHERE e.health_status = 'CRITICAL'
ORDER BY e.true_rul_cycles LIMIT 10;

-- name: Open work orders by priority
SELECT w.priority, w.work_order_id, a.tail_number, w.task_type, w.status, COALESCE(w.repair_slot_id, '-') AS slot, w.due_date
FROM work_orders w JOIN aircraft a ON a.aircraft_id = w.aircraft_id
WHERE w.status NOT IN ('COMPLETED', 'CANCELLED')
ORDER BY w.priority, w.due_date LIMIT 12;

-- name: Aircraft on ground (AOG) and the part they are waiting for
SELECT a.tail_number, a.base_code, w.work_order_id, p.name AS missing_part, p.part_id, p.lead_time_days
FROM aircraft a
JOIN work_orders w ON w.aircraft_id = a.aircraft_id AND w.status = 'WAITING_PARTS'
JOIN work_order_parts l ON l.work_order_id = w.work_order_id
JOIN spare_parts p ON p.part_id = l.part_id
WHERE a.status = 'AOG';

-- name: Next free repair slots per hangar
SELECT base_code, slot_id, bay, start_time, end_time
FROM repair_slots WHERE is_available = 1
ORDER BY start_time LIMIT 8;

-- name: Spare parts at or below reorder level
SELECT warehouse_code AS warehouse, name, quantity_in_stock AS stock, reorder_level, lead_time_days
FROM spare_parts WHERE quantity_in_stock <= reorder_level AND is_critical = 1
ORDER BY quantity_in_stock, lead_time_days DESC LIMIT 10;
