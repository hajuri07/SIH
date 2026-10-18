"""
Maintenance Scheduling, Repair Slot & Spare Parts Allocation Router.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any

from ..schemas import MaintenanceScheduleResponse
from ..db import get_fleet_health_summary, get_recent_telemetry
from ..services import registry

router = APIRouter(prefix="/maintenance", tags=["Maintenance Scheduling & Logistics"])


@router.get("/schedule", response_model=List[MaintenanceScheduleResponse])
async def get_maintenance_schedule():
    """
    Evaluates fleet health and schedules inspection and overhaul actions:
    - IMMEDIATE_AOG (Aircraft On Ground): RUL <= 25 cycles or CRITICAL status.
    - EXPEDITED: RUL <= 60 cycles or WARNING status.
    - ROUTINE: Nominal healthy status.
    """
    fleet = await get_fleet_health_summary()
    schedule = []

    for eng in fleet.get("engines", []):
        unit_nr = eng["unit_nr"]
        rul = eng["predicted_rul"]
        hi = eng["health_index"]
        status = eng["status"]

        if status == "CRITICAL" or rul <= 25.0:
            urgency = "IMMEDIATE_AOG"
            action = "Emergency Borescope Inspection & Hot Section Overhaul (Immediate Grounding)"
            slot = f"Hangar-A1 (Priority Express Bay)"
            hours = 18.0
            parts = [
                "HPC Stage 1-4 Rotor Blades Kit (P/N 40821-HPC)",
                "High Pressure Turbine Nozzle Guide Vanes (P/N 19830-NGV)",
                "Gas-Path Interstage Dynamic Seal Assembly (P/N 77210-SL)"
            ]
        elif status == "WARNING" or rul <= 60.0:
            urgency = "EXPEDITED"
            action = "Compressor Wash & Optical Fiber Alignment Inspection within 48 flight hours"
            slot = f"Hangar-B2 (Scheduled Line Bay)"
            hours = 8.5
            parts = [
                "Combustion Liner Fuel Injector Set (P/N 33010-FI)",
                "Static Pressure Transmitter Probe (P/N 60412-PS30)",
                "Engine Bleed Air Valve Overhaul Kit (P/N 88310-BAV)"
            ]
        else:
            urgency = "ROUTINE"
            action = "Routine A-Check visual inspection and diagnostic data dump"
            slot = f"Ramp Stand 14 (Standard Turnaround)"
            hours = 2.0
            parts = [
                "Oil Filter Cartridge (P/N 12004-OF)",
                "Magnetic Chip Detector O-Ring Set (P/N 09210-OR)"
            ]

        schedule.append(MaintenanceScheduleResponse(
            unit_nr=unit_nr,
            urgency=urgency,
            predicted_rul=round(rul, 1),
            health_index=round(hi, 3),
            recommended_action=action,
            target_hangar_slot=slot,
            estimated_turnaround_hours=hours,
            required_parts=parts
        ))

    # Sort so most critical maintenance items appear first
    priority_order = {"IMMEDIATE_AOG": 0, "EXPEDITED": 1, "ROUTINE": 2}
    schedule.sort(key=lambda x: (priority_order.get(x.urgency, 3), x.predicted_rul))
    return schedule


@router.get("/slots")
async def get_repair_slots():
    """
    Returns available airport MRO hangar capacity and allocated maintenance bays.
    """
    return {
        "mro_facility": "Air India / Boeing PHM Service Center",
        "total_hangar_bays": 6,
        "occupied_bays": 2,
        "available_bays": [
            {"bay_id": "Hangar-A1", "type": "Engine Drop Overhaul", "status": "RESERVED_FOR_AOG"},
            {"bay_id": "Hangar-A2", "type": "Engine Drop Overhaul", "status": "AVAILABLE"},
            {"bay_id": "Hangar-B1", "type": "Line Maintenance & Borescope", "status": "AVAILABLE"},
            {"bay_id": "Hangar-B2", "type": "Line Maintenance & Borescope", "status": "OCCUPIED"},
            {"bay_id": "Hangar-C1", "type": "Avionics & Sensor Calibration", "status": "AVAILABLE"},
            {"bay_id": "Hangar-C2", "type": "General Inspection", "status": "OCCUPIED"}
        ]
    }


@router.get("/parts")
async def get_spare_parts_inventory():
    """
    Returns inventory of critical PHM gas-path turbine and compressor components.
    """
    return {
        "warehouse_location": "Chhatrapati Shivaji Maharaj International (BOM) MRO Depot",
        "inventory": [
            {"part_number": "40821-HPC", "description": "HPC Stage 1-4 Rotor Blades Kit", "stock": 4, "lead_time_days": 2},
            {"part_number": "19830-NGV", "description": "HPT Nozzle Guide Vanes", "stock": 3, "lead_time_days": 3},
            {"part_number": "77210-SL", "description": "Interstage Dynamic Seal Assembly", "stock": 8, "lead_time_days": 1},
            {"part_number": "60412-PS30", "description": "Static Pressure Transmitter Ps30", "stock": 12, "lead_time_days": 1},
            {"part_number": "33010-FI", "description": "Fuel Injector Assembly", "stock": 10, "lead_time_days": 2},
            {"part_number": "12004-OF", "description": "Oil Filter Cartridges (Pack of 10)", "stock": 45, "lead_time_days": 0}
        ]
    }
