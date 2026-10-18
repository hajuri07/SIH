"""
Anomaly Alerts Query Router.
"""

from fastapi import APIRouter, Query
from typing import List, Optional

from ..schemas import AnomalyAlertItem
from ..db import get_alerts

router = APIRouter(prefix="/alerts", tags=["Anomaly Alerts"])


@router.get("", response_model=List[AnomalyAlertItem])
async def list_anomaly_alerts(
    unit_nr: Optional[int] = Query(default=None, ge=1, description="Filter by engine unit"),
    alert_level: Optional[str] = Query(default=None, description="Filter by severity: 'WARNING' or 'CRITICAL'"),
    limit: int = Query(default=50, ge=1, le=200, description="Max alerts to retrieve")
):
    """
    Queries anomaly alerts from the database with optional engine and severity filters.
    """
    records = await get_alerts(unit_nr=unit_nr, alert_level=alert_level, limit=limit)
    return [AnomalyAlertItem(**r) for r in records]
