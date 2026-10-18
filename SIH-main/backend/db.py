"""
Database Layer for Telemetry, Predictions, and Alert Persistence.
Supports PostgreSQL / TimescaleDB with automatic lightweight SQLite fallback.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from sqlalchemy import Column, Integer, Float, String, Text, DateTime, select, desc
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from .config import settings
from .retry import async_retry

Base = declarative_base()


class TelemetryRecord(Base):
    __tablename__ = "telemetry_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unit_nr = Column(Integer, index=True, nullable=False)
    time_cycles = Column(Integer, index=True, nullable=False)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    setting_1 = Column(Float, default=0.0)
    setting_2 = Column(Float, default=0.0)
    setting_3 = Column(Float, default=100.0)
    sensor_data = Column(Text, nullable=False)  # JSON-serialized sensor readings


class RULPredictionRecord(Base):
    __tablename__ = "rul_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unit_nr = Column(Integer, index=True, nullable=False)
    time_cycles = Column(Integer, nullable=False)
    model_used = Column(String(50), nullable=False)
    predicted_rul = Column(Float, nullable=False)
    health_index = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AnomalyAlertRecord(Base):
    __tablename__ = "anomaly_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unit_nr = Column(Integer, index=True, nullable=False)
    time_cycles = Column(Integer, nullable=False)
    alert_level = Column(String(20), nullable=False)  # WARNING, CRITICAL
    health_index = Column(Float, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Async Engine and Session
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
_db_initialized = False


async def init_db():
    """Initializes database tables asynchronously."""
    global _db_initialized
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _db_initialized = True


async def ensure_db():
    global _db_initialized
    if not _db_initialized:
        await init_db()


async def get_db():
    """Dependency for obtaining an async session."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        yield session


async def check_db_readiness() -> Dict[str, Any]:
    """Probes database connection readiness by executing a lightweight test query."""
    try:
        await ensure_db()
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_type = "timescaledb" if "postgres" in settings.DATABASE_URL else "sqlite"
        return {
            "ready": True,
            "status": "CONNECTED",
            "database_type": db_type,
            "error": None
        }
    except Exception as e:
        return {
            "ready": False,
            "status": "ERROR",
            "database_type": "unknown",
            "error": str(e)
        }


@async_retry(max_retries=3, initial_delay=0.05, backoff_factor=2.0)
async def save_telemetry(unit_nr: int, time_cycles: int, settings_dict: Dict[str, float], sensors: Dict[str, float]):
    """Inserts a single telemetry cycle with exponential backoff retry."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        record = TelemetryRecord(
            unit_nr=unit_nr,
            time_cycles=time_cycles,
            setting_1=settings_dict.get("setting_1", 0.0),
            setting_2=settings_dict.get("setting_2", 0.0),
            setting_3=settings_dict.get("setting_3", 100.0),
            sensor_data=json.dumps(sensors)
        )
        session.add(record)
        await session.commit()


async def get_latest_cycle_for_unit(unit_nr: int) -> Optional[int]:
    """Retrieves the highest time_cycle recorded for a given engine."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        stmt = (
            select(TelemetryRecord.time_cycles)
            .where(TelemetryRecord.unit_nr == unit_nr)
            .order_by(TelemetryRecord.time_cycles.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def get_recent_telemetry(unit_nr: int, limit: int = 30) -> List[Dict]:
    """Retrieves up to `limit` recent telemetry cycles for a given engine."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        stmt = (
            select(TelemetryRecord)
            .where(TelemetryRecord.unit_nr == unit_nr)
            .order_by(TelemetryRecord.time_cycles.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()
        rows = []
        for r in reversed(records):
            sensors = json.loads(r.sensor_data)
            row = {
                "unit_nr": r.unit_nr,
                "time_cycles": r.time_cycles,
                "setting_1": r.setting_1,
                "setting_2": r.setting_2,
                "setting_3": r.setting_3,
                **sensors
            }
            rows.append(row)
        return rows


@async_retry(max_retries=3, initial_delay=0.05, backoff_factor=2.0)
async def save_prediction(unit_nr: int, cycle: int, model_used: str, rul: float, hi: float, status: str):
    """Stores model prediction record with exponential backoff retry."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        pred = RULPredictionRecord(
            unit_nr=unit_nr,
            time_cycles=cycle,
            model_used=model_used,
            predicted_rul=rul,
            health_index=hi,
            status=status
        )
        session.add(pred)
        await session.commit()


async def get_prediction_history(unit_nr: int, limit: int = 50) -> List[Dict]:
    """Retrieves prediction history for an engine unit."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        stmt = (
            select(RULPredictionRecord)
            .where(RULPredictionRecord.unit_nr == unit_nr)
            .order_by(RULPredictionRecord.time_cycles.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()
        return [
            {
                "id": r.id,
                "unit_nr": r.unit_nr,
                "time_cycles": r.time_cycles,
                "model_used": r.model_used,
                "predicted_rul": r.predicted_rul,
                "health_index": r.health_index,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in records
        ]


@async_retry(max_retries=3, initial_delay=0.05, backoff_factor=2.0)
async def save_alert(unit_nr: int, time_cycles: int, alert_level: str, health_index: float, message: str):
    """Persists an anomaly alert with exponential backoff retry."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        alert = AnomalyAlertRecord(
            unit_nr=unit_nr,
            time_cycles=time_cycles,
            alert_level=alert_level,
            health_index=health_index,
            message=message
        )
        session.add(alert)
        await session.commit()


async def get_alerts(unit_nr: Optional[int] = None, alert_level: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """Queries anomaly alerts with optional filters."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        stmt = select(AnomalyAlertRecord).order_by(AnomalyAlertRecord.id.desc())
        if unit_nr is not None:
            stmt = stmt.where(AnomalyAlertRecord.unit_nr == unit_nr)
        if alert_level is not None:
            stmt = stmt.where(AnomalyAlertRecord.alert_level == alert_level.upper())
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        records = result.scalars().all()
        return [
            {
                "id": a.id,
                "unit_nr": a.unit_nr,
                "time_cycles": a.time_cycles,
                "alert_level": a.alert_level,
                "health_index": a.health_index,
                "message": a.message,
                "created_at": a.created_at.isoformat() if a.created_at else None
            }
            for a in records
        ]


async def get_fleet_health_summary() -> Dict:
    """Aggregates health status across all active engine units in telemetry database."""
    await ensure_db()
    async with AsyncSessionLocal() as session:
        # Get distinct units
        stmt = select(TelemetryRecord.unit_nr).distinct()
        result = await session.execute(stmt)
        units = result.scalars().all()

        fleet = []
        tier_counts = {"HEALTHY": 0, "WARNING": 0, "CRITICAL": 0}

        for u in units:
            # Get latest prediction
            pred_stmt = (
                select(RULPredictionRecord)
                .where(RULPredictionRecord.unit_nr == u)
                .order_by(RULPredictionRecord.time_cycles.desc())
                .limit(1)
            )
            p_res = await session.execute(pred_stmt)
            latest_pred = p_res.scalar_one_or_none()

            # Latest cycle
            cyc_stmt = (
                select(TelemetryRecord.time_cycles)
                .where(TelemetryRecord.unit_nr == u)
                .order_by(TelemetryRecord.time_cycles.desc())
                .limit(1)
            )
            c_res = await session.execute(cyc_stmt)
            latest_cyc = c_res.scalar_one_or_none() or 0

            status = latest_pred.status if latest_pred else "HEALTHY"
            rul = latest_pred.predicted_rul if latest_pred else 125.0
            hi = latest_pred.health_index if latest_pred else 1.0

            tier_counts[status] = tier_counts.get(status, 0) + 1

            fleet.append({
                "unit_nr": u,
                "latest_cycle": latest_cyc,
                "status": status,
                "predicted_rul": rul,
                "health_index": hi
            })

        return {
            "total_engines": len(units),
            "tier_counts": tier_counts,
            "engines": fleet
        }
