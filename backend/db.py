"""
Database Layer for Telemetry, Predictions, and Alert Persistence.
Supports PostgreSQL / TimescaleDB with automatic lightweight SQLite fallback.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, select, desc

from .config import settings

Base = declarative_base()


class TelemetryRecord(Base):
    __tablename__ = "telemetry_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unit_nr = Column(Integer, index=True, nullable=False)
    time_cycles = Column(Integer, index=True, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
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
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyAlertRecord(Base):
    __tablename__ = "anomaly_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unit_nr = Column(Integer, index=True, nullable=False)
    time_cycles = Column(Integer, nullable=False)
    alert_level = Column(String(20), nullable=False)  # WARNING, CRITICAL
    health_index = Column(Float, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Async Engine and Session
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db():
    """Initializes database tables asynchronously."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency for obtaining an async session."""
    async with AsyncSessionLocal() as session:
        yield session


async def save_telemetry(unit_nr: int, time_cycles: int, settings_dict: Dict[str, float], sensors: Dict[str, float]):
    """Inserts a single telemetry cycle."""
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


async def get_recent_telemetry(unit_nr: int, limit: int = 30) -> List[Dict]:
    """Retrieves up to `limit` recent telemetry cycles for a given engine."""
    async with AsyncSessionLocal() as session:
        stmt = (
            select(TelemetryRecord)
            .where(TelemetryRecord.unit_nr == unit_nr)
            .order_by(TelemetryRecord.time_cycles.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()
        # Return sorted ascending by time_cycles
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


async def save_prediction(unit_nr: int, cycle: int, model_used: str, rul: float, hi: float, status: str):
    """Stores model prediction record."""
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
