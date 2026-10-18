"""
Pytest configuration and global fixtures.
"""

import pytest
import asyncio
from backend.db import engine, Base, init_db
from backend.services import registry
from backend.kafka_client import kafka_manager
from backend.main import handle_telemetry_inference


@pytest.fixture(autouse=True, scope="session")
def setup_test_suite():
    """Initializes a clean database, loads models, and registers Kafka consumer handler for the test session."""
    async def _init_clean_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_clean_db())
    if not registry.is_loaded:
        registry.load_models()
    kafka_manager.register_handler(handle_telemetry_inference)
