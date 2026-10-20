"""
Shared fixtures for the Arena 6 tests.

* Data/seed tests need nothing but pytest (standard library only).
* API tests need the backend requirements (fastapi, httpx, sqlalchemy, aiosqlite, pytest-asyncio ...)
  and are skipped automatically when those are not installed.

API tests run against a throw-away SQLite file, so they never write into the repo's telemetry.db
(unless backend.db was already imported by another test module earlier in the same session).
"""

import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "backend.db" not in sys.modules and "DATABASE_URL" not in os.environ:
    _tmp_dir = tempfile.mkdtemp(prefix="arena6_api_")
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///" + (Path(_tmp_dir) / "arena6_test.db").as_posix()

try:
    import pytest_asyncio
except ImportError:  # API tests will be skipped via importorskip
    pytest_asyncio = None


@pytest.fixture
def fresh_unit():
    """Returns a function giving a random engine id that no other test uses."""
    def _make() -> int:
        return random.randint(10_000_000, 99_999_999)
    return _make


if pytest_asyncio is not None:

    @pytest_asyncio.fixture
    async def client():
        """HTTP client wired straight to the FastAPI app (no server needed)."""
        from httpx import AsyncClient, ASGITransport
        from backend.main import app
        from backend.db import init_db, engine
        from backend.services import registry

        # httpx's ASGITransport does not run the app 'lifespan', so do the startup work here.
        await init_db()
        if not registry.is_loaded:
            registry.load_models()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
        await engine.dispose()  # avoid "event loop closed" errors between tests


@pytest.fixture
def models_ready(client):
    """Skip a test when the trained model files are not present/loaded."""
    from backend.services import registry
    if not (registry.feature_engineer and registry.xgboost_model and registry.anomaly_detector):
        pytest.skip("Model artifacts not loaded - run: python -m src.pipeline.train_all")
    return registry
