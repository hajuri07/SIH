"""
Application Configuration Settings.
Loads environment variables with sensible defaults for local development and Docker containers.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "SIH Turbofan Predictive Maintenance & RUL System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database: TimescaleDB / PostgreSQL or SQLite Fallback
    DATABASE_URL: str = "sqlite+aiosqlite:///./telemetry.db"
    USE_SQLITE_FALLBACK: bool = True

    # Kafka configuration (optional broker)
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_TELEMETRY: str = "turbofan-telemetry"
    KAFKA_TOPIC_ALERTS: str = "turbofan-alerts"

    # Model Artifacts
    ARTIFACTS_DIR: Path = ROOT_DIR / "models" / "artifacts"
    RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"


settings = Settings()
