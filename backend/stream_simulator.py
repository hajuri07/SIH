"""
Streaming Telemetry Simulator.
Simulates a live IoT sensor stream (e.g. from an aircraft Kafka / MQTT bus)
by emitting sequential flight cycles from test engines into the system.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import settings
from .db import save_telemetry, save_prediction
from .services import registry


class TelemetryStreamSimulator:
    def __init__(self):
        self.is_running: bool = False
        self.current_unit: int = 1
        self.current_cycle: int = 0
        self.total_emitted: int = 0
        self.delay_seconds: float = 0.5
        self._task: Optional[asyncio.Task] = None
        self._test_df: Optional[pd.DataFrame] = None

    def load_data(self):
        test_path = settings.RAW_DATA_DIR / "test_FD001.txt"
        if test_path.exists():
            col_names = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [
                f"s_{i}" for i in range(1, 22)
            ]
            self._test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
            if self._test_df.shape[1] > 26:
                self._test_df = self._test_df.iloc[:, :26]
            self._test_df.columns = col_names[:self._test_df.shape[1]]

    async def _stream_loop(self):
        if self._test_df is None:
            self.load_data()
        if self._test_df is None:
            print("[ERROR] StreamSimulator: No test dataset available.")
            return

        unit_df = self._test_df[self._test_df["unit_nr"] == self.current_unit].sort_values("time_cycles")
        print(f"[INFO] StreamSimulator started for Engine Unit {self.current_unit} ({len(unit_df)} cycles).")

        for _, row in unit_df.iterrows():
            if not self.is_running:
                break

            self.current_cycle = int(row["time_cycles"])
            sensors = {k: float(v) for k, v in row.items() if k.startswith("s_")}
            settings_dict = {
                "setting_1": float(row.get("setting_1", 0.0)),
                "setting_2": float(row.get("setting_2", 0.0)),
                "setting_3": float(row.get("setting_3", 100.0))
            }

            # Ingest into time-series database
            await save_telemetry(self.current_unit, self.current_cycle, settings_dict, sensors)
            self.total_emitted += 1

            # Proactive model inference if registry is loaded
            if registry.is_loaded and registry.feature_engineer:
                try:
                    # Ingest single slice for inference
                    recent_cycles = unit_df[unit_df["time_cycles"] <= self.current_cycle].tail(30)
                    rul = registry.predict_rul("xgboost", recent_cycles)
                    is_anom, hi, status = registry.detect_anomaly(recent_cycles)
                    await save_prediction(self.current_unit, self.current_cycle, "xgboost", rul, hi, status)
                except Exception as e:
                    pass

            await asyncio.sleep(self.delay_seconds)

        self.is_running = False
        print(f"[INFO] StreamSimulator completed for Engine Unit {self.current_unit}.")

    def start(self, unit_id: int = 1, speed: float = 0.5):
        if self.is_running:
            return False
        self.is_running = True
        self.current_unit = unit_id
        self.delay_seconds = max(0.05, speed)
        self._task = asyncio.create_task(self._stream_loop())
        return True

    def stop(self):
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        return True

    def get_status(self):
        return {
            "is_active": self.is_running,
            "current_unit": self.current_unit,
            "current_cycle": self.current_cycle,
            "total_cycles_emitted": self.total_emitted,
            "speed_seconds": self.delay_seconds
        }


simulator = TelemetryStreamSimulator()
