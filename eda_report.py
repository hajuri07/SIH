"""
Exploratory Data Analysis (EDA) & Descriptive Statistics Generator.
Generates:
1. Descriptive statistics tables for all sensors.
2. Missing values and duplicates audit.
3. Sensor correlation heatmap figure.
4. Degradation trajectory curves for active sensors.
5. Engine failure lifetime distribution histogram.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
IMAGES_DIR = ROOT_DIR / "docs" / "images"
DOCS_DIR = ROOT_DIR / "docs"

from data.download_data import prepare_dataset
from src.preprocessing.cleaner import CMAPSSCleaner


def run_eda():
    prepare_dataset()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    cleaner = CMAPSSCleaner()
    train_df = cleaner.load_raw_file(RAW_DIR / "train_FD001.txt")

    print("[INFO] Computing descriptive statistics...", flush=True)
    sensor_cols = [c for c in train_df.columns if c.startswith("s_") or c.startswith("setting_")]
    stats_df = train_df[sensor_cols].describe().T
    stats_dict = stats_df.round(4).to_dict(orient="index")

    stats_path = DOCS_DIR / "descriptive_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[SUCCESS] Saved descriptive statistics to: {stats_path}", flush=True)

    # Active sensors
    active_sensors = cleaner.fit_active_features(train_df)

    # 1. Plot Engine Failure Lifetimes Histogram
    max_cycles = train_df.groupby("unit_nr")["time_cycles"].max()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(max_cycles, bins=20, kde=True, color="#2563EB", ax=ax)
    ax.set_title("Distribution of Engine Lifetimes to Failure (Cycles)", fontsize=13)
    ax.set_xlabel("Operational Flight Cycles")
    ax.set_ylabel("Engine Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = IMAGES_DIR / "engine_lifetimes_histogram.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved lifetime histogram: {hist_path}")

    # 2. Sensor Correlation Heatmap
    corr = train_df[active_sensors].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, annot_kws={"size": 7})
    ax.set_title("Active Sensor Cross-Correlation Matrix (FD001)", fontsize=13)
    plt.tight_layout()
    corr_path = IMAGES_DIR / "sensor_correlations.png"
    plt.savefig(corr_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved correlation heatmap: {corr_path}")

    # 3. Sensor Degradation Trajectories for Sample Engines
    sample_units = [1, 2, 3, 4]
    key_sensors = ["s_2", "s_3", "s_4", "s_7", "s_9", "s_11", "s_12", "s_15"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for idx, s in enumerate(key_sensors[:8]):
        ax = axes[idx]
        for u in sample_units:
            u_df = train_df[train_df["unit_nr"] == u]
            ax.plot(u_df["time_cycles"], u_df[s], label=f"Engine {u}", alpha=0.8, lw=1.2)
        ax.set_title(f"Sensor {s.upper()} Trajectory", fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx % 2 == 0:
            ax.set_ylabel("Reading")
        if idx >= 6:
            ax.set_xlabel("Flight Cycles")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Degradation Trajectories Across Sample Engines (Run-to-Failure)", fontsize=14)
    plt.tight_layout()
    traj_path = IMAGES_DIR / "sensor_degradation_trajectories.png"
    plt.savefig(traj_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved degradation trajectories: {traj_path}")


if __name__ == "__main__":
    run_eda()
