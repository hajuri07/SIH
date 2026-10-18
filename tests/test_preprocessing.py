"""
Unit Tests for Data Cleaning and Feature Engineering Pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing.cleaner import CMAPSSCleaner
from src.preprocessing.feature_engineering import CMAPSSFeatureEngineer
from src.preprocessing.dataset import CMAPSSSequenceDataset, build_sequence_data


@pytest.fixture
def sample_telemetry_df():
    """Generates synthetic run-to-failure data for 2 engines."""
    rows = []
    # Engine 1: 50 cycles
    for c in range(1, 51):
        rows.append({
            "unit_nr": 1,
            "time_cycles": c,
            "setting_1": 0.001,
            "setting_2": 0.0002,
            "setting_3": 100.0,  # Invariant
            "s_1": 518.67,       # Invariant
            "s_2": 642.0 + c * 0.05,
            "s_3": 1580.0 + c * 0.5,
            "s_4": 1400.0 + c * 0.4
        })
    # Engine 2: 40 cycles
    for c in range(1, 41):
        rows.append({
            "unit_nr": 2,
            "time_cycles": c,
            "setting_1": -0.001,
            "setting_2": -0.0002,
            "setting_3": 100.0,  # Invariant
            "s_1": 518.67,       # Invariant
            "s_2": 641.5 + c * 0.04,
            "s_3": 1582.0 + c * 0.4,
            "s_4": 1398.0 + c * 0.35
        })
    return pd.DataFrame(rows)


def test_cleaner_quality_audit(sample_telemetry_df):
    cleaner = CMAPSSCleaner(variance_threshold=1e-5)
    audit = cleaner.perform_quality_audit(sample_telemetry_df)

    assert audit["total_records"] == 90
    assert audit["total_units"] == 2
    assert audit["missing_values"] == 0
    assert audit["duplicate_records"] == 0
    assert "setting_3" in audit["zero_variance_columns"]
    assert "s_1" in audit["zero_variance_columns"]


def test_cleaner_filtering(sample_telemetry_df):
    cleaner = CMAPSSCleaner()
    active_cols = cleaner.fit_active_features(sample_telemetry_df)

    assert "s_1" not in active_cols
    assert "setting_3" not in active_cols
    assert "s_2" in active_cols
    assert "s_3" in active_cols

    cleaned = cleaner.transform(sample_telemetry_df)
    assert "s_1" not in cleaned.columns
    assert "setting_3" not in cleaned.columns
    assert "unit_nr" in cleaned.columns
    assert "time_cycles" in cleaned.columns


def test_feature_engineering_piecewise_rul(sample_telemetry_df):
    fe = CMAPSSFeatureEngineer(max_rul=30, window_sizes=[5, 10])
    df_with_rul = fe.add_rul_target(sample_telemetry_df)

    # Unit 1 runs 50 cycles: at cycle 1, raw RUL is 49, piecewise is min(30, 49) = 30
    u1_cycle1 = df_with_rul[(df_with_rul["unit_nr"] == 1) & (df_with_rul["time_cycles"] == 1)].iloc[0]
    assert u1_cycle1["rul"] == 30.0

    # At cycle 50, RUL is 0
    u1_cycle50 = df_with_rul[(df_with_rul["unit_nr"] == 1) & (df_with_rul["time_cycles"] == 50)].iloc[0]
    assert u1_cycle50["rul"] == 0.0


def test_sequence_dataset_sliding_windows(sample_telemetry_df):
    fe = CMAPSSFeatureEngineer(max_rul=30, window_sizes=[5])
    featured_df, feat_cols = fe.fit_transform(sample_telemetry_df, ["s_2", "s_3"], is_train=True)

    features, targets, unit_indices = build_sequence_data(featured_df, feat_cols, sequence_length=15)
    ds = CMAPSSSequenceDataset(features, targets, unit_indices, sequence_length=15)

    assert len(ds) > 0
    seq_tensor, target_tensor = ds[0]
    assert seq_tensor.shape == (15, len(feat_cols))
    assert isinstance(target_tensor.item(), float)
