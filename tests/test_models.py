"""
Unit Tests for Machine Learning and Deep Learning Models.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from src.models.evaluator import compute_nasa_score, evaluate_rul_predictions
from src.models.rul_xgboost import RULXGBoostModel
from src.models.rul_lstm import RULLSTMModel
from src.models.rul_tcn import RULTCNModel
from src.models.anomaly_detector import EngineAnomalyDetector
from src.models.autoencoder_detector import AutoencoderAnomalyDetector


def test_nasa_score_asymmetry():
    y_true = np.array([50.0])

    # Early prediction by 10 cycles (pred = 40)
    score_early = compute_nasa_score(y_true, np.array([40.0]))
    # Late prediction by 10 cycles (pred = 60)
    score_late = compute_nasa_score(y_true, np.array([60.0]))

    # Catastrophic penalty: late predictions MUST incur a strictly higher penalty than early ones
    assert score_late > score_early
    assert score_late > 1.45 * score_early

    # For larger error (20 cycles), exponential asymmetry grows (> 1.7x)
    score_early_20 = compute_nasa_score(y_true, np.array([30.0]))
    score_late_20 = compute_nasa_score(y_true, np.array([70.0]))
    assert score_late_20 > 1.7 * score_early_20


def test_xgboost_fit_predict():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(np.random.uniform(0, 125, size=100))

    model = RULXGBoostModel(n_estimators=10, max_depth=3)
    metrics = model.fit(X, y)

    assert "rmse" in metrics
    preds = model.predict(X)
    assert len(preds) == 100
    assert np.all(preds >= 0.0)


def test_lstm_forward_and_prediction():
    batch_size = 16
    seq_len = 20
    input_dim = 6

    model = RULLSTMModel(input_dim=input_dim, hidden_dim=32, num_layers=1)
    dummy_input = np.random.randn(seq_len, input_dim).astype(np.float32)
    pred_rul = model.predict_single_sequence(dummy_input)

    assert isinstance(pred_rul, float)
    assert pred_rul >= 0.0


def test_tcn_forward_and_prediction():
    seq_len = 20
    input_dim = 6

    model = RULTCNModel(input_dim=input_dim, num_channels=[16, 32], kernel_size=2)
    dummy_input = np.random.randn(seq_len, input_dim).astype(np.float32)
    pred_rul = model.predict_single_sequence(dummy_input)

    assert isinstance(pred_rul, float)
    assert pred_rul >= 0.0


def test_tcn_dataloader_prediction():
    seq_len = 20
    input_dim = 6
    batch_size = 4
    model = RULTCNModel(input_dim=input_dim, num_channels=[16, 32], kernel_size=2)
    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    dummy_y = torch.tensor([50.0, 40.0, 30.0, 20.0])
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)
    preds = model.predict(loader)
    assert len(preds) == 4
    assert np.all(preds >= 0.0)


def test_isolation_forest_anomaly_detector():
    np.random.seed(42)
    # 50 healthy cycles
    X_healthy = pd.DataFrame(np.random.normal(loc=10.0, scale=0.5, size=(50, 4)), columns=[f"s_{i}" for i in range(4)])
    # 5 outlier cycles
    X_outlier = pd.DataFrame(np.random.normal(loc=30.0, scale=2.0, size=(5, 4)), columns=[f"s_{i}" for i in range(4)])

    detector = EngineAnomalyDetector(contamination=0.05, n_estimators=50)
    detector.fit(X_healthy)

    is_anom, hi, status = detector.predict(X_outlier)
    assert len(is_anom) == 5
    assert np.all((hi >= 0.0) & (hi <= 1.0))
    # Outliers should trigger WARNING or CRITICAL
    assert any(s in ("WARNING", "CRITICAL") for s in status)


def test_autoencoder_anomaly_detector():
    np.random.seed(42)
    torch.manual_seed(42)
    # 60 baseline cycles
    X_baseline = pd.DataFrame(
        np.random.normal(loc=5.0, scale=0.2, size=(60, 8)),
        columns=[f"feat_{i}" for i in range(8)]
    )
    # Outlier cycles with severe sensor shift
    X_outlier = pd.DataFrame(
        np.random.normal(loc=15.0, scale=1.5, size=(10, 8)),
        columns=[f"feat_{i}" for i in range(8)]
    )

    ae = AutoencoderAnomalyDetector(input_dim=8, latent_dim=4, lr=0.01)
    stats = ae.fit(X_baseline, epochs=15, batch_size=16)

    assert "baseline_mean_mse" in stats
    assert stats["baseline_samples"] == 60

    is_anom, hi, status = ae.predict(X_outlier)
    assert len(is_anom) == 10
    assert np.all((hi >= 0.0) & (hi <= 1.0))
    # Outliers should trigger anomaly flag
    assert np.sum(is_anom) > 0
    assert any(s in ("WARNING", "CRITICAL") for s in status)

