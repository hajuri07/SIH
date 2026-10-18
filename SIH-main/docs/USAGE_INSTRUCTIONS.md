# SIH Turbofan Predictive Maintenance: Usage Instructions

This guide provides instructions to run data cleaning, model training, the FastAPI backend microservice, and the interactive Streamlit monitoring dashboard.

---

## 1. Environment Setup

Activate the project virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

If dependencies need re-installation:
```powershell
pip install -r requirements.txt
```

---

## 2. Step 1: Exploratory Data Analysis & Quality Audit

Generate descriptive statistics, missing value audits, and high-resolution sensor plots:

```powershell
python eda_report.py
```

Outputs generated:
- `docs/descriptive_statistics.json`: Complete statistical summaries for all 21 sensors.
- `docs/images/sensor_correlations.png`: Heatmap of active sensor cross-correlations.
- `docs/images/sensor_degradation_trajectories.png`: Degradation wear curves across engines.
- `docs/images/engine_lifetimes_histogram.png`: Lifetime-to-failure distribution.

---

## 3. Step 2: Run Unified Training & Benchmark Pipeline

Execute the end-to-end training pipeline for all models (XGBoost, Stacked LSTM, Dilated TCN, Isolation Forest, and SHAP):

```powershell
python -m src.pipeline.train_all
```

Outputs generated in `models/artifacts/`:
- `feature_engineer.joblib`: Fitted scaler, invariant sensor pruner, and rolling configuration.
- `xgboost_rul.joblib`: Trained XGBoost regressor model.
- `lstm_rul.pt`: PyTorch Stacked LSTM state dictionary.
- `tcn_rul.pt`: PyTorch Dilated TCN state dictionary.
- `isolation_forest.joblib`: Trained Isolation Forest anomaly model.
- `shap_explainer.joblib`: Fitted SHAP explainer configuration.
- `benchmark_metrics.json`: Consolidated benchmark evaluation scores.
- `docs/MODEL_EVALUATION_REPORT.md`: Formatted Markdown report comparing models on RMSE, MAE, and the NASA Asymmetric Score.

---

## 4. Step 3: Run Unit & Integration Tests

Verify that data cleaners, feature engineering, neural network architectures, and API endpoints pass all assertions:

```powershell
pytest tests/ -v
```

---

## 5. Step 4: Launch FastAPI Microservice (Arena 2)

Start the production REST API server with automatic hot-reloading:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **Health Check**: `http://127.0.0.1:8000/health`

### Key API Endpoints:
- `POST /api/v1/telemetry`: Ingest flight cycle sensor packets.
- `POST /api/v1/predict/rul`: Predict RUL using XGBoost, LSTM, or TCN.
- `POST /api/v1/detect/anomaly`: Evaluate Health Index (HI) and trigger alerts.
- `GET /api/v1/explain/{unit_nr}`: Retrieve real-time SHAP root-cause feature attributions.
- `POST /api/v1/stream/control`: Start/stop background live streaming simulation.

---

## 6. Step 5: Launch Interactive Fleet Dashboard

Launch the Streamlit monitoring cockpit:

```powershell
streamlit run dashboard/app.py
```

The browser will open `http://localhost:8501`, featuring:
- **Fleet Overview**: Health status, critical engine alerts, active flight cycles.
- **Degradation Trajectories**: Live RUL predictions vs critical failure thresholds.
- **Anomaly Radar**: Isolation Forest continuous Health Index and lead-time tracking.
- **AI Explainability Lab**: Interactive SHAP waterfall plots pinpointing exact sensor degradation drivers.
- **Leaderboard**: Side-by-side benchmark table comparing TCN, LSTM, and XGBoost.

---

## 7. Step 6: Docker Compose Multi-Container Deployment

To run the complete production stack (TimescaleDB, Kafka, Zookeeper, FastAPI, and Streamlit):

```powershell
cd deployment
docker-compose up --build
```
