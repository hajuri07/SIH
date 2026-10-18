# Turbofan Engine Predictive Maintenance & RUL Prognostics System
### Smart India Hackathon (SIH) — Industry 4.0 / Aeronautics Prognostics & Health Management (PHM)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-11B0C4?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com)

---

## 1. Project Overview
This project is an end-to-end, production-grade **Predictive Maintenance & Remaining Useful Life (RUL) Prognostics System** designed for commercial aircraft turbofan jet engines using the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) benchmark dataset.

The system is built to solve four major industrial challenges in aviation:
1. **Unplanned Aircraft Groundings (AOG)**: Inability to anticipate compressor and turbine degradation before functional failure.
2. **Asymmetric Risk in Aviation**: Predicting an engine will last longer than it actually can is catastrophic; late predictions must be penalized exponentially harder than early maintenance warnings.
3. **Black-Box AI Acceptance**: Maintenance engineers require explainable root-cause attribution (e.g. knowing *why* an engine has low RUL and *which physical sensor* caused the alert).
4. **Real-Time High-Throughput Ingestion**: Ingesting high-frequency telemetry into time-series hypertables and running sub-second inference.

---

## 2. Key Capabilities by Arena

### Arena 1: Dataset Preparation & Prognostic Modeling
- **C-MAPSS Data Ingestion**: Automated downloader for FD001–FD004 with high-fidelity offline fallback generator.
- **Data Quality Audit**: Automated missing value audits, duplicate checks, and invariant flatline sensor detection (`s_1`, `s_5`, `s_6`, `s_10`, `s_16`, `s_18`, `s_19`, `setting_3`).
- **Feature Engineering**: Multi-scale rolling averages, rolling standard deviations, differential features, and piecewise linear RUL target clipping ($R_{\max} = 125$ cycles).
- **Model Suite**:
  - **XGBoost Regressor**: High-speed, interpretable gradient boosted tree baseline.
  - **Stacked PyTorch LSTM**: 2-layer recurrent network with Dropout and LayerNorm for sequence modeling.
  - **Dilated Temporal Convolutional Network (TCN)**: 1D causal convolutions with exponentially growing receptive fields ($2^0, 2^1, 2^2$).
- **Benchmark Metrics**: Evaluated against RMSE, MAE, $R^2$, and the official **NASA Asymmetric Scoring Function**.

### Arena 2: Backend Microservice & System Integration
- **FastAPI Microservice**: Async REST endpoints with Pydantic validation.
- **Time-Series Persistence**: TimescaleDB hypertables with automatic SQLite fallback for zero-dependency standalone execution.
- **Streaming Telemetry Simulator**: In-memory and Kafka-compatible flight cycle replay worker.

### Arena 3: Anomaly Detection & AI Explainability
- **Isolation Forest**: Unsupervised anomaly detection trained strictly on nominal early-life cycles ($t \le 30$).
- **Health Index (HI)**: Continuous degradation index $[0.0, 1.0]$ with `HEALTHY`, `WARNING`, and `CRITICAL` triage states.
- **Proactive Lead Time**: Evaluates Detection Lead Time (cycles before failure alert triggers), False-Alert Rate (FAR), and Missed Failure Rate (MFR).
- **TreeSHAP Explainability**: Local waterfall attributions and global feature rankings explaining exact sensor contributions.

---

## 3. Project Directory Structure

```
c:\Ibrahim\Projects\SIH\
├── data/
│   ├── raw/                        # C-MAPSS raw files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt)
│   ├── processed/                  # Normalized datasets
│   ├── download_data.py            # Automated downloader with offline fallback
│   ├── DATA_DICTIONARY.md          # 26 columns, units, operating ranges & physics
│   └── GLOSSARY.md                 # PHM, aeronautics, and ML domain glossary
├── docs/
│   ├── ARCHITECTURE.md             # System topology and data flow diagram
│   ├── MODEL_EVALUATION_REPORT.md  # Detailed benchmark metrics across all models
│   ├── USAGE_INSTRUCTIONS.md       # Setup, training, and execution guide
│   └── images/                     # Generated EDA figures & plots
├── src/
│   ├── preprocessing/              # Cleaner, feature engineer, sequence dataset
│   ├── models/                     # XGBoost, Stacked LSTM, TCN, Anomaly Detector, Evaluator
│   ├── explainability/             # TreeSHAP explainer
│   └── pipeline/                   # Unified training orchestrator
├── backend/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── config.py                   # App configuration
│   ├── db.py                       # TimescaleDB & SQLite async persistence
│   ├── schemas.py                  # Pydantic V2 schemas
│   ├── services.py                 # In-memory model registry
│   ├── stream_simulator.py         # Telemetry flight streaming simulator
│   └── routers/                    # Endpoints for telemetry, prediction, anomaly, SHAP, stream
├── dashboard/
│   └── app.py                      # Interactive Streamlit Fleet Cockpit
├── deployment/
│   ├── Dockerfile                  # Container definition
│   ├── docker-compose.yml          # TimescaleDB, Kafka, Zookeeper, FastAPI, Streamlit
│   └── init_timescaledb.sql        # TimescaleDB hypertable DDL
├── tests/                          # Pytest test suite for preprocessing, models, and API
├── eda_report.py                   # EDA generator script
├── requirements.txt
└── README.md
```

---

## 4. Quick Start Guide

### 1. Run Data Preparation & EDA
```powershell
python eda_report.py
```

### 2. Train All Models & Benchmark
```powershell
python -m src.pipeline.train_all
```

### 3. Run Test Suite
```powershell
pytest tests/ -v
```

### 4. Start FastAPI Microservice
```powershell
uvicorn backend.main:app --port 8000 --reload
```
Interactive API documentation available at `http://localhost:8000/docs`.

### 5. Launch Interactive Operator Dashboard
```powershell
streamlit run dashboard/app.py
```
Open `http://localhost:8501` to view the live fleet cockpit.
