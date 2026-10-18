# Turbofan Engine Predictive Maintenance: Architecture & Data Flow

## 1. System Topology Overview
The system is architected as an event-driven, production-grade microservice pipeline that continuously ingests multi-channel sensor telemetry, evaluates real-time degradation indices, isolates early anomalies, forecasts Remaining Useful Life (RUL), and delivers explainable root-cause diagnostics to aircraft maintenance operators.

```
+-----------------------------------------------------------------------------------+
|                            AIRCRAFT TELEMETRY INGESTION                           |
|  NASA C-MAPSS FD001-FD004 Telemetry Bus / IoT Stream Simulator / Kafka Broker     |
+-----------------------------------------------------------------------------------+
                                       │
                                       ▼ (JSON Telemetry Packets)
+-----------------------------------------------------------------------------------+
|                                FASTAPI MICROSERVICE                               |
|  • /api/v1/telemetry        • /api/v1/predict/rul       • /api/v1/detect/anomaly  |
|  • /api/v1/explain/{unit}   • /api/v1/stream/control    • /health                 |
+-----------------------------------------------------------------------------------+
         │                                                            │
         ▼ (Async Session)                                            ▼ (Tensor Tensors)
+----------------------------+                     +---------------------------------+
|   TIME-SERIES DATA LAYER   |                     |      PROGNOSTIC ML/DL ENGINE    |
| • TimescaleDB Hypertables  |                     | • Invariant Channel Pruner      |
| • PostgreSQL / SQLite      |                     | • Multi-Scale Rolling Features  |
| • Continuous Aggregates    |                     | • Min-Max Spatial Normalizer    |
+----------------------------+                     +---------------------------------+
                                                                      │
                                    ┌─────────────────────────────────┼──────────────────────────────┐
                                    ▼                                 ▼                              ▼
                    +------------------------------+   +------------------------------+   +--------------------+
                    |     RUL PREDICTION CORE      |   |   ANOMALY RADAR (ARENA 3)    |   | SHAP EXPLAINABILITY|
                    | • Dilated TCN (Causal Conv)  |   | • Isolation Forest           |   | • TreeSHAP Local   |
                    | • Stacked 2-Layer LSTM       |   | • Health Index (HI) [0..1]   |   |   Waterfall Plots  |
                    | • XGBoost Regressor Baseline |   | • Lead-Time / FAR Evaluation |   | • Global Ranking   |
                    +------------------------------+   +------------------------------+   +--------------------+
                                    │                                 │                              │
                                    └─────────────────────────────────┼──────────────────────────────┘
                                                                      ▼
+-------------------------------------------------------------------------------------------------------+
|                                    OPERATOR DASHBOARD (STREAMLIT)                                     |
|  • Fleet Status Cockpit      • Live Degradation Curves    • Sensor Gas-Path Gauges                    |
|  • Anomaly Alert Timeline    • SHAP Root-Cause Lab        • Architectural Model Leaderboard           |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

### 2.1 Invariant Sensor Filter & Feature Engineering Pipeline
- Raw C-MAPSS FD001 telemetry has 26 columns per cycle.
- **Invariant Sensor Removal**: Channels with zero variance (`s_1`, `s_5`, `s_6`, `s_10`, `s_16`, `s_18`, `s_19`, `setting_3`) are dynamically filtered out to eliminate non-informative numerical noise.
- **Temporal Feature Construction**:
  - Multi-scale rolling averages (windows = 5, 10, 20 cycles) to track moving baseline trends.
  - Multi-scale standard deviation to capture aerodynamic flutter and vibration instability.
  - Rolling differential features ($\Delta s_t = s_t - \mu_{t, w}$) to capture instantaneous thermodynamic drift.
- **Piecewise Linear Target**: Target RUL is capped at $R_{\max} = 125$ cycles to reflect realistic machinery run-in behavior prior to defect initiation.

### 2.2 Prognostic Model Architectures
1. **XGBoost Regressor**:
   - Tabular gradient boosted decision trees trained on multi-scale rolling temporal feature vectors.
   - Extremely low inference latency ($< 1 \text{ ms}$), interpretable tree structures compatible with TreeSHAP.
2. **PyTorch Stacked LSTM**:
   - 2-layer recurrent network consuming 30-cycle sliding sequences.
   - Uses Dropout and LayerNorm before final linear projection.
3. **Dilated Temporal Convolutional Network (TCN)**:
   - State-of-the-art causal 1D dilated convolutions with receptive field spanning $2^0, 2^1, 2^2$ dilations.
   - Strictly preserves causality (no future information leakage) via causal padding and chomp layers.

### 2.3 Anomaly Radar & Health Index (Arena 3)
- An unsupervised **Isolation Forest** is fitted on pristine early-life operational cycles ($t \le 30$).
- As internal components wear, high-dimensional sensor vectors deviate into low-density regions, triggering early anomaly flags.
- Normalized into a continuous **Health Index (HI)**:
  - $\text{HI} > 0.65$: **HEALTHY** (Nominal operating regime).
  - $0.35 < \text{HI} \le 0.65$: **WARNING** (Early thermodynamic drift detected).
  - $\text{HI} \le 0.35$: **CRITICAL** (Imminent failure alert; maintenance mandatory).

### 2.4 Time-Series Data Layer & Kafka Streaming (Arena 2)
- **TimescaleDB / PostgreSQL**: Optimized time-series hypertables partitioned on timestamp with hypertable indexes on `(unit_nr, time_cycles)`.
- **Zero-Dependency Fallback**: Automatic async SQLite engine (`telemetry.db`) activated when external PostgreSQL services are not present, enabling standalone execution on any laptop for hackathon presentations.
- **Kafka Stream Simulation**: Background worker that replays engine flights at configurable speeds, simulating real-time telemetry streaming from aircraft telemetry buses.
