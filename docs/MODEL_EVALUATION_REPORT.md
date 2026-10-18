# C-MAPSS Turbofan Model Evaluation & Benchmark Report

## 1. Executive Summary
This report benchmarks three prognostic architectures (**XGBoost Regressor**, **Stacked LSTM**, and **Dilated TCN**) along with an unsupervised **Isolation Forest** anomaly detector on the **NASA C-MAPSS FD001** benchmark.

Models are evaluated on Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), $R^2$, and the official **NASA Asymmetric Scoring Function** (which penalizes late predictions where $\hat{y} > y$ significantly harder than early alerts).

---

## 2. Model Performance Leaderboard (Cutoff Test Set)

| Model Architecture | RMSE (Cycles) | MAE (Cycles) | $R^2$ Score | NASA Asymmetric Score |
| :--- | :---: | :---: | :---: | :---: |
| **Dilated TCN** | **10.6048** | **5.8121** | **0.9307** | **350.74** |
| **Stacked LSTM** | 10.9007 | 6.2268 | 0.9268 | 370.7 |
| **XGBoost (Baseline)** | 11.4012 | 6.1335 | 0.9199 | 454.5 |

---

## 3. Anomaly Detection & Reliability Metrics (Arena 3)

The unsupervised **Isolation Forest** model was trained on pristine early-life cycles ($t \le 30$) to establish nominal operating distributions.

| Evaluation Metric | Measured Value | Target Criterion | Operational Status |
| :--- | :---: | :---: | :---: |
| **False Alert Rate (FAR)** | **4.0%** | $< 5.0\%$ | PASS |
| **Avg Detection Lead Time** | **237.82 cycles** | $> 40\text{ cycles}$ | PASS |
| **Missed Failure Rate (MFR)** | **0.0%** | $< 2.0\%$ | PASS |

---

## 4. Feature Quality & Data Audit

- **Total Training Telemetry**: 24177 cycles across 100 engines.
- **Missing Values**: 0 (0%).
- **Duplicate Cycles**: 0 (0%).
- **Invariant Sensor Channels Filtered**: `['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']`.
- **Engineered Channels**: 140 features including rolling means, standard deviations, and multi-scale difference vectors.
