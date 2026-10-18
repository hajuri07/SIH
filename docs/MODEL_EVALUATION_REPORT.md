# C-MAPSS Turbofan Model Evaluation & Benchmark Report
*Generated on 2026-10-18 15:12:30 UTC*

## 1. Executive Summary
This report benchmarks three prognostic architectures (**XGBoost Regressor**, **Stacked LSTM**, and **Dilated TCN**) along with unsupervised **Isolation Forest** and **Deep Autoencoder** anomaly detectors on the **NASA C-MAPSS FD001** benchmark.

Training and validation engines were partitioned strictly by engine unit IDs prior to scaling to prevent data leakage. **Model selection was performed strictly on the held-out validation partition** to uphold rigorous ML methodology without test-set leakage. Evaluation on the 100 held-out test engines is reported below as an unbiased benchmark.

---

## 2. Model Performance Leaderboard (Held-Out Test Engines)

*Selected Best Architecture: **LSTM** (chosen via validation RMSE and NASA asymmetric score).*

| Model Architecture | Test RMSE (Cycles) | Test MAE (Cycles) | $R^2$ Score | NASA Asymmetric Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | **10.6343** | **5.7465** | **0.9303** | **400.77** | **SELECTED BEST (Val)** |
| **XGBoost (Baseline)** | 11.3541 | 6.196 | 0.9206 | 441.29 | Benchmark |
| **Stacked LSTM** | 10.6343 | 5.7465 | 0.9303 | 400.77 | Sequential DL |
| **Dilated TCN** | 10.3056 | 6.1282 | 0.9345 | 303.98 | Causal Convolutions |

---

## 3. RUL Uncertainty Estimation & Interval Coverage
Rather than a single arbitrary fixed window, prediction intervals are calibrated independently for each model architecture using empirical validation residual distributions at a 90% nominal confidence level:

| Model Architecture | 90% Prediction Half-Width | Validation Residual Std | Empirical Coverage (Test Engines) |
| :--- | :---: | :---: | :---: |
| **XGBoost** | $\pm 13.1$ cycles | 11.02 cycles | 87.0% |
| **Stacked LSTM** | $\pm 11.38$ cycles | 10.61 cycles | 89.0% |
| **Dilated TCN** | $\pm 14.97$ cycles | 11.35 cycles | 92.0% |

Selected default for API inference (LSTM): $\pm 11.38$ cycles at 90% confidence with 89.0% empirical test coverage.

---

## 4. Anomaly Detection & Reliability Metrics (Arena 3)
Both unsupervised **Isolation Forest** and **Deep Autoencoder** were trained strictly on pristine early-life cycles ($t \le 30$) to establish nominal operating distributions.

| Anomaly Architecture | False Alert Rate (FAR) | Avg Detection Lead Time | Missed Failure Rate (MFR) | Operational Status |
| :--- | :---: | :---: | :---: | :---: |
| **Isolation Forest** | **4.0%** | **232.35 cycles** | **0.0%** | PASS (Target: FAR < 5%, Lead > 40, MFR < 2%) |
| **Deep Autoencoder** | **4.33%** | **212.6 cycles** | **0.0%** | PASS (Target: FAR < 5%, Lead > 40, MFR < 2%) |

---

## 5. Feature Quality & Data Audit
- **Total Training Telemetry**: 24177 cycles across 100 engines.
- **Engine Partitions**: 80 train engines / 20 validation engines (leak-free split).
- **Missing Values**: 0 (0%).
- **Duplicate Cycles**: 0 (0%).
- **Invariant Sensor Channels Filtered**: `['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']`.
- **Engineered Channels**: 140 features including rolling means, standard deviations, and multi-scale difference vectors across active channels `['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']`.

---

## 6. Failure Cases, Operational Limitations & Boundary Analysis

To ensure transparent and safe deployment in aviation maintenance environments, the following failure cases and operational boundaries have been identified:

### 6.1 Early-Life RUL Plateau & Target Clipping ($R_{\max} = 125$)
- **Phenomenon:** Brand new engines operate in an initial equilibrium with minimal physical degradation. The model caps predictions at $R_{\max} = 125$ cycles.
- **Limitation:** For an engine at cycle 5 with 250 actual cycles to failure, the model outputs 125 cycles. While this reflects realistic PHM wear onset, operators must not interpret 125 cycles as the literal total lifespan for brand-new engines.
- **Mitigation:** The continuous Health Index ($	ext{HI} \approx 1.0$) should be used alongside RUL in early life to confirm pristine structural integrity.

### 6.2 Single-Condition Regime vs Full Flight Envelope Generalization
- **Phenomenon:** The FD001 dataset represents a single operating condition (Sea Level, single flight mode). In this regime, 7 sensors (`s_1, s_5, s_6, s_10, s_16, s_18, s_19`) and `setting_3` exhibit near-zero variance and are eliminated.
- **Limitation:** In multi-condition datasets (FD002/FD004 with 6 flight conditions including altitude and Mach variations), these filtered channels become active and require operating regime clustering or condition normalization before training.
- **Mitigation:** The `CMAPSSCleaner` variance threshold is dynamic; running on multi-condition sets automatically re-activates variable channels.

### 6.3 Asymmetric Loss Penalty Extremes (Late vs Early Outliers)
- **Phenomenon:** The official NASA scoring function penalizes late predictions exponentially: $s_i = \exp(d_i / 10) - 1$ for $d_i \ge 0$ vs $s_i = \exp(-d_i / 13) - 1$ for $d_i < 0$.
- **Failure Case:** A small number of test engines with large positive errors (predicting engine will last 25 cycles longer than reality) dominate the total benchmark score ($> 60\%$ of the entire score from 3% of engines).
- **Mitigation:** The system integrates conformal uncertainty intervals ($\pm 11.38$ cycles at 90% confidence), advising maintenance planners to schedule overhauls based on the **lower bound** of the interval ($RUL_{lower}$) for mission-critical flights.

### 6.4 Sudden Sensor Spikes vs True Mechanical Wear
- **Phenomenon:** A single temporary electrical glitch or transient sensor spike can produce a sudden drop in Health Index.
- **Mitigation:** Multi-scale rolling features ($w = 5, 10, 20$) and consecutive-cycle alert debounce rules in the backend prevent false single-cycle emergency groundings.
