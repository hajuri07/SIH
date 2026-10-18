# Predictive Maintenance & Aeronautics Domain Glossary

## 1. Aeronautics & Turbomachinery Terminology

### Turbofan Engine
An airbreathing jet engine featuring a ducted fan that propels bypass air around the core engine, generating the majority of thrust at subsonic and transonic cruise speeds.

### HPC (High-Pressure Compressor)
Compressor stage positioned directly upstream of the combustor. Operates under extreme mechanical stresses and thermal loads. HPC degradation (blade erosion, aerodynamic stall margin reduction) is the primary failure mode simulated in C-MAPSS FD001 and FD002.

### LPC (Low-Pressure Compressor) / Fan
The initial compression stage directly driven by the low-pressure turbine. Susceptible to foreign object debris (FOD) ingestion, fan blade imbalance, and aerodynamic flutter.

### HPT & LPT (High & Low-Pressure Turbines)
Expansion stages that extract kinetic and thermal energy from combustion gases to drive the HPC and LPC respectively. Bleed cooling flows ($W_{31}$ and $W_{32}$) maintain turbine blade metal temperatures below metallurgical melting limits.

### Bleed Air & Bleed Enthalpy
Compressed air tapped from the compressor stages to provide cabin pressurization, anti-icing, and internal turbine cooling. Variations in bleed enthalpy (`htBleed` / $s_{17}$) reflect thermodynamic inefficiencies caused by internal gas-path seal leakage.

### Bypass Ratio (BPR)
The ratio between the mass flow rate of air bypassing the engine core to the mass flow rate passing through the core. In degraded engines, core flow resistance shifts flow outward, increasing BPR.

---

## 2. Prognostics & Health Management (PHM) Terminology

### Remaining Useful Life (RUL)
The expected operating time (measured in flight cycles or flight hours) remaining before an engine component reaches an unacceptable risk of catastrophic failure or exceeds safe operating clearances.

### Health Index (HI)
A continuous normalized prognostic score spanning $[0.0, 1.0]$:
- **$1.0$**: Pristine, nominal baseline condition.
- **$0.5 – 0.7$**: Early detectable thermodynamic drift (Advisory warning).
- **$< 0.2$**: Critical degradation state requiring immediate unscheduled maintenance or grounding.

### Detection Lead Time (DLT)
The number of operational flight cycles elapsed between the initial generation of an anomaly alert and the eventual functional failure of the engine. A high lead time allows maintenance crews to schedule depot-level repairs without disrupting airline flight schedules.

### False Alert Rate (FAR)
The percentage of healthy baseline operating cycles incorrectly flagged as anomalies by the detection algorithm:
$$\text{FAR} = \frac{\text{False Positives}}{\text{Total Healthy Baseline Cycles}} \times 100\%$$

### Missed Failure Rate (MFR)
The percentage of engines that degraded to failure without an alert being triggered prior to the critical threshold:
$$\text{MFR} = \frac{\text{Unflagged Failing Engines}}{\text{Total Failing Engines}} \times 100\%$$

---

## 3. Machine Learning & Explainability Terminology

### NASA Asymmetric Scoring Function
The official evaluation metric defined by Saxena et al. (2008) for C-MAPSS benchmark competitions:
$$d_i = \hat{y}_i - y_i$$
$$S = \sum_{i=1}^{N} s_i, \quad s_i = \begin{cases} e^{-d_i / 13} - 1, & d_i < 0 \text{ (Early prediction)} \\ e^{d_i / 10} - 1, & d_i \ge 0 \text{ (Late prediction)} \end{cases}$$
In aviation safety, predicting that an engine will last longer than it actually will (late prediction, $d_i > 0$) is catastrophic; the exponential penalty for positive errors is steeper than for early alerts ($e^{d/10}$ vs $e^{-d/13}$).

### Dilated Temporal Convolutional Network (TCN)
A deep neural network architecture combining 1D causal convolutions with exponentially increasing dilation factors ($2^0, 2^1, 2^2, \dots$) and residual connections. TCNs capture long-range temporal dependencies over sequential sensor histories without the vanishing/exploding gradient problems common in RNNs.

### SHAP (SHapley Additive exPlanations)
A cooperative game-theoretic framework that computes the exact marginal contribution of each sensor measurement to the final RUL prediction or anomaly score, satisfying the mathematical properties of local accuracy, missingness, and consistency.
