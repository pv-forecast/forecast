# PV Power Forecasting

Comparing ML models for photovoltaic power forecasting:
- **Linear Regression** (OLS, Ridge, Lasso) - domain-engineered features
- **LSTM Neural Network** - learns temporal patterns from sequences
- **ARIMAX** - autoregressive with exogenous variables

## Project Structure

```
forecast/
├── data/                         # Data (private repo, see setup below)
├── models/
│   ├── linear/
│   │   └── regression.py         # OLS, Ridge, Lasso (5-180min horizons)
│   ├── lstm/
│   │   ├── lstm.py               # One-to-Many LSTM (seq_dim=1)
│   │   ├── lstm_manytomany.py    # Many-to-Many LSTM (seq_dim=60)
│   │   ├── lstm_model.py         # PyTorch LSTM class
│   │   ├── treat_nans.py         # NaN handling utilities
│   │   ├── postprocess_lstm.py   # LSTM evaluation
│   │   └── saved/                # Saved model weights
│   └── timeseries/
│       ├── ar.py                 # AutoRegression (excluded from comparison)
│       └── arma.py               # ARIMAX(12,1,1) with exogenous
├── forecasts/
│   └── linear/                   # Linear model predictions (HDF5)
├── metrics/
│   └── linear/                   # Linear model metrics (MAE, RMSE, Skill)
├── results/
│   └── lstm/                     # LSTM metrics (CSV)
├── figures/                      # All plots
├── data_management.py            # Data loading & feature engineering
├── postprocess.py                # Evaluation metrics (MAE, RMSE, Skill)
├── visualize_models.py           # Plot predictions vs actual
├── forecast_kt.py                # Irradiance (kt) forecasting
└── data_analysis.py              # Exploratory data analysis
```

## Reading Results

Results are stored in HDF5 format:

```python
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Forecasts (predictions vs actual)
df = pd.read_hdf("forecasts/linear/forecasts_30min_BOTH.h5", key="df")
print(df.head())

# Metrics (MAE, RMSE, Skill per horizon)
df = pd.read_hdf("metrics/linear/results_BOTH_intra-hour.h5", key="df")
print(df.head())
```

## Requirements

```bash
pip install -r requirements.txt
```

## Running Models

Run all commands from the project root directory.

### Linear Regression (OLS, Ridge, Lasso)

```bash
python models/linear/regression.py   # Train & generate forecasts
python postprocess.py                # Evaluate (MAE, RMSE, Skill)
python visualize_models.py           # Plot predictions vs actual
```

### LSTM Neural Networks

```bash
python models/lstm/lstm.py              # One-to-Many (seq_dim=1)
python models/lstm/lstm_manytomany.py   # Many-to-Many (seq_dim=60)
python models/lstm/postprocess_lstm.py  # Evaluate
```

### ARIMAX

```bash
python models/timeseries/arma.py     # ARIMAX with exogenous variables
```

## Data

The data is stored in a separate private repository: [pv-forecast/data](https://github.com/pv-forecast/data)

**Setup for collaborators with data access:**

```bash
# Clone this repo
git clone git@github.com:pv-forecast/forecast.git
cd forecast

# Clone data repo into data folder
git clone git@github.com:pv-forecast/data.git data
```

**Data contents:**
- Source: `data/PVAMM_201911-202011_PT5M_merged.csv`
- 1 year of 5-minute interval measurements (Nov 2019 - Nov 2020)
- Features: GHI, BNI, DHI, ENI, temperature, humidity, pressure, wind, solar angles
- Target: Pdc (DC power output)

## Train/Test Split

- **Method:** Chronological split (no shuffling, preserves time order)
- **Split ratio:** 80% train / 20% test
- **Train period:** Nov 2019 - Aug 2020 (first 80% of data)
- **Test period:** Aug 2020 - Nov 2020 (last 20% of data)

## Forecasting Task

**Goal:** Predict PV power output for the next 30-180 minutes (multi-step ahead forecasting)

- **Timestep:** 5 minutes
- **Horizons:** 6, 12, 18, 24, 30, 36 steps = 30, 60, 90, 120, 150, 180 minutes
- **Output:** 36 predictions per forecast (all horizons at once)

## Baseline: Smart Persistence

**Definition:** Smart Persistence scales the current power by the expected change in clear-sky irradiance.

```
P_predicted(t+h) = P_actual(t) * (CSGHI(t+h) / CSGHI(t))
```

Where:
- `P_actual(t)` = current measured power
- `CSGHI(t+h)` = Clear-Sky GHI at future time (calculable from sun position)
- `CSGHI(t)` = Clear-Sky GHI at current time

**Why CSGHI instead of ENI?**
- ENI varies only ~3% over the year (Earth-Sun distance)
- CSGHI varies significantly throughout the day (accounts for sun angle)

**Why this baseline?**
- Better than naive persistence (`P(t+h) = P(t)`) because it accounts for sun movement
- CSGHI is "free" information - calculable for any future time
- Physically meaningful for solar forecasting

## Features per Model

### Linear Regression (77 features)

| Category                       | Features                                                                                                                      | Count |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-------|
| Backward avg GHI clearness     | B_GHI_kt_0, B_GHI_kt_1, B_GHI_kt_2, B_GHI_kt_3, B_GHI_kt_4, B_GHI_kt_5, B_GHI_kt_6, B_GHI_kt_7, B_GHI_kt_8, B_GHI_kt_9, B_GHI_kt_10, B_GHI_kt_11 | 12    |
| Backward avg BNI clearness     | B_BNI_kt_0, B_BNI_kt_1, B_BNI_kt_2, B_BNI_kt_3, B_BNI_kt_4, B_BNI_kt_5, B_BNI_kt_6, B_BNI_kt_7, B_BNI_kt_8, B_BNI_kt_9, B_BNI_kt_10, B_BNI_kt_11 | 12    |
| Variability GHI                | V_GHI_kt_0, V_GHI_kt_1, V_GHI_kt_2, V_GHI_kt_3, V_GHI_kt_4, V_GHI_kt_5, V_GHI_kt_6, V_GHI_kt_7, V_GHI_kt_8, V_GHI_kt_9, V_GHI_kt_10, V_GHI_kt_11 | 12    |
| Variability BNI                | V_BNI_kt_0, V_BNI_kt_1, V_BNI_kt_2, V_BNI_kt_3, V_BNI_kt_4, V_BNI_kt_5, V_BNI_kt_6, V_BNI_kt_7, V_BNI_kt_8, V_BNI_kt_9, V_BNI_kt_10, V_BNI_kt_11 | 12    |
| Lagged GHI clearness           | L_GHI_kt_0, L_GHI_kt_1, L_GHI_kt_2, L_GHI_kt_3, L_GHI_kt_4, L_GHI_kt_5, L_GHI_kt_6, L_GHI_kt_7, L_GHI_kt_8, L_GHI_kt_9, L_GHI_kt_10, L_GHI_kt_11 | 12    |
| Lagged BNI clearness           | L_BNI_kt_0, L_BNI_kt_1, L_BNI_kt_2, L_BNI_kt_3, L_BNI_kt_4, L_BNI_kt_5, L_BNI_kt_6, L_BNI_kt_7, L_BNI_kt_8, L_BNI_kt_9, L_BNI_kt_10, L_BNI_kt_11 | 12    |
| Cross-irradiance (1-3 days)    | BNI_kt_one, BNI_kt_two, BNI_kt_three, gti_kt_one, gti_kt_two, gti_kt_three                                                    | 6     |
| Weather                        | Ta, vw, RH, wdir, tpw, Patm, TL, kd                                                                                           | 8     |
| Solar geometry                 | El, Az                                                                                                                        | 2     |
| Current power                  | Pdc_33                                                                                                                        | 1     |
| Clear-sky GHI                  | CSGHI                                                                                                                         | 1     |
| CSGHI ratio (per horizon)      | CSGHI_ratio_5min, 10min, ..., 180min                                                                                          | 36    |
| **Total**                      |                                                                                                                               | **114**|

### LSTM (17 features)

| Category              | Features                                                | Count |
|-----------------------|---------------------------------------------------------|-------|
| Raw irradiance        | GHI, BNI                                                | 2     |
| Weather               | Ta                                                      | 1     |
| Solar geometry        | El, Az                                                  | 2     |
| Future elevation      | El_future_30min, 60min, 90min, 120min, 150min, 180min   | 6     |
| CSGHI ratio (future)  | CSGHI_ratio_30min, 60min, 90min, 120min, 150min, 180min | 6     |
| **Total**             |                                                         | **17**|

### ARIMAX (6 features)

| Category                      | Features          | Count |
|-------------------------------|-------------------|-------|
| Power (12 lags)               | Pdc               | 1     |
| Exogenous (persistence)       | GHI, kt           | 2     |
| Exogenous (known future)      | CSGHI, El, ENI    | 3     |
| **Total**                     |                   | **6** |

**Notes:**
- Linear Regression uses heavily engineered features (domain knowledge)
- LSTM learns from raw features over a sequence
- ARIMAX uses exogenous variables with **persistence assumption**: X(t+h) ≈ X(t)
- For known future variables (CSGHI, El, ENI), actual future values are used instead of persistence

## Model Architecture Details

| Aspect             | Linear Regression         | LSTM                         | ARIMAX         |
|--------------------|---------------------------|------------------------------|----------------|
| **File**           | `regression.py`           | `lstm_manytomany.py`         | `arma.py`      |
| **Library**        | scikit-learn              | PyTorch                      | statsmodels    |
| **Variants**       | OLS, Ridge (L2), Lasso (L1) | Many-to-Many               | ARIMAX(12,1,1) |
| **Hidden units**   | -                         | 136                          | -              |
| **Layers**         | 1                         | 3 LSTM + dropout             | -              |
| **Normalization**  | StandardScaler            | MinMaxScaler                 | None           |
| **Regularization** | Ridge: L2, Lasso: L1      | Dropout 0.45                 | -              |
| **Training**       | Closed-form / CV          | Early stopping, LR scheduler | Closed-form    |

## Key Differences

| Aspect                       | Linear | LSTM           | ARIMAX             |
|------------------------------|--------|----------------|--------------------|
| Uses measured irradiance?    | Yes    | Yes            | Yes (persistence)  |
| Uses engineered features?    | Yes    | No             | No                 |
| Learns from sequence?        | No     | Yes (60 steps) | Yes (12 lags)      |
| Needs future X values?       | No     | No             | Yes (persistence)  |
| Captures non-linearity?      | No     | Yes            | No                 |

## Results

**Model Comparison (30-180 min horizons, Smart Persistence baseline):**

| Model | Avg Skill | Features | Comment |
|-------|-----------|----------|---------|
| **LSTM** | **50.2%** | 17 | 3-layer, 136 hidden, dropout 0.45 |
| ARIMAX | 44.9% | 6 | CSGHI baseline + exogenous |
| Linear | 4.8% | 77+ | Domain-engineered features |

All models use **Smart Persistence** as baseline: `P(t+h) = P(t) × CSGHI(t+h)/CSGHI(t)`

*Note: AR model excluded from comparison - univariate (no CSGHI access), uses different baseline*
