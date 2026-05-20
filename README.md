# PV Power Forecasting

Comparing traditional ML models (Linear Regression, Ridge, Lasso) with LSTM neural networks for photovoltaic power forecasting.

## Project Structure

```
forecast/
├── Daten/                        # Data (private repo, see setup below)
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
│       ├── ar.py                 # AutoRegression (1000 lags)
│       └── arma.py               # ARIMA(30,1,0)
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

### Time Series

```bash
python models/timeseries/ar.py       # AutoRegression
python models/timeseries/arma.py     # ARIMA
```

## Data

The data is stored in a separate private repository: [pv-forecast/data](https://github.com/pv-forecast/data)

**Setup for collaborators with data access:**

```bash
# Clone this repo
git clone git@github.com:pv-forecast/forecast.git
cd forecast

# Clone data repo into Daten folder
git clone git@github.com:pv-forecast/data.git Daten
```

**Data contents:**
- Source: `Daten/PVAMM_201911-202011_PT5M_merged.csv`
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

**Definition:** Smart Persistence scales the current power by the expected change in solar irradiance.

```
P_predicted(t+h) = P_actual(t) * (ENI(t+h) / ENI(t))
```

Where:
- `P_actual(t)` = current measured power
- `ENI(t+h)` = extraterrestrial normal irradiance at future time (calculated from solar geometry)
- `ENI(t)` = extraterrestrial normal irradiance at current time

**Why this baseline?**
- Better than naive persistence (`P(t+h) = P(t)`) because it accounts for sun movement
- Physically meaningful for solar forecasting
- Same baseline used for ALL models (fair comparison)

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
| **Total**                      |                                                                                                                               | **77**|

### LSTM (5 features)

| Category         | Features    | Count |
|------------------|-------------|-------|
| Raw irradiance   | GHI, BNI    | 2     |
| Weather          | Ta          | 1     |
| Solar geometry   | El, Az      | 2     |
| **Total**        |             | **5** |

### AR (1 feature)

| Category           | Features | Count |
|--------------------|----------|-------|
| Power (1000 lags)  | Pdc      | 1     |
| **Total**          |          | **1** |

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
- AR uses only historical power - no weather/irradiance data
- ARIMAX uses exogenous variables with **persistence assumption**: X(t+h) ≈ X(t)
- For known future variables (CSGHI, El, ENI), actual future values are used instead of persistence

## Model Architecture Details

| Aspect             | Linear Regression         | LSTM                         | AR           | ARIMAX         |
|--------------------|---------------------------|------------------------------|--------------|----------------|
| **File**           | `regression.py`           | `lstm_manytomany.py`         | `ar.py`      | `arma.py`      |
| **Library**        | scikit-learn              | PyTorch                      | statsmodels  | statsmodels    |
| **Variants**       | OLS, Ridge (L2), Lasso (L1) | Many-to-Many               | AR(1000)     | ARIMAX(12,1,1) |
| **Hidden units**   | -                         | 75                           | -            | -              |
| **Layers**         | 1                         | 2 LSTM + dropout             | -            | -              |
| **Normalization**  | StandardScaler            | MinMaxScaler                 | None         | None           |
| **Regularization** | Ridge: L2, Lasso: L1      | Dropout 0.5                  | -            | -              |
| **Training**       | Closed-form / CV          | Early stopping, LR scheduler | Closed-form  | Closed-form    |

## Key Differences

| Aspect                       | Linear | LSTM           | AR             | ARIMAX             |
|------------------------------|--------|----------------|----------------|--------------------|
| Uses measured irradiance?    | Yes    | Yes            | No             | Yes (persistence)  |
| Uses engineered features?    | Yes    | No             | No             | No                 |
| Learns from sequence?        | No     | Yes (60 steps) | Yes (1000 lags)| Yes (12 lags)      |
| Needs future X values?       | No     | No             | No             | Yes (persistence)  |
| Captures non-linearity?      | No     | Yes            | No             | No                 |
