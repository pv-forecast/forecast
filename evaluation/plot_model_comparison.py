"""Plot forecasts from all models for a single day - with proper alignment."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from sklearn.preprocessing import MinMaxScaler
from data_management import DataManager
import datetime

# Configuration
HORIZON_MIN = 60  # minutes ahead
HORIZON_IDX = 11  # 0-indexed: 60min = index 11

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FORECASTS_DIR = PROJECT_ROOT / "forecasts"
MODELS_DIR = PROJECT_ROOT / "models/lstm/saved"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Selected day
# 2020-04-26: Cloudy/variable day (high variability)
# 2020-05-28: Clear sky day (lowest variability)
selected_date = datetime.date(2020, 5, 28)

print("Loading data...")
feat = DataManager()

# ============ RECREATE LINEAR REGRESSION FILTERING ============
# This matches exactly what regression.py does
window_tar = 36
features = feat.get_features(window_ft=12, window_tar=window_tar, dropnight="true")
tar = feat.get_target_Pdc(window_tar=window_tar, dropnight="true")

test_x = features[features["dataset"] == "Test"]
test_y = tar[tar["dataset"] == "Test"].drop('dataset', axis=1)
test_merged = test_x.merge(test_y, on="t")

# Apply El >= 10 filter (same as regression.py line 54)
test_filtered = test_merged[test_merged["El_x"] >= 10].copy()
test_filtered = test_filtered.dropna()
test_filtered = test_filtered.reset_index(drop=True)

# Add datetime columns
test_filtered['datetime'] = pd.to_datetime(test_filtered['t'])
test_filtered['date'] = test_filtered['datetime'].dt.date
test_filtered['hour'] = test_filtered['datetime'].dt.hour

print(f"Test samples (El>=10 filter): {len(test_filtered)}")

# ============ LOAD LINEAR REGRESSION FORECASTS ============
print("\nLoading Linear Regression forecasts...")
lr_df = pd.read_hdf(FORECASTS_DIR / 'linear/forecasts_60min_BOTH.h5', key='df')
lr_test = lr_df[lr_df['dataset'] == 'Test'].reset_index(drop=True)

# Add timestamps from filtered test data (same order!)
lr_test['datetime'] = test_filtered['datetime'].values
lr_test['date'] = test_filtered['date'].values
lr_test['hour'] = test_filtered['hour'].values

print(f"Linear test samples: {len(lr_test)}")

# Filter to selected day (daytime only)
day_mask = (lr_test['date'] == selected_date) & \
           (lr_test['hour'] >= 6) & (lr_test['hour'] < 18)
lr_day = lr_test[day_mask].copy()

print(f"Selected day: {selected_date}")
print(f"Samples for this day: {len(lr_day)}")

if len(lr_day) == 0:
    print(f"ERROR: No data found for {selected_date}")
    sys.exit(1)

# Extract values
times = pd.to_datetime(lr_day['datetime']).values
actual = lr_day['Pdc_BOTH_actual'].values
smart_persistence = lr_day['Pdc_BOTH_sp'].values
lr_ols = lr_day['Pdc_BOTH_ols'].values

print(f"\nActual power range: {actual.min():.0f} - {actual.max():.0f} W")
print(f"Smart Persistence range: {smart_persistence.min():.0f} - {smart_persistence.max():.0f} W")
print(f"Linear OLS range: {lr_ols.min():.0f} - {lr_ols.max():.0f} W")

# ============ LSTM ============
print("\nLoading LSTM model and making predictions...")

model_path = MODELS_DIR / 'LSTM_m2m_Layer_2_Input_17_hidden_75_future_El'
model = torch.load(model_path, weights_only=False)
model.eval()

# Prepare LSTM data (uses hour 6-18 filter, different from Linear's El>=10)
feature_str = ["GHI", "BNI", "Ta", "El", "Az"]
window_LSTM = 36
seq_dim = 60

train_X, test_X_features = feat.get_features_LSTM(window_LSTM, feature_str,
                                                   include_future_el=True,
                                                   include_csghi_ratio=True)
train_Y, test_Y = feat.get_target_LSTM(window_LSTM)

# Scale features
train_X_np = train_X.fillna(0).values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_X_np)

# Filter to daytime
test_Y_copy = test_Y.copy()
test_Y_copy['datetime'] = pd.to_datetime(test_Y_copy['t'])
test_Y_copy['hour'] = test_Y_copy['datetime'].dt.hour
test_Y_copy['date'] = test_Y_copy['datetime'].dt.date
daytime_mask = (test_Y_copy['hour'] >= 6) & (test_Y_copy['hour'] < 18)

test_X_filtered = test_X_features[daytime_mask.values[:len(test_X_features)]].reset_index(drop=True)
test_Y_filtered = test_Y_copy[daytime_mask].reset_index(drop=True)

test_X_scaled = scaler.transform(test_X_filtered.fillna(0).values)

# Find day indices for LSTM
lstm_day_mask = test_Y_filtered['date'] == selected_date
lstm_day_indices = test_Y_filtered[lstm_day_mask].index.tolist()

print(f"  LSTM day samples: {len(lstm_day_indices)}")

# Make LSTM predictions
lstm_predictions = []
lstm_times = []

for idx in lstm_day_indices:
    if idx >= seq_dim - 1:
        seq_start = idx - seq_dim + 1
        X_seq = test_X_scaled[seq_start:idx + 1]

        if len(X_seq) == seq_dim:
            X_tensor = torch.from_numpy(X_seq).float().unsqueeze(0)
            with torch.no_grad():
                pred = model(X_tensor)
            pred_value = pred[0, HORIZON_IDX].item() * feat.CAPACITY
            lstm_predictions.append(pred_value)
            lstm_times.append(test_Y_filtered.iloc[idx]['datetime'])

print(f"  LSTM predictions: {len(lstm_predictions)}")
if len(lstm_predictions) > 0:
    print(f"  LSTM range: {min(lstm_predictions):.0f} - {max(lstm_predictions):.0f} W")

# ============ ARIMAX ============
print("\nLoading ARIMAX forecasts...")

# Load ARIMAX predictions
arimax_file = FORECASTS_DIR / 'timeseries/arima_forecasts.npz'
if arimax_file.exists():
    arimax_data = np.load(arimax_file, allow_pickle=True)
    arimax_pred = arimax_data[f'{HORIZON_MIN}min_predicted']
    arimax_actual = arimax_data[f'{HORIZON_MIN}min_actual']

    # Use saved timestamps if available, otherwise recreate
    if 'timestamps' in arimax_data.files:
        arimax_timestamps = pd.to_datetime(arimax_data['timestamps'])
    else:
        # Fallback: recreate timestamps (for old files with sample_step=50)
        test_timestamps = pd.to_datetime(feat.test['t'])
        test_hours = test_timestamps.dt.hour
        arimax_daytime_mask = (test_hours >= 6) & (test_hours < 18)
        daytime_timestamps = test_timestamps[arimax_daytime_mask].reset_index(drop=True)
        sample_step = len(daytime_timestamps) // len(arimax_pred)  # Auto-detect
        max_h = 36
        test_end = len(daytime_timestamps) - max_h
        test_indices = list(range(0, test_end, sample_step))
        arimax_timestamps = daytime_timestamps.iloc[test_indices].reset_index(drop=True)

    arimax_dates = arimax_timestamps.date

    # Filter to selected day
    arimax_day_mask = arimax_dates == selected_date
    arimax_day_times = arimax_timestamps[arimax_day_mask].values
    arimax_day_pred = arimax_pred[arimax_day_mask]
    arimax_day_actual = arimax_actual[arimax_day_mask]

    print(f"  ARIMAX samples for this day: {len(arimax_day_pred)}")
    if len(arimax_day_pred) > 0:
        print(f"  ARIMAX range: {arimax_day_pred.min():.0f} - {arimax_day_pred.max():.0f} W")
else:
    print("  ARIMAX file not found")
    arimax_day_times = []
    arimax_day_pred = []

# ============ PLOT ============
print("\nCreating plot...")
fig, ax = plt.subplots(figsize=(14, 7))

# Plot from Linear Regression data (properly aligned)
ax.plot(times, actual, 'k-', linewidth=2.5, label='Actual', zorder=5)
ax.plot(times, smart_persistence, 'r--', linewidth=2, label='Smart Persistence', alpha=0.8, zorder=4)
ax.plot(times, lr_ols, 'b-', linewidth=1.8, label='Linear (OLS)', alpha=0.9, zorder=3)

# Plot LSTM
if len(lstm_predictions) > 0:
    ax.plot(lstm_times, lstm_predictions, color='green', linewidth=1.8, label='LSTM', alpha=0.9, zorder=3)

# Plot ARIMAX
if len(arimax_day_pred) > 0:
    ax.plot(arimax_day_times, arimax_day_pred, color='purple', linewidth=1.8,
            label='ARIMAX', alpha=0.9, zorder=3)

# Labels and formatting
ax.set_xlabel('Time of Day', fontsize=12)
ax.set_ylabel('Power [W]', fontsize=12)
ax.set_title(f'PV Power Forecasts - {selected_date} ({HORIZON_MIN}-min Horizon)', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

plt.tight_layout()
output_name = f'model_comparison_{selected_date}.png'
plt.savefig(FIGURES_DIR / output_name, dpi=150, bbox_inches='tight')
print(f"\nSaved to {FIGURES_DIR / output_name}")
plt.close()
