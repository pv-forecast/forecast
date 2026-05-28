"""Plot forecasts from all models for a single day."""

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
HORIZON = 60  # minutes ahead (12 steps * 5min)
HORIZON_IDX = 11  # 0-indexed: 60min = index 11 (12th output)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FORECASTS_DIR = PROJECT_ROOT / "forecasts"
MODELS_DIR = PROJECT_ROOT / "models/lstm/saved"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

print("Loading data...")
feat = DataManager()
test_data = feat.test.copy()
test_data['datetime'] = pd.to_datetime(test_data['t'])
test_data['date'] = test_data['datetime'].dt.date
test_data = test_data.reset_index(drop=True)

# Selected day with valid data
selected_date = datetime.date(2020, 4, 26)
print(f"Selected day: {selected_date}")

# ============ LINEAR REGRESSION ============
print("\nLoading Linear Regression forecasts...")
lr_df = pd.read_hdf(FORECASTS_DIR / 'linear/forecasts_60min_BOTH.h5', key='df')
lr_test = lr_df[lr_df['dataset'] == 'Test'].reset_index(drop=True)

# Find alignment
lr_actual = lr_test['Pdc_BOTH_actual'].values
test_pdc = test_data['Pdc_33'].values
lr_start_idx = None
for i in range(len(test_pdc) - 10):
    if np.allclose(test_pdc[i:i+10], lr_actual[:10], rtol=0.001, equal_nan=True):
        lr_start_idx = i
        break

lr_test['datetime'] = test_data.iloc[lr_start_idx:lr_start_idx+len(lr_test)]['datetime'].values
lr_test['date'] = pd.to_datetime(lr_test['datetime']).dt.date
lr_day = lr_test[lr_test['date'] == selected_date].copy()
lr_day = lr_day[(pd.to_datetime(lr_day['datetime']).dt.hour >= 6) &
                (pd.to_datetime(lr_day['datetime']).dt.hour < 18)]
print(f"  Linear samples: {len(lr_day)}")

# ============ AR ============
print("Loading AR forecasts...")
ar_data = np.load(FORECASTS_DIR / 'timeseries/ar_forecasts.npz')
ar_actual = ar_data['60min_actual']
ar_pred = ar_data['60min_predicted']

ar_start_idx = len(test_data) - len(ar_actual)
ar_df = pd.DataFrame({
    'datetime': test_data.iloc[ar_start_idx:ar_start_idx+len(ar_actual)]['datetime'].values,
    'actual': ar_actual,
    'predicted': ar_pred
})
ar_df['date'] = pd.to_datetime(ar_df['datetime']).dt.date
ar_day = ar_df[ar_df['date'] == selected_date].copy()
ar_day = ar_day[(pd.to_datetime(ar_day['datetime']).dt.hour >= 6) &
                (pd.to_datetime(ar_day['datetime']).dt.hour < 18)]
print(f"  AR samples: {len(ar_day)}")

# ============ LSTM ============
print("Loading LSTM model and making predictions...")

# Load saved model
model_path = MODELS_DIR / 'LSTM_m2m_Layer_2_Input_17_hidden_75_future_El'
model = torch.load(model_path, weights_only=False)
model.eval()

# Prepare features for LSTM (same as training)
feature_str = ["GHI", "BNI", "Ta", "El", "Az"]
window_LSTM = 36
seq_dim = 60

# Get features with CSGHI ratio
train_X, test_X_features = feat.get_features_LSTM(window_LSTM, feature_str,
                                                   include_future_el=True,
                                                   include_csghi_ratio=True)

# Get targets
train_Y, test_Y = feat.get_target_LSTM(window_LSTM)

# Filter to daytime (same as training)
test_Y_copy = test_Y.copy()
test_Y_copy['datetime'] = pd.to_datetime(test_Y_copy['t'])
test_Y_copy['hour'] = test_Y_copy['datetime'].dt.hour
daytime_mask = (test_Y_copy['hour'] >= 6) & (test_Y_copy['hour'] < 18)

# IMPORTANT: Reset index BEFORE converting to numpy to keep alignment
test_X_filtered = test_X_features[daytime_mask.values[:len(test_X_features)]].reset_index(drop=True)
test_Y_filtered = test_Y_copy[daytime_mask].reset_index(drop=True)

# Prepare data
test_X_np = test_X_filtered.fillna(0).values

# Scale features (fit on training data)
train_X_np = train_X.fillna(0).values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_X_np)
test_X_scaled = scaler.transform(test_X_np)

# Find indices for selected day (now indices are sequential 0, 1, 2, ...)
test_Y_filtered['date'] = test_Y_filtered['datetime'].dt.date
day_mask = test_Y_filtered['date'] == selected_date
day_indices = test_Y_filtered[day_mask].index.tolist()

print(f"  LSTM day indices: {len(day_indices)} samples")

# Make predictions for the day
lstm_predictions = []
lstm_times = []
lstm_actual = []
lstm_sp = []

if len(day_indices) > 0:
    for idx in day_indices:
        if idx >= seq_dim - 1:  # Need seq_dim timesteps of history
            # Get sequence
            seq_start = idx - seq_dim + 1
            seq_end = idx + 1
            X_seq = test_X_scaled[seq_start:seq_end]

            if len(X_seq) == seq_dim:
                X_tensor = torch.from_numpy(X_seq).float().unsqueeze(0)

                with torch.no_grad():
                    pred = model(X_tensor)

                # Get 60-min horizon prediction (index 11)
                pred_60min = pred[0, HORIZON_IDX].item() * feat.CAPACITY
                lstm_predictions.append(pred_60min)
                lstm_times.append(test_Y_filtered.iloc[idx]['datetime'])

                # Get actual and SP for this horizon
                actual_col = f'Pdc_{HORIZON}min'
                sp_col = f'Pdc_sp_{HORIZON}min'
                if actual_col in test_Y_filtered.columns:
                    lstm_actual.append(test_Y_filtered.iloc[idx][actual_col] * feat.CAPACITY)
                if sp_col in test_Y_filtered.columns:
                    lstm_sp.append(test_Y_filtered.iloc[idx][sp_col])

print(f"  LSTM predictions: {len(lstm_predictions)}")

# ============ PLOT ============
print("\nCreating plot...")
fig, ax = plt.subplots(figsize=(14, 7))

# Plot Linear Regression data
if len(lr_day) > 0:
    times_lr = pd.to_datetime(lr_day['datetime']).values
    ax.plot(times_lr, lr_day['Pdc_BOTH_actual'].values, 'k-', linewidth=2.5, label='Actual', zorder=5)
    ax.plot(times_lr, lr_day['Pdc_BOTH_sp'].values, 'r--', linewidth=2, label='Smart Persistence', alpha=0.8, zorder=4)
    ax.plot(times_lr, lr_day['Pdc_BOTH_ols'].values, 'b-', linewidth=1.8, label='Linear (OLS)', alpha=0.9, zorder=3)

# Plot AR
if len(ar_day) > 0:
    times_ar = pd.to_datetime(ar_day['datetime']).values
    ax.plot(times_ar, ar_day['predicted'].values, color='orange', linewidth=1.8, label='AR', alpha=0.9, zorder=2)

# Plot LSTM
if len(lstm_predictions) > 0:
    ax.plot(lstm_times, lstm_predictions, color='green', linewidth=1.8, label='LSTM', alpha=0.9, zorder=3)

# Labels and formatting
ax.set_xlabel('Time of Day', fontsize=12)
ax.set_ylabel('Power [W]', fontsize=12)
ax.set_title(f'PV Power Forecasts - {selected_date} ({HORIZON}-min Horizon)', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_comparison_day.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to {FIGURES_DIR / 'model_comparison_day.png'}")
plt.close()
