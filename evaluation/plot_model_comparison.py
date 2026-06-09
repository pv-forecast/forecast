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

# Selected days
# 2020-04-26: Cloudy/variable day (high variability)
# 2020-05-28: Clear sky day (lowest variability)
selected_dates = [
    (datetime.date(2020, 5, 28), "sunny"),
    (datetime.date(2020, 4, 26), "cloudy"),
]

print("Loading data...")
feat = DataManager()

# ============ RECREATE LINEAR REGRESSION FILTERING ============
window_tar = 36
features = feat.get_features(window_ft=12, window_tar=window_tar, dropnight="true")
tar = feat.get_target_Pdc(window_tar=window_tar, dropnight="true")

test_x = features[features["dataset"] == "Test"]
test_y = tar[tar["dataset"] == "Test"].drop('dataset', axis=1)
test_merged = test_x.merge(test_y, on="t")

test_filtered = test_merged[test_merged["El_x"] >= 10].copy()
test_filtered = test_filtered.dropna()
test_filtered = test_filtered.reset_index(drop=True)

test_filtered['datetime'] = pd.to_datetime(test_filtered['t'])
test_filtered['date'] = test_filtered['datetime'].dt.date
test_filtered['hour'] = test_filtered['datetime'].dt.hour

# ============ LOAD LINEAR REGRESSION FORECASTS ============
print("Loading Linear Regression forecasts...")
lr_df = pd.read_hdf(FORECASTS_DIR / 'linear/forecasts_60min_BOTH.h5', key='df')
lr_test = lr_df[lr_df['dataset'] == 'Test'].reset_index(drop=True)
lr_test['datetime'] = test_filtered['datetime'].values
lr_test['date'] = test_filtered['date'].values
lr_test['hour'] = test_filtered['hour'].values

# ============ LSTM ============
print("Loading LSTM model...")
model_path = MODELS_DIR / 'LSTM_m2m_Layer_2_Input_17_hidden_75_future_El'
model = torch.load(model_path, weights_only=False)
model.eval()

feature_str = ["GHI", "BNI", "Ta", "El", "Az"]
window_LSTM = 36
seq_dim = 60

train_X, test_X_features = feat.get_features_LSTM(window_LSTM, feature_str,
                                                   include_future_el=True,
                                                   include_csghi_ratio=True)
train_Y, test_Y = feat.get_target_LSTM(window_LSTM)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_X.fillna(0).values)

test_Y_copy = test_Y.copy()
test_Y_copy['datetime'] = pd.to_datetime(test_Y_copy['t'])
test_Y_copy['hour'] = test_Y_copy['datetime'].dt.hour
test_Y_copy['date'] = test_Y_copy['datetime'].dt.date
daytime_mask = (test_Y_copy['hour'] >= 6) & (test_Y_copy['hour'] < 18)

test_X_filtered = test_X_features[daytime_mask.values[:len(test_X_features)]].reset_index(drop=True)
test_Y_filtered = test_Y_copy[daytime_mask].reset_index(drop=True)
test_X_scaled = scaler.transform(test_X_filtered.fillna(0).values)

# ============ ARIMAX ============
print("Loading ARIMAX forecasts...")
arimax_file = FORECASTS_DIR / 'timeseries/arima_forecasts.npz'
arimax_available = arimax_file.exists()

if arimax_available:
    arimax_data = np.load(arimax_file, allow_pickle=True)
    arimax_pred_all = arimax_data[f'{HORIZON_MIN}min_predicted']

    if 'timestamps' in arimax_data.files:
        arimax_timestamps = pd.to_datetime(arimax_data['timestamps'])
    else:
        test_timestamps = pd.to_datetime(feat.test['t'])
        test_hours = test_timestamps.dt.hour
        arimax_daytime_mask = (test_hours >= 6) & (test_hours < 18)
        daytime_timestamps = test_timestamps[arimax_daytime_mask].reset_index(drop=True)
        sample_step = len(daytime_timestamps) // len(arimax_pred_all)
        test_end = len(daytime_timestamps) - 36
        test_indices = list(range(0, test_end, sample_step))
        arimax_timestamps = daytime_timestamps.iloc[test_indices].reset_index(drop=True)

    arimax_dates = arimax_timestamps.date

# ============ GENERATE PLOTS FOR EACH DAY ============
for selected_date, day_type in selected_dates:
    print(f"\n{'='*50}")
    print(f"Plotting {selected_date} ({day_type})")
    print('='*50)

    # Linear data for this day
    day_mask = (lr_test['date'] == selected_date) & (lr_test['hour'] >= 6) & (lr_test['hour'] < 18)
    lr_day = lr_test[day_mask].copy()

    if len(lr_day) == 0:
        print(f"No data for {selected_date}, skipping...")
        continue

    times = pd.to_datetime(lr_day['datetime']).values
    actual = lr_day['Pdc_BOTH_actual'].values
    smart_persistence = lr_day['Pdc_BOTH_sp'].values
    lr_ols = lr_day['Pdc_BOTH_ols'].values

    # LSTM predictions
    lstm_day_mask = test_Y_filtered['date'] == selected_date
    lstm_day_indices = test_Y_filtered[lstm_day_mask].index.tolist()

    lstm_predictions, lstm_times = [], []
    for idx in lstm_day_indices:
        if idx >= seq_dim - 1:
            X_seq = test_X_scaled[idx - seq_dim + 1:idx + 1]
            if len(X_seq) == seq_dim:
                X_tensor = torch.from_numpy(X_seq).float().unsqueeze(0)
                with torch.no_grad():
                    pred = model(X_tensor)
                lstm_predictions.append(pred[0, HORIZON_IDX].item() * feat.CAPACITY)
                lstm_times.append(test_Y_filtered.iloc[idx]['datetime'])

    # ARIMAX predictions
    if arimax_available:
        arimax_day_mask = arimax_dates == selected_date
        arimax_day_times = arimax_timestamps[arimax_day_mask].values
        arimax_day_pred = arimax_pred_all[arimax_day_mask]
    else:
        arimax_day_times, arimax_day_pred = [], []

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(times, actual, 'k-', linewidth=2.5, label='Actual', zorder=5)
    ax.plot(times, smart_persistence, 'r--', linewidth=2, label='Smart Persistence', alpha=0.8)
    ax.plot(times, lr_ols, 'b-', linewidth=1.8, label='Linear (OLS)', alpha=0.9)

    if lstm_predictions:
        ax.plot(lstm_times, lstm_predictions, 'g-', linewidth=1.8, label='LSTM', alpha=0.9)

    if len(arimax_day_pred) > 0:
        ax.plot(arimax_day_times, arimax_day_pred, color='purple', linewidth=1.8, label='ARIMAX', alpha=0.9)

    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Power [W]', fontsize=12)
    ax.set_title(f'PV Power Forecasts - {selected_date} ({day_type.capitalize()} Day, {HORIZON_MIN}-min Horizon)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    output_name = f'model_comparison_{day_type}.png'
    plt.savefig(FIGURES_DIR / output_name, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_name}")
    plt.close()

print("\nDone!")
