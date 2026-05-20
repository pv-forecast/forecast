"""Create LSTM daily prediction plots for different times of day."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
from data_management import DataManager

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "lstm" / "saved"

CAPACITY = 20808.66  # Wp

def load_model_and_make_predictions():
    """Load model, prepare data, make predictions."""
    from sklearn.preprocessing import MinMaxScaler
    from models.lstm.treat_nans import split_sequences

    # Load model
    model_files = list(MODELS_DIR.glob("LSTM_m2m*"))
    if not model_files:
        print("No LSTM model found")
        return None, None, None, None
    model = torch.load(model_files[0], weights_only=False)
    model.eval()

    # Prepare data
    feature_str = ["GHI", "BNI", "Ta", "El", "Az"]
    window_LSTM = 36
    seq_dim = 60

    feat = DataManager()
    train_X, test_X = feat.get_features_LSTM(window_LSTM, feature_str, include_future_el=False)
    train_Y, test_Y = feat.get_target_LSTM(window_LSTM)

    # Get timestamps before filtering
    test_timestamps = pd.to_datetime(test_Y['t'])

    # Filter to daytime (6-18)
    test_Y_copy = test_Y.copy()
    test_Y_copy['datetime'] = test_timestamps
    test_Y_copy['hour'] = test_Y_copy['datetime'].dt.hour
    daytime_mask = (test_Y_copy['hour'] >= 6) & (test_Y_copy['hour'] < 18)

    test_X = test_X.loc[test_Y_copy[daytime_mask].index]
    test_Y = test_Y.loc[test_Y_copy[daytime_mask].index]
    test_timestamps = test_timestamps.loc[test_Y_copy[daytime_mask].index].reset_index(drop=True)

    # Extract components
    test_y = test_Y[["Pdc_5min"]].fillna(0)
    Pdc_sp_test = test_Y[["Pdc_sp"]].fillna(0)
    test_X = test_X.fillna(0)

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_raw, _ = feat.get_features_LSTM(window_LSTM, feature_str, include_future_el=False)
    train_X_raw = train_X_raw.fillna(0).values
    scaler.fit(train_X_raw)
    test_X_np = scaler.transform(test_X.values)

    # Stack and split sequences
    testdata_stacked = np.hstack((test_X_np, test_y.values, Pdc_sp_test.values, test_Y[["ENI"]].fillna(0).values))
    X_test, Y_test, sp_test, _ = split_sequences(testdata_stacked, seq_dim, window_LSTM)

    # Convert to tensors
    X_test = torch.from_numpy(X_test).float()
    Y_test_tensor = torch.from_numpy(Y_test).float()

    # Make predictions
    with torch.no_grad():
        predictions = model(X_test).numpy()

    # Denormalize (Y_test and predictions are normalized 0-1)
    Y_actual = Y_test * CAPACITY
    Y_pred = predictions * CAPACITY
    # sp_test is already in Watts (not normalized)
    Y_sp = sp_test.flatten() * CAPACITY if sp_test.max() <= 1 else sp_test.flatten()

    return Y_actual, Y_pred, Y_sp, test_timestamps.values[:len(Y_actual)]


def plot_daily_predictions():
    """Create plots for morning, noon, afternoon, evening."""
    print("Loading model and making predictions...")
    Y_actual, Y_pred, Y_sp, timestamps = load_model_and_make_predictions()

    if Y_actual is None:
        return

    # Convert timestamps
    ts = pd.to_datetime(timestamps)
    hours = ts.hour
    months = ts.month

    # Time slots to plot
    time_slots = [
        (8, "Morgen (8:00)"),
        (12, "Mittag (12:00)"),
        (15, "Nachmittag (15:00)"),
        (17, "Abend (17:00)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, (hour, label) in enumerate(time_slots):
        # Find samples for this hour in summer with good power
        mask = (hours == hour) & (months >= 5) & (months <= 8)
        indices = np.where(mask)[0]

        # Filter for samples with meaningful actual power
        good_indices = [i for i in indices if Y_actual[i, 0] > 1000]  # >1kW

        if len(good_indices) == 0:
            print(f"No good samples for {label}")
            ax = axes[ax_idx]
            ax.text(0.5, 0.5, f"Keine Daten für {label}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Pick the first good sample
        idx = good_indices[0]

        # Get data for this sample
        actual = Y_actual[idx]
        pred = Y_pred[idx]
        sp = Y_sp[idx]

        # Create time axis
        start_time = ts[idx]
        time_axis = [start_time + timedelta(minutes=5*(i+1)) for i in range(36)]
        time_labels = [t.strftime('%H:%M') for t in time_axis]

        # Plot
        ax = axes[ax_idx]
        x_range = range(36)
        ax.plot(x_range, actual, 'b-', linewidth=2, label='Actual')
        ax.plot(x_range, pred, 'r--', linewidth=2, label='LSTM')
        ax.axhline(y=sp, color='gray', linestyle=':', linewidth=1.5, label=f'Smart Persistence ({sp:.0f}W)')

        ax.set_xticks(range(0, 36, 6))
        ax.set_xticklabels([time_labels[i] for i in range(0, 36, 6)], rotation=45)
        ax.set_xlabel('Zeit')
        ax.set_ylabel('Leistung [Watt]')
        ax.set_title(f'{label}\nStart: {start_time.strftime("%Y-%m-%d %H:%M")}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_file = FIGURES_DIR / "lstm_daily_predictions.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    plot_daily_predictions()
