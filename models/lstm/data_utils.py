"""
Data utilities for LSTM ablation study.

Extracted from lstm_manytomany.py for reusability across experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from data_management import DataManager
from models.lstm.treat_nans import split_sequences


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_daytime(X: pd.DataFrame, Y: pd.DataFrame,
                   start_hour: int = 6, end_hour: int = 18):
    """
    Filter data to daytime hours only.

    Args:
        X: Feature DataFrame
        Y: Target DataFrame (must contain 't' column with timestamps)
        start_hour: Start of daytime (inclusive)
        end_hour: End of daytime (exclusive)

    Returns:
        X_filtered, Y_filtered: Filtered DataFrames
    """
    Y = Y.copy()
    Y['datetime'] = pd.to_datetime(Y['t'])
    Y['hour'] = Y['datetime'].dt.hour
    daytime_mask = (Y['hour'] >= start_hour) & (Y['hour'] < end_hour)
    Y_filtered = Y[daytime_mask].drop(columns=['datetime', 'hour'])
    X_filtered = X.loc[Y_filtered.index]
    print(f"  Daytime filter: {len(Y)} -> {len(Y_filtered)} samples ({100*len(Y_filtered)/len(Y):.1f}%)")
    return X_filtered, Y_filtered


def load_and_prepare_data(window_LSTM: int = 36, seq_dim: int = 60):
    """
    Load and prepare data for LSTM training.

    Args:
        window_LSTM: Output horizon (36 = 5-180 min in 5-min steps)
        seq_dim: Input sequence length (60 timesteps = 5h of history)

    Returns:
        dict with keys:
            - X_train, X_test: Input tensors (batch, seq_dim, features)
            - y_train, y_test: Target tensors (batch, window_LSTM)
            - sp_train, sp_test: Smart Persistence tensors (batch, window_LSTM)
            - ENI_train, ENI_test: ENI tensors (batch, window_LSTM)
            - n_features: Number of input features
            - CAPACITY: PV system capacity in Wp
    """
    feature_str = ["GHI", "BNI", "Ta", "El", "Az"]

    print("Loading data...")
    feat = DataManager()

    # Get features with future elevation and CSGHI ratio
    train_X, test_X = feat.get_features_LSTM(
        window_LSTM, feature_str,
        include_future_el=True,
        include_csghi_ratio=True
    )
    train_Y, test_Y = feat.get_target_LSTM(window_LSTM)

    # Apply daytime filter
    print("Applying daytime filter (6:00-18:00)...")
    train_X, train_Y = filter_daytime(train_X, train_Y)
    test_X, test_Y = filter_daytime(test_X, test_Y)

    # Extract target power and Smart Persistence
    train_y = train_Y[["Pdc_5min"]]
    test_y = test_Y[["Pdc_5min"]]
    train_ENI = train_Y[["ENI"]]
    test_ENI = test_Y[["ENI"]]

    # Smart Persistence columns per horizon (in Watts)
    sp_cols = [f"Pdc_sp_{h*5}min" for h in range(1, window_LSTM + 1)]
    Pdc_sp_train = train_Y[sp_cols]
    Pdc_sp_test = test_Y[sp_cols]

    # Fill NaN values
    train_X = train_X.fillna(value=0)
    test_X = test_X.fillna(value=0)
    train_y = train_y.fillna(value=0)

    # Convert to numpy
    train_X = train_X.values
    test_X = test_X.values
    train_y = train_y.values
    test_y = test_y.values

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # Stack and split sequences
    traindata_stacked = np.hstack((train_X, train_y, Pdc_sp_train, train_ENI))
    testdata_stacked = np.hstack((test_X, test_y, Pdc_sp_test, test_ENI))

    X, Y, train_sp, train_ENI = split_sequences(
        traindata_stacked, seq_dim, window_LSTM, n_sp_cols=window_LSTM
    )
    test_X, Y_test, test_sp, test_ENI = split_sequences(
        testdata_stacked, seq_dim, window_LSTM, n_sp_cols=window_LSTM
    )

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X).float()
    X_test = torch.from_numpy(test_X).float()
    y_train = torch.from_numpy(Y).float()
    y_test = torch.from_numpy(Y_test).float()
    sp_train = torch.from_numpy(train_sp).float()
    sp_test = torch.from_numpy(test_sp).float()
    ENI_train = torch.from_numpy(train_ENI).float()
    ENI_test = torch.from_numpy(test_ENI).float()

    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[2]}, Sequence length: {seq_dim}, Output horizon: {window_LSTM}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sp_train': sp_train,
        'sp_test': sp_test,
        'ENI_train': ENI_train,
        'ENI_test': ENI_test,
        'n_features': X_train.shape[2],
        'CAPACITY': feat.CAPACITY,
    }
