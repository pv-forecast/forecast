import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from models.lstm import lstm_model
from models.lstm import treat_nans
from models.lstm.treat_nans import IndicatorNaN, split_sequences
from data_management import DataManager

# Reproducibility: Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Output directories
METRICS_DIR = "metrics/lstm"
FORECASTS_DIR = "forecasts/lstm"
FIGURES_DIR = "figures"
MODELS_DIR = "models/lstm/saved"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FORECASTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# BEFORE RUN:
# check window_tar of DataManagement.py, window_tar has to be 36 for Ouput 36
# check if seasonal or chronological Dataset is choosen
# check Model parameters below (layer, Inputs, model_load, hidden, treatnans, batch size, epochs, optimizer, ...)

# Trainings- /Test Set

feature_str = ["GHI", "BNI", "Ta", "El", "Az"]
window_LSTM = 36
seq_dim = 60
feat = DataManager()
# include_future_el=True and include_csghi_ratio=True for fair comparison with Linear Regression
train_X, test_X = feat.get_features_LSTM(window_LSTM, feature_str, include_future_el=True, include_csghi_ratio=True)
train_Y, test_Y = feat.get_target_LSTM(window_LSTM)

# Filter to daytime only (6:00-18:00) to avoid night-time bias
# This ensures forecasts up to 3h ahead still have meaningful power values
def filter_daytime(X, Y, start_hour=6, end_hour=18):
    """Filter data to daytime hours only."""
    Y = Y.copy()
    Y['datetime'] = pd.to_datetime(Y['t'])
    Y['hour'] = Y['datetime'].dt.hour
    daytime_mask = (Y['hour'] >= start_hour) & (Y['hour'] < end_hour)
    Y_filtered = Y[daytime_mask].drop(columns=['datetime', 'hour'])
    X_filtered = X.loc[Y_filtered.index]
    print(f"  Daytime filter: {len(Y)} -> {len(Y_filtered)} samples ({100*len(Y_filtered)/len(Y):.1f}%)")
    return X_filtered, Y_filtered

print("Applying daytime filter (6:00-18:00)...")
train_X, train_Y = filter_daytime(train_X, train_Y)
test_X, test_Y = filter_daytime(test_X, test_Y)

# Extract target power (first horizon for training loss) and Smart Persistence per horizon
train_y = train_Y[["Pdc_5min"]]
test_y = test_Y[["Pdc_5min"]]
train_ENI = train_Y[["ENI"]]
test_ENI = test_Y[["ENI"]]

# Smart Persistence columns per horizon (in Watts) - for fair comparison with CSGHI-based baseline
sp_cols = [f"Pdc_sp_{h*5}min" for h in range(1, window_LSTM + 1)]
Pdc_sp_train = train_Y[sp_cols]  # Shape: (n_samples, 36)
Pdc_sp_test = test_Y[sp_cols]    # Shape: (n_samples, 36)

# nan values
train_X = train_X.fillna(value=0) # train_X = train_X.fillna(value=0), train_X = train_X.fillna(value=-100000)
test_X = test_X.fillna(value=0)
train_y = train_y.fillna(value=0)

# Indicator on missing values
"""train_X, test_X, train_Y = IndicatorNaN(train_X, test_X, train_Y)"""

# numpy.ndarray
train_X = train_X.values
test_X = test_X.values
train_y = train_y.values
test_y = test_y.values

# Scaler !oder MinMaxScaler: auch y gescaled!
"""scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)"""

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

"""train_ENI = train_ENI.fillna(method="ffill")
test_ENI = test_ENI.fillna(method="ffill")"""
traindata_stacked = np.hstack((train_X, train_y, Pdc_sp_train, train_ENI))
testdata_stacked = np.hstack((test_X, test_y, Pdc_sp_test, test_ENI))
X, Y, train_sp, train_ENI = split_sequences(traindata_stacked, seq_dim, window_LSTM, n_sp_cols=window_LSTM)
test_X, Y_test, test_sp, test_ENI = split_sequences(testdata_stacked, seq_dim, window_LSTM, n_sp_cols=window_LSTM)

# to torch
X_train = torch.from_numpy(X).float()
X_test = torch.from_numpy(test_X).float()
y_train = torch.from_numpy(Y).float()
y_test = torch.from_numpy(Y_test).float()
sp_train = torch.from_numpy(train_sp).float()
sp_test = torch.from_numpy(test_sp).float()
ENI_train = torch.from_numpy(train_ENI).float()
ENI_test = torch.from_numpy(test_ENI).float()

def initializeNewModel(input_dim, hidden_dim, layer_dim, output_dim):

    # Initializing LSTM
    # input_dim = number of features
    # hidden_dim = number of hidden layer
    # layer_dim = number of stacked LSTM's
    # output_dim = output horizon

    model = lstm_model.LSTM(input_dim, hidden_dim, layer_dim, output_dim)

    return model

def trainModel(model, batch_size, seq_dim, epochs, patience=7, initial_lr=1e-3):
    """
    Train LSTM model with learning rate scheduler and early stopping.

    Args:
        patience: Number of epochs without improvement before early stopping
        initial_lr: Starting learning rate (use lower value for fine-tuning)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    train_loss = []
    metric = pd.DataFrame()
    result = pd.DataFrame()

    # Store best results
    best_rmse = best_mae = best_mbe = best_rmse_sp = None
    best_p_real = best_p_pred = best_p_sp = None

    print(f"Starting training: {epochs} epochs, {int(len(X_train)/batch_size)} steps per epoch")

    for epoch in range(epochs):
        model.train()
        epoch_loss = []

        # Training loop
        num_steps = int(len(X_train)/batch_size)
        for step in range(0, num_steps):
            if step == 0:
                print(f"Epoch {epoch} started...")
            train_load = X_train[step * batch_size:batch_size * (step+1)].view(-1, seq_dim, X_train.shape[2])
            y = y_train[step * batch_size:(step + 1) * batch_size]

            optimizer.zero_grad()
            y_pred = model(train_load)
            RMSE = model.loss(y_pred, y)

            epoch_loss.append(RMSE.item())
            train_loss.append(RMSE.data)

            RMSE.backward()
            optimizer.step()

        # Validation at end of each epoch
        model.eval()
        batch_test = 200
        test_len = int(len(X_test)/batch_test)

        rmse, rmse_sp, mae, mbe = [], [], [], []
        p_sp, p_pred, p_real = [], [], []

        with torch.no_grad():
            for testrun in range(test_len):
                test_load = X_test[testrun * batch_test:(testrun + 1) * batch_test, :].view(-1, seq_dim, X_test.shape[2])
                y = y_test[testrun * batch_test:(testrun + 1) * batch_test]
                y_sp = sp_test[testrun * batch_test:(testrun + 1) * batch_test]

                y_pred = model(test_load)

                # Denormalize
                test_pred = y_pred * feat.CAPACITY
                observ = y * feat.CAPACITY

                # Plot prediction (once per epoch) - in Watts
                if testrun == 120:
                    P = y.transpose(0,1) * feat.CAPACITY  # Denormalize to Watts
                    P_pred = y_pred.transpose(0, 1) * feat.CAPACITY
                    plt.figure(figsize=(12, 5))
                    plt.plot(P[2].numpy(), label='Actual', linewidth=2)
                    plt.plot(P_pred[2].numpy(), label='Predicted', linewidth=2, alpha=0.8)
                    plt.xlabel('Forecast Horizon [5min steps]')
                    plt.ylabel('Power [Watt]')
                    plt.legend()
                    plt.title(f'LSTM Prediction - Epoch {epoch}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(PATH_fig, dpi=150)
                    plt.close()

                p_sp.append(y_sp.numpy())
                p_pred.append(y_pred.numpy())
                p_real.append(y.numpy())

                # Compute metrics
                error = observ.numpy() - test_pred.numpy()
                error_sp = observ.numpy() - y_sp.numpy()

                rmse_sp.append(np.sqrt(np.nanmean(error_sp ** 2, axis=0)))
                rmse.append(np.sqrt(np.nanmean(error ** 2, axis=0)))
                mae.append(np.nanmean(np.abs(error), axis=0))
                mbe.append(np.nanmean(error, axis=0))

        # Epoch metrics
        avg_train_rmse = np.mean(epoch_loss) * feat.CAPACITY  # Denormalize for comparison
        avg_test_rmse = np.nanmean(rmse)
        avg_rmse_sp = np.nanmean(rmse_sp)
        avg_mae = np.nanmean(mae)
        avg_mbe = np.nanmean(mbe)
        skill = (1 - avg_test_rmse / avg_rmse_sp) * 100 if avg_rmse_sp > 0 else 0

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}: Train_RMSE={avg_train_rmse:.0f}, Test_RMSE={avg_test_rmse:.0f}, '
              f'RMSE_sp={avg_rmse_sp:.0f}, Skill={skill:.1f}%, MAE={avg_mae:.0f}, MBE={avg_mbe:.0f}, '
              f'LR={current_lr:.2e}')

        # Learning rate scheduler
        scheduler.step(avg_test_rmse)

        # Early stopping
        if avg_test_rmse < best_val_loss:
            best_val_loss = avg_test_rmse
            epochs_without_improvement = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_rmse, best_mae, best_mbe, best_rmse_sp = rmse, mae, mbe, rmse_sp
            best_p_real, best_p_pred, best_p_sp = p_real, p_pred, p_sp
            print(f'  -> New best model! Test_RMSE: {best_val_loss:.0f}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'\nEarly stopping after {epoch + 1} epochs')
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        rmse, mae, mbe, rmse_sp = best_rmse, best_mae, best_mbe, best_rmse_sp
        p_real, p_pred, p_sp = best_p_real, best_p_pred, best_p_sp

    # Save results
    metric.insert(metric.shape[1], "MAE", value=mae)
    metric.insert(metric.shape[1], "MBE", value=mbe)
    metric.insert(metric.shape[1], "RMSE", value=rmse)
    metric.insert(metric.shape[1], "RMSE_sp", value=rmse_sp)
    result.insert(result.shape[1], "P_real", value=p_real)
    result.insert(result.shape[1], "P_pred", value=p_pred)
    result.insert(result.shape[1], "P_sp", value=p_sp)
    metric.to_csv(PATH_save_met)
    result.to_csv(PATH_save_res)
    torch.save(model, PATH_save)

    print(f'\nBest Test_RMSE: {best_val_loss:.0f}')
    return metric, result

# START
# define which Model to load or name Model to be initialized (layer = ...)

batch_size = 128  # Larger batch for more stable gradients
layer = 2
hidden = 75
epochs = 100  # More epochs to see full convergence curve

# CHECK if train/test Set seasonal or chronological
file = "LSTM_m2m_Layer_{}_Input_{}_hidden_{}_future_El".format(layer, X_train.shape[2], hidden)
PATH_load = os.path.join(MODELS_DIR, file)
PATH_save = os.path.join(MODELS_DIR, file)
PATH_save_met = os.path.join(METRICS_DIR, "{}.csv".format(file))
PATH_save_res = os.path.join(FORECASTS_DIR, "result_{}.csv".format(file))
PATH_fig = os.path.join(FIGURES_DIR, "lstm_prediction.png")
# Training mode
# True = load existing model and fine-tune
# False = train new model from scratch
load_model = False

if load_model:
    model = torch.load(PATH_load, weights_only=False)
    print(f"Model loaded from: {PATH_load}")
    test_loss = trainModel(model, batch_size, seq_dim, epochs, initial_lr=5e-4)
    print("finished fine-tuning")
else:
    model = initializeNewModel(input_dim=X_train.shape[2], hidden_dim=hidden, layer_dim=layer, output_dim=36)
    print(model)
    test_loss = trainModel(model, batch_size, seq_dim, epochs, patience=15, initial_lr=1e-3)
    print("finished training")
