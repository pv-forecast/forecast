import numpy as np
import pandas as pd


def IndicatorNaN(train_X, test_X, train_Y):

    for i, m in zip(range(1, train_X.shape[1]*2, 2), range(0, train_X.shape[1]*2 + 1, 2)):
        bool = train_X[train_X.columns[m]].isna()
        bool = np.multiply(bool, 1)
        train_X.insert(i, "Indicator_{}".format(m), value=bool)
        bool = test_X[test_X.columns[m]].isna()
        bool = np.multiply(bool, 1)
        test_X.insert(i, "Indicator_{}".format(m), value=bool)

    train_X = train_X.fillna(value=0.00)
    test_X = test_X.fillna(value=0.00)
    train_Y = train_Y.fillna(value=0.00)

    return train_X, test_X, train_Y

def split_sequences(sequences, n_steps_in, n_steps_out, n_sp_cols=36):
    """
    Split sequences for LSTM training.

    Args:
        sequences: Stacked array with columns:
            - Features (n_features columns)
            - Target power (1 column, normalized)
            - Smart Persistence per horizon (n_sp_cols columns, in Watts)
            - ENI (1 column)
        n_steps_in: Input sequence length (seq_dim)
        n_steps_out: Output horizon (window_LSTM)
        n_sp_cols: Number of Smart Persistence columns (default 36 for 5-180min)
    """
    X, y, y_sp, ENI = list(), list(), list(), list()

    # Column layout: [features..., target, sp_5min, sp_10min, ..., sp_180min, ENI]
    n_cols = sequences.shape[1]
    n_features = n_cols - 1 - n_sp_cols - 1  # Subtract target, sp columns, and ENI

    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(sequences):
            break

        # X: input features for seq_dim timesteps
        seq_x = sequences[i:end_ix, 0:n_features]

        # Y: target power for n_steps_out future timesteps (from consecutive rows)
        seq_y = sequences[end_ix - 1:out_end_ix, n_features]

        # Smart Persistence: all horizon-specific values at current time (row end_ix-1)
        # Shape: (n_sp_cols,) = (36,) for each horizon
        seq_ysp = sequences[end_ix - 1, n_features + 1:n_features + 1 + n_sp_cols]

        # ENI for future timesteps
        seq_ENI = sequences[end_ix - 1:out_end_ix, -1]

        X.append(seq_x)
        y.append(seq_y)
        y_sp.append(seq_ysp)
        ENI.append(seq_ENI)

    print("done")

    return np.array(X), np.array(y), np.array(y_sp), np.array(ENI)


