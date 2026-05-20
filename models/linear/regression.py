import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tables
from sklearn import linear_model, ensemble, neural_network
from sklearn.preprocessing import StandardScaler
import data_management
from data_management import DataManager

# Output directories
FORECASTS_DIR = "forecasts/linear"
os.makedirs(FORECASTS_DIR, exist_ok=True)

# BEFORE RUN: check if "window_tar" fits to "horizon"

feat = DataManager()
dropnight = "true"
window_tar = 36
features = feat.get_features(window_ft=12, window_tar=window_tar, dropnight=dropnight)
tar = feat.get_target_Pdc(window_tar=window_tar, dropnight=dropnight)

target = ["GHI", "BNI", "BOTH"]
horizon = ["5min", "10min", "15min", "20min", "25min", "30min", "35min", "40min", "45min", "50min", "55min", "60min",
"65min", "70min", "75min", "80min", "85min", "90min", "95min", "100min", "105min", "110min", "115min",
"120min", "125min", "130min", "135min", "140min", "145min", "150min", "155min", "160min", "165min",
"170min", "175min", "180min"]
window_tar = len(horizon)

""", "35min", "40min", "45min", "50min", "55min", "60min",
"65min", "70min", "75min", "80min", "85min", "90min", "95min", "100min", "105min", "110min", "115min",
"120min", "125min", "130min", "135min", "140min", "145min", "150min", "155min", "160min", "165min",
"170min", "175min", "180min"]"""

def run_forecast(target,horizon):

    train_x = features[features["dataset"] == "Train"]
    test_x = features[features["dataset"] == "Test"]
    train_y = tar[tar["dataset"] == "Train"]
    test_y = tar[tar["dataset"] == "Test"]

    train_y = train_y.drop('dataset', axis=1)
    test_y = test_y.drop('dataset', axis=1)

    train = train_x.merge(train_y, on="t")
    test = test_x.merge(test_y, on="t")

    # delete night time values
    train = train.drop(train.index[train["El_x"] < 10])
    test = test.drop(test.index[test["El_x"] < 10])

    train = train.dropna()
    test = test.dropna()

    # Select features based on target variant
    feature_cols_G = features.filter(regex="GHI").columns.tolist()
    feature_cols_B = features.filter(regex="BNI").columns.tolist()

    if target == "GHI":
        feature_cols = feature_cols_G
    elif target == "BNI":
        feature_cols = feature_cols_B
    else:  # BOTH
        feature_cols = feature_cols_G + feature_cols_B

    # Cross-irradiance features + weather variables + CSGHI
    # Use set to avoid duplicates (BNI_kt_* already in feature_cols_B, gti_kt_* already in feature_cols_G)
    cross_irr_cols = ["BNI_kt_one", "BNI_kt_two", "BNI_kt_three",
                      "gti_kt_one", "gti_kt_two", "gti_kt_three"]
    weather_cols = ["wdir", "tpw", "Az", "TL", "kd", "Ta", "vw", "RH", "El_x", "Patm", "Pdc_33", "CSGHI"]
    all_cols = list(dict.fromkeys(feature_cols + cross_irr_cols + weather_cols))  # preserves order, removes duplicates
    train_X = train[all_cols].values
    test_X = test[all_cols].values

    #+ ["Pdc_33"] + ["BNI_kt_one"] + ["BNI_kt_two"] + ["BNI_kt_three"] + ["gti_kt_one"] + ["gti_kt_two"] + ["gti_kt_three"]
    # + ["wdir"] + ["tpw"] + ["Az"] + ["TL"] + ["kd"] + ["Ta"] + ["vw"] + ["RH"] + ["CS"] + ["Patm"]

    train_Y = train['Pdc_{}'.format(horizon)].values
    test_Y = test['Pdc_{}'.format(horizon)].values

    # Ordinary Least-Squares (OLS)
    # Ridge Regression (OLS + L2-regularizer)
    # Lasso (OLS, L1-regularizer)

    models = [["ols", linear_model.LinearRegression()],
             ["ridge", linear_model.RidgeCV(cv=10)],
             ["lasso", linear_model.LassoCV(cv=10, max_iter=10000)]]

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    for name, model in models:
        model.fit(train_X, train_Y)
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)

        """plt.plot(test_Y)
        plt.plot(test_pred)"""

        # Denormalization of prediction
        train_pred = train_pred * feat.CAPACITY
        test_pred = test_pred * feat.CAPACITY

        train.insert(train.shape[1], "Pdc_{}_{}".format(target,name), train_pred)
        test.insert(test.shape[1], "Pdc_{}_{}".format(target,name), test_pred)

    # Denormalization of Power
    train_Y = train_Y * feat.CAPACITY
    test_Y = test_Y * feat.CAPACITY

    # Smart Persistence forecast: P_sp(t,h) = P(t) * ENI(t+h) / ENI(t)
    # Use horizon-specific smart persistence column
    sp_col = "Pdc_sp_{}".format(horizon)
    tmp = np.squeeze(train[sp_col].values) * feat.CAPACITY
    train.insert(train.shape[1], "Pdc_{}_sp".format(target), tmp)
    tmp = np.squeeze(test[sp_col].values) * feat.CAPACITY
    test.insert(test.shape[1], "Pdc_{}_sp".format(target), tmp)

    # save actual values to compare with predicted values in Postprozess
    train.insert(train.shape[1], "Pdc_{}_actual".format(target), train_Y)
    test.insert(test.shape[1], "Pdc_{}_actual".format(target), test_Y)

    # save forecasts
    # only keep essential forecast columns
    cols = train.columns[train.columns.str.startswith("Pdc_{}".format(target))]
    train = train[cols]
    test = test[cols]

    train.insert(train.shape[1], "dataset", "Train")
    test.insert(test.shape[1], "dataset", "Test")
    df = pd.concat([train, test], axis=0)
    df.insert(df.shape[1], "target", target)
    df.insert(df.shape[1], "horizon", horizon)
    df.to_hdf(
        os.path.join(FORECASTS_DIR, "forecasts_{}_{}.h5".format(horizon, target)),
        key="df",
        mode="w"
    )

# delete all files in folder to avoid having files of a bigger horizon in the folder when testing a smaller
# horizon -> for postprocess
for filename in os.listdir(FORECASTS_DIR):
    os.remove(os.path.join(FORECASTS_DIR, filename))

for t in target:
    for h in horizon:
        print("{} Pdc forecast for {}".format(h,t))
        run_forecast(t,h)
