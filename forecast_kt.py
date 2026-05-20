import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tables
import DataManagement
from sklearn import linear_model, ensemble, neural_network
from sklearn.preprocessing import StandardScaler
from DataManagement import get_features, get_target_Irr

#Capacity = 20808.66
features = get_features()
tar = get_target_Irr()
features.insert(features.shape[1], column="key", value = np.array(range(0,len(features))))
tar.insert(tar.shape[1], column="key", value = np.array(range(0,len(tar))))

def run_forecast(target,horizon):

    cols = [
        "{}_{}".format(target, horizon),  # actual
        "{}_kt_{}".format(target, horizon),  # clear-sky index
        "{}_clear_{}".format(target, horizon),  # clear-sky model
        "El_{}".format(horizon), # solar elevation
        "ENI_{}".format(horizon)] # extraterrestrial Irradiation

    train_x = features[features["dataset"] == "Train"]
    test_x = features[features["dataset"] == "Test"]
    train_y = tar[tar["dataset"] == "Train"]
    test_y = tar[tar["dataset"] == "Test"]

    train_y = train_y.drop('dataset', axis=1)
    test_y = test_y.drop('dataset', axis=1)

    train = train_x.merge(train_y, on="key")
    test = test_x.merge(test_y, on="key")

    """train = train.drop(train.index[train["El_{}".format(horizon)] < 5])
    test = test.drop(test.index[test["El_{}".format(horizon)] < 5])"""

    feature_cols = features.filter(regex=target).columns.tolist()

    train = train[cols + feature_cols].dropna(how="any")
    test = test[cols + feature_cols].dropna(how="any")

    train_X = train[feature_cols].values
    test_X = test[feature_cols].values

    train_y = train['{}_kt_{}'.format(target, horizon)].values
    elev_train = train["El_{}".format(horizon)].values
    elev_test  = test["El_{}".format(horizon)].values

    train_clear = train["{}_clear_{}".format(target,horizon)].values
    test_clear = test["{}_clear_{}".format(target,horizon)].values

    train_ENI = train["ENI_{}".format(horizon)].values
    test_ENI = test["ENI_{}".format(horizon)].values

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
        model.fit(train_X, train_y)
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)

        # convert from kt back to irradiance
        """train_pred *= train_clear
        test_pred *= test_clear"""

        if target == "BNI":
            train_pred *= train_ENI
            test_pred *= test_ENI
        else:
            train_pred *= train_ENI * np.sin(np.deg2rad(elev_train))
            test_pred *= test_ENI * np.sin(np.deg2rad(elev_test))

        # removes nighttime values (solar elevation < 5)
        train_pred[elev_train < 5] = np.nan
        test_pred[elev_test < 5] = np.nan

        train.insert(train.shape[1], "{}_{}".format(target,name), train_pred)
        test.insert(test.shape[1], "{}_{}".format(target,name), test_pred)

    # smart persistence forecast
    # uses the shortest backward average as the "current" kt value
    if target == "BNI":
        tmp = np.squeeze(train["B_{}_kt_0".format(target)].values) * train_ENI # train_clear
        #tmp[elev_train < 5] = np.nan
        train.insert(train.shape[1], "{}_sp".format(target), tmp)
        tmp = np.squeeze(test["B_{}_kt_0".format(target)].values) * test_ENI # test_clear
        #tmp[elev_test <5] = np.nan
        test.insert(test.shape[1], "{}_sp".format(target), tmp)
    else:
        tmp = np.squeeze(train["B_{}_kt_0".format(target)].values) * train_ENI * np.sin(np.deg2rad(elev_train))  # train_clear
        # tmp[elev_train < 5] = np.nan
        train.insert(train.shape[1], "{}_sp".format(target), tmp)
        tmp = np.squeeze(test["B_{}_kt_0".format(target)].values) * test_ENI * np.sin(np.deg2rad(elev_test))  # test_clear
        # tmp[elev_test <5] = np.nan
        test.insert(test.shape[1], "{}_sp".format(target), tmp)

    # save forecasts
    # only keep essential forecast columns
    cols = train.columns[train.columns.str.startswith("{}".format(target))]
    train = train[cols]
    test = test[cols]

    train.insert(train.shape[1], "dataset", "Train")
    test.insert(test.shape[1], "dataset", "Test")
    df = pd.concat([train, test], axis=0)
    df.insert(df.shape[1], "target", target)
    df.insert(df.shape[1], "horizon", horizon)
    df.to_hdf(os.path.join("forecasts",
                "forecasts_{}_{}.h5".format
                (horizon,target),
                           ), "df", mode="w",
              )

# delete all files in folder to avoid having files of a bigger horizon in the folder when testing a smaller
# horizon -> for postprocess
for filename in os.listdir("forecasts"):
    os.remove(os.path.join("forecasts", filename))

target = ["GHI", "BNI"]
horizon = ["5min", "10min", "15min", "20min", "25min", "30min"]
#horizon_tar = []


for t in target:
    for h in horizon:
        print("{} Pdc forecast for {}".format(h,t))
        run_forecast(t,h)
