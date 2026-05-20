import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import statistics
from data_management import DataManager

# Postprocess

def computeMetrics(path):

    file = pd.read_csv(path, sep=",")
    mae = file["MAE"]
    mbe = file["MBE"]
    rmse = file["RMSE"]
    rmse_sp = file["RMSE_sp"]

    file_res = pd.read_csv(path_res, sep=",")
    p = file_res["P_real"]
    pred = file_res["P_pred"]
    sp = file_res["P_sp"]

    MAE = []
    MBE = []
    RMSE = []
    RMSE_sp = []
    skill = []
    P_act = []
    P_pred = []
    P_sp = []

    for line in range(len(file)):
        row_mae = mae[line].strip("[]").split()
        row_mbe = mbe[line].strip("[]").split()
        row_rmse = rmse[line].strip("[]").split()
        row_rmse_sp = rmse_sp[line].strip("[]").split()
        for i, (item_mae, item_mbe, item_rmse, item_rmse_sp) in enumerate(zip(row_mae, row_mbe, row_rmse, row_rmse_sp)):
            row_mae[i] = float(item_mae)
            row_mbe[i] = float(item_mbe)
            row_rmse[i] = float(item_rmse)
            row_rmse_sp[i] = float(item_rmse_sp)
        MAE.append(row_mae)
        MBE.append(row_mbe)
        RMSE.append(row_rmse)
        RMSE_sp.append(row_rmse_sp)


    """for t in range(len(file_res)):
        row_real = p[t].strip("[]").split()
        row_pred = pred[t].strip("[]").split()
        row_sp = sp[t].strip("[]").split()
        for m, (item_p, item_pred, item_sp) in enumerate(zip(row_real, row_pred, row_sp)):
            row_real[m+1] = float(item_p)
            row_pred[m+1] = float(item_pred)
            row_sp[m+1] = float(item_sp)
        P_act.append(row_real)
        P_pred.append(row_pred)
        P_sp.append(row_sp)"""


    mae_mean = np.nanmean(MAE, axis=0)
    mbe_mean = np.nanmean(MBE, axis=0)
    rmse_mean = np.nanmean(RMSE, axis=0)
    rmse_sp_mean = np.nanmean(RMSE_sp, axis=0)

    for i in range(len(rmse_mean)):
        val = 1 - (rmse_mean[i]/rmse_sp_mean[i])
        skill.append(val)

    for t in range(len(mae_mean)-1):
        horizon = (t+1) * 5
        print("forecast {}min: MAE: {:.2f} std: {:.1f}, MBE: {:.2f} std: {:.1f}, RMSE: {:.2f} std: {:.1f},"
              " skill: {:.2f} std: {:.1f}, RMSE_sp: {:.2f} std: {:.1f}".format
              (horizon, mae_mean[t], np.nanstd(np.transpose(MAE)[t]), mbe_mean[t], np.nanstd(np.transpose(MBE)[t]),
               rmse_mean[t], np.nanstd(np.transpose(RMSE)[t]), skill[t] * 100, np.nanstd(np.transpose(skill)[t]),
              rmse_sp_mean[t], np.nanstd(np.transpose(RMSE_sp)[t])))

    return None

# metricLSTM_m2m_layer_2_Input_5_hidden_100_highneg_cr
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_cr
# metricLSTM_m2m_layer_2_Input_5_hidden_100_highneg_seas
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_seas
# metricLSTM_m2m_layer_2_Input_3_hidden_100_0_seas_gtiBNIRHkd_sc
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_seas_minmax
# metricLSTM_m2m_layer_2_Input_8_hidden_100_0_seas_shift_Ind
# metricLSTM_m2m_layer_2_Input_8_hidden_100_0_seas_Ind

# GHI, BNI, Ta, El, Az
# LSTM_m2m_Layer_2_Input_5_hidden_150_0_shift_denorm_minmax_El_Az.csv not usable
# LSTM_m2m_Layer_2_Input_5_hidden_75_0_shift_denorm_minmax_El_Az_bt15.csv verwendbar!

# GHI, BNI, Ta, TL
# LSTM_m2m_Layer_2_Input_4_hidden_150_0_seas_shift_night_denorm_minmax.csv

# LSTM_m2m_Layer_2_Input_5_hidden_50_0_denorm_minmax
# LSTM_m2m_Layer_2_Input_5_hidden_100_0_seas_minmax_bt10.csv
# LSTM_m2m_Layer_2_Input_5_hidden_100_0_seas_minmax_bt30_GHIBNITaTLAMa #bt10 MSE
# LSTM_m2m_Layer_2_Input_5_hidden_100_0_seas_scaler_bt10_ktBNIktTaTLvw_MSE.csv

file = "LSTM_m2m_Layer_2_Input_5_hidden_75_0_shift_denorm_minmax_El_Az_bt15.csv"
path = "results/{}".format(file)
path_res = "results/result{}".format(file)

"""file = "LSTM_Layer_2_Input_5_hidden_75_0_Scaler_GHIBNITaTLvw.csv" #bt10 L1
path = "C:/Users/sophi/OneDrive/Dokumente/TU_Stuttgart/Studienarbeit/Git/PV-forecasting_1year/Vulcan/{}".format(file)
path_res = "C:/Users/sophi/OneDrive/Dokumente/TU_Stuttgart/Studienarbeit/Git/PV-forecasting_1year/Vulcan/result{}".format(file)"""

# LSTM_Layer_2_Input_5_hidden_75_0_Scaler_GHIBNITaTLvw.csv L1, bt=10, dropout=false
# LSTM_m2m_Layer_2_Input_5_hidden_75_0_denorm_minmax_L1.csv GHI, BNI, Ta, El, Az
# LSTM_m2m_Layer_2_Input_5_hidden_50_0_denorm_minmax_month  GHI, BNI, Ta, El, Az loss=L1

computeMetrics(path)

"""window_LSTM = 36
tar = DataManager()
train_y, test_y = tar.get_target_LSTM(window_LSTM)
# rmse_sp = get_rmse_sp

test_y = test_y.drop(test_y.index[test_y["El"] < 15])
tar_cols = test_y.filter(regex="min").columns.tolist()

rmse_sp = []
for step in range(0, len(tar_cols)):
    min = tar_cols[step]
    Pdc_act = test_y["{}".format(min)] * tar.CAPACITY
    error_sp = np.sqrt(np.nanmean((Pdc_act - test_y["Pdc_sp"]).values ** 2))
    rmse_sp.append(error_sp)"""