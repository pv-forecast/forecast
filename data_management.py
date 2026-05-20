import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

class DataManager:

    def __init__(self):

        self.CAPACITY = 20808.66
        self.delta = 5 # step size [min]

        filename = 'Daten/PVAMM_201911-202011_PT5M_merged.csv'
        data = pd.read_csv(filename)

        # Calculate clearness indices
        gti_kt = data["gti30t187a"].div(data["ENI"])
        BNI_kt = data["BNI"].div(data["ENI"])

        # Calculate lagged features (1, 2, 3 days = 288, 576, 864 intervals)
        new_cols = pd.DataFrame({
            "gti_kt": gti_kt,
            "BNI_kt": BNI_kt,
            "BNI_kt_one": BNI_kt.shift(periods=288),
            "BNI_kt_two": BNI_kt.shift(periods=288*2),
            "BNI_kt_three": BNI_kt.shift(periods=288*3),
            "gti_kt_one": gti_kt.shift(periods=288),
            "gti_kt_two": gti_kt.shift(periods=288*2),
            "gti_kt_three": gti_kt.shift(periods=288*3),
        })

        data = pd.concat([data, new_cols], axis=1)
        self.data = data

        train = pd.DataFrame()
        test = pd.DataFrame()
        t = pd.to_datetime(self.data["t"]).array
        month_number = t.month

        for m in range(1, 13):
            in_month_m = month_number == m
            this_month = self.data.iloc[in_month_m, :]
            # splitting by day number ensures you have whole days on each data set
            day_number = t[in_month_m].day
            in_training_set = day_number < np.percentile(day_number, 80)
            train = pd.concat([train, this_month.iloc[in_training_set, :]], axis=0)
            test = pd.concat([test, this_month.iloc[~in_training_set, :]], axis=0)

        """first = ["Sep", "Dec", "Mar", "Jun"]
        second = ["Oct", "Jan", "Apr", "Jul"]
        third = ["Nov", "Feb", "May", "Aug"]  # 2019 und 2020
        autumn = pd.DataFrame()
        winter = pd.DataFrame()
        spring = pd.DataFrame()
        summer = pd.DataFrame()

        for t in [first, second, third]:
            aut = self.data[self.data.t.str.contains(t[0])]
            win = self.data[self.data.t.str.contains(t[1])]
            spr = self.data[self.data.t.str.contains(t[2])]
            sum = self.data[self.data.t.str.contains(t[3])]
            autumn = pd.concat([autumn, aut], axis=0)
            winter = pd.concat([winter, win], axis=0)
            spring = pd.concat([spring, spr], axis=0)
            summer = pd.concat([summer, sum], axis=0)"""

        """train, test = self.data[0:int(len(self.data) * 0.8)], self.data[int(len(self.data) * 0.8):len(self.data)]"""

        self.train = train
        self.test = test

    def get_data(self, deep_copy = True):
        return self.data.copy(deep_copy)

    def get_features (self, window_ft, window_tar, dropnight, deep_copy = True):
        # data have to be stored in a pandas DataFrame
        # for Input -> X (B = backward Average, L = lagged Average, V = Variability)
        # build feature Normalization

        # Trainings features

        B_BNI_kt_1 = self.train["BNI_kt"]
        B_BNI_kt = self.train["BNI_kt"]
        B_GHI_kt_1 = self.train["gti_kt"]
        B_GHI_kt = self.train["gti_kt"]

        featuresB_GHI = pd.DataFrame()
        featuresB_BNI = pd.DataFrame()
        featuresL_GHI = pd.DataFrame()
        featuresL_BNI = pd.DataFrame()
        featuresV_GHI = pd.DataFrame()
        featuresV_BNI = pd.DataFrame()
        featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
        featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)

        for col in range(0, window_ft):
            featuresB_GHI.insert(col, column='B_GHI_kt_%i' % col, value=B_GHI_kt)
            featuresB_BNI.insert(col, column='B_BNI_kt_%i' % col, value=B_BNI_kt)
            BGHI_shift = B_GHI_kt_1.shift(periods=col+1)
            BBNI_shift = B_BNI_kt_1.shift(periods=col+1)
            clmn = col + 1
            featuresL_GHI.insert(clmn, column='L_GHI_kt_%i' % clmn, value=BGHI_shift)
            featuresL_BNI.insert(clmn, column='L_BNI_kt_%i' % clmn, value=BBNI_shift)
            B_GHI_kt = featuresL_GHI.mean(axis=1)
            B_BNI_kt = featuresL_BNI.mean(axis=1)

        featuresL_GHI = featuresL_GHI.drop('L_GHI_kt_{}'.format(window_ft), axis=1)
        featuresL_BNI = featuresL_BNI.drop('L_BNI_kt_{}'.format(window_ft), axis=1)
        GHI_kt_mean = featuresB_GHI["B_GHI_kt_{}".format(window_ft-1)].to_frame()
        BNI_kt_mean = featuresB_BNI["B_BNI_kt_{}".format(window_ft-1)].to_frame()

        for col in range(0, window_ft):
            delta_kt_GHI = featuresL_GHI.iloc[:, 0:col+1].sub(GHI_kt_mean.values.reshape(len(GHI_kt_mean), col+1)).pow(2)
            delta_kt_BNI = featuresL_BNI.iloc[:, 0:col+1].sub(BNI_kt_mean.values.reshape(len(BNI_kt_mean), col+1)).pow(2)
            GHI_kt_mean.insert(col+1, column="times_%i" % col, value=GHI_kt_mean["B_GHI_kt_{}".format(window_ft-1)].values)
            BNI_kt_mean.insert(col+1, column="times_%i" % col, value=BNI_kt_mean["B_BNI_kt_{}".format(window_ft-1)].values)
            V_GHI_kt = pd.Series(np.sqrt(np.divide(delta_kt_GHI.sum(axis=1), col+1)))
            V_BNI_kt = pd.Series(np.sqrt(np.divide(delta_kt_BNI.sum(axis=1), col+1)))
            featuresV_GHI.insert(col, column='V_GHI_kt_%i' % col, value=V_GHI_kt)
            featuresV_BNI.insert(col, column='V_BNI_kt_%i' % col, value=V_BNI_kt)

        # Include Pdc to Trainingsset, which is not within the Horizon
        back = window_ft + 1
        Pdc_35_train = self.train["Pdc_33"].shift(periods=back)
        Pdc_35_train = Pdc_35_train.div(self.CAPACITY)

        features_train = pd.concat([self.train["t"], featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                              featuresL_GHI, featuresL_BNI, self.train["BNI_kt_one"], self.train["BNI_kt_two"],
                              self.train["BNI_kt_three"], self.train["gti_kt_one"], self.train["gti_kt_two"],
                              self.train["gti_kt_three"], self.train["Ta"], self.train["vw"], self.train["RH"], self.train["wdir"], self.train["tpw"],
                              self.train["Az"], self.train["TL"], Pdc_35_train, self.train["AMa"], self.train["kd"],
                              self.train["El"], self.train["CS"], self.train["Patm"],
                              self.train["CSGHI"], self.train["GHI"]], axis=1)

        # , self.train["BNI_kt_one"], self.train["BNI_kt_two"],
        #                               self.train["BNI_kt_three"], self.train["Ta"], self.train["CS"], self.train["Patm"]
        #                               self.train["vw"], self.train["RH"], self.train["wdir"], self.train["tpw"],
        #                               self.train["Az"], , self.train["TL"], Pdc_35_train, self.train["AMa"], self.train["kd"],
        #, self.train["gti_kt_one"], self.train["gti_kt_two"],
        #                               self.train["gti_kt_three"],

        ft_train = features_train[0:len(features_train) - window_tar]
        ft_train.insert(ft_train.shape[1], "dataset", "Train")

        # Test features

        B_BNI_kt_1 = self.test["BNI_kt"]
        B_BNI_kt = self.test["BNI_kt"]
        B_GHI_kt_1 = self.test["gti_kt"]
        B_GHI_kt = self.test["gti_kt"]

        featuresB_GHI = pd.DataFrame()
        featuresB_BNI = pd.DataFrame()
        featuresL_GHI = pd.DataFrame()
        featuresL_BNI = pd.DataFrame()
        featuresV_GHI = pd.DataFrame()
        featuresV_BNI = pd.DataFrame()
        featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
        featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)

        for col in range(0, window_ft):
            featuresB_GHI.insert(col, column='B_GHI_kt_%i' % col, value=B_GHI_kt)
            featuresB_BNI.insert(col, column='B_BNI_kt_%i' % col, value=B_BNI_kt)
            BGHI_shift = B_GHI_kt_1.shift(periods=col + 1)
            BBNI_shift = B_BNI_kt_1.shift(periods=col + 1)
            clmn = col + 1
            featuresL_GHI.insert(clmn, column='L_GHI_kt_%i' % clmn, value=BGHI_shift)
            featuresL_BNI.insert(clmn, column='L_BNI_kt_%i' % clmn, value=BBNI_shift)
            B_GHI_kt = featuresL_GHI.mean(axis=1)
            B_BNI_kt = featuresL_BNI.mean(axis=1)

        featuresL_GHI = featuresL_GHI.drop('L_GHI_kt_{}'.format(window_ft), axis=1)
        featuresL_BNI = featuresL_BNI.drop('L_BNI_kt_{}'.format(window_ft), axis=1)
        GHI_kt_mean = featuresB_GHI["B_GHI_kt_{}".format(window_ft-1)].to_frame()
        BNI_kt_mean = featuresB_BNI["B_BNI_kt_{}".format(window_ft-1)].to_frame()

        for col in range(0, window_ft):
            delta_kt_GHI = featuresL_GHI.iloc[:, 0:col + 1].sub(GHI_kt_mean.values.reshape(len(GHI_kt_mean), col + 1)).pow(
                2)
            delta_kt_BNI = featuresL_BNI.iloc[:, 0:col + 1].sub(BNI_kt_mean.values.reshape(len(BNI_kt_mean), col + 1)).pow(
                2)
            GHI_kt_mean.insert(col + 1, column="times_%i" % col, value=GHI_kt_mean["B_GHI_kt_{}".format(window_ft-1)].values)
            BNI_kt_mean.insert(col + 1, column="times_%i" % col, value=BNI_kt_mean["B_BNI_kt_{}".format(window_ft-1)].values)
            V_GHI_kt = pd.Series(np.sqrt(np.divide(delta_kt_GHI.sum(axis=1), col + 1)))
            V_BNI_kt = pd.Series(np.sqrt(np.divide(delta_kt_BNI.sum(axis=1), col + 1)))
            featuresV_GHI.insert(col, column='V_GHI_kt_%i' % col, value=V_GHI_kt)
            featuresV_BNI.insert(col, column='V_BNI_kt_%i' % col, value=V_BNI_kt)

        # Include Pdc to Testset, which is not within the Horizon
        Pdc_35_test = self.test["Pdc_33"].shift(periods=back)
        Pdc_35_test = Pdc_35_test.div(self.CAPACITY)

        features_test = pd.concat([self.test["t"], featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                              featuresL_GHI, featuresL_BNI, self.test["BNI_kt_one"], self.test["BNI_kt_two"],
                              self.test["BNI_kt_three"], self.test["gti_kt_one"], self.test["gti_kt_two"],
                              self.test["gti_kt_three"], self.test["Ta"], self.test["vw"], self.test["RH"], self.test["wdir"], self.test["tpw"],
                              self.test["Az"], self.test["TL"], Pdc_35_test, self.test["AMa"], self.test["kd"],
                              self.test["El"], self.test["CS"], self.test["Patm"],
                              self.test["CSGHI"], self.test["GHI"]], axis=1)

        # , self.test["BNI_kt_one"], self.test["BNI_kt_two"],
        #                               self.test["BNI_kt_three"], self.test["Ta"], self.test["CS"], self.test["Patm"]
        #                               self.test["vw"], self.test["RH"], self.test["wdir"], self.test["tpw"],
        #                               self.test["Az"], Pdc_35_test, self.test["TL"],self.test["AMa"], self.test["kd"]
        # , self.test["gti_kt_one"], self.test["gti_kt_two"],
        #                                       self.test["gti_kt_three"],

        ft_test = features_test[0:len(features_test) - window_tar]
        ft_test.insert(ft_test.shape[1], "dataset", "Test")

        """if dropnight == "true":
            ft_train = ft_train.drop(ft_train.index[ft_train["El"] < 15])
            ft_test = ft_test.drop(ft_test.index[ft_test["El"] < 15])"""

        features = pd.concat([ft_train, ft_test], axis=0)

        return features.copy(deep_copy)

    def get_target_Pdc (self, window_tar, dropnight, deep_copy = True):
        # for Output -> Y (Power)
        # Train target
        # For Linear Regression for time t: x(t), y(t+1) in one row, SEE
        # https://ichi.pro/de/so-formen-sie-daten-neu-und-fuhren-mit-lstm-eine-regression-fur-zeitreihen-durch-21155626274048

        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.train["Pdc_33"].div(self.CAPACITY)
        # Use Clear-Sky GHI for Smart Persistence (accounts for sun angle)
        CSGHI_current = self.train["CSGHI"].copy()

        # Smart Persistence for each horizon: P_sp(t,h) = P(t) * CSGHI(t+h) / CSGHI(t)
        # Create horizon-specific smart persistence columns
        for col in range(1, window_tar + 1):
            horizon_min = self.delta * col
            CSGHI_future = self.train["CSGHI"].shift(periods=-col)
            # Avoid division by zero
            CSGHI_ratio = CSGHI_future / CSGHI_current.replace(0, np.nan)
            CSGHI_ratio = CSGHI_ratio.fillna(1.0)  # If CSGHI_current is 0, use ratio of 1
            Pdc_sp_horizon = Pdc_norm * CSGHI_ratio
            Pdc_shift.insert(col-1, column='Pdc_sp_{}min'.format(horizon_min), value=Pdc_sp_horizon)

        # Add actual power targets for each horizon
        Pdc_norm_shifted = self.train["Pdc_33"].div(self.CAPACITY)
        for col in range(1, window_tar + 1):
            Pdc_norm_shifted = Pdc_norm_shifted.shift(periods=-1)
            Pdc_shift.insert(Pdc_shift.shape[1], column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm_shifted)

        target_train = pd.concat([self.train["t"], Pdc_shift, self.train["ENI"], self.train["El"]], axis=1)
        target_train.insert(target_train.shape[1], "dataset", "Train")
        t_train = target_train[0:len(target_train) - window_tar]

        # Test target
        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.test["Pdc_33"].div(self.CAPACITY)
        CSGHI_current = self.test["CSGHI"].copy()

        # Smart Persistence for each horizon: P_sp(t,h) = P(t) * CSGHI(t+h) / CSGHI(t)
        for col in range(1, window_tar + 1):
            horizon_min = self.delta * col
            CSGHI_future = self.test["CSGHI"].shift(periods=-col)
            CSGHI_ratio = CSGHI_future / CSGHI_current.replace(0, np.nan)
            CSGHI_ratio = CSGHI_ratio.fillna(1.0)
            Pdc_sp_horizon = Pdc_norm * CSGHI_ratio
            Pdc_shift.insert(col-1, column='Pdc_sp_{}min'.format(horizon_min), value=Pdc_sp_horizon)

        # Add actual power targets for each horizon
        Pdc_norm_shifted = self.test["Pdc_33"].div(self.CAPACITY)
        for col in range(1, window_tar + 1):
            Pdc_norm_shifted = Pdc_norm_shifted.shift(periods=-1)
            Pdc_shift.insert(Pdc_shift.shape[1], column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm_shifted)

        target_test = pd.concat([self.test["t"], Pdc_shift, self.test["ENI"], self.test["El"]], axis=1)
        target_test.insert(target_test.shape[1], "dataset", "Test")
        t_test = target_test[0:len(target_test) - window_tar]

        """if dropnight == "true":
            t_train = t_train.drop(t_train.index[t_train["El"] < 15])
            t_test = t_test.drop(t_test.index[t_test["El"] < 15])"""

        target = pd.concat([t_train, t_test], axis=0)

        return target.copy(deep_copy)

    def get_features_LSTM (self, window_LSTM, feature_str, include_future_el=True):
        """
        Get features for LSTM model.

        Args:
            window_LSTM: Output window size (number of future timesteps to predict)
            feature_str: List of feature column names
            include_future_el: If True, add future sun elevation for each output timestep
        """
        # Train features
        features_train = self.train[feature_str].copy()

        # Add future elevation at key horizons only (every 30min)
        # 6 features instead of 36 to reduce redundancy
        if include_future_el and "El" in self.train.columns:
            horizons = [6, 12, 18, 24, 30, 36]  # 30, 60, 90, 120, 150, 180 min
            for h in horizons:
                if h <= window_LSTM:
                    el_future = self.train["El"].shift(periods=-h)
                    features_train[f"El_future_{h*5}min"] = el_future

        X_train = features_train[0:len(features_train) - window_LSTM]

        # Test features
        features_test = self.test[feature_str].copy()

        if include_future_el and "El" in self.test.columns:
            horizons = [6, 12, 18, 24, 30, 36]  # 30, 60, 90, 120, 150, 180 min
            for h in horizons:
                if h <= window_LSTM:
                    el_future = self.test["El"].shift(periods=-h)
                    features_test[f"El_future_{h*5}min"] = el_future

        X_test = features_test[0:len(features_test) - window_LSTM]

        return X_train, X_test

    def get_target_LSTM(self, window_LSTM):
        # for Output -> Y (Power)
        # Train target
        # for LSTM Input for time t: x(t), y(t) in one row
        # take the shortest backwards step as Smart Persistence Model (sp)

        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.train["Pdc_33"].div(self.CAPACITY)
        # Smart Persistence: current value at t predicts all future horizons
        Pdc_sp = self.train["Pdc_33"].copy()  # In Watts (not normalized)
        Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

        for col in range(1, window_LSTM + 1):
            Pdc_shift.insert(col, column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm)
            Pdc_norm = Pdc_norm.shift(periods=-1)

        target_train = pd.concat([self.train["t"], Pdc_shift, self.train["ENI"], self.train["El"], self.train["Pdc_33"]], axis=1)
        Y_train = target_train[0:len(target_train) - window_LSTM]

        # Test target

        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.test["Pdc_33"].div(self.CAPACITY)
        # Smart Persistence: current value at t predicts all future horizons
        Pdc_sp = self.test["Pdc_33"].copy()  # In Watts (not normalized)
        Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

        for col in range(1, window_LSTM + 1):
            Pdc_shift.insert(col, column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm)
            Pdc_norm = Pdc_norm.shift(periods=-1)

        target_test = pd.concat([self.test["t"], Pdc_shift, self.test["ENI"], self.test["El"], self.test["Pdc_33"]], axis=1)
        Y_test = target_test[0:len(target_test) - window_LSTM]

        return Y_train, Y_test

    def get_target_Irr(self, window_tar, deep_copy = True):
        # for Output -> Y (Irradiance, kt)
        # Train target

        BNI_train = self.train["BNI"]
        GHI_train = self.train["GHI"]
        El_train = self.train["El"]
        CSGHI_train = self.train["CSGHI"]
        CSBNI_train = self.train["CSBNI"]
        GHI_KT_train = self.train["kt"]
        ENI_train = self.train["ENI"]

        target_Irr_train = pd.DataFrame()
        BNI_kt_train = BNI_train.div(ENI_train)

        for blk in range(0, window_tar):
            GHI_train = GHI_train.shift(periods=-1)
            BNI_train = BNI_train.shift(periods=-1)
            CSGHI_train = CSGHI_train.shift(periods=-1)
            CSBNI_train = CSBNI_train.shift(periods=-1)
            GHI_KT_train = GHI_KT_train.shift(periods=-1)
            BNI_kt_train = BNI_kt_train.shift(periods=-1)
            El_train = El_train.shift(periods=-1)
            ENI_train = ENI_train.shift(periods=-1)
            block = pd.DataFrame()
            block.insert(0, column="GHI_{}min".format(self.delta * (blk + 1)), value=GHI_train)
            block.insert(1, column="BNI_{}min".format(self.delta * (blk + 1)), value=BNI_train)
            block.insert(2, column="GHI_clear_{}min".format(self.delta * (blk + 1)), value=CSGHI_train)
            block.insert(3, column="BNI_clear_{}min".format(self.delta * (blk + 1)), value=CSBNI_train)
            block.insert(4, column="GHI_kt_{}min".format(self.delta * (blk + 1)), value=GHI_KT_train)
            block.insert(5, column="BNI_kt_{}min".format(self.delta * (blk + 1)), value=BNI_kt_train)
            block.insert(6, column="El_{}min".format(self.delta * (blk + 1)), value=El_train)
            block.insert(7, column="ENI_{}min".format(self.delta * (blk + 1)), value=ENI_train)
            target_Irr_train = pd.concat([target_Irr_train, block], axis=1)

        target_Irr_train = pd.concat([self.train["t"], target_Irr_train], axis=1) # del later, just for validation

        target_Irr_train.insert(target_Irr_train.shape[1], "dataset", "Train")
        t_Irr_train = target_Irr_train[0:len(target_Irr_train) - window_tar]

        # key = pd.DataFrame(np.array(range(0, len(t_Irr_train))), columns=["key"])  # del later, just for validation
        # tar_Irr_train = pd.concat([key, t_Irr_train], axis=1)  # del later, just for validation

        # Test target

        BNI_test = self.test["BNI"]
        GHI_test = self.test["GHI"]
        El_test = self.test["El"]
        CSGHI_test = self.test["CSGHI"]
        CSBNI_test = self.test["CSBNI"]
        GHI_KT_test = self.test["kt"]
        ENI_test = self.test["ENI"]

        target_Irr_test = pd.DataFrame()
        BNI_kt_test = BNI_test.div(ENI_test)

        for blk in range(0, window_tar):
            GHI_test = GHI_test.shift(periods=-1)
            BNI_test = BNI_test.shift(periods=-1)
            CSGHI_test = CSGHI_test.shift(periods=-1)
            CSBNI_test = CSBNI_test.shift(periods=-1)
            GHI_KT_test = GHI_KT_test.shift(periods=-1)
            BNI_kt_test = BNI_kt_test.shift(periods=-1)
            El_test = El_test.shift(periods=-1)
            ENI_test = ENI_test.shift(periods=-1)
            block = pd.DataFrame()
            block.insert(0, column="GHI_{}min".format(self.delta * (blk + 1)), value=GHI_test)
            block.insert(1, column="BNI_{}min".format(self.delta * (blk + 1)), value=BNI_test)
            block.insert(2, column="GHI_clear_{}min".format(self.delta * (blk + 1)), value=CSGHI_test)
            block.insert(3, column="BNI_clear_{}min".format(self.delta * (blk + 1)), value=CSBNI_test)
            block.insert(4, column="GHI_kt_{}min".format(self.delta * (blk + 1)), value=GHI_KT_test)
            block.insert(5, column="BNI_kt_{}min".format(self.delta * (blk + 1)), value=BNI_kt_test)
            block.insert(6, column="El_{}min".format(self.delta * (blk + 1)), value=El_test)
            block.insert(7, column="ENI_{}min".format(self.delta * (blk + 1)), value=ENI_test)
            target_Irr_test = pd.concat([target_Irr_test, block], axis=1)

        target_Irr_test = pd.concat([self.test["t"], target_Irr_test], axis=1) # del later, just for validation

        target_Irr_test.insert(target_Irr_test.shape[1], "dataset", "Test")
        t_Irr_test = target_Irr_test[0:len(target_Irr_test) - window_tar]

        # key = pd.DataFrame(np.array(range(0, len(t_Irr_test))), columns=["key"]) # del later, just for validation
        # tar_Irr_test = pd.concat([key, t_Irr_test], axis=1) # del later, just for validation

        target = pd.concat([t_Irr_train, t_Irr_test], axis=0)

        return target.copy(deep_copy)


# Wrapper functions for backwards compatibility with LSTM.py
_default_manager = None

def _get_manager():
    global _default_manager
    if _default_manager is None:
        _default_manager = DataManager()
    return _default_manager

def get_features(window_ft=12, window_tar=36, dropnight="true"):
    """Wrapper function for DataManager.get_features()"""
    return _get_manager().get_features(window_ft, window_tar, dropnight)

def get_target_LSTM(window_LSTM=36):
    """Wrapper function for DataManager.get_target_LSTM()"""
    manager = _get_manager()
    train_y, test_y = manager.get_target_LSTM(window_LSTM)
    # Combine train and test with dataset labels for compatibility
    train_y.insert(train_y.shape[1], "dataset", "Train")
    test_y.insert(test_y.shape[1], "dataset", "Test")
    return pd.concat([train_y, test_y], axis=0)

def get_target_Pdc(window_tar=36, dropnight="true"):
    """Wrapper function for DataManager.get_target_Pdc()"""
    return _get_manager().get_target_Pdc(window_tar, dropnight)

def get_target(window_tar=6, dropnight="true"):
    """Alias for get_target_Pdc for backwards compatibility"""
    return get_target_Pdc(window_tar, dropnight)

def get_data():
    """Wrapper function for DataManager.get_data()"""
    return _get_manager().get_data()
