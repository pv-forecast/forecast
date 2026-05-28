import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import pandas as pd
import statsmodels as stat
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot, lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from data_management import DataManager

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def get_data():
    """Wrapper for backwards compatibility."""
    return DataManager().get_data()

# GHI - Global Horizontal Irradiance [W/m²], data[8]
# BNI - Beam Normal Irradiance (modeled) [W/m²], data[23]
# DHI - Diffuse Horizontal Irradiance (modeled) [W/m²], data[22]
# gti30t187a - Global Irradiance on a tilted plane (30° tilt, 187° azimuth) [W/m²], data[9]
# ENI - Extraterrestrial Normal Irradiance [W/m²], data[19]
# 
# CSGHI, CSBNI, CSDHI - (modeled) Clear-sky irradiances [W/m²], data[14], data[12], data[13]
# TL - (modeled) Linke Turbidity Factor, data[20]
# CS - boolean flag (modeled), does sky seem clear? (from GHI), data[18]
#
# Ta - Ambient temperature (°C), data[5]
# vw - wind speed [m/s], data[7]
# wdir - wind direction (degrees), data[6]
# Patm - ambient pressure [Pa], data[1]
# RH - relative humidity [0-1], data[4]
# tpw - Total Precipitable Water [mm], data[15]
# 
# AMa - Absolute Air-Mass (equivalent clean atmospheres), data[21]
# kd - Diffuse fraction kd = DHI/GHI, data[25]
# kt - Clearness Index kt = GHI/(ENI·cos(zenith)), data[24]
# 
# Az - Solar azimuth angle (degrees) - convention: Equator = 90°, positive around zenith., data[27]
# El - Apparent Solar elevation angle (degrees), considering atmospheric refraction, data[28]
# w - Solar hour angle (degrees), data[29]
# dec - Solar declination angle (degrees), data[30]
# 
# Pdc - Measured plant output (mean of 40 monitored inverters) [W], data[32]
# PR - Performance Ratio: output / nominal installed power (W/Wp), data[31]

# time - data[0]
# windir - , data[10]

# Import testing data

data = get_data()

time = data.t
gti30t187a = data.gti30t187a
GHI = data.GHI
Ta = data.Ta
BNI = data.BNI
Pdc = data.Pdc_1
Pdcmean = data.iloc[:, 109:].mean(axis=1)
DHI = data.DHI
ENI = data.ENI
CSGHI = data.CSGHI
CSBNI = data.CSBNI
CSDHI = data.CSDHI
TL = data.TL
CS = data.CS
vw = data.vw
wdir = data.wdir
Patm = data.Patm
RH = data.RH
tpw = data.tpw
AMa = data.AMa
kd = data.kd
kt = data.kt
Az = data.Az
El = data.El
w = data.w
dec = data.dec

Indata_nlist = []

Pdcmean = Pdcmean.fillna(0.01)
Pdc = Pdc.fillna(0.01)
gti30t187a = gti30t187a.fillna(0.01)
GHI = GHI.fillna(0.01)
Ta = Ta.fillna(0.01)
BNI = BNI.fillna(0.01)
DHI = DHI.fillna(0.01)
ENI = ENI.fillna(0.01)
CSGHI = CSGHI.fillna(0.01)
CSBNI = CSBNI.fillna(0.01)
CSDHI = CSDHI.fillna(0.01)
TL = TL.fillna(0.01)
CS = CS.fillna(0.01)
vw = vw.fillna(0.01)
wdir = wdir.fillna(0.01)
Patm = Patm.fillna(0.01)
RH = RH.fillna(0.01)
tpw = tpw.fillna(0.01)
AMa = AMa.fillna(0.01)
kd = kd.fillna(0.01)
kt = kt.fillna(0.01)
Az = Az.fillna(0.01)
El = El.fillna(0.01)
w = w.fillna(0.01)
dec = dec.fillna(0.01)

Pdc_norm = Pdc/max(Pdc)
gti30t187a_norm = gti30t187a/max(gti30t187a)
GHI_norm = GHI/max(GHI)
Ta_norm = Ta/max(Ta)
BNI_norm = BNI/max(BNI)
DHI_norm = DHI/max(DHI)
ENI_norm  = ENI/max(ENI)
# CSGHI_norm  = CSGHI/max(CSGHI)
# CSBNI_norm  = CSBNI/max(CSGHI)
# CSDHI_norm  = CSDHI/max(CSDHI)
# TL_norm  = TL/max(TL)
# CS_norm  = CS/max(CS)
# vw_norm  = vw/max(vw)
# wdir_norm  = wdir/max(wdir)
# Patm_norm  = Patm/max(Patm)
# RH_norm  = RH/max(RH)
# tpw_norm  = tpw/max(tpw)
# AMa_norm  = AMa/max(AMa)
# kd_norm  = kd/max(kd)
# kt_norm  = kt/max(kt)
# Az_norm  = Az/max(Az)
# El_norm  = El/max(El)
# w_norm  = w/max(w)
# dec_norm  = dec/max(dec)

# data analysis

# plt.figure()
# plt.plot(time[0:28293], Pdc_norm[0:28293])
# plt.figure()
# plt.plot(time[28294:56587], Pdc_norm[28294:56587])
# plt.figure()
# plt.plot(time[56588:84879], Pdc_norm[56588:84879])
# plt.figure()
# plt.plot(time[84880:len(gti30t187a_norm)], Pdc_norm[84880:len(gti30t187a_norm)])
# plt.figure()
autocorrelation_plot(Pdc_norm)
plt.suptitle('Autocorrelation Pdc (single)')
plt.savefig(os.path.join(FIGURES_DIR, 'Autocorrelation_Pdc_single'))
#plt.figure()
plot_acf(Pdc_norm, lags=100)
plt.suptitle('ACF Pdc (single)')
plt.savefig(os.path.join(FIGURES_DIR, 'ACF_Pdc_single'))
plt.show()

plt.figure(figsize=(15,5))
plt.subplot(151)
plt.plot(gti30t187a_norm, Pdc_norm, 'ro')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('gti30t187a')
plt.ylabel('Pdc')
plt.subplot(152)
plt.plot(GHI_norm, Pdc_norm, 'go')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('GHI')
plt.subplot(153)
plt.plot(BNI_norm, Pdc_norm, 'bo')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('BNI')
plt.subplot(154)
plt.plot(ENI_norm, Pdc_norm, 'yo')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('ENI')
plt.subplot(155)
plt.plot(DHI_norm, Pdc_norm, 'mo')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('DNI')
#plt.show()
#plt.savefig('Irradiation_OneYear')

# plt.subplot(153)
# plt.boxplot(gti30t187a, notch=1)
# plt.suptitle('Normalized Data')