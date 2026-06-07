"""Plot Skill Score vs Forecast Horizon for all models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Horizons
horizons = [30, 60, 90, 120, 150, 180]

# ============ ARIMAX ============
arimax = pd.read_csv(PROJECT_ROOT / 'metrics/timeseries/arima_results.csv')
arimax_skill = arimax['Skill'].values
arimax_rmse = arimax['RMSE'].values
arimax_rmse_sp = arimax['RMSE_sp'].values

# ============ LSTM ============
lstm = pd.read_csv(PROJECT_ROOT / 'metrics/lstm/LSTM_m2m_Layer_2_Input_17_hidden_75_future_El.csv')

# Parse all rows into arrays
all_rmse = []
all_rmse_sp = []
for idx, row in lstm.iterrows():
    rmse_arr = np.fromstring(row['RMSE'].strip('[]'), sep=' ')
    rmse_sp_arr = np.fromstring(row['RMSE_sp'].strip('[]'), sep=' ')
    all_rmse.append(rmse_arr)
    all_rmse_sp.append(rmse_sp_arr)

all_rmse = np.array(all_rmse)
all_rmse_sp = np.array(all_rmse_sp)

# Average across epochs
lstm_rmse_mean = np.nanmean(all_rmse, axis=0)
lstm_rmse_sp_mean = np.nanmean(all_rmse_sp, axis=0)
lstm_skill_all = (1 - lstm_rmse_mean / lstm_rmse_sp_mean) * 100

# Extract specific horizons (indices 5, 11, 17, 23, 29, 35)
horizon_indices = [5, 11, 17, 23, 29, 35]
lstm_skill = [lstm_skill_all[i] for i in horizon_indices]
lstm_rmse = [lstm_rmse_mean[i] for i in horizon_indices]
lstm_rmse_sp = [lstm_rmse_sp_mean[i] for i in horizon_indices]

# ============ LINEAR ============
linear = pd.read_hdf(PROJECT_ROOT / 'metrics/linear/results_BOTH_intra-hour.h5', key='df')
linear_ols = linear[(linear['dataset'] == 'Test') & (linear['model'] == 'ols')].copy()
linear_ols['horizon_min'] = linear_ols['horizon'].str.replace('min', '').astype(int)
linear_filtered = linear_ols[linear_ols['horizon_min'].isin(horizons)].sort_values('horizon_min')
linear_skill = (linear_filtered['skill'] * 100).values
linear_rmse = linear_filtered['RMSE'].values
linear_rmse_sp = linear_filtered['RMSE_p'].values

# ============ PLOT ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Skill vs Horizon
ax1 = axes[0]
ax1.plot(horizons, arimax_skill, 'o-', color='purple', linewidth=2, markersize=8, label='ARIMAX')
ax1.plot(horizons, lstm_skill, 's-', color='green', linewidth=2, markersize=8, label='LSTM')
ax1.plot(horizons, linear_skill, '^-', color='blue', linewidth=2, markersize=8, label='Linear (OLS)')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline (Smart Persistence)')

ax1.set_xlabel('Forecast Horizon [min]', fontsize=12)
ax1.set_ylabel('Skill Score [%]', fontsize=12)
ax1.set_title('Skill Score vs Forecast Horizon', fontsize=14)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(horizons)

# Right plot: RMSE comparison (Model vs Smart Persistence)
ax2 = axes[1]
ax2.plot(horizons, arimax_rmse_sp, '--', color='red', linewidth=2, label='Smart Persistence (Baseline)')
ax2.plot(horizons, arimax_rmse, 'o-', color='purple', linewidth=2, markersize=8, label='ARIMAX')
ax2.plot(horizons, lstm_rmse, 's-', color='green', linewidth=2, markersize=8, label='LSTM')
ax2.plot(horizons, linear_rmse, '^-', color='blue', linewidth=2, markersize=8, label='Linear (OLS)')

ax2.set_xlabel('Forecast Horizon [min]', fontsize=12)
ax2.set_ylabel('RMSE [W]', fontsize=12)
ax2.set_title('RMSE vs Forecast Horizon', fontsize=14)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(horizons)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'skill_vs_horizon.png', dpi=150, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'skill_vs_horizon.png'}")

# Print summary table
print("\n" + "="*70)
print("SKILL SCORE COMPARISON")
print("="*70)
print(f"{'Horizon':<12} {'ARIMAX':<12} {'LSTM':<12} {'Linear':<12} {'SP RMSE':<12}")
print("-"*70)
for i, h in enumerate(horizons):
    print(f"{h} min{'':<6} {arimax_skill[i]:<12.1f} {lstm_skill[i]:<12.1f} {linear_skill[i]:<12.1f} {arimax_rmse_sp[i]:<12.0f}")
print("-"*70)
print(f"{'Average':<12} {np.mean(arimax_skill):<12.1f} {np.mean(lstm_skill):<12.1f} {np.mean(linear_skill):<12.1f}")
