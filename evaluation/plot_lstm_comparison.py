"""Compare Baseline vs Improved LSTM performance."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
METRICS_DIR = PROJECT_ROOT / "metrics/lstm"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Load metrics
baseline = pd.read_csv(METRICS_DIR / "LSTM_m2m_Layer_2_Input_17_hidden_75_baseline.csv")
improved = pd.read_csv(METRICS_DIR / "LSTM_m2m_Layer_3_Input_17_hidden_128_improved.csv")

# Calculate skill
baseline['Skill'] = (1 - baseline['RMSE'] / baseline['RMSE_sp']) * 100
improved['Skill'] = (1 - improved['RMSE'] / improved['RMSE_sp']) * 100

# Horizons
horizons = [30, 60, 90, 120, 150, 180]
horizon_indices = [5, 11, 17, 23, 29, 35]

# Extract values
baseline_skill = [baseline.iloc[i]['Skill'] for i in horizon_indices]
improved_skill = [improved.iloc[i]['Skill'] for i in horizon_indices]
baseline_rmse = [baseline.iloc[i]['RMSE'] for i in horizon_indices]
improved_rmse = [improved.iloc[i]['RMSE'] for i in horizon_indices]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Skill comparison
ax1 = axes[0]
x = np.arange(len(horizons))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_skill, width, label='Baseline (2L, 75h, d=0.5)', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, improved_skill, width, label='Improved (3L, 128h, d=0.2)', color='forestgreen', alpha=0.8)

ax1.set_xlabel('Forecast Horizon [min]', fontsize=12)
ax1.set_ylabel('Skill Score [%]', fontsize=12)
ax1.set_title('Skill Score: Baseline vs Improved LSTM', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(horizons)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# RMSE comparison
ax2 = axes[1]
ax2.plot(horizons, baseline_rmse, 'o-', color='steelblue', linewidth=2, markersize=8, label='Baseline')
ax2.plot(horizons, improved_rmse, 's-', color='forestgreen', linewidth=2, markersize=8, label='Improved')

ax2.set_xlabel('Forecast Horizon [min]', fontsize=12)
ax2.set_ylabel('RMSE [W]', fontsize=12)
ax2.set_title('RMSE: Baseline vs Improved LSTM', fontsize=14)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(horizons)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lstm_baseline_vs_improved.png', dpi=150, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'lstm_baseline_vs_improved.png'}")

# Print summary table
print("\n" + "="*60)
print("LSTM COMPARISON: Baseline vs Improved")
print("="*60)
print(f"\n{'Horizon':<10} {'Baseline':<12} {'Improved':<12} {'Diff':<10}")
print("-"*44)
for i, h in enumerate(horizons):
    diff = improved_skill[i] - baseline_skill[i]
    print(f"{h} min{'':<4} {baseline_skill[i]:<12.1f} {improved_skill[i]:<12.1f} {diff:+.1f}%")
print("-"*44)
print(f"{'Average':<10} {np.mean(baseline_skill):<12.1f} {np.mean(improved_skill):<12.1f} {np.mean(improved_skill)-np.mean(baseline_skill):+.1f}%")
print("\nModel Parameters:")
print("  Baseline: 76,536 (2 layers, 75 hidden, dropout=0.5)")
print("  Improved: 344,100 (3 layers, 128 hidden, dropout=0.2)")
