"""Visualize OLS, Ridge, Lasso predictions vs actual values."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Load forecast data for a specific horizon
horizon = "30min"
target = "BOTH"

df = pd.read_hdf(f"forecasts/linear/forecasts_{horizon}_{target}.h5", key="df")
df_test = df[df["dataset"] == "Test"]

actual = df_test[f"Pdc_{target}_actual"]
ols = df_test[f"Pdc_{target}_ols"]
ridge = df_test[f"Pdc_{target}_ridge"]
lasso = df_test[f"Pdc_{target}_lasso"]
sp = df_test[f"Pdc_{target}_sp"]

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Perfect prediction line
max_val = max(actual.max(), ols.max()) * 1.1
line = [0, max_val]

# Plot 1: OLS vs Actual
ax = axes[0, 0]
ax.scatter(actual, ols, alpha=0.3, s=5)
ax.plot(line, line, 'r--', label='Perfect prediction')
ax.set_xlabel('Actual [W]')
ax.set_ylabel('Predicted [W]')
ax.set_title(f'OLS - {horizon} {target}')
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend()

# Plot 2: Ridge vs Actual
ax = axes[0, 1]
ax.scatter(actual, ridge, alpha=0.3, s=5)
ax.plot(line, line, 'r--', label='Perfect prediction')
ax.set_xlabel('Actual [W]')
ax.set_ylabel('Predicted [W]')
ax.set_title(f'Ridge - {horizon} {target}')
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend()

# Plot 3: Lasso vs Actual
ax = axes[1, 0]
ax.scatter(actual, lasso, alpha=0.3, s=5)
ax.plot(line, line, 'r--', label='Perfect prediction')
ax.set_xlabel('Actual [W]')
ax.set_ylabel('Predicted [W]')
ax.set_title(f'Lasso - {horizon} {target}')
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend()

# Plot 4: Smart Persistence vs Actual
ax = axes[1, 1]
ax.scatter(actual, sp, alpha=0.3, s=5)
ax.plot(line, line, 'r--', label='Perfect prediction')
ax.set_xlabel('Actual [W]')
ax.set_ylabel('Predicted [W]')
ax.set_title(f'Smart Persistence - {horizon} {target}')
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend()

plt.tight_layout()
plt.savefig(f"figures/model_comparison_{horizon}_{target}.png", dpi=150)
plt.show()

print(f"\nSaved to: figures/model_comparison_{horizon}_{target}.png")

# Also show error distribution
fig, ax = plt.subplots(figsize=(10, 6))

errors_ols = (ols - actual).values
errors_ridge = (ridge - actual).values
errors_lasso = (lasso - actual).values
errors_sp = (sp - actual).values

ax.hist(errors_ols, bins=50, alpha=0.5, label=f'OLS (MAE={np.mean(np.abs(errors_ols)):.0f}W)')
ax.hist(errors_lasso, bins=50, alpha=0.5, label=f'Lasso (MAE={np.mean(np.abs(errors_lasso)):.0f}W)')
ax.hist(errors_sp, bins=50, alpha=0.5, label=f'Smart Pers. (MAE={np.mean(np.abs(errors_sp)):.0f}W)')

ax.set_xlabel('Prediction Error [W]')
ax.set_ylabel('Count')
ax.set_title(f'Error Distribution - {horizon} {target}')
ax.legend()
ax.axvline(x=0, color='black', linestyle='--')

plt.tight_layout()
plt.savefig(f"figures/error_distribution_{horizon}_{target}.png", dpi=150)
plt.show()

print(f"Saved to: figures/error_distribution_{horizon}_{target}.png")
