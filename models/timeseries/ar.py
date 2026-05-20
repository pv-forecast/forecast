"""AutoRegression model for PV power forecasting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from data_management import DataManager

# Output directories
METRICS_DIR = Path(__file__).parent.parent.parent / "metrics" / "timeseries"
FORECASTS_DIR = Path(__file__).parent.parent.parent / "forecasts" / "timeseries"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
np.random.seed(42)


def run_ar_forecast(lags=12, horizons=[6, 12, 18, 24, 30, 36]):
    """
    Run AutoRegression forecast and save predictions.

    Args:
        lags: Number of lag observations (12 = 1 hour of 5-min intervals)
        horizons: Forecast horizons in 5-min steps (6=30min, 12=60min, etc.)

    Returns:
        DataFrame with metrics per horizon
    """
    print("="*60)
    print("AR MODEL TRAINING & EVALUATION")
    print("="*60)

    print("\nLoading data...")
    feat = DataManager()

    # Get power data in Watts (same train/test split as other models)
    train_data = feat.train["Pdc_33"].ffill().dropna()
    test_data = feat.test["Pdc_33"].ffill().dropna()

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"AR lags: {lags} (= {lags*5} min)")

    # Fit AR model on training data
    print(f"\nFitting AR({lags}) model...")
    model = AutoReg(train_data.values, lags=lags)
    model_fit = model.fit()

    # Get AR coefficients for manual prediction
    params = model_fit.params
    intercept = params[0]
    ar_coefs = params[1:]

    results = []
    all_forecasts = {}

    # Combine history for prediction
    history = np.concatenate([train_data.values[-lags:], test_data.values])

    for h in horizons:
        horizon_min = h * 5
        print(f"\nHorizont {horizon_min} min...", end=" ")

        predictions = []
        actuals = []
        sp_predictions = []  # Smart Persistence

        # Predict for all test samples
        for t in range(lags, len(history) - h):
            # Multi-step AR prediction
            pred_seq = list(history[t-lags:t])
            for step in range(h):
                pred = intercept + np.dot(ar_coefs, pred_seq[-lags:][::-1])
                pred_seq.append(pred)

            predictions.append(pred_seq[-1])
            actuals.append(history[t + h])
            sp_predictions.append(history[t])  # Smart Persistence: current value

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        sp_predictions = np.array(sp_predictions)

        print(f"({len(predictions)} samples)", end=" ")

        # Store forecasts
        all_forecasts[f'{horizon_min}min'] = {
            'actual': actuals,
            'predicted': predictions,
            'smart_persistence': sp_predictions
        }

        # Calculate metrics
        error = actuals - predictions
        error_sp = actuals - sp_predictions

        rmse = np.sqrt(np.nanmean(error ** 2))
        rmse_sp = np.sqrt(np.nanmean(error_sp ** 2))
        mae = np.nanmean(np.abs(error))
        mbe = np.nanmean(error)
        skill = (1 - rmse / rmse_sp) * 100 if rmse_sp > 0 else 0

        results.append({
            'model': 'AR',
            'horizon_min': horizon_min,
            'RMSE': rmse,
            'RMSE_sp': rmse_sp,
            'Skill': skill,
            'MAE': mae,
            'MBE': mbe,
            'n_samples': len(predictions)
        })

        print(f"RMSE={rmse:.0f}, Skill={skill:.1f}%")

    # Save metrics
    df_metrics = pd.DataFrame(results)
    metrics_file = METRICS_DIR / "ar_results.csv"
    df_metrics.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")

    # Save forecasts
    forecasts_file = FORECASTS_DIR / "ar_forecasts.npz"
    np.savez(forecasts_file, **{
        f"{k}_{sub}": v[sub]
        for k, v in all_forecasts.items()
        for sub in ['actual', 'predicted', 'smart_persistence']
    })
    print(f"Forecasts saved to {forecasts_file}")

    # Summary
    print("\n" + "="*60)
    print("AR MODEL SUMMARY")
    print("="*60)
    print(f"{'Horizont':<12} {'RMSE':<10} {'Skill':<10} {'MAE':<10}")
    print("-"*42)
    for r in results:
        print(f"{r['horizon_min']} min{'':<6} {r['RMSE']:<10.0f} {r['Skill']:<10.1f}% {r['MAE']:<10.0f}")
    print("-"*42)
    print(f"{'Durchschnitt':<12} {df_metrics['RMSE'].mean():<10.0f} {df_metrics['Skill'].mean():<10.1f}% {df_metrics['MAE'].mean():<10.0f}")

    return df_metrics


if __name__ == "__main__":
    df = run_ar_forecast(lags=12)
