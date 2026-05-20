"""ARIMAX model for PV power forecasting with exogenous variables."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from data_management import DataManager

# Output directories
METRICS_DIR = Path(__file__).parent.parent.parent / "metrics" / "timeseries"
FORECASTS_DIR = Path(__file__).parent.parent.parent / "forecasts" / "timeseries"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
np.random.seed(42)


def filter_daytime(df, start_hour=6, end_hour=18):
    """Filter dataframe to daytime hours only."""
    hours = pd.to_datetime(df.index).hour
    mask = (hours >= start_hour) & (hours < end_hour)
    return df[mask]


def run_arima_forecast(order=(12, 1, 1), horizons=[6, 12, 18, 24, 30, 36], sample_step=50, use_exog=True):
    """
    Run ARIMAX forecast with exogenous variables and save predictions.

    Args:
        order: (p, d, q) - AR order, differencing, MA order
        horizons: Forecast horizons in 5-min steps (6=30min, 12=60min, etc.)
        sample_step: Evaluate every Nth sample (default 50 for speed)
        use_exog: If True, use exogenous variables (GHI, kt, El, ENI)

    Returns:
        DataFrame with metrics per horizon
    """
    print("="*60)
    print("ARIMAX MODEL TRAINING & EVALUATION")
    print("="*60)
    print(flush=True)

    print("Loading data...")
    feat = DataManager()

    # Get timestamps
    train_timestamps = pd.to_datetime(feat.train["t"])
    test_timestamps = pd.to_datetime(feat.test["t"])

    # Get power data in Watts
    train_P = feat.train["Pdc_33"].ffill().fillna(0)
    test_P = feat.test["Pdc_33"].ffill().fillna(0)

    # Get exogenous variables
    # Split into: persistence (unknown future) vs known (calculable future)
    persistence_cols = ["GHI", "kt"]           # Use current value for future
    known_future_cols = ["CSGHI", "El", "ENI"] # Use actual future values
    exog_cols = persistence_cols + known_future_cols

    if use_exog:
        print(f"Exogenous variables: {exog_cols}")
        print(f"  - Persistence (unknown future): {persistence_cols}")
        print(f"  - Known future (calculable):    {known_future_cols}")
        train_exog = feat.train[exog_cols].ffill().fillna(0)
        test_exog = feat.test[exog_cols].ffill().fillna(0)

    # Add hour column for filtering
    train_hours = train_timestamps.dt.hour
    test_hours = test_timestamps.dt.hour

    # Filter to daytime (6:00 - 18:00)
    print("Applying daytime filter (6:00-18:00)...", flush=True)
    train_mask = (train_hours >= 6) & (train_hours < 18)
    test_mask = (test_hours >= 6) & (test_hours < 18)

    train_P = train_P[train_mask.values]
    test_P = test_P[test_mask.values]
    if use_exog:
        train_exog = train_exog[train_mask.values]
        test_exog = test_exog[test_mask.values]

    print(f"  Train: {train_mask.sum()} daytime samples")
    print(f"  Test: {test_mask.sum()} daytime samples")

    print(f"ARIMA order: {order}")
    print(flush=True)

    # Fit ARIMAX on training data
    print(f"Fitting ARIMAX{order} model...", flush=True)
    warnings.filterwarnings('ignore')

    if use_exog:
        model = ARIMA(train_P.values, order=order, exog=train_exog.values)
    else:
        model = ARIMA(train_P.values, order=order)

    model_fit = model.fit()
    print("Model fitted.", flush=True)

    results = []
    all_forecasts = {}
    max_h = max(horizons)

    # For rolling forecast, we need contiguous test data
    # Reset index for clean iteration
    test_P = test_P.reset_index(drop=True)
    if use_exog:
        test_exog = test_exog.reset_index(drop=True)

    # Calculate test indices with sample_step for speed
    test_start = 0
    test_end = len(test_P) - max_h
    test_indices = list(range(test_start, test_end, sample_step))
    n_points = len(test_indices)

    print(f"\nEvaluating {n_points} test points (sample_step={sample_step})...", flush=True)

    # Store predictions per horizon
    horizon_predictions = {h: [] for h in horizons}
    horizon_actuals = {h: [] for h in horizons}
    horizon_sp = {h: [] for h in horizons}

    # Use a simpler approach: fit once, forecast from different starting points
    # by using the forecast method with dynamic prediction

    for i, t in enumerate(test_indices):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_points} ({100*i/n_points:.0f}%)", flush=True)

        try:
            # Get current values
            P_current = test_P.iloc[t]

            if use_exog:
                # Build future exogenous:
                # - Persistence for GHI, kt (unknown future)
                # - Actual future values for CSGHI, El, ENI (calculable)
                future_exog = np.zeros((max_h, len(exog_cols)))

                for h in range(max_h):
                    future_idx = t + h + 1
                    if future_idx < len(test_exog):
                        for col_idx, col in enumerate(exog_cols):
                            if col in persistence_cols:
                                # Use current value (persistence)
                                future_exog[h, col_idx] = test_exog.iloc[t][col]
                            else:
                                # Use actual future value (known/calculable)
                                future_exog[h, col_idx] = test_exog.iloc[future_idx][col]
                    else:
                        # Fallback to current values if out of bounds
                        future_exog[h, :] = test_exog.iloc[t].values

                # Refit model with train + test up to point t for better accuracy
                # But this is slow, so we use a simpler approach:
                # Just use the trained model and append test data
                if t == 0:
                    current_model = model_fit
                else:
                    # Extend with test data up to t
                    extend_P = test_P.iloc[:t].values
                    extend_exog = test_exog.iloc[:t].values
                    current_model = model_fit.apply(extend_P, exog=extend_exog)

                forecast = current_model.forecast(steps=max_h, exog=future_exog)
            else:
                if t == 0:
                    current_model = model_fit
                else:
                    extend_P = test_P.iloc[:t].values
                    current_model = model_fit.apply(extend_P)

                forecast = current_model.forecast(steps=max_h)

            # Get ENI for Smart Persistence
            ENI_current = test_exog.iloc[t]["ENI"] if use_exog else 1.0

            # Collect predictions for each horizon
            for h in horizons:
                pred = max(0, forecast[h-1])  # Clip to non-negative
                actual = test_P.iloc[t + h]

                # Smart Persistence: P_sp = P(t) * ENI(t+h) / ENI(t)
                if use_exog:
                    ENI_future = test_exog.iloc[t + h]["ENI"]
                    if ENI_current > 0:
                        sp_pred = P_current * (ENI_future / ENI_current)
                    else:
                        sp_pred = P_current
                else:
                    sp_pred = P_current  # Naive persistence if no exog

                horizon_predictions[h].append(pred)
                horizon_actuals[h].append(actual)
                horizon_sp[h].append(sp_pred)

        except Exception as e:
            # Skip problematic points
            continue

    print(f"  Progress: {n_points}/{n_points} (100%)", flush=True)

    # Calculate metrics for each horizon
    for h in horizons:
        horizon_min = h * 5
        print(f"\nHorizon {horizon_min} min...", end=" ", flush=True)

        predictions = np.array(horizon_predictions[h])
        actuals = np.array(horizon_actuals[h])
        sp_predictions = np.array(horizon_sp[h])

        if len(predictions) == 0:
            print("No predictions")
            continue

        print(f"({len(predictions)} samples)", end=" ", flush=True)

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
            'model': 'ARIMAX' if use_exog else 'ARIMA',
            'horizon_min': horizon_min,
            'RMSE': rmse,
            'RMSE_sp': rmse_sp,
            'Skill': skill,
            'MAE': mae,
            'MBE': mbe,
            'n_samples': len(predictions)
        })

        print(f"RMSE={rmse:.0f}, Skill={skill:.1f}%", flush=True)

    # Save metrics
    df_metrics = pd.DataFrame(results)
    metrics_file = METRICS_DIR / "arima_results.csv"
    df_metrics.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")

    # Save forecasts
    forecasts_file = FORECASTS_DIR / "arima_forecasts.npz"
    np.savez(forecasts_file, **{
        f"{k}_{sub}": v[sub]
        for k, v in all_forecasts.items()
        for sub in ['actual', 'predicted', 'smart_persistence']
    })
    print(f"Forecasts saved to {forecasts_file}")

    # Summary
    print("\n" + "="*60)
    print("ARIMAX MODEL SUMMARY" if use_exog else "ARIMA MODEL SUMMARY")
    print("="*60)
    print(f"{'Horizon':<12} {'RMSE':<10} {'Skill':<10} {'MAE':<10}")
    print("-"*42)
    for r in results:
        print(f"{r['horizon_min']} min{'':<6} {r['RMSE']:<10.0f} {r['Skill']:<10.1f}% {r['MAE']:<10.0f}")
    print("-"*42)
    print(f"{'Average':<12} {df_metrics['RMSE'].mean():<10.0f} {df_metrics['Skill'].mean():<10.1f}% {df_metrics['MAE'].mean():<10.0f}")

    return df_metrics


if __name__ == "__main__":
    # Run ARIMAX with exogenous variables (GHI, kt, El, ENI)
    # Future exogenous values use persistence: X(t+h) = X(t)
    # sample_step=50 for faster evaluation
    df = run_arima_forecast(order=(12, 1, 1), use_exog=True, sample_step=50)
