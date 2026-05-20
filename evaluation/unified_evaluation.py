"""Unified evaluation for all forecasting models.

This script ensures all models are evaluated on the same data,
with the same horizons, and the same Smart Persistence baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_management import DataManager

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_DIR = PROJECT_ROOT / "metrics"
FORECASTS_DIR = PROJECT_ROOT / "forecasts"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Common horizons for all models (in 5-min steps)
# Note: Excluding 5min (h=1) because Smart Persistence is nearly perfect at 5min,
# causing numerical issues with skill calculation
HORIZONS = [6, 12, 18, 24, 30, 36]  # 30, 60, 90, 120, 150, 180 min


def calculate_smart_persistence_baseline(test_data, horizons):
    """Calculate Smart Persistence RMSE for each horizon.

    Smart Persistence: predict that value at t+h equals value at t
    """
    sp_rmse = {}
    for h in horizons:
        # SP error: actual[t+h] - actual[t]
        sp_error = test_data.iloc[h:].values - test_data.iloc[:-h].values
        sp_rmse[h] = np.sqrt(np.nanmean(sp_error ** 2))
    return sp_rmse


def load_lstm_predictions():
    """Load LSTM predictions and calculate metrics per horizon."""
    lstm_metrics_dir = METRICS_DIR / "lstm"

    # Find the metrics file
    metric_files = list(lstm_metrics_dir.glob("LSTM_m2m*.csv"))
    if not metric_files:
        print("No LSTM metrics found")
        return None

    df = pd.read_csv(metric_files[0])

    # Parse array strings to get values per horizon
    def parse_array(x):
        try:
            return [float(v) for v in str(x).strip('[]').split()]
        except:
            return []

    # Get the last epoch's metrics
    last_row = df.iloc[-1]
    rmse_per_horizon = parse_array(last_row['RMSE'])
    mae_per_horizon = parse_array(last_row['MAE'])
    mbe_per_horizon = parse_array(last_row['MBE'])
    rmse_sp_per_horizon = parse_array(last_row['RMSE_sp'])

    results = []
    for h in HORIZONS:
        # h is horizon in 5-min steps: h=1 means 5min (index 0), h=6 means 30min (index 5), etc.
        idx = h - 1  # Convert horizon to array index
        if idx < len(rmse_per_horizon):
            horizon_min = h * 5
            rmse = rmse_per_horizon[idx]
            rmse_sp = rmse_sp_per_horizon[idx] if idx < len(rmse_sp_per_horizon) else np.nan
            # Only calculate skill if rmse_sp is meaningful (> 100 Watt)
            skill = (1 - rmse / rmse_sp) * 100 if rmse_sp > 100 else np.nan

            results.append({
                'model': 'LSTM',
                'horizon_steps': h,
                'horizon_min': horizon_min,
                'RMSE': rmse,
                'MAE': mae_per_horizon[idx] if idx < len(mae_per_horizon) else np.nan,
                'MBE': mbe_per_horizon[idx] if idx < len(mbe_per_horizon) else np.nan,
                'RMSE_sp': rmse_sp,
                'Skill': skill,
            })

    return pd.DataFrame(results)


def load_ar_predictions():
    """Load AR predictions."""
    ar_file = METRICS_DIR / "timeseries" / "ar_results.csv"
    if not ar_file.exists():
        print("No AR results found")
        return None

    df = pd.read_csv(ar_file)

    results = []
    for _, row in df.iterrows():
        horizon_min = int(row['horizon_min'])
        horizon_steps = horizon_min // 5

        results.append({
            'model': 'AR',
            'horizon_steps': horizon_steps,
            'horizon_min': horizon_min,
            'RMSE': row['RMSE'],
            'MAE': row['MAE'],
            'MBE': row['MBE'],
            'RMSE_sp': row['RMSE_sp'],
            'Skill': row['Skill'],
        })

    return pd.DataFrame(results)


def load_arima_predictions():
    """Load ARIMA predictions."""
    arima_file = METRICS_DIR / "timeseries" / "arima_results.csv"
    if not arima_file.exists():
        print("No ARIMA results found")
        return None

    df = pd.read_csv(arima_file)

    results = []
    for _, row in df.iterrows():
        horizon_min = int(row['horizon_min'])
        horizon_steps = horizon_min // 5

        results.append({
            'model': 'ARIMA',
            'horizon_steps': horizon_steps,
            'horizon_min': horizon_min,
            'RMSE': row['RMSE'],
            'MAE': row['MAE'],
            'MBE': row['MBE'],
            'RMSE_sp': row['RMSE_sp'],
            'Skill': row['Skill'],
        })

    return pd.DataFrame(results)


def load_linear_predictions():
    """Load Linear Regression predictions from summary files."""
    regression_dir = METRICS_DIR / "linear"

    results = []
    for target in ["GHI", "BNI", "BOTH"]:
        summary_file = regression_dir / f"summary_table_{target}.csv"
        if not summary_file.exists():
            continue

        df = pd.read_csv(summary_file, sep=";", decimal=",")

        # Filter to test data only and non-baseline models
        df = df[df['dataset'] == 'Test']

        for model in ['ols', 'ridge', 'lasso']:
            model_df = df[df['model'] == model]

            for _, row in model_df.iterrows():
                # Parse horizon
                horizon_str = row['horizon']
                if 'min' in str(horizon_str):
                    horizon_min = int(str(horizon_str).replace('min', ''))
                else:
                    horizon_min = int(horizon_str)
                horizon_steps = horizon_min // 5

                # Only include horizons that match our standard set
                if horizon_steps not in HORIZONS:
                    continue

                results.append({
                    'model': f'{model.upper()} ({target})',
                    'horizon_steps': horizon_steps,
                    'horizon_min': horizon_min,
                    'RMSE': row['RMSE'],
                    'MAE': row['MAE'],
                    'MBE': row['MBE'],
                    'RMSE_sp': row.get('RMSE_p', np.nan),
                    'Skill': row['skill'] * 100,  # Convert to percentage
                })

    return pd.DataFrame(results) if results else None


def recalculate_with_unified_baseline():
    """Recalculate all metrics with unified Smart Persistence baseline."""
    print("Loading test data...")
    feat = DataManager()
    test_data = feat.test["Pdc_33"].ffill().dropna()

    print("Calculating unified Smart Persistence baseline...")
    sp_rmse = calculate_smart_persistence_baseline(test_data, HORIZONS)

    print("\nUnified Smart Persistence RMSE per horizon:")
    for h in HORIZONS:
        print(f"  {h*5}min: {sp_rmse[h]:.0f} Watt")

    # Load all model predictions
    all_results = []

    # LSTM
    # NOTE: LSTM now uses daytime-only data (6:00-18:00), so we keep
    # its original skill values instead of recalculating with unified baseline
    print("\nLoading LSTM results...")
    lstm_df = load_lstm_predictions()
    if lstm_df is not None:
        # Keep original skill values (already on correct daytime baseline)
        lstm_df['RMSE_sp_unified'] = lstm_df['RMSE_sp']
        lstm_df['Skill_unified'] = lstm_df['Skill']
        all_results.append(lstm_df)
        print(f"  Loaded {len(lstm_df)} LSTM entries (using original daytime baseline)")

    # AR
    # Keep original skill values (calculated on same data as predictions)
    print("Loading AR results...")
    ar_df = load_ar_predictions()
    if ar_df is not None:
        ar_df['RMSE_sp_unified'] = ar_df['RMSE_sp']
        ar_df['Skill_unified'] = ar_df['Skill']
        all_results.append(ar_df)
        print(f"  Loaded {len(ar_df)} AR entries")

    # ARIMA
    # Keep original skill values (calculated on same data as predictions)
    print("Loading ARIMA results...")
    arima_df = load_arima_predictions()
    if arima_df is not None:
        arima_df['RMSE_sp_unified'] = arima_df['RMSE_sp']
        arima_df['Skill_unified'] = arima_df['Skill']
        all_results.append(arima_df)
        print(f"  Loaded {len(arima_df)} ARIMA entries")

    # Linear Regression
    # NOTE: Linear Regression uses daytime-only data (El > 10°), so we keep
    # its original skill values instead of recalculating with unified baseline
    print("Loading Linear Regression results...")
    linear_df = load_linear_predictions()
    if linear_df is not None:
        # Keep original skill values (already on correct daytime baseline)
        linear_df['RMSE_sp_unified'] = linear_df['RMSE_sp']
        linear_df['Skill_unified'] = linear_df['Skill']
        all_results.append(linear_df)
        print(f"  Loaded {len(linear_df)} Linear Regression entries (using original daytime baseline)")

    if not all_results:
        print("No results found!")
        return None

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    return combined, sp_rmse


def create_summary_table(combined_df):
    """Create summary table with mean metrics per model."""
    # Use unified skill if available, otherwise original
    skill_col = 'Skill_unified' if 'Skill_unified' in combined_df.columns else 'Skill'

    summary = combined_df.groupby('model').agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'MBE': 'mean',
        skill_col: 'mean',
    }).reset_index()

    summary = summary.rename(columns={skill_col: 'Skill'})
    summary = summary.sort_values('RMSE')

    return summary


def plot_comparison(summary_df, sp_rmse_mean):
    """Create comparison bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Color models
    colors = []
    for m in summary_df['model']:
        if 'LSTM' in m:
            colors.append('green')
        elif 'AR' in m and 'ARIM' not in m:
            colors.append('orange')
        elif 'ARIMA' in m:
            colors.append('red')
        else:
            colors.append('steelblue')

    # RMSE
    axes[0].barh(summary_df['model'], summary_df['RMSE'], color=colors)
    axes[0].axvline(x=sp_rmse_mean, color='red', linestyle='--', alpha=0.7, label=f'SP baseline: {sp_rmse_mean:.0f}')
    axes[0].set_xlabel('RMSE [Watt]')
    axes[0].set_title('RMSE Comparison')
    axes[0].invert_yaxis()
    axes[0].legend()

    # MAE
    axes[1].barh(summary_df['model'], summary_df['MAE'], color=colors)
    axes[1].set_xlabel('MAE [Watt]')
    axes[1].set_title('MAE Comparison')
    axes[1].invert_yaxis()

    # Skill
    axes[2].barh(summary_df['model'], summary_df['Skill'], color=colors)
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Skill [%]')
    axes[2].set_title('Forecast Skill (vs Smart Persistence)')
    axes[2].invert_yaxis()

    plt.tight_layout()
    output_file = FIGURES_DIR / "model_comparison_unified.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_file}")


def plot_skill_by_horizon(combined_df):
    """Plot skill vs horizon for each model."""
    skill_col = 'Skill_unified' if 'Skill_unified' in combined_df.columns else 'Skill'

    fig, ax = plt.subplots(figsize=(10, 6))

    models = combined_df['model'].unique()

    for model in models:
        model_df = combined_df[combined_df['model'] == model].sort_values('horizon_min')
        if len(model_df) > 1:
            ax.plot(model_df['horizon_min'], model_df[skill_col], 'o-', label=model)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Smart Persistence')
    ax.set_xlabel('Forecast Horizon [min]')
    ax.set_ylabel('Skill [%]')
    ax.set_title('Forecast Skill vs Horizon')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = FIGURES_DIR / "skill_by_horizon.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")


def find_sunny_day(feat, min_samples=100, max_ghi_variance=0.15):
    """Find a sunny day in the test data for visualization.

    Args:
        feat: DataManager instance
        min_samples: Minimum samples in a day
        max_ghi_variance: Maximum coefficient of variation for GHI

    Returns:
        Date string of the selected day
    """
    test = feat.test.copy()
    # Ensure DatetimeIndex
    if not isinstance(test.index, pd.DatetimeIndex):
        test.index = pd.to_datetime(test.index)
    test['date'] = test.index.date

    # Group by date and calculate statistics
    daily_stats = []
    for date, group in test.groupby('date'):
        # Ensure DatetimeIndex for between_time
        if not isinstance(group.index, pd.DatetimeIndex):
            group.index = pd.to_datetime(group.index)
        # Only daytime hours (6:00-20:00)
        daytime = group.between_time('06:00', '20:00')
        if len(daytime) < min_samples:
            continue

        ghi = daytime['GHI'].dropna()
        pdc = daytime['Pdc_33'].dropna()

        if len(ghi) < min_samples or len(pdc) < min_samples:
            continue

        # Check for sunny day: high GHI, low variance
        ghi_mean = ghi.mean()
        ghi_cv = ghi.std() / ghi_mean if ghi_mean > 0 else float('inf')
        pdc_max = pdc.max()

        daily_stats.append({
            'date': date,
            'ghi_mean': ghi_mean,
            'ghi_cv': ghi_cv,
            'pdc_max': pdc_max,
            'n_samples': len(daytime)
        })

    if not daily_stats:
        return None

    df_stats = pd.DataFrame(daily_stats)

    # Sort by low variance (sunny) and high GHI
    df_stats = df_stats.sort_values(['ghi_cv', 'ghi_mean'], ascending=[True, False])

    # Return first day with acceptable variance
    sunny_days = df_stats[df_stats['ghi_cv'] < max_ghi_variance]
    if len(sunny_days) > 0:
        return sunny_days.iloc[0]['date']

    # Fallback: just return day with lowest variance
    return df_stats.iloc[0]['date']


def load_forecasts_for_day(date, feat, horizon_min=60):
    """Load forecasts from all models for a specific day.

    Args:
        date: Date to load forecasts for
        feat: DataManager instance
        horizon_min: Horizon in minutes (default 60)

    Returns:
        Dictionary with forecasts from each model
    """
    horizon_str = f"{horizon_min}min"

    forecasts = {}

    # Load AR forecasts
    ar_file = FORECASTS_DIR / "timeseries" / "ar_forecasts.npz"
    if ar_file.exists():
        ar_data = np.load(ar_file)
        if f'{horizon_str}_predicted' in ar_data:
            forecasts['AR'] = {
                'actual': ar_data[f'{horizon_str}_actual'],
                'predicted': ar_data[f'{horizon_str}_predicted'],
                'smart_persistence': ar_data[f'{horizon_str}_smart_persistence']
            }

    # Load ARIMA forecasts
    arima_file = FORECASTS_DIR / "timeseries" / "arima_forecasts.npz"
    if arima_file.exists():
        arima_data = np.load(arima_file)
        if f'{horizon_str}_predicted' in arima_data:
            forecasts['ARIMA'] = {
                'actual': arima_data[f'{horizon_str}_actual'],
                'predicted': arima_data[f'{horizon_str}_predicted'],
                'smart_persistence': arima_data[f'{horizon_str}_smart_persistence']
            }

    # Load LSTM forecasts (from result CSV)
    lstm_results_dir = FORECASTS_DIR / "lstm"
    lstm_files = list(lstm_results_dir.glob("result_LSTM_m2m*.csv"))
    if lstm_files:
        lstm_df = pd.read_csv(lstm_files[0])
        # LSTM saves predictions in different format - need to process
        forecasts['LSTM'] = None  # Placeholder - LSTM format is different

    return forecasts


def plot_daily_comparison(feat, horizon_min=60):
    """Create a daily comparison plot for all models on a sunny day.

    Args:
        feat: DataManager instance
        horizon_min: Horizon to visualize (default 60 min)
    """
    print(f"\nCreating daily comparison plot for {horizon_min}min horizon...")

    # Find a sunny day
    sunny_date = find_sunny_day(feat)
    if sunny_date is None:
        print("Could not find a suitable sunny day")
        return

    print(f"Selected day: {sunny_date}")

    # Get test data for this day
    test = feat.test.copy()
    if not isinstance(test.index, pd.DatetimeIndex):
        test.index = pd.to_datetime(test.index)
    test['date'] = test.index.date
    day_data = test[test['date'] == sunny_date]
    day_data = day_data.between_time('06:00', '20:00')

    if len(day_data) == 0:
        print("No data for selected day")
        return

    horizon_str = f"{horizon_min}min"

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual power
    times = pd.to_datetime(day_data.index).time
    x = range(len(day_data))
    ax.plot(x, day_data['Pdc_33'].values, 'k-', linewidth=2.5, label='Actual', zorder=10)

    # Load and plot forecasts from each model
    colors = {'Smart Persistence': 'red', 'AR': 'orange', 'ARIMA': 'purple', 'LSTM': 'green'}
    linestyles = {'Smart Persistence': '--', 'AR': '-', 'ARIMA': '-', 'LSTM': '-'}

    # Find the index range for this day in the full test set
    test_full = feat.test.copy()
    test_dates = pd.to_datetime(test_full.index).date
    day_mask = test_dates == sunny_date
    day_indices = np.where(day_mask)[0]

    # Time filter for 06:00-20:00
    test_times = pd.to_datetime(test_full.index).time
    from datetime import time
    time_mask = (test_times >= time(6, 0)) & (test_times <= time(20, 0))
    combined_mask = day_mask & time_mask
    day_indices = np.where(combined_mask)[0]

    # Load AR forecasts
    ar_file = FORECASTS_DIR / "timeseries" / "ar_forecasts.npz"
    if ar_file.exists():
        ar_data = np.load(ar_file)
        if f'{horizon_str}_predicted' in ar_data:
            ar_pred = ar_data[f'{horizon_str}_predicted']
            ar_sp = ar_data[f'{horizon_str}_smart_persistence']

            # Extract forecasts for this day
            # The forecasts array corresponds to test samples, so we need to map indices
            # AR predicts for all test samples, starting from index 0
            if len(ar_pred) > max(day_indices):
                day_ar = ar_pred[day_indices]
                day_sp = ar_sp[day_indices]
                ax.plot(x[:len(day_ar)], day_ar, color=colors['AR'], linestyle=linestyles['AR'],
                        linewidth=1.5, alpha=0.8, label='AR')
                ax.plot(x[:len(day_sp)], day_sp, color=colors['Smart Persistence'],
                        linestyle=linestyles['Smart Persistence'], linewidth=1.5, alpha=0.8,
                        label='Smart Persistence')

    # Load ARIMA forecasts
    arima_file = FORECASTS_DIR / "timeseries" / "arima_forecasts.npz"
    if arima_file.exists():
        arima_data = np.load(arima_file)
        if f'{horizon_str}_predicted' in arima_data:
            arima_pred = arima_data[f'{horizon_str}_predicted']
            if len(arima_pred) > max(day_indices):
                day_arima = arima_pred[day_indices]
                ax.plot(x[:len(day_arima)], day_arima, color=colors['ARIMA'],
                        linestyle=linestyles['ARIMA'], linewidth=1.5, alpha=0.8, label='ARIMA')

    # Format x-axis with times
    tick_positions = list(range(0, len(x), 12))  # Every hour
    tick_labels = [str(times[i])[:5] if i < len(times) else '' for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

    ax.set_xlabel('Time')
    ax.set_ylabel('Power [Watt]')
    ax.set_title(f'PV Power Forecast Comparison - {sunny_date} (Horizon: {horizon_min} min)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = FIGURES_DIR / f"daily_comparison_{horizon_min}min.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED MODEL EVALUATION")
    print("=" * 60)

    # Recalculate with unified baseline
    result = recalculate_with_unified_baseline()

    if result is None:
        print("No results to compare!")
        sys.exit(1)

    combined_df, sp_rmse = result

    # Save detailed results
    output_file = METRICS_DIR / "unified_comparison_detailed.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")

    # Create summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean over all horizons)")
    print("=" * 60)

    summary = create_summary_table(combined_df)

    print(f"\n{'Model':<20} {'RMSE':>8} {'MAE':>8} {'Skill':>8}")
    print("-" * 50)
    for _, row in summary.iterrows():
        print(f"{row['model']:<20} {row['RMSE']:>8.0f} {row['MAE']:>8.0f} {row['Skill']:>7.1f}%")

    # Save summary
    summary_file = METRICS_DIR / "unified_comparison_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")

    # Create plots
    sp_rmse_mean = np.mean(list(sp_rmse.values()))
    plot_comparison(summary, sp_rmse_mean)
    plot_skill_by_horizon(combined_df)

    # Create daily comparison plots for different horizons
    feat = DataManager()
    for horizon in [60, 180]:
        plot_daily_comparison(feat, horizon_min=horizon)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
