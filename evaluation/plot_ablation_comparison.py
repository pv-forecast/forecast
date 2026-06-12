#!/usr/bin/env python3
"""
Plot ablation study results to identify which hyperparameter causes the crossover.

Loads all LSTM_ablation_*.csv files and creates comparison plots.

Usage:
    python evaluation/plot_ablation_comparison.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS_DIR = "metrics/lstm"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Define experiment configurations for labeling
EXPERIMENT_CONFIGS = {
    'baseline': {'layers': 2, 'hidden': 75, 'dropout': 0.5, 'color': 'blue', 'linestyle': '-'},
    'improved': {'layers': 3, 'hidden': 128, 'dropout': 0.2, 'color': 'green', 'linestyle': '-'},
    'L3_H128_D05': {'layers': 3, 'hidden': 128, 'dropout': 0.5, 'color': 'red', 'linestyle': '--'},
    'L3_H75_D02': {'layers': 3, 'hidden': 75, 'dropout': 0.2, 'color': 'orange', 'linestyle': '--'},
    'L2_H128_D02': {'layers': 2, 'hidden': 128, 'dropout': 0.2, 'color': 'purple', 'linestyle': '--'},
    'L3_H128_D03': {'layers': 3, 'hidden': 128, 'dropout': 0.3, 'color': 'cyan', 'linestyle': '--'},
    'L3_H140_D02': {'layers': 3, 'hidden': 140, 'dropout': 0.2, 'color': 'magenta', 'linestyle': '--'},
    'L4_H128_D02': {'layers': 4, 'hidden': 128, 'dropout': 0.2, 'color': 'brown', 'linestyle': '--'},
    'optuna_best': {'layers': 3, 'hidden': 136, 'dropout': 0.45, 'color': 'gold', 'linestyle': '-', 'linewidth': 3},
}


def load_all_metrics():
    """Load all ablation study metrics files."""
    pattern = os.path.join(METRICS_DIR, "LSTM_ablation_*.csv")
    files = glob.glob(pattern)

    # Also look for baseline metrics files
    baseline_pattern = os.path.join(METRICS_DIR, "LSTM_m2m_*.csv")
    baseline_files = glob.glob(baseline_pattern)

    # Optuna results
    optuna_pattern = os.path.join(METRICS_DIR, "LSTM_optuna_*.csv")
    optuna_files = glob.glob(optuna_pattern)

    metrics = {}

    # Load ablation results
    for f in files:
        name = os.path.basename(f).replace("LSTM_ablation_", "").replace(".csv", "")
        df = pd.read_csv(f)
        metrics[name] = df
        print(f"Loaded {name}: {len(df)} horizons")

    # Load baseline/improved if available
    for f in baseline_files:
        basename = os.path.basename(f)
        if 'improved' in basename.lower():
            name = 'improved'
        elif 'baseline' in basename.lower() or 'Layer_2' in basename:
            name = 'baseline'
        else:
            continue

        df = pd.read_csv(f)
        # Check if this file has the right structure
        if 'RMSE' in df.columns and 'RMSE_sp' in df.columns:
            # Calculate skill if not present
            if 'Skill' not in df.columns:
                df['Skill'] = (1 - df['RMSE'] / df['RMSE_sp']) * 100
            # Add horizon_min if not present
            if 'horizon_min' not in df.columns:
                df['horizon_min'] = (df.index + 1) * 5
            metrics[name] = df
            print(f"Loaded {name}: {len(df)} horizons")

    # Load Optuna results
    for f in optuna_files:
        df = pd.read_csv(f)
        if 'Skill' in df.columns:
            if 'horizon_min' not in df.columns and 'horizon_steps' in df.columns:
                df['horizon_min'] = df['horizon_steps'] * 5
            metrics['optuna_best'] = df
            print(f"Loaded optuna_best: {len(df)} horizons")

    return metrics


def find_crossover(skill1, skill2, horizons):
    """Find the horizon where skill1 crosses below skill2."""
    diff = skill1 - skill2
    for i in range(len(diff) - 1):
        if diff.iloc[i] > 0 and diff.iloc[i+1] <= 0:
            # Linear interpolation
            x1, x2 = horizons.iloc[i], horizons.iloc[i+1]
            y1, y2 = diff.iloc[i], diff.iloc[i+1]
            crossover = x1 - y1 * (x2 - x1) / (y2 - y1)
            return crossover
    return None


def plot_skill_comparison(metrics):
    """Plot skill vs horizon for all configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, df in metrics.items():
        config = EXPERIMENT_CONFIGS.get(name, {
            'color': 'gray', 'linestyle': '-'
        })

        # Use key horizons (30, 60, 90, 120, 150, 180) if available
        if 'horizon_min' in df.columns:
            key_mask = df['horizon_min'].isin([30, 60, 90, 120, 150, 180])
            plot_df = df[key_mask] if key_mask.sum() > 0 else df
        else:
            plot_df = df

        label = f"{name}"
        if name in EXPERIMENT_CONFIGS:
            cfg = EXPERIMENT_CONFIGS[name]
            label = f"{name} (L={cfg['layers']}, H={cfg['hidden']}, D={cfg['dropout']})"

        ax.plot(
            plot_df['horizon_min'],
            plot_df['Skill'],
            color=config.get('color', 'gray'),
            linestyle=config.get('linestyle', '-'),
            marker='o',
            markersize=8 if name == 'optuna_best' else 6,
            linewidth=config.get('linewidth', 2),
            label=label
        )

    ax.set_xlabel('Forecast Horizon [min]', fontsize=12)
    ax.set_ylabel('Skill Score [%]', fontsize=12)
    ax.set_title('LSTM Ablation Study: Skill vs Horizon', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(25, 185)

    # Add reference line at y=0
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "ablation_skill_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_single_factor_analysis(metrics):
    """Plot analysis of single-factor changes."""
    if 'improved' not in metrics:
        print("Warning: 'improved' baseline not found. Skipping single-factor analysis.")
        return

    improved = metrics['improved']
    key_mask = improved['horizon_min'].isin([30, 60, 90, 120, 150, 180])
    improved = improved[key_mask] if key_mask.sum() > 0 else improved

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    factors = [
        ('L3_H128_D05', 'Dropout: 0.2 -> 0.5', 'H1: Higher dropout reduces overfitting'),
        ('L3_H75_D02', 'Hidden: 128 -> 75', 'H2: Fewer hidden units reduce capacity'),
        ('L2_H128_D02', 'Layers: 3 -> 2', 'H3: Shallower network generalizes better'),
    ]

    for ax, (exp_name, change, hypothesis) in zip(axes, factors):
        # Plot improved baseline
        ax.plot(
            improved['horizon_min'],
            improved['Skill'],
            'g-o',
            linewidth=2,
            markersize=6,
            label='Improved (3L, 128h, d=0.2)'
        )

        # Plot variant if available
        if exp_name in metrics:
            variant = metrics[exp_name]
            key_mask = variant['horizon_min'].isin([30, 60, 90, 120, 150, 180])
            variant = variant[key_mask] if key_mask.sum() > 0 else variant

            ax.plot(
                variant['horizon_min'],
                variant['Skill'],
                'r--o',
                linewidth=2,
                markersize=6,
                label=f'{exp_name}'
            )

            # Find crossover
            crossover = find_crossover(
                improved['Skill'].reset_index(drop=True),
                variant['Skill'].reset_index(drop=True),
                improved['horizon_min'].reset_index(drop=True)
            )

            if crossover:
                ax.axvline(x=crossover, color='gray', linestyle=':', alpha=0.7)
                ax.text(crossover, ax.get_ylim()[1] * 0.95, f'Cross: {crossover:.0f}min',
                       ha='center', fontsize=9, color='gray')
        else:
            ax.text(0.5, 0.5, f'No data for {exp_name}',
                   transform=ax.transAxes, ha='center', fontsize=12, color='red')

        ax.set_xlabel('Horizon [min]', fontsize=11)
        ax.set_ylabel('Skill [%]', fontsize=11)
        ax.set_title(f'{change}\n{hypothesis}', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "ablation_single_factor_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_summary_table(metrics):
    """Print summary table with crossover analysis."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    # Key horizons
    key_horizons = [30, 60, 90, 120, 150, 180]

    # Build summary table
    rows = []
    for name, df in metrics.items():
        config = EXPERIMENT_CONFIGS.get(name, {})

        # Filter to key horizons
        if 'horizon_min' in df.columns:
            key_df = df[df['horizon_min'].isin(key_horizons)]
        else:
            key_df = df

        if len(key_df) == 0:
            continue

        row = {
            'Name': name,
            'L': config.get('layers', '?'),
            'H': config.get('hidden', '?'),
            'D': config.get('dropout', '?'),
        }

        for h in key_horizons:
            h_row = key_df[key_df['horizon_min'] == h]
            if len(h_row) > 0:
                row[f'{h}min'] = h_row['Skill'].values[0]
            else:
                row[f'{h}min'] = np.nan

        row['Avg'] = np.nanmean([row.get(f'{h}min', np.nan) for h in key_horizons])
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Print table
    print(f"\n{'Name':<12} | {'L':>2} | {'H':>4} | {'D':>4} |", end="")
    for h in key_horizons:
        print(f" {h}min |", end="")
    print(f" {'Avg':>6} |")
    print("-" * 80)

    for _, row in summary_df.iterrows():
        print(f"{row['Name']:<12} | {row['L']:>2} | {row['H']:>4} | {row['D']:>4} |", end="")
        for h in key_horizons:
            val = row.get(f'{h}min', np.nan)
            if np.isnan(val):
                print(f" {'--':>5} |", end="")
            else:
                print(f" {val:>5.1f}% |", end="")
        print(f" {row['Avg']:>5.1f}% |")

    print("=" * 80)

    # Crossover analysis
    if 'improved' in metrics and len(metrics) > 1:
        print("\nCROSSOVER ANALYSIS:")
        print("-" * 60)

        improved = metrics['improved']
        key_mask = improved['horizon_min'].isin(key_horizons)
        improved = improved[key_mask].reset_index(drop=True)

        for name, df in metrics.items():
            if name == 'improved':
                continue

            key_mask = df['horizon_min'].isin(key_horizons)
            variant = df[key_mask].reset_index(drop=True)

            if len(variant) == len(improved):
                crossover = find_crossover(
                    improved['Skill'],
                    variant['Skill'],
                    improved['horizon_min']
                )

                if crossover:
                    print(f"  {name} vs improved: crossover at {crossover:.0f} min")
                else:
                    # Check which is better overall
                    diff = improved['Skill'].mean() - variant['Skill'].mean()
                    if diff > 0:
                        print(f"  {name}: improved is better at all horizons (avg +{diff:.1f}%)")
                    else:
                        print(f"  {name}: variant is better at all horizons (avg {diff:.1f}%)")

    # Recommendations
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("-" * 80)
    print("If crossover is ELIMINATED by a single-factor change:")
    print("  - L3_H128_D05 (dropout 0.2->0.5): Dropout is the main factor")
    print("  - L3_H75_D02 (hidden 128->75): Hidden units are the main factor")
    print("  - L2_H128_D02 (layers 3->2): Network depth is the main factor")
    print("If crossover PERSISTS in all variants: combination effect")
    print("=" * 80)


def main():
    print("Loading ablation study results...")
    metrics = load_all_metrics()

    if len(metrics) == 0:
        print("No ablation study results found!")
        print(f"Expected files in: {METRICS_DIR}/LSTM_ablation_*.csv")
        print("\nRun ablation experiments first:")
        print("  python models/lstm/ablation_study.py --layers 3 --hidden 128 --dropout 0.5 --name L3_H128_D05")
        print("  python models/lstm/ablation_study.py --layers 3 --hidden 75 --dropout 0.2 --name L3_H75_D02")
        print("  python models/lstm/ablation_study.py --layers 2 --hidden 128 --dropout 0.2 --name L2_H128_D02")
        return

    print(f"\nLoaded {len(metrics)} experiment(s)")

    # Generate plots
    plot_skill_comparison(metrics)
    plot_single_factor_analysis(metrics)

    # Print summary
    print_summary_table(metrics)


if __name__ == "__main__":
    main()
