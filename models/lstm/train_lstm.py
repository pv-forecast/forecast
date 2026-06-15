#!/usr/bin/env python3
"""
Unified LSTM training script for PV power forecasting.

Supports presets, custom hyperparameters, and loading from Optuna results.

Usage:
    # Train with preset configurations
    python models/lstm/train_lstm.py --preset baseline
    python models/lstm/train_lstm.py --preset improved

    # Train with custom hyperparameters
    python models/lstm/train_lstm.py --layers 3 --hidden 136 --dropout 0.45 --name optuna_best

    # Load best params from Optuna results
    python models/lstm/train_lstm.py --from-optuna

    # Custom training settings
    python models/lstm/train_lstm.py --preset improved --epochs 50 --patience 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import numpy as np
import pandas as pd
import torch

from models.lstm.data_utils import set_seed, load_and_prepare_data
from models.lstm import lstm_model

# Output directories
METRICS_DIR = "metrics/lstm"
MODELS_DIR = "models/lstm/saved"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Preset configurations
PRESETS = {
    'baseline': {
        'layers': 2,
        'hidden': 75,
        'dropout': 0.5,
        'batch_size': 128,
        'description': 'Original baseline (2L, 75h, d=0.5)'
    },
    'improved': {
        'layers': 3,
        'hidden': 128,
        'dropout': 0.2,
        'batch_size': 128,
        'description': 'Improved config (3L, 128h, d=0.2)'
    },
    'best': {
        'layers': 3,
        'hidden': 136,
        'dropout': 0.45,
        'batch_size': 128,
        'description': 'Optuna best (3L, 136h, d=0.45)'
    }
}


def train_lstm(model, data, batch_size, seq_dim, epochs, patience, CAPACITY, verbose=True):
    """
    Train LSTM model with early stopping.

    Returns:
        best_skills: Per-horizon skill scores (36 values)
        best_model: Trained model with best validation performance
    """
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    sp_test = data['sp_test']
    window_LSTM = y_train.shape[1]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_skills = None
    best_rmse = None
    best_rmse_sp = None
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        num_steps = int(len(X_train) / batch_size)
        train_loss = 0

        for step in range(num_steps):
            train_load = X_train[step * batch_size:batch_size * (step+1)].view(-1, seq_dim, X_train.shape[2])
            y = y_train[step * batch_size:(step + 1) * batch_size]

            optimizer.zero_grad()
            y_pred = model(train_load)
            loss = model.loss(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= num_steps

        # Validation
        model.eval()
        batch_test = 200
        test_len = int(len(X_test) / batch_test)

        sum_sq_error = np.zeros(window_LSTM)
        sum_sq_error_sp = np.zeros(window_LSTM)
        n_valid = np.zeros(window_LSTM)
        n_valid_sp = np.zeros(window_LSTM)

        with torch.no_grad():
            for testrun in range(test_len):
                test_load = X_test[testrun * batch_test:(testrun + 1) * batch_test].view(-1, seq_dim, X_test.shape[2])
                y = y_test[testrun * batch_test:(testrun + 1) * batch_test]
                y_sp = sp_test[testrun * batch_test:(testrun + 1) * batch_test]

                y_pred = model(test_load)
                test_pred = y_pred * CAPACITY
                observ = y * CAPACITY

                error = observ.numpy() - test_pred.numpy()
                error_sp = observ.numpy() - y_sp.numpy()

                for h in range(window_LSTM):
                    valid_mask = ~np.isnan(error[:, h])
                    valid_mask_sp = ~np.isnan(error_sp[:, h])
                    sum_sq_error[h] += np.sum(error[valid_mask, h] ** 2)
                    sum_sq_error_sp[h] += np.sum(error_sp[valid_mask_sp, h] ** 2)
                    n_valid[h] += np.sum(valid_mask)
                    n_valid_sp[h] += np.sum(valid_mask_sp)

        rmse = np.sqrt(sum_sq_error / np.maximum(n_valid, 1))
        rmse_sp = np.sqrt(sum_sq_error_sp / np.maximum(n_valid_sp, 1))
        skill = (1 - rmse / rmse_sp) * 100

        avg_train_rmse = train_loss ** 0.5 * CAPACITY
        avg_test_rmse = np.nanmean(rmse)
        avg_skill = np.nanmean(skill)

        scheduler.step(avg_test_rmse)
        current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f"Epoch {epoch}: Train_RMSE={avg_train_rmse:.0f}, Test_RMSE={avg_test_rmse:.0f}, "
                  f"RMSE_sp={np.nanmean(rmse_sp):.0f}, Skill={avg_skill:.1f}%, LR={current_lr:.2e}")

        if avg_test_rmse < best_val_loss:
            best_val_loss = avg_test_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            best_skills = skill.copy()
            best_rmse = rmse.copy()
            best_rmse_sp = rmse_sp.copy()
            best_model_state = model.state_dict().copy()
            if verbose:
                print(f"  -> New best model! Test_RMSE: {avg_test_rmse:.0f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"\nBest model at epoch {best_epoch}, Test_RMSE: {best_val_loss:.0f}")
        print(f"Average Skill: {np.nanmean(best_skills):.1f}%")

    return {
        'skills': best_skills,
        'rmse': best_rmse,
        'rmse_sp': best_rmse_sp,
        'best_epoch': best_epoch,
        'model': model
    }


def print_summary(results, name):
    """Print summary table for key horizons."""
    key_horizons = [30, 60, 90, 120, 150, 180]
    key_indices = [5, 11, 17, 23, 29, 35]

    skills = results['skills']
    rmse = results['rmse']
    rmse_sp = results['rmse_sp']

    print("\n" + "=" * 60)
    print(f"Summary for key horizons: {name}")
    print("-" * 60)
    print(f"   Horizon |       RMSE |    RMSE_sp |      Skill")
    print("-" * 60)
    for h, idx in zip(key_horizons, key_indices):
        print(f"    {h:>3} min | {rmse[idx]:>10.1f} | {rmse_sp[idx]:>10.1f} | {skills[idx]:>9.1f}%")
    print("-" * 60)
    print(f"   Average |          - |          - | {np.mean(skills[key_indices]):>9.1f}%")
    print("=" * 60)


def save_results(results, config, name):
    """Save metrics and model."""
    skills = results['skills']
    rmse = results['rmse']
    rmse_sp = results['rmse_sp']
    model = results['model']

    # Save metrics
    metrics_df = pd.DataFrame({
        'horizon_steps': list(range(1, 37)),
        'horizon_min': [h * 5 for h in range(1, 37)],
        'RMSE': rmse,
        'RMSE_sp': rmse_sp,
        'Skill': skills
    })

    metrics_path = os.path.join(METRICS_DIR, f"LSTM_{name}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save model
    model_name = f"LSTM_{name}_Layer_{config['layers']}_Input_17_hidden_{config['hidden']}"
    model_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model, model_path)
    print(f"Model saved to: {model_path}")

    return metrics_path, model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for PV power forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preset configurations
  python models/lstm/train_lstm.py --preset baseline
  python models/lstm/train_lstm.py --preset improved
  python models/lstm/train_lstm.py --preset best

  # Custom hyperparameters
  python models/lstm/train_lstm.py --layers 3 --hidden 136 --dropout 0.45 --name my_model

  # Load from Optuna
  python models/lstm/train_lstm.py --from-optuna
        """
    )

    # Configuration source
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--preset", choices=['baseline', 'improved', 'best'],
                              help="Use preset configuration")
    config_group.add_argument("--from-optuna", action='store_true',
                              help="Load hyperparameters from Optuna results")
    config_group.add_argument("--custom", action='store_true',
                              help="Use custom hyperparameters (requires --layers, --hidden, --dropout)")

    # Custom hyperparameters
    parser.add_argument("--layers", type=int, help="Number of LSTM layers")
    parser.add_argument("--hidden", type=int, help="Hidden units per layer")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--name", type=str, help="Model name (required for --custom)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Determine configuration
    if args.preset:
        config = PRESETS[args.preset].copy()
        name = args.preset
        print(f"Using preset: {args.preset}")
        print(f"  {config['description']}")

    elif args.from_optuna:
        # Load from Optuna results
        optuna_path = os.path.join(METRICS_DIR, "optuna_best_lstm_optimization.csv")
        if not os.path.exists(optuna_path):
            print(f"Error: {optuna_path} not found!")
            print("Run optuna_optimize.py first.")
            return

        params_df = pd.read_csv(optuna_path)
        params = dict(zip(params_df['parameter'], params_df['value']))

        config = {
            'layers': int(params['layers']),
            'hidden': int(params['hidden']),
            'dropout': float(params['dropout']),
            'batch_size': int(params['batch_size']),
            'description': 'Loaded from Optuna optimization'
        }
        name = 'optuna_best'
        print("Loaded hyperparameters from Optuna results")

    elif args.custom:
        # Custom configuration
        if not all([args.layers, args.hidden, args.dropout, args.name]):
            parser.error("--custom requires --layers, --hidden, --dropout, and --name")

        config = {
            'layers': args.layers,
            'hidden': args.hidden,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'description': f'Custom: {args.layers}L, {args.hidden}h, d={args.dropout}'
        }
        name = args.name
        print(f"Using custom configuration: {args.name}")

    # Override batch_size if specified
    if args.batch_size != 128:
        config['batch_size'] = args.batch_size

    # Print configuration
    print("\n" + "=" * 60)
    print(f"LSTM Training: {name}")
    print(f"Configuration: layers={config['layers']}, hidden={config['hidden']}, "
          f"dropout={config['dropout']}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    set_seed(args.seed)
    data = load_and_prepare_data(window_LSTM=36, seq_dim=60)

    print(f"Data loaded: {len(data['X_train'])} train, {len(data['X_test'])} test samples")
    print(f"Features: {data['n_features']}, Sequence length: 60, Output horizon: 36")

    # Create model
    print("\nModel architecture:")
    model = lstm_model.LSTM(
        input_dim=data['n_features'],
        hidden_dim=config['hidden'],
        layer_dim=config['layers'],
        output_dim=36,
        dropout=config['dropout']
    )
    print(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Train
    print(f"\nStarting training: {args.epochs} max epochs, early stopping patience={args.patience}")
    results = train_lstm(
        model=model,
        data=data,
        batch_size=config['batch_size'],
        seq_dim=60,
        epochs=args.epochs,
        patience=args.patience,
        CAPACITY=data['CAPACITY'],
        verbose=True
    )

    # Print summary
    print_summary(results, name)

    # Save results
    save_results(results, config, name)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
