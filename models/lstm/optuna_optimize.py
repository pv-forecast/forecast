#!/usr/bin/env python3
"""
Hyperparameter optimization for LSTM using Optuna.

Based on ablation study findings:
- Hidden units: 75 and 128 work well, 100 is bad
- Layers: 3 is better than 2
- Dropout: 0.2-0.5 range

Usage:
    python models/lstm/optuna_optimize.py --n_trials 20
    python models/lstm/optuna_optimize.py --n_trials 50 --objective weighted
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.trial import TrialState

from models.lstm.data_utils import set_seed, load_and_prepare_data
from models.lstm import lstm_model

# Output directories
METRICS_DIR = "metrics/lstm"
MODELS_DIR = "models/lstm/saved"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_evaluate(model, data, batch_size, seq_dim, epochs, patience, CAPACITY):
    """Train model and return per-horizon skills."""
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
    epochs_without_improvement = 0
    best_skills = None

    for epoch in range(epochs):
        model.train()
        num_steps = int(len(X_train) / batch_size)

        for step in range(num_steps):
            train_load = X_train[step * batch_size:batch_size * (step+1)].view(-1, seq_dim, X_train.shape[2])
            y = y_train[step * batch_size:(step + 1) * batch_size]

            optimizer.zero_grad()
            y_pred = model(train_load)
            loss = model.loss(y_pred, y)
            loss.backward()
            optimizer.step()

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

        avg_test_rmse = np.nanmean(rmse)
        scheduler.step(avg_test_rmse)

        if avg_test_rmse < best_val_loss:
            best_val_loss = avg_test_rmse
            epochs_without_improvement = 0
            best_skills = skill.copy()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return best_skills


def create_objective(data, objective_type='average'):
    """Create Optuna objective function."""

    # Key horizon indices (0-indexed): 30, 60, 90, 120, 150, 180 min
    key_indices = [5, 11, 17, 23, 29, 35]  # steps 6, 12, 18, 24, 30, 36
    short_indices = [5, 11]      # 30, 60 min
    long_indices = [29, 35]      # 150, 180 min

    def objective(trial):
        # Hyperparameter search space based on ablation findings
        hidden = trial.suggest_int('hidden', 64, 160, step=8)
        layers = trial.suggest_int('layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        set_seed(42)

        model = lstm_model.LSTM(
            input_dim=data['n_features'],
            hidden_dim=hidden,
            layer_dim=layers,
            output_dim=36,
            dropout=dropout
        )

        skills = train_and_evaluate(
            model=model,
            data=data,
            batch_size=batch_size,
            seq_dim=60,
            epochs=50,  # Reduced for faster optimization
            patience=10,
            CAPACITY=data['CAPACITY']
        )

        # Calculate objective based on type
        key_skills = skills[key_indices]

        if objective_type == 'average':
            return np.mean(key_skills)
        elif objective_type == 'weighted':
            # Weight short horizons more (often more valuable operationally)
            short_skill = np.mean(skills[short_indices])
            long_skill = np.mean(skills[long_indices])
            return 0.6 * short_skill + 0.4 * long_skill
        elif objective_type == 'min':
            # Maximize the minimum skill (robust across all horizons)
            return np.min(key_skills)
        elif objective_type == 'long':
            # Focus on long horizons
            return np.mean(skills[long_indices])
        else:
            return np.mean(key_skills)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna LSTM Hyperparameter Optimization")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--objective", type=str, default="average",
                        choices=["average", "weighted", "min", "long"],
                        help="Objective type: average, weighted, min, or long")
    parser.add_argument("--study_name", type=str, default="lstm_optimization",
                        help="Name for the Optuna study")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Optuna LSTM Optimization")
    print(f"Trials: {args.n_trials}, Objective: {args.objective}")
    print("=" * 60)

    # Load data once
    print("\nLoading data...")
    set_seed(42)
    data = load_and_prepare_data(window_LSTM=36, seq_dim=60)

    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    objective = create_objective(data, args.objective)

    print(f"\nStarting optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best {args.objective} skill: {study.best_value:.2f}%")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best params
    print("\n" + "-" * 60)
    print("Training final model with best parameters...")

    set_seed(42)
    best_params = study.best_params

    model = lstm_model.LSTM(
        input_dim=data['n_features'],
        hidden_dim=best_params['hidden'],
        layer_dim=best_params['layers'],
        output_dim=36,
        dropout=best_params['dropout']
    )

    # Full training with more epochs
    skills = train_and_evaluate(
        model=model,
        data=data,
        batch_size=best_params['batch_size'],
        seq_dim=60,
        epochs=100,
        patience=15,
        CAPACITY=data['CAPACITY']
    )

    # Save results
    key_horizons = [30, 60, 90, 120, 150, 180]
    key_indices = [5, 11, 17, 23, 29, 35]

    print("\nFinal model performance:")
    print("-" * 40)
    for h, idx in zip(key_horizons, key_indices):
        print(f"  {h:>3} min: {skills[idx]:>6.1f}%")
    print("-" * 40)
    print(f"  Average: {np.mean(skills[key_indices]):>6.1f}%")

    # Save best parameters
    results_path = os.path.join(METRICS_DIR, f"optuna_best_{args.study_name}.csv")
    results_df = pd.DataFrame({
        'parameter': list(best_params.keys()) + ['best_skill', 'objective_type'],
        'value': list(best_params.values()) + [study.best_value, args.objective]
    })
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save full metrics
    metrics_df = pd.DataFrame({
        'horizon_steps': list(range(1, 37)),
        'horizon_min': [h * 5 for h in range(1, 37)],
        'Skill': skills,
    })
    metrics_path = os.path.join(METRICS_DIR, f"LSTM_optuna_{args.study_name}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    # Save model
    model_name = f"LSTM_optuna_Layer_{best_params['layers']}_Input_{data['n_features']}_hidden_{best_params['hidden']}_dropout_{best_params['dropout']:.2f}"
    model_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model, model_path)
    print(f"Model saved to: {model_path}")

    # Top 5 trials
    print("\n" + "=" * 60)
    print("TOP 5 TRIALS")
    print("=" * 60)

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['state'] == 'COMPLETE']
    trials_df = trials_df.sort_values('value', ascending=False).head(5)

    for _, row in trials_df.iterrows():
        print(f"Trial {int(row['number']):>2}: skill={row['value']:.1f}% | "
              f"L={int(row['params_layers'])}, H={int(row['params_hidden'])}, "
              f"D={row['params_dropout']:.2f}, BS={int(row['params_batch_size'])}")

    print("=" * 60)


if __name__ == "__main__":
    main()
