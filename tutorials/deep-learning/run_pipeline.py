#!/usr/bin/env python3
"""
NHANES glucose prediction pipeline.
Run: python run_pipeline.py [--config path/to/config.yaml]
"""
import argparse
import os
import sys

# Ensure pipeline directory is on path when run from elsewhere
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import yaml
import pandas as pd
import numpy as np

from steps.upload import load_csv
from steps.audit import run_audit, get_numeric_columns
from steps.preprocess1 import preprocess1
from steps.feature_selection import select_features_by_pvalue
from steps.preprocess2 import preprocess2_split
from steps.train import train_nn, _hidden_layers_from_config
from steps.evaluate import (
    evaluate_and_save,
    save_learning_curve,
    save_pred_vs_actual,
    save_residual_plot,
    save_results_table,
)
from models.nn_whuber import NNWeightedHuberWrapper


def load_config(path: str = None) -> dict:
    path = path or os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="NHANES glucose prediction pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = cfg.get("data_path", "nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv")
    if not os.path.isabs(data_path):
        data_path = os.path.join(_SCRIPT_DIR, data_path)
    target_col = cfg.get("target_col", "glucose")
    feature_cols = cfg.get("feature_cols") or []
    use_stat_filter = cfg.get("use_statistical_significance_filter", False)
    p_threshold = cfg.get("feature_selection_p_value_threshold", 0.1)
    train_size = cfg.get("train_size", 0.7)
    val_size = cfg.get("val_size", 0.15)
    test_size = cfg.get("test_size", 0.15)
    random_state = cfg.get("random_state", 42)
    target_trim_lo = cfg.get("target_trim_lower_percentile", 0.05)
    target_trim_hi = cfg.get("target_trim_upper_percentile", 0.95)
    imputation = cfg.get("imputation", "median")
    nn_cfg = cfg.get("nn", {})
    use_hpo = cfg.get("use_hyperparameter_optimization", "none")
    optuna_cfg = cfg.get("optuna", {}) or {}
    output_dir = cfg.get("output_dir", "outputs")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_SCRIPT_DIR, output_dir)

    # 1. Upload
    print("Step 1: Loading data...")
    df = load_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Audit
    print("Step 2: Audit...")
    audit = run_audit(df)
    print(f"  Missing cols: {len(audit['missing'])}, Duplicates: {audit['duplicates']}, Constant: {len(audit['constant_cols'])}, ID-like: {len(audit['id_like_cols'])}")

    # 3. Preprocess 1
    print("Step 3: First pre-processing...")
    df = preprocess1(df, target_col, drop_duplicates=True, drop_constant_cols=False)
    print(f"  Rows after clean: {len(df)}")

    # 4. Feature selection (optional)
    if use_stat_filter:
        print("Step 4: Statistical significance filtering (p < {})...".format(p_threshold))
        numeric = get_numeric_columns(df)
        candidates = [c for c in numeric if c != target_col]
        feature_cols = select_features_by_pvalue(df, target_col, candidate_features=candidates, p_threshold=p_threshold)
        print(f"  Selected {len(feature_cols)} features (may be dramatically fewer than candidates)")
    else:
        print("Step 4: Skipping statistical filter; using config or all numeric features.")
        if not feature_cols:
            numeric = get_numeric_columns(df)
            feature_cols = [c for c in numeric if c != target_col]
        print(f"  Using {len(feature_cols)} features")

    if not feature_cols:
        print("ERROR: No features selected. Enable use_statistical_significance_filter or set feature_cols in config.")
        sys.exit(1)

    # 5. Preprocess 2 + split
    print("Step 5: Second pre-processing (scaler, target trim) and split...")
    scaler, imputer, X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess2_split(
        df, target_col, feature_cols,
        train_size=train_size, val_size=val_size, test_size=test_size,
        random_state=random_state,
        target_trim_lower=target_trim_lo, target_trim_upper=target_trim_hi,
        imputation=imputation,
    )
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # 6. Train (with optional Optuna)
    if use_hpo == "optuna":
        print("Step 6: Hyperparameter optimization (Optuna)...")
        try:
            import optuna
        except ImportError:
            print("  Optuna not installed. Install with: pip install optuna. Falling back to config defaults.")
            use_hpo = "none"

    if use_hpo == "optuna":
        n_trials = optuna_cfg.get("n_trials", 30)
        timeout = optuna_cfg.get("timeout", 600)

        def objective(trial):
            num_layers = trial.suggest_int("num_layers", 1, 4)
            layer_width = trial.suggest_int("layer_width", 16, 128)
            pattern = trial.suggest_categorical("architecture_pattern", ["constant", "pyramid", "funnel"])
            activation = trial.suggest_categorical("activation", ["relu", "tanh", "elu"])
            dropout = trial.suggest_float("dropout", 0.0, 0.4)
            epochs = trial.suggest_int("epochs", 80, 250)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            patience = trial.suggest_int("patience", 15, 40)
            hidden = _hidden_layers_from_config(num_layers, layer_width, pattern)
            m = NNWeightedHuberWrapper(hidden_layers=hidden, dropout=dropout, task_type="regression", activation=activation)
            res = m.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=lr,
                        weight_decay=weight_decay, patience=patience, random_seed=random_state)
            hist = res.get("history", {})
            val_rmse = hist.get("val_rmse", [])
            return val_rmse[-1] if val_rmse else float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        best = study.best_params
        nn_cfg = {**nn_cfg, **best}
        nn_cfg["num_layers"] = best["num_layers"]
        nn_cfg["layer_width"] = best["layer_width"]
        nn_cfg["architecture_pattern"] = best["architecture_pattern"]
        nn_cfg["activation"] = best["activation"]
        nn_cfg["dropout"] = best["dropout"]
        nn_cfg["epochs"] = best["epochs"]
        nn_cfg["batch_size"] = best["batch_size"]
        nn_cfg["lr"] = best["lr"]
        nn_cfg["weight_decay"] = best["weight_decay"]
        nn_cfg["patience"] = best["patience"]
        print(f"  Best val RMSE: {study.best_value:.4f}")

    print("Step 6: Training neural network...")
    model, results = train_nn(X_train, y_train, X_val, y_val, nn_cfg, random_state=random_state)
    y_pred = model.predict(X_test)

    # 7. Evaluate
    print("Step 7: Performance assessment...")
    metrics = evaluate_and_save(
        y_test, y_pred, output_dir, "metrics.json",
        extra={"feature_cols": feature_cols, "n_train": len(y_train), "n_test": len(y_test)},
    )
    # Plots and results table
    history = getattr(model, "history", None) or results.get("history", {})
    if history:
        save_learning_curve(history, output_dir)
        print(f"  Learning curve saved to {output_dir}/learning_curve.png")
    save_pred_vs_actual(y_test, y_pred, output_dir)
    save_residual_plot(y_test, y_pred, output_dir)
    print(f"  Predicted vs actual saved to {output_dir}/predicted_vs_actual.png")
    print(f"  Residuals saved to {output_dir}/residuals.png")
    save_results_table(
        metrics, output_dir,
        extra={"n_train": len(y_train), "n_test": len(y_test), "n_features": len(feature_cols)},
    )
    print(f"  Results table saved to {output_dir}/results_table.csv and {output_dir}/results_table.txt")
    print("  Test metrics:", metrics)
    print(f"  Metrics written to {output_dir}/metrics.json")
    print("Done.")


if __name__ == "__main__":
    main()
