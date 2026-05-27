"""Step 7: Model performance assessment — regression metrics, plots, results table."""
import json
import os
from typing import Dict, Any, Optional

import numpy as np

from ._eval import calculate_regression_metrics


def evaluate_and_save(
    y_test: Any,
    y_pred: Any,
    output_dir: str,
    metrics_path: str = "metrics.json",
    extra: Dict[str, Any] = None,
) -> Dict[str, float]:
    """Compute regression metrics, write to output_dir/metrics_path, return metrics dict."""
    metrics = calculate_regression_metrics(y_test, y_pred)
    os.makedirs(output_dir, exist_ok=True)
    out = {"metrics": metrics}
    if extra:
        out.update(extra)
    path = os.path.join(output_dir, metrics_path)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return metrics


def save_learning_curve(history: Dict[str, list], output_dir: str, filename: str = "learning_curve.png") -> str:
    """Plot train/val loss and val RMSE vs epoch; save to output_dir. Returns path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history.get("train_loss", []), label="Train loss")
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Learning curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if history.get("val_rmse"):
        ax2.plot(epochs, history["val_rmse"], color="green", label="Val RMSE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("RMSE")
        ax2.set_title("Validation RMSE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_pred_vs_actual(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    filename: str = "predicted_vs_actual.png",
) -> str:
    """Scatter plot predicted vs actual; save to output_dir. Returns path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="y = x")
    plt.xlabel("Actual glucose")
    plt.ylabel("Predicted glucose")
    plt.title("Predictions vs actual (test set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_residual_plot(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    filename: str = "residuals.png",
) -> str:
    """Residual (y_true - y_pred) vs predicted scatter; save to output_dir. Returns path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_test - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted glucose")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs predicted (test set)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_results_table(
    metrics: Dict[str, float],
    output_dir: str,
    csv_path: str = "results_table.csv",
    txt_path: str = "results_table.txt",
    extra: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Write metrics (and optional extra rows) to a CSV and a human-readable text table. Returns (csv_path, txt_path)."""
    os.makedirs(output_dir, exist_ok=True)
    extra = extra or {}

    # Build table: one row per metric/field
    rows = [("Metric", "Value")]
    for k, v in metrics.items():
        rows.append((k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v)))
    for k, v in extra.items():
        if k in metrics:
            continue
        rows.append((str(k), str(v)))

    # CSV
    csv_full = os.path.join(output_dir, csv_path)
    with open(csv_full, "w") as f:
        f.write("Metric,Value\n")
        for r in rows[1:]:
            f.write(f'"{r[0]}","{r[1]}"\n')

    # Human-readable text table (fixed-width style)
    txt_full = os.path.join(output_dir, txt_path)
    col1_w = max(len(str(r[0])) for r in rows)
    col2_w = max(len(str(r[1])) for r in rows)
    with open(txt_full, "w") as f:
        f.write("Test set results\n")
        f.write("=" * (col1_w + col2_w + 5) + "\n")
        for r in rows:
            f.write(f"{str(r[0]):<{col1_w}}   {str(r[1]):<{col2_w}}\n")
        f.write("=" * (col1_w + col2_w + 5) + "\n")

    return csv_full, txt_full
