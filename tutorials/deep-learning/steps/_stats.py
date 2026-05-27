"""Minimal stats for feature selection (correlation test)."""
import numpy as np
from scipy import stats
from typing import Tuple


def correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson",
) -> Tuple[float, float, str]:
    """Correlation between two numeric vectors with p-value. Returns (r, p, test_name)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 3 or len(y) < 3:
        return (float("nan"), float("nan"), "correlation (insufficient n)")
    if method == "spearman":
        r, p = stats.spearmanr(x, y)
        return (float(r), float(p), "Spearman")
    r, p = stats.pearsonr(x, y)
    return (float(r), float(p), "Pearson")
