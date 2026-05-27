"""Step 4: Optional statistical significance filtering — keep features with p < threshold."""
import pandas as pd
import numpy as np
from typing import List, Optional

from ._stats import correlation_test
from .audit import get_numeric_columns


def select_features_by_pvalue(
    df: pd.DataFrame,
    target_col: str,
    candidate_features: Optional[List[str]] = None,
    p_threshold: float = 0.1,
    method: str = "pearson",
) -> List[str]:
    """
    Keep only features with correlation p-value < p_threshold.
    May dramatically reduce the number of features.
    """
    if candidate_features is None:
        numeric = get_numeric_columns(df)
        candidate_features = [c for c in numeric if c != target_col]
    if not candidate_features:
        return []

    y = np.asarray(df[target_col].values, dtype=float)
    selected = []
    for col in candidate_features:
        if col not in df.columns:
            continue
        x = np.asarray(df[col].values, dtype=float)
        _, p, _ = correlation_test(x, y, method=method)
        if np.isfinite(p) and p < p_threshold:
            selected.append(col)
    return selected
