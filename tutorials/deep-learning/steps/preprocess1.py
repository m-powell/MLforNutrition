"""Step 3: First pre-processing — drop rows with missing target, optional duplicates/constant cols."""
import pandas as pd
from typing import List, Optional


def preprocess1(
    df: pd.DataFrame,
    target_col: str,
    drop_duplicates: bool = True,
    drop_constant_cols: bool = False,
) -> pd.DataFrame:
    """Return cleaned DataFrame. Drops rows with missing target; optionally duplicates and constant columns."""
    out = df.copy()
    out = out[out[target_col].notna()].reset_index(drop=True)
    if drop_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)
    if drop_constant_cols:
        constant = [c for c in out.columns if c != target_col and out[c].nunique() <= 1]
        out = out.drop(columns=[c for c in constant if c in out.columns])
    return out
