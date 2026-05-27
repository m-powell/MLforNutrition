"""Step 2: Data audit (missingness, dtypes, duplicates, constant cols, ID-like)."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Numeric columns with at least one non-null."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if df[c].notna().sum() > 0]


def run_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Return audit dict: missing, dtypes, duplicates, constant_cols, id_like_cols."""
    audit = {}
    # Missing
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    audit["missing"] = [
        {"column": c, "count": int(missing_counts[c]), "pct": float(missing_pct[c])}
        for c in missing_counts.index if missing_counts[c] > 0
    ]
    # Dtypes
    audit["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
    # Duplicates
    audit["duplicates"] = int(df.duplicated().sum())
    # Constant columns
    audit["constant_cols"] = [c for c in df.columns if df[c].nunique() <= 1]
    # ID-like (numeric, unique per row)
    numeric_cols = get_numeric_columns(df)
    audit["id_like_cols"] = [
        c for c in numeric_cols
        if df[c].nunique() == len(df) and df[c].dtype in [np.int64, np.int32]
    ]
    return audit
