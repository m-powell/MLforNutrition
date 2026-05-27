"""Step 5: Second pre-processing — impute, StandardScaler, target trim (top/bottom %), split."""
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def preprocess2_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    target_trim_lower: float = 0.05,
    target_trim_upper: float = 0.95,
    imputation: str = "median",
) -> Tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Trim target by percentiles (remove rows in bottom target_trim_lower and top (1-target_trim_upper)),
    impute features, split train/val/test, fit scaler on train only.
    Returns: scaler, imputer, X_train, X_val, X_test, y_train, y_val, y_test, feature_names.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    # Convert to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    # Drop rows with missing target
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Target trimming: remove rows in bottom and top percentile of target
    q_lo = y.quantile(target_trim_lower)
    q_hi = y.quantile(target_trim_upper)
    mask = (y >= q_lo) & (y <= q_hi)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Impute
    if imputation == "median":
        imp = SimpleImputer(strategy="median")
    elif imputation == "mean":
        imp = SimpleImputer(strategy="mean")
    else:
        imp = SimpleImputer(strategy="constant", fill_value=0)
    X_imp = imp.fit_transform(X)
    X = pd.DataFrame(X_imp, columns=feature_cols)

    # Split
    idx = np.arange(len(X))
    i_train, i_temp = train_test_split(idx, test_size=(val_size + test_size), random_state=random_state)
    rel_val = val_size / (val_size + test_size)
    i_val, i_test = train_test_split(i_temp, test_size=(1.0 - rel_val), random_state=random_state)

    X_train = X.iloc[i_train]
    X_val = X.iloc[i_val]
    X_test = X.iloc[i_test]
    y_train = y.iloc[i_train].values
    y_val = y.iloc[i_val].values
    y_test = y.iloc[i_test].values

    # Scale on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return scaler, imp, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, feature_cols
