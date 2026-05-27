"""Step 1: Load CSV from path."""
import os
import pandas as pd
from typing import Union


def load_csv(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Load CSV and return DataFrame. Path can be relative to cwd."""
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(p)
