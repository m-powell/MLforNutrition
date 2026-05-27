"""
Base model wrapper interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseModelWrapper(ABC):
    """Base class for all model wrappers."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None

    def get_model(self) -> Any:
        return self.model

    def supports_proba(self) -> bool:
        return False
