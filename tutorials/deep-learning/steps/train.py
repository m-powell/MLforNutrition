"""Step 6: Train neural network."""
import numpy as np
from typing import Dict, Any, List

from models.nn_whuber import NNWeightedHuberWrapper


def _hidden_layers_from_config(
    num_layers: int,
    layer_width: int,
    pattern: str,
) -> List[int]:
    if pattern == "constant":
        return [layer_width] * num_layers
    if pattern == "pyramid":
        return [layer_width * (2 ** i) for i in range(num_layers)]
    if pattern == "funnel":
        mw = layer_width * (2 ** (num_layers - 1))
        return [mw // (2 ** i) for i in range(num_layers)]
    return [layer_width] * num_layers


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    nn_config: Dict[str, Any],
    random_state: int = 42,
) -> tuple:
    """Train NN; return (model, results_dict)."""
    nc = nn_config
    hidden = _hidden_layers_from_config(
        nc.get("num_layers", 2),
        nc.get("layer_width", 32),
        nc.get("architecture_pattern", "constant"),
    )
    model = NNWeightedHuberWrapper(
        hidden_layers=hidden,
        dropout=nc.get("dropout", 0.1),
        task_type="regression",
        activation=nc.get("activation", "relu"),
    )
    results = model.fit(
        X_train, y_train, X_val, y_val,
        epochs=nc.get("epochs", 200),
        batch_size=nc.get("batch_size", 256),
        lr=nc.get("lr", 0.0015),
        weight_decay=nc.get("weight_decay", 0.0002),
        patience=nc.get("patience", 30),
        random_seed=random_state,
    )
    return model, results
