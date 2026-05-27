"""
Neural Network wrapper using weighted Huber loss (regression).
Self-contained copy for NHANES glucose pipeline.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any

from .base import BaseModelWrapper


class SimpleMLP(nn.Module):
    """Simplified MLP for regression and classification."""

    def __init__(self, input_dim: int, hidden: list = None, dropout: float = 0.1,
                 output_dim: int = 1, activation: str = 'relu'):
        super().__init__()
        hidden = hidden or [32, 32]
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
        }
        activation_fn = activation_map.get(activation.lower(), nn.ReLU())
        layers = []
        prev_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.layers(x)


def weighted_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                        t0: float = 180.0, s: float = 20.0, alpha: float = 2.5) -> torch.Tensor:
    """Weighted Huber loss for regression."""
    errors = y_true - y_pred
    abs_errors = torch.abs(errors)
    delta = 1.0
    huber_loss = torch.where(
        abs_errors <= delta,
        0.5 * errors ** 2,
        delta * abs_errors - 0.5 * delta ** 2
    )
    w = 1.0 + alpha * torch.exp(-((y_true - t0) / s) ** 2)
    w = torch.clamp(w, min=0.1, max=10.0)
    return (w * huber_loss).mean()


class NNWeightedHuberWrapper(BaseModelWrapper):
    """Wrapper for Neural Network with weighted Huber loss (regression)."""

    def __init__(self, hidden_layers: List[int] = None, dropout: float = 0.1,
                 task_type: str = 'regression', activation: str = 'relu'):
        super().__init__("Neural Network")
        self.hidden_layers = hidden_layers or [32, 32]
        self.dropout = dropout
        self.task_type = task_type
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = None
        self.classes_ = None
        self.n_classes_ = None

    def get_architecture_summary(self) -> Dict[str, Any]:
        return {
            'hidden_layers': self.hidden_layers,
            'num_layers': len(self.hidden_layers),
            'dropout': self.dropout,
            'activation': self.activation,
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 200,
            batch_size: int = 256,
            lr: float = 0.0015,
            weight_decay: float = 0.0002,
            patience: int = 30,
            progress_callback: Optional[callable] = None,
            random_seed: Optional[int] = None,
            **kwargs) -> Dict[str, Any]:
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)

        if self.task_type == 'classification':
            self.classes_ = np.unique(y_train)
            self.n_classes_ = len(self.classes_)
            class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
            y_train_mapped = np.array([class_to_idx[cls] for cls in y_train])
            output_dim = 1 if self.n_classes_ == 2 else self.n_classes_
            X_train_t = torch.FloatTensor(X_train).to(self.device)
            y_train_t = torch.LongTensor(y_train_mapped).to(self.device)
            if X_val is not None and y_val is not None:
                y_val_mapped = np.array([class_to_idx.get(cls, 0) for cls in y_val])
                X_val_t = torch.FloatTensor(X_val).to(self.device)
                y_val_t = torch.LongTensor(y_val_mapped).to(self.device)
            else:
                X_val_t = None
                y_val_t = None
            criterion = nn.BCEWithLogitsLoss() if self.n_classes_ == 2 else nn.CrossEntropyLoss()
        else:
            output_dim = 1
            X_train_t = torch.FloatTensor(X_train).to(self.device)
            y_train_t = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
            if X_val is not None and y_val is not None:
                X_val_t = torch.FloatTensor(X_val).to(self.device)
                y_val_t = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
            else:
                X_val_t = None
                y_val_t = None
            criterion = None

        self.model = SimpleMLP(
            input_dim=X_train.shape[1],
            hidden=self.hidden_layers,
            dropout=self.dropout,
            output_dim=output_dim,
            activation=self.activation
        )
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if X_val_t is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
            )

        best_val_metric = float('inf') if self.task_type == 'regression' else 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [] if self.task_type == 'regression' else [],
            'val_accuracy': [] if self.task_type == 'classification' else []
        }
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True
        )

        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                if self.task_type == 'classification':
                    loss = criterion(y_pred.squeeze() if self.n_classes_ == 2 else y_pred,
                                    y_batch.float() if self.n_classes_ == 2 else y_batch)
                else:
                    loss = weighted_huber_loss(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)

            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_t)
                    if self.task_type == 'classification':
                        val_loss = criterion(y_val_pred.squeeze() if self.n_classes_ == 2 else y_val_pred,
                                            y_val_t.float() if self.n_classes_ == 2 else y_val_t)
                        val_pred_labels = (torch.sigmoid(y_val_pred.squeeze()) > 0.5).long() if self.n_classes_ == 2 else torch.argmax(y_val_pred, dim=1)
                        val_accuracy = (val_pred_labels == y_val_t).float().mean().item()
                        val_metric = val_accuracy
                    else:
                        val_loss = weighted_huber_loss(y_val_pred, y_val_t)
                        val_rmse = torch.sqrt(torch.mean((y_val_pred - y_val_t) ** 2)).item()
                        val_metric = val_rmse
                history['val_loss'].append(val_loss.item())
                if self.task_type == 'regression':
                    history['val_rmse'].append(val_rmse)
                else:
                    history['val_accuracy'].append(val_accuracy)
                if progress_callback and self.task_type == 'regression':
                    progress_callback(epoch + 1, train_loss, val_loss.item(), val_rmse)
                elif progress_callback:
                    progress_callback(epoch + 1, train_loss, val_loss.item(), val_accuracy)
                if self.task_type == 'regression':
                    scheduler.step(val_rmse)
                else:
                    scheduler.step(1.0 - val_accuracy)
                is_better = val_rmse < best_val_metric - 0.0001 if self.task_type == 'regression' else val_accuracy > best_val_metric + 0.0001
                if is_better:
                    best_val_metric = val_metric
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.model.load_state_dict(best_model_state)
                        break
        if X_val_t is not None and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        self.history = history
        return {'history': history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            if self.task_type == 'classification':
                if self.n_classes_ == 2:
                    pred_labels = (torch.sigmoid(logits.squeeze()) > 0.5).long().cpu().numpy()
                else:
                    pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
                if self.classes_ is not None:
                    pred_labels = self.classes_[pred_labels]
                return pred_labels
            return logits.cpu().numpy().flatten()

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.task_type != 'classification' or not self.is_fitted:
            return None
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            if self.n_classes_ == 2:
                prob_positive = torch.sigmoid(logits.squeeze()).cpu().numpy()
                return np.column_stack([1 - prob_positive, prob_positive])
            return torch.softmax(logits, dim=1).cpu().numpy()

    def supports_proba(self) -> bool:
        return self.task_type == 'classification'
