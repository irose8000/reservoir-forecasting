"""Feed-forward neural network: the feature row in, next month's fraction full out.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast.models.neural_networks.trainer import count_parameters, to_tensor, train_network


def build_ffnn(n_inputs: int, hidden_sizes: list[int]) -> nn.Sequential:
    """Input -> [Linear -> ReLU] per hidden layer -> Linear(1)."""
    layers, width = [], n_inputs
    for size in hidden_sizes:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    layers.append(nn.Linear(width, 1))       # one output: fraction full next month
    return nn.Sequential(*layers)


class FFNN:
    """Multilayer perceptron trained by trainer.train_network()."""

    def __init__(self, feature_cols: list[str], series: pd.DataFrame,
                 hidden_sizes=(32, 16), learning_rate=1e-3, batch_size=32,
                 epochs=500, patience=30):
        # Keeps feature_cols (which columns are inputs). series is not used by this model.
        self.feature_cols = feature_cols
        self.hidden_sizes = list(hidden_sizes)
        self.training = {"learning_rate": float(learning_rate), "batch_size": int(batch_size),
                         "epochs": int(epochs), "patience": int(patience)}
        self.net = None
        self.best_epoch_ = None   # set by fit(); tuning.py records it as `epochs`

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "FFNN":
        # Built fresh for each fit, after set_seed(), so the starting weights are repeatable.
        self.net = build_ffnn(len(self.feature_cols), self.hidden_sizes)

        X_train = to_tensor(train[self.feature_cols])
        y_train = to_tensor(train["pred_target_frac"])
        X_val = to_tensor(val[self.feature_cols]) if val is not None else None
        y_val = to_tensor(val["pred_target_frac"]) if val is not None else None


        #train_network() is in trainer.py, and returns the trained network and the best epoch
        self.net, self.best_epoch_ = train_network(self.net, X_train, y_train, X_val, y_val,
                                                   **self.training)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            return self.net(to_tensor(rows[self.feature_cols])).numpy().ravel()

    def n_parameters(self) -> int:
        return count_parameters(self.net)