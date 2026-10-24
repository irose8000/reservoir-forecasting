"""Long short-term memory network: past months as a sequence in, next month's fraction full out.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast.models.neural_networks.trainer import count_parameters, to_tensor, train_network


class LSTMNet(nn.Module):
    """LSTM over the monthly sequence, then [last hidden state + static inputs] -> Linear(1)."""

    def __init__(self, n_step_inputs: int, n_static: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(n_step_inputs, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size + n_static, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sequence, static = x
        outputs, _ = self.lstm(sequence)          # (rows, steps, hidden_size)
        last = outputs[:, -1, :]                  # state after the most recent month (the origin)
        return self.head(torch.cat([last, static], dim=1))


class LSTM:
    """LSTM on volume + rainfall sequences, trained by trainer.train_network()."""

    def __init__(self, feature_cols: list[str], series: pd.DataFrame,
                 hidden_size=32, num_layers=1, learning_rate=1e-3, batch_size=32,
                 epochs=500, patience=30):

        frac_lags = [c for c in feature_cols if c.startswith("frac_lag")]
        rain_lags = [c for c in feature_cols if c.startswith("rain_lag")]
        # if frac_lags and rain_lags are not the same length, the LSTM cannot be built because it expects a sequence of pairs (frac, rain) for each lag
        if len(frac_lags) != len(rain_lags):
            raise ValueError(f"lstm needs n_rain_lags == n_lags (got {len(rain_lags)} and {len(frac_lags)})")

        n = len(frac_lags)
        # Oldest month first: lag n-1, ..., lag 0
        self.step_cols = [[f"frac_lag{k}", f"rain_lag{k}"] for k in reversed(range(n))]
        self.static_cols = [c for c in feature_cols if c not in frac_lags + rain_lags]

        self.hidden_size, self.num_layers = int(hidden_size), int(num_layers)
        self.training = {"learning_rate": float(learning_rate), "batch_size": int(batch_size),
                         "epochs": int(epochs), "patience": int(patience)}
        self.net = None
        self.best_epoch_ = None   # set by fit(); tuning.py records it as `epochs`

    def _inputs(self, rows: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows -> (sequence tensor (rows, steps, 2), static tensor (rows, n_static))."""
        sequence = np.stack([rows[cols].to_numpy(dtype=np.float32) for cols in self.step_cols], axis=1)
        return to_tensor(sequence), to_tensor(rows[self.static_cols])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "LSTM":
        # Built fresh for each fit, after set_seed(), so the starting weights are repeatable.
        self.net = LSTMNet(n_step_inputs=2, n_static=len(self.static_cols),
                           hidden_size=self.hidden_size, num_layers=self.num_layers)

        train_x = self._inputs(train)
        train_y = to_tensor(train["pred_target_frac"])
        val_x = self._inputs(val) if val is not None else None
        val_y = to_tensor(val["pred_target_frac"]) if val is not None else None

        #train_network() is in trainer.py, and returns the trained network and the best epoch
        self.net, self.best_epoch_ = train_network(self.net, train_x, train_y, val_x, val_y,
                                                   **self.training)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            return self.net(self._inputs(rows)).numpy().ravel()

    def n_parameters(self) -> int:
        return count_parameters(self.net)