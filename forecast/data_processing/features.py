"""Turn data/processed/combined.csv into a model-ready table with one row per forecast origin.

Decisions come from notebooks/eda/EDA_processed.ipynb:
- target scale is fraction of capacity, unscaled, and may exceed 1.0 (Warren, Kangaroo Creek)
- frac and yoy inputs stay unscaled (same units as the target)
- only rainfall is standard-scaled, with statistics from training rows only (RainScaler)
"""
import numpy as np
import pandas as pd

from forecast import configure


def load_combined() -> pd.DataFrame:
    """Read the merged, cleaned dataset."""
    df = pd.read_csv(configure.COMBINED, parse_dates=["month"], dtype={"station": str})
    return df.sort_values(["reservoir", "month"]).reset_index(drop=True)

#opposite to frac_to_ml() in metrics.py, which converts fractions to megalitres
def frac_full(df: pd.DataFrame) -> pd.Series:
    """Volume as a fraction of capacity (may exceed 1.0)"""
    return df["volume_ml"] / df["capacity_ml"]

# models using build_series: Seasonal Naive, SARIMA
def build_series(df: pd.DataFrame) -> pd.DataFrame:
    """ Build model inputs and the target for every forecast origin for models that read the full series of past data.
    """
    return df.assign(frac_full=frac_full(df)).pivot(index="month", columns="reservoir", values="frac_full")

#models using build_features: MLR, SVR, FFNN, LSTM
def build_features(df: pd.DataFrame, n_lags: int, n_rain_lags: int) -> tuple[pd.DataFrame, list[str]]:
    """Build model inputs and the target for every forecast origin for models that read only a fixed number of past months (lags).

    Returns (table, feature_columns). The table keeps identifying columns next to the inputs for traceability.
    """
    # n_rain_lags = 0 means no rainfall inputs at all (used by the rainfall ablation: 03_execute_models.py --no-rain)
    if n_lags < 1 or n_rain_lags < 0:
        raise ValueError("n_lags must be at least 1 and n_rain_lags at least 0")

    out = df.sort_values(["reservoir", "month"]).copy()

    # Every shift is done within a reservoir (groupby), so a target can never come from a different reservoir's series.
    by_res = out.groupby("reservoir")

    # Scale volumes by capacity
    out["frac_full"] = frac_full(out)
    out["yoy_frac"] = out["yoy_change_ml"] / out["capacity_ml"]   # change vs 12 months earlier, at month t

    # Target: fraction full NEXT month
    out["pred_target_frac"] = by_res["frac_full"].shift(-1)
    out["pred_target_month"] = by_res["month"].shift(-1)

    # Volume lags: frac_lag0 = month t, frac_lag1 = month t-1, ...
    for k in range(n_lags):
        out[f"frac_lag{k}"] = by_res["frac_full"].shift(k)

    # Rainfall lags: raw mm here; standard-scaled later by RainScaler
    for k in range(n_rain_lags):
        out[f"rain_lag{k}"] = by_res["rain_mm"].shift(k)

    # Calendar month of the TARGET, as a point on a circle so December (12) and January (1) are neighbours rather than 11 apart.
    cal = out["pred_target_month"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * cal / 12)
    out["month_cos"] = np.cos(2 * np.pi * cal / 12)

    # Reservoir identity as one-hot columns
    onehot_cols = []
    onehot = pd.get_dummies(out["reservoir"], prefix="res", dtype=float)
    onehot.columns = [c.replace(" Reservoir", "").replace(" ", "_") for c in onehot.columns]
    out = pd.concat([out, onehot], axis=1)
    onehot_cols = list(onehot.columns)

    feature_cols = ([f"frac_lag{k}" for k in range(n_lags)]
                    + ["yoy_frac"]
                    + [f"rain_lag{k}" for k in range(n_rain_lags)]
                    + ["month_sin", "month_cos"]
                    + onehot_cols)

    keep = ["reservoir", "month", "pred_target_month", "capacity_ml",
            "pred_target_frac"] + feature_cols
    return out[keep].reset_index(drop=True), feature_cols


class RainScaler:
    """Standard-scale the rainfall columns using statistics from the rows it is fitted on.
    """

    def __init__(self):
        # Set by fit(). 
        # The trailing underscore follows the scikit-learn convention for values learned from data 
        self.mean_ = None
        self.std_ = None

    @staticmethod
    def _rain_cols(df: pd.DataFrame) -> list[str]:
        # Identify all rainfall inputs: rain_lag0 (month t), rain_lag1 (month t-1), ...
        return [c for c in df.columns if c.startswith("rain_lag")]

    def fit(self, df: pd.DataFrame) -> "RainScaler":
        """Learn the mean and standard deviation of rainfall from the given rows (training rows only)."""
        # One mean and std across ALL stations and months (global scaling)
        # That way, differences between stations survive scaling (e.g. a wetter station keeps higher scaled values.)

        cols = self._rain_cols(df)
        # No rainfall columns (rainfall ablation): nothing to learn, so store an identity scaling (mean 0, std 1)
        if not cols:
            self.mean_, self.std_ = 0.0, 1.0
            return self

        values = df[cols].to_numpy().ravel()

        self.mean_ = float(np.mean(values))
        self.std_ = float(np.std(values))

        # A std of 0 would mean dividing by zero in transform(). 
        # It can only happen if every fitted month had identical rainfall, which signals a bug upstream.
        if self.std_ == 0:
            raise ValueError("rainfall has zero spread in the fitting rows")

        # Returning self allows the one line: RainScaler().fit(train_rows)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale the rainfall columns with the stored mean and std; other columns are untouched."""
        # Guard against using the scaler before fit()
        if self.mean_ is None:
            raise RuntimeError("RainScaler must be fitted before transform")

        # Work on a copy so the inputted table is never changed by surprise.
        out = df.copy()
        cols = self._rain_cols(out)

        if not cols:            # rainfall ablation: nothing to scale
            return out

        # The same mean and std (learned from the training rows) are applied to every row passed in, including validation and test 
        out[cols] = (out[cols] - self.mean_) / self.std_
        return out