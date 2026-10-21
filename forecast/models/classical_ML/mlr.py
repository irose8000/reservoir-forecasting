"""Multiple linear regression: prediction as a weighted sum of the inputs.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class MLR:
    """Ordinary least-squares regression on the feature columns."""

    def __init__(self, feature_cols: list[str], series: pd.DataFrame, **params):
        # Series is not used by this model (it reads the features directly), but is kept so the pipeline can call fit() on every model in exactly the same way.
        self.feature_cols = feature_cols
        self.model = LinearRegression(**params)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "MLR":
        # Validation is not used by this model because least squares is solved in one step (no "early stopping")
        self.model.fit(train[self.feature_cols], train["pred_target_frac"])
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(rows[self.feature_cols])

    def n_parameters(self) -> int:
        # One coefficient per input column, plus the intercept.
        return int(self.model.coef_.size + 1)