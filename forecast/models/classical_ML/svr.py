"""Support vector regression with an RBF kernel

The RBF kernel compares rows by distance, so inputs need similar scales:
- fractions (~0-1)
- standard-scaled rainfall
- sin/cos (-1 to 1) months
- one-hot (0/1) reservoir names
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR as SklearnSVR


class SVR:
    """Support vector regression on the feature columns."""

    def __init__(self, feature_cols: list[str], series: pd.DataFrame, **params):
        # Series is not used by this model (it reads the features directly), but is kept so the pipeline can call fit() on every model in exactly the same way.
        self.feature_cols = feature_cols
        # params (kernel, C, epsilon, gamma) are passed straight to scikit-learn's SVR.
        self.model = SklearnSVR(**params)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "SVR":
        # Validation is not used by this model because SVR solves one optimisation problem (no "early stopping")
        self.model.fit(train[self.feature_cols], train["pred_target_frac"])
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(rows[self.feature_cols])

    def n_parameters(self) -> int:
        # What SVR learns is its support vectors, each with a weight. 
        return int(self.model.support_.size)