"""Seasonal naive baseline: next month's volume = the same month's volume last year.
"""
import numpy as np
import pandas as pd


class SeasonalNaive:
    """Predict the target month's fraction full as its value 12 months earlier.
    """

    def __init__(self, feature_cols: list[str], series: pd.DataFrame, **params):
        # This model reads past values directly, so it keeps `series`.
        self.series = series

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "SeasonalNaive":
        # Nothing to learn. Still defined so the pipeline can call fit() on every model in exactly the same way.
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        predictions = []

        # One forecast per row
        for reservoir, origin, target in zip(rows["reservoir"], rows["month"], rows["pred_target_month"]):

            # Everything known at the end of the origin month (the leakage rule above).
            history = self.series.loc[:origin, reservoir]

            # The target's month one year earlier
            same_month_last_year = target - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)

            # Only the earliest rows could lack a year of history; stop with a clear message
            if same_month_last_year not in history.index:
                raise ValueError(f"seasonal naive: no value for {reservoir} in {same_month_last_year.date()} "
                                 f"(needs 12 months of history before the target)")

            predictions.append(history.loc[same_month_last_year])

        return np.array(predictions, dtype=float)

    def n_parameters(self) -> int:
        # Nothing is learned from data.
        return 0