"""SARIMA: one seasonal ARIMA per reservoir, forecasting one month ahead.

The model is fitted once, then to predict month t+1, the observed values up to month t are appended WITHOUT refitting and a one-step forecast is made.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMA:
    """One SARIMAX model per reservoir, with no exogenous inputs (volume history only)."""

    def __init__(self, feature_cols: list[str], series: pd.DataFrame,
                 order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), **params):
        # This model reads past values directly, so it keeps `series`.
        self.series = series.asfreq("ME")        # gives the index an explicit monthly frequency

        # Unlike scikit-learn models, a statsmodels SARIMAX needs its data when created.
        # The ten per-reservoir models must be built from the training series in fit() instead of initialized here.
        # Any extra SARIMAX settings from default.yaml (normally none) are stored in self.params and passed to SARIMAX() in fit().
        self.params = params    
        # YAML stores the orders as lists; statsmodels expects tuples.
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)

        #initialize the results dict and fit_end to None
        self.fitted_models = {}               # reservoir -> fitted statsmodels results
        self.fit_end = None             # last month used for fitting

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "SARIMA":
        # Validation is not used by this model because coefficients are estimated in one step by maximum likelihood (no "early stopping")
        self.fit_end = train["pred_target_month"].max()

        n_warnings = 0
        for reservoir in sorted(train["reservoir"].unique()):
            y = self.series.loc[:self.fit_end, reservoir]
            model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, **self.params)

            # Count the warnings instead of flooding the output, and report the total once.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                self.fitted_models[reservoir] = model.fit(disp=False)

            n_warnings += len(caught)

        if n_warnings:
            # statsmodels warns freely on short series (e.g. starting values, convergence).
            print(f"  sarima: {n_warnings} statsmodels warning(s) while fitting "
                  f"{len(self.fitted_models)} reservoirs (usually harmless on short series)")
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        #initialize a series of NaNs to hold the predictions, indexed by the rows to predict
        predictions = pd.Series(np.nan, index=rows.index)

        for reservoir, group in rows.groupby("reservoir"):
            model = self.fitted_models[reservoir]

            last_origin = group["month"].max()

            # Look up the observed values in self.series that correspond to all rows which the model wants to predict, and give them to the model.
            new_obs = self.series.loc[self.fit_end + pd.offsets.MonthEnd(1):last_origin, reservoir]
            if len(new_obs): #check to make sure that there are new observations to append, otherwise statsmodels will throw an error
                # give the model the new observations to condition on, but don't refit the coefficients
                model = model.append(new_obs, refit=False) 

            targets = group["pred_target_month"]

            # dynamic = False means the forecast for month t+1 uses the observed values up to t, not its own predictions
            forecast = model.predict(start=targets.min(), end=targets.max(), dynamic=False) 

            # copy the predicted value for each row’s target month into the output, in row order.
            predictions.loc[group.index] = forecast.loc[targets].to_numpy()

        return predictions.to_numpy()

    def n_parameters(self) -> int:
        # Estimated coefficients (AR terms, variance, ...) summed over the ten reservoir models.
        return int(sum(len(r.params) for r in self.fitted_models.values()))