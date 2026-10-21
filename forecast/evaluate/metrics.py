"""Metrics for next-month reservoir volume forecasts.

    relative RMSE  RMSE / mean observed volume x 100            (%, primary metric; comparable across reservoir sizes)
    MAE            mean absolute error                          (megalitres)
    R              Pearson correlation of observed vs predicted (unitless, -1 to 1)

They are reported per reservoir and for the sum across the reservoir network.
"""
import numpy as np
import pandas as pd


def relative_rmse(observed, predicted) -> float:
    """RMSE divided by the mean observed value, as a percentage.
    """
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))
    return float(rmse / np.mean(observed) * 100)


def mae(observed, predicted) -> float:
    """Mean absolute error, in the same units as the inputs (megalitres)."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(observed - predicted)))


def pearson_r(observed, predicted) -> float:
    """Pearson correlation between observed and predicted values.
    """
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if np.std(observed) == 0 or np.std(predicted) == 0:        
        # Returns NaN when either series is constant (correlation is undefined)
        # This way, one flat series can't stop a whole evaluation.
        return float("nan")
    # The 1/P normalising factors in the formula cancel, so corrcoef gives the same value, see https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
    return float(np.corrcoef(observed, predicted)[0, 1])

#opposite to frac_full() in features.py, which converts megalitres to a fraction of capacity
def frac_to_ml(frac, capacity_ml):
    """Convert a fraction of capacity back to megalitres (models predict fractions)."""
    return np.asarray(frac, float) * np.asarray(capacity_ml, float)


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score one model's predictions per reservoir and for the network total.

    Returns a results table with one row per reservoir plus a final "Reservoirs Total" row
    """
    required = {"reservoir", "pred_target_month", "y_true_ml", "y_pred_ml"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    if predictions[["y_true_ml", "y_pred_ml"]].isna().any().any():
        raise ValueError("predictions contain missing values")
    if predictions.duplicated(["reservoir", "pred_target_month"]).any():
        raise ValueError("more than one prediction for the same reservoir and month")

    def score(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(group),
            "relative_rmse_pct": relative_rmse(group["y_true_ml"], group["y_pred_ml"]),
            "mae_ml": mae(group["y_true_ml"], group["y_pred_ml"]),
            "r": pearson_r(group["y_true_ml"], group["y_pred_ml"]),
        })


    

    # Per reservoir
    per_reservoir = {name: score(g) for name, g in predictions.groupby("reservoir")}
    #initialize a table with per-reservoir scores
    table = pd.DataFrame(per_reservoir).T
    table.index.name = "reservoir"



    # Reservoirs total: sum observed and predicted volumes across reservoirs each month
    n_reservoirs = predictions["reservoir"].nunique()
    
    #creates a dataframe with the sum of observed and predicted volumes for each month, and the count of reservoirs with data for that month
    monthly = predictions.groupby("pred_target_month").agg(
        y_true_ml=("y_true_ml", "sum"), y_pred_ml=("y_pred_ml", "sum"), count=("reservoir", "size"))

    #drops any months that don't have data for all reservoirs, so that the total score is based on complete data
    complete = monthly[monthly["count"] == n_reservoirs]
    if len(complete) < len(monthly):
        print(f"  reservoirs total: skipped {len(monthly) - len(complete)} month(s) missing a reservoir")

    #add the summary stats to the same table containing the per-reservoir scores
    table.loc["Reservoirs Total"] = score(complete)

    #since the count is a whole integer and not a decimal like the metrics, convert it to an integer type for clarity
    table["n"] = table["n"].astype(int)

    return table

#takes as input the table produced by evaluate()
def mean_relative_rmse(table: pd.DataFrame) -> float:
    """Average relative RMSE across reservoirs (each weighted equally); used to pick the best tuning trial."""
    return float(table.drop(index="Reservoirs Total")["relative_rmse_pct"].mean())