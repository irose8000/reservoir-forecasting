"""How long each model takes to tune/fit/predict, and how many parameters it learns.
"""
import pandas as pd

from forecast import configure

#Cost evaluation from tuning
def load_trials(folder=None) -> pd.DataFrame:
    """Stack every model's trial table into one, from results/tuning/metrics/ by default."""
    folder = folder or configure.TUNING_METRICS
    files = sorted(folder.glob(f"*{configure.TRIALS_SUFFIX}"))
    if not files:
        raise FileNotFoundError(f"no *{configure.TRIALS_SUFFIX} files in {folder}; run 02_tuning.py first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def tuning_cost_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per model summarising its cost during tuning.

    total tuning time is (fit + predict) x n_seeds summed over trials.
    """
    trials = trials.copy()
    # Older trial files (round 1) have no seed columns: treat them as a single seed.
    trials["n_seeds"] = trials["n_seeds"].fillna(1) if "n_seeds" in trials else 1
    trials["score_std_pct"] = trials["score_std_pct"].fillna(0.0) if "score_std_pct" in trials else 0.0
    trials["trial_seconds"] = (trials["fit_seconds"] + trials["predict_seconds"]) * trials["n_seeds"]

    # The best trial per model: lowest mean relative RMSE on validation
    best = trials.loc[trials.groupby("model")["mean_relative_rmse_pct"].idxmin()].set_index("model")

    table = pd.DataFrame({
        "n_trials": trials.groupby("model").size(),
        "n_seeds": best["n_seeds"].astype(int),
        "tuning_seconds": trials.groupby("model")["trial_seconds"].sum(),
        "best_fit_seconds": best["fit_seconds"],          # one fit, averaged over seeds
        "best_predict_seconds": best["predict_seconds"],
        "n_parameters": best["n_parameters"],
        "mean_relative_rmse_pct": best["mean_relative_rmse_pct"],
        "relative_rmse_sd_pct": best["score_std_pct"],    # spread across seeds for the best trial
    })
    table.index.name = "model"
    return table.sort_values("mean_relative_rmse_pct")

#Cost evaluation from the final run: refit on train+val, scored on test
def final_cost_table(runs: pd.DataFrame, tuning_costs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per model summarising the final run.
    """
    g = runs.groupby("model")
    table = pd.DataFrame({
        "n_seeds": g.size(),
        "fit_seconds": g["fit_seconds"].mean(),
        "predict_seconds": g["predict_seconds"].mean(),
        "n_parameters": g["n_parameters"].first(),
        "mean_relative_rmse_pct": g["mean_relative_rmse_pct"].mean(),
        "relative_rmse_sd_pct": g["mean_relative_rmse_pct"].std(ddof=1).fillna(0.0),   # spread across seeds
    })
    # One number for "cost of producing a forecast from scratch" (used on the cost-vs-accuracy plot)
    table["fit_predict_seconds"] = table["fit_seconds"] + table["predict_seconds"]
    if tuning_costs is not None:
        table["tuning_seconds"] = tuning_costs["tuning_seconds"]
    table.index.name = "model"
    return table.sort_values("mean_relative_rmse_pct")