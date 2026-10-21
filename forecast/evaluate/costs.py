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
    """One row per model summarising its cost during tuning
    """
    trials = trials.assign(trial_seconds=trials["fit_seconds"] + trials["predict_seconds"])

    # The best trial per model: lowest mean relative RMSE on validation
    best = trials.loc[trials.groupby("model")["mean_relative_rmse_pct"].idxmin()].set_index("model")

    table = pd.DataFrame({
        "n_trials": trials.groupby("model").size(),
        "tuning_seconds": trials.groupby("model")["trial_seconds"].sum(),
        "best_fit_seconds": best["fit_seconds"],
        "best_predict_seconds": best["predict_seconds"],
        "n_parameters": best["n_parameters"],
        "mean_relative_rmse_pct": best["mean_relative_rmse_pct"],
    })
    table.index.name = "model"
    return table.sort_values("mean_relative_rmse_pct")

#Cost evaluation from final fit/predict of best performer on train+val, evaluated on test
