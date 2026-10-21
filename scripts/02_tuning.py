"""Tune each model on the validation period and save everything needed to compare them.

Run from the repo root:
    python scripts/02_tuning.py                 # every model in default.yaml -> models.run
    python scripts/02_tuning.py --models naive  # only the models named
"""
import argparse

import joblib
import pandas as pd

from forecast import configure
from forecast.evaluate.costs import tuning_cost_table, load_trials
from forecast.evaluate.plots import (MODEL_COLOURS, plot_cost_vs_accuracy, plot_metric_bars,
                                     plot_predictions, save_figure)
from forecast.tune import load_search_spaces, tune_model, update_best_params


def save_model_results(name: str, result: dict) -> None:
    """Save one model's tuning outputs to the four tuning-results folders."""
    joblib.dump(result["bundle"], configure.TUNING_MODELS / f"{name}.joblib")
    result["predictions"].to_csv(configure.TUNING_PREDICTIONS / f"{name}.csv", index=False)
    result["trials"].to_csv(configure.TUNING_METRICS / f"{name}_trials.csv", index=False)
    result["metrics"].to_csv(configure.TUNING_METRICS / f"{name}_metrics.csv")

    save_figure(plot_predictions({name: result["predictions"]}, title=f"{name}: validation predictions"),
                configure.TUNING_FIGURES / f"{name}_predictions.png")
    save_figure(plot_metric_bars({name: result["metrics"]}, title=f"{name}: validation relative RMSE"),
                configure.TUNING_FIGURES / f"{name}_relative_rmse.png")


def save_comparison() -> None:
    """Redraw the cross-model comparison from every model's saved tuning results."""

    # Models in a fixed order (the colour order), so figures always list them the same way.
    saved = {p.stem for p in configure.TUNING_PREDICTIONS.glob("*.csv")}
    names = [m for m in MODEL_COLOURS if m in saved] + sorted(saved - set(MODEL_COLOURS))

    predictions = {m: pd.read_csv(configure.TUNING_PREDICTIONS / f"{m}.csv", parse_dates=["pred_target_month"])
                   for m in names}
    metrics = {m: pd.read_csv(configure.TUNING_METRICS / f"{m}_metrics.csv", index_col=0) for m in names}

    costs = tuning_cost_table(load_trials())
    costs.to_csv(configure.TUNING_METRICS / "costs.csv")

    save_figure(plot_predictions(predictions, title="Validation predictions, all models"),
                configure.TUNING_FIGURES / "all_predictions.png")
    save_figure(plot_metric_bars(metrics, title="Validation relative RMSE, all models"),
                configure.TUNING_FIGURES / "all_relative_rmse.png")
    save_figure(plot_cost_vs_accuracy(costs, title="Tuning cost vs validation accuracy"),
                configure.TUNING_FIGURES / "cost_vs_accuracy.png")
    print(f"comparison updated for: {', '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune models on the validation period.")
    parser.add_argument("--models", nargs="+", help="model names to tune (default: models.run in default.yaml)")
    args = parser.parse_args()

    # Start every model from its defaults, not from earlier tuning results.
    cfg = configure.load_config(use_best_params=False)
    spaces = load_search_spaces()
    model_names = args.models or cfg["models"]["run"]

    for folder in (configure.TUNING_MODELS, configure.TUNING_PREDICTIONS,
                   configure.TUNING_METRICS, configure.TUNING_FIGURES):
        folder.mkdir(parents=True, exist_ok=True)

    for name in model_names:
        if name not in spaces:
            print(f"{name}: not in tuning.yaml, so one trial with its defaults")
        result = tune_model(name, cfg, spaces.get(name) or {})
        save_model_results(name, result)
        update_best_params(name, result["best_params"])

    save_comparison()


if __name__ == "__main__":
    main()