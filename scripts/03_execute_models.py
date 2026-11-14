"""Final run: refit every model with its tuned settings on train + validation, then score ONCE on test.
"""
import argparse
import copy
import time

import joblib
import pandas as pd
import yaml

from forecast import configure
from forecast.evaluate.costs import final_cost_table
from forecast.evaluate.metrics import evaluate, mean_relative_rmse
from forecast.evaluate.plots import (MODEL_COLOURS, plot_cost_vs_accuracy, plot_metric_bars,
                                     plot_predictions, save_figure)
from forecast.pipeline import run_model
from forecast.tune import STOCHASTIC


# Rainfall ablation (--no-rain): only the feature-based models use rainfall, so only they are rerun.
ABLATION_MODELS = ["mlr", "svr", "ffnn", "lstm"]
ABLATION_DIR = configure.RESULTS / "ablation_no_rain"


def use_ablation_folders() -> None:
    """Point every final-results folder at results/ablation_no_rain/, so results/final/ is never overwritten."""
    configure.FINAL_MODELS = ABLATION_DIR / "models"
    configure.FINAL_PREDICTIONS = ABLATION_DIR / "predictions"
    configure.FINAL_METRICS = ABLATION_DIR / "metrics"
    configure.FINAL_FIGURES = ABLATION_DIR / "figures"


def check_tuned(model_names: list[str]) -> None:
    """load_config() silently falls back to defaults for a missing model, so fail loudly instead."""
    with open(configure.BEST_PARAMS_PATH) as f:
        tuned = (yaml.safe_load(f) or {}).get("models", {})
    missing = [m for m in model_names if m not in tuned]
    if missing:
        raise SystemExit(f"no tuned settings for {missing} in best_params.yaml; run 02_tuning.py first")


def run_final(name: str, cfg: dict) -> dict:
    """Fit on train + val and predict test, once per seed for the neural networks."""
    seeds = cfg["nn_seeds"] if name in STOCHASTIC else [cfg["seed"]]
    runs = []
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["seed"] = seed                       # fit_model() calls set_seed(cfg["seed"])
        predictions, cost, bundle = run_model(name, run_cfg, final=True)
        metrics = evaluate(predictions)
        score = mean_relative_rmse(metrics)
        runs.append({"seed": seed, "predictions": predictions, "cost": cost,
                     "bundle": bundle, "metrics": metrics, "score": score})
        print(f"    seed {seed}: test mean relative RMSE {score:.2f}% "
              f"(fit {cost['fit_seconds']:.2f}s)", flush=True)

    mean_score = sum(r["score"] for r in runs) / len(runs)
    rep = min(runs, key=lambda r: abs(r["score"] - mean_score))          # representative seed
    avg_metrics = sum(r["metrics"] for r in runs) / len(runs)            # same index/columns every run
    avg_metrics["n"] = avg_metrics["n"].astype(int)

    seed_rows = pd.DataFrame([{"model": name, "seed": r["seed"],
                               "mean_relative_rmse_pct": r["score"],
                               "fit_seconds": r["cost"]["fit_seconds"],
                               "predict_seconds": r["cost"]["predict_seconds"],
                               "n_parameters": r["cost"]["n_parameters"]} for r in runs])
    return {"bundle": rep["bundle"], "predictions": rep["predictions"],
            "metrics": avg_metrics, "seeds": seed_rows, "score": mean_score}


def save_model_results(name: str, result: dict) -> None:
    """Save one model's final outputs to the four final-results folders."""
    joblib.dump(result["bundle"], configure.FINAL_MODELS / f"{name}.joblib")
    result["predictions"].to_csv(configure.FINAL_PREDICTIONS / f"{name}.csv", index=False)
    result["metrics"].to_csv(configure.FINAL_METRICS / f"{name}_metrics.csv")
    result["seeds"].to_csv(configure.FINAL_METRICS / f"{name}_seeds.csv", index=False)

    save_figure(plot_predictions({name: result["predictions"]}, title=f"{name}: test predictions"),
                configure.FINAL_FIGURES / f"{name}_predictions.png")
    save_figure(plot_metric_bars({name: result["metrics"]}, title=f"{name}: test relative RMSE"),
                configure.FINAL_FIGURES / f"{name}_relative_rmse.png")


def save_comparison() -> None:
    """Redraw the cross-model tables and figures from every model's saved final results."""
    saved = {p.stem for p in configure.FINAL_PREDICTIONS.glob("*.csv")}
    names = [m for m in MODEL_COLOURS if m in saved] + sorted(saved - set(MODEL_COLOURS))

    predictions = {m: pd.read_csv(configure.FINAL_PREDICTIONS / f"{m}.csv", parse_dates=["pred_target_month"])
                   for m in names}
    metrics = {m: pd.read_csv(configure.FINAL_METRICS / f"{m}_metrics.csv", index_col=0) for m in names}
    runs = pd.concat([pd.read_csv(configure.FINAL_METRICS / f"{m}_seeds.csv") for m in names],
                     ignore_index=True)

    tuning_path = configure.TUNING_METRICS / "costs.csv"
    tuning_costs = pd.read_csv(tuning_path, index_col=0) if tuning_path.exists() else None
    costs = final_cost_table(runs, tuning_costs)
    costs.to_csv(configure.FINAL_METRICS / "costs.csv")

    save_figure(plot_predictions(predictions, title="Test predictions, all models"),
                configure.FINAL_FIGURES / "all_predictions.png")
    save_figure(plot_metric_bars(metrics, title="Test relative RMSE, all models"),
                configure.FINAL_FIGURES / "all_relative_rmse.png")
    save_figure(plot_cost_vs_accuracy(costs, title="Cost vs test accuracy",
                                      x_col="fit_predict_seconds",
                                      x_label="Fit + predict time on train+val (s, log scale)",
                                      y_label="Mean relative RMSE, test (%)",
                                      yerr_col="relative_rmse_sd_pct"),
                configure.FINAL_FIGURES / "cost_vs_accuracy.png")
    print(f"comparison updated for: {', '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit on train + validation and score on test.")
    parser.add_argument("--models", nargs="+", help="model names to run (default: models.run in default.yaml)")
    parser.add_argument("--no-rain", action="store_true",
                        help="rainfall ablation: tuned settings with no rainfall inputs, saved to results/ablation_no_rain/")
    args = parser.parse_args()

    cfg = configure.load_config(use_best_params=True)       # defaults overlaid with the tuned settings
    model_names = args.models or (ABLATION_MODELS if args.no_rain else cfg["models"]["run"])
    check_tuned(model_names)

    if args.no_rain:
        not_feature_based = [m for m in model_names if m not in ABLATION_MODELS]
        if not_feature_based:
            raise SystemExit(f"--no-rain only applies to {ABLATION_MODELS}; {not_feature_based} use no rainfall")
        for name in model_names:
            cfg["models"][name]["n_rain_lags"] = 0          # everything else stays at its tuned value
        use_ablation_folders()

    # Make sure the output folders exist (fitted models are not committed to the repo)
    for folder in (configure.FINAL_MODELS, configure.FINAL_PREDICTIONS,
                   configure.FINAL_METRICS, configure.FINAL_FIGURES):
        folder.mkdir(parents=True, exist_ok=True)

    check_tuned(model_names)

    start = time.perf_counter()
    for i, name in enumerate(model_names, 1):
        print(f"[{i}/{len(model_names)}] {name}: {cfg['models'][name]}", flush=True)
        result = run_final(name, cfg)

        months = result["predictions"]["pred_target_month"]
        print(f"    test months {months.min():%b %Y} - {months.max():%b %Y} "
              f"({months.nunique()}) | mean relative RMSE {result['score']:.2f}% | "
              f"elapsed {time.perf_counter() - start:.0f}s", flush=True)
        save_model_results(name, result)

    save_comparison()


if __name__ == "__main__":
    main()