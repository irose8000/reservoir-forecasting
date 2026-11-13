"""Grid search over each model's settings, scored on the validation period.

Every trial fits on TRAIN and is scored on VALIDATION (pipeline.run_model with final=False).
"""
import copy
import itertools

import time
import pandas as pd
import numpy as np
import yaml

from forecast import configure
from forecast.evaluate.metrics import evaluate, mean_relative_rmse
from forecast.pipeline import run_model


def load_search_spaces() -> dict:
    """The values to try for each model, from configs/tuning.yaml."""
    with open(configure.CONFIGS / "tuning.yaml") as f:
        spaces = yaml.safe_load(f) or {}
    return spaces.get("models") or {}

# Per-model rules for which settings can be combined. Models not listed here are unconstrained.
# Each rule gets the full settings dict (defaults overlaid with the trial's values).
CONSTRAINTS = {
    "lstm": lambda p: p["n_lags"] == p["n_rain_lags"],   # one (volume, rain) pair per step
}

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def all_combinations(search_space: dict, is_valid=None) -> list[dict]:
    """Every combination of the listed values (minus any that is_valid rejects), one dict per trial."""
    names = list(search_space)
    value_lists = [search_space[name] for name in names]
    combos = [dict(zip(names, values)) for values in itertools.product(*value_lists)]

    if is_valid is not None:
        n_before = len(combos)
        combos = [c for c in combos if is_valid(c)]
        if n_before != len(combos):
            print(f"  dropped {n_before - len(combos)} invalid combination(s)")
    if not combos:
        raise ValueError("no valid combinations in the search space")
    return combos





STOCHASTIC = {"ffnn", "lstm"}   # models whose result depends on the random seed


def tune_model(model_name: str, cfg: dict, search_space: dict) -> dict:
    """Try every valid combination. For the neural networks, run each one once per seed.

        Returns a dict with:
        trials       one row per trial: model, trial, each setting, mean_relative_rmse_pct,
                     fit_seconds, predict_seconds, n_parameters
        best_params  the winning settings (written to best_params.yaml by the caller)
        bundle       the best trial's fitted bundle
        predictions  the best trial's validation predictions
        metrics      the best trial's per-reservoir validation scores

    A trial's score is the mean relative RMSE across seeds; its epochs is the median best epoch.
    The saved bundle/predictions come from the seed whose score is closest to the trial mean (not the best seed, which would be cherry-picking). 
    The saved metrics are averaged over seeds.
    """

    defaults = cfg["models"][model_name]
    rule = CONSTRAINTS.get(model_name)
    is_valid = (lambda c: rule({**defaults, **c})) if rule else None
    combos = all_combinations(search_space, is_valid)

    seeds = cfg["nn_seeds"] if model_name in STOCHASTIC else [cfg["seed"]]
    print(f"{model_name}: {len(combos)} trial(s) x {len(seeds)} seed(s)")

    start = time.perf_counter()

    rows = []
    best = None


    for i, combo in enumerate(combos):
        runs = []
        for seed in seeds:
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg["seed"] = seed                      # fit_model() calls set_seed(cfg["seed"])
            trial_cfg["models"][model_name].update(combo)

            predictions, cost, bundle = run_model(model_name, trial_cfg, final=False)
            metrics = evaluate(predictions)
            runs.append({"seed": seed, "predictions": predictions, "cost": cost, "bundle": bundle,
                         "metrics": metrics, "score": mean_relative_rmse(metrics),
                         "best_epoch": getattr(bundle["model"], "best_epoch_", None)})

        scores = np.array([r["score"] for r in runs])
        mean_score = float(scores.mean())

        params = dict(combo)
        epochs = [r["best_epoch"] for r in runs if r["best_epoch"] is not None]
        if epochs:
            params["epochs"] = int(round(np.median(epochs)))

        rows.append({"model": model_name, "trial": i, **params,
                     "mean_relative_rmse_pct": mean_score,
                     "score_std_pct": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                     "n_seeds": len(seeds),
                     # per-fit averages; the cost table multiplies by n_seeds for total tuning time
                     "fit_seconds": float(np.mean([r["cost"]["fit_seconds"] for r in runs])),
                     "predict_seconds": float(np.mean([r["cost"]["predict_seconds"] for r in runs])),
                     "n_parameters": runs[0]["cost"]["n_parameters"]})
        
        elapsed = time.perf_counter() - start
        eta = elapsed / (i + 1) * (len(combos) - i - 1)
        print(f"  [{i + 1}/{len(combos)}] {params or '(defaults)'} -> {mean_score:.2f}% "
              f"(sd {rows[-1]['score_std_pct']:.2f}) | best so far "
              f"{min(r['mean_relative_rmse_pct'] for r in rows):.2f}% | "
              f"elapsed {_fmt(elapsed)}, ETA {_fmt(eta)}", flush=True)
        print(f"  trial {i}: {params or '(defaults)'} -> {mean_score:.2f}% (sd {rows[-1]['score_std_pct']:.2f})")

        if best is None or mean_score < best["score"]:
            rep = min(runs, key=lambda r: abs(r["score"] - mean_score))   # representative seed
            avg_metrics = sum(r["metrics"] for r in runs) / len(runs)     # same index/columns each run
            avg_metrics["n"] = avg_metrics["n"].astype(int)
            best = {"score": mean_score, "best_params": params, "bundle": rep["bundle"],
                    "predictions": rep["predictions"], "metrics": avg_metrics}

    print(f"  best: {best['best_params'] or '(defaults)'} ({best['score']:.2f}%)")
    return {"trials": pd.DataFrame(rows), **best}


def update_best_params(model_name: str, params: dict) -> None:
    """Record one model's winning settings in configs/best_params.yaml.
    """
    best = {}
    if configure.BEST_PARAMS_PATH.exists():
        with open(configure.BEST_PARAMS_PATH) as f:
            best = yaml.safe_load(f) or {}
    best.setdefault("models", {})[model_name] = params

    with open(configure.BEST_PARAMS_PATH, "w") as f:
        yaml.safe_dump(best, f, sort_keys=False)