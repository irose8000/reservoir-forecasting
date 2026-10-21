"""Grid search over each model's settings, scored on the validation period.

Every trial fits on TRAIN and is scored on VALIDATION (pipeline.run_model with final=False).
"""
import copy
import itertools

import pandas as pd
import yaml

from forecast import configure
from forecast.evaluate.metrics import evaluate, mean_relative_rmse
from forecast.pipeline import run_model


def load_search_spaces() -> dict:
    """The values to try for each model, from configs/tuning.yaml."""
    with open(configure.CONFIGS / "tuning.yaml") as f:
        spaces = yaml.safe_load(f) or {}
    return spaces.get("models") or {}


def all_combinations(search_space: dict) -> list[dict]:
    """Every combination of the listed values, one dict per trial.
    """
    names = list(search_space)
    value_lists = [search_space[name] for name in names]
    
    #An empty search space gives one trial with no changes, i.e. the defaults.
    return [dict(zip(names, values)) for values in itertools.product(*value_lists)]


def tune_model(model_name: str, cfg: dict, search_space: dict) -> dict:
    """Try every combination in the search space and keep the best.

    Returns a dict with:
        trials       one row per trial: model, trial, each setting, mean_relative_rmse_pct,
                     fit_seconds, predict_seconds, n_parameters
        best_params  the winning settings (written to best_params.yaml by the caller)
        bundle       the best trial's fitted bundle
        predictions  the best trial's validation predictions
        metrics      the best trial's per-reservoir validation scores
    """
    combos = all_combinations(search_space)
    print(f"{model_name}: {len(combos)} trial(s)")

    rows = []
    best = None
    for i, combo in enumerate(combos):
        # A copy of the config with this trial's values in place of the defaults.
        trial_cfg = copy.deepcopy(cfg)        # deepcopy so one trial's changes can never leak into the next.
        trial_cfg["models"][model_name].update(combo)

        predictions, cost, bundle = run_model(model_name, trial_cfg, final=False)
        metrics = evaluate(predictions)
        score = mean_relative_rmse(metrics)

        # Settings worth recording for this trial
        params = dict(combo)
        # NNs will also report the epoch where early stopping found the lowest validation loss
        best_epoch = getattr(bundle["model"], "best_epoch_", None)

        if best_epoch is not None:
            params["epochs"] = int(best_epoch)

        rows.append({"model": model_name, "trial": i, **params,
                     "mean_relative_rmse_pct": score,
                     "fit_seconds": cost["fit_seconds"],
                     "predict_seconds": cost["predict_seconds"],
                     "n_parameters": cost["n_parameters"]})
        print(f"  trial {i}: {params or '(defaults)'} -> mean relative RMSE {score:.2f}%")

        #update best if RMSE is lower than the previous best (or if this is the first trial)
        if best is None or score < best["score"]:
            best = {"score": score, "best_params": params, "bundle": bundle,
                    "predictions": predictions, "metrics": metrics}

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