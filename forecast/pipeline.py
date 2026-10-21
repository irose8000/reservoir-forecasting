"""Run models from data to scored predictions, in two halves.

    fit_model(model_name, cfg, final)   -> bundle        (training: the slow half)
    predict_model(bundle, split)        -> predictions   (inference only: no training)
    run_model(model_name, cfg, final)   -> both halves in one call

Nothing is saved here. Where results go is decided by the script calling these functions:
    Tuning (final=False):  fit on train, predict validation.          
    Final  (final=True):   fit on train + validation, predict test.   

"""
import numpy as np
import pandas as pd

from forecast.seed import set_seed

from forecast.data_processing.features import RainScaler, build_series, build_features, load_combined
from forecast.data_processing.splits import assign_splits, select

from forecast.evaluate.metrics import frac_to_ml
from forecast.evaluate.timing import Timer

from forecast.models.baseline.naive import SeasonalNaive
from forecast.models.statistical_models.sarima import SARIMA
from forecast.models.classical_ML.mlr import MLR
from forecast.models.classical_ML.svr import SVR


# Model name (as used in default.yaml) -> model class
MODELS = {
    "naive": SeasonalNaive,
    "sarima": SARIMA,
    "mlr": MLR,
    "svr": SVR,
}


def _prepare_rows(df: pd.DataFrame, n_lags: int, n_rain_lags: int,
                  train_end, val_end) -> tuple[pd.DataFrame, list[str]]:
    """Call build_features() from features.py and assign_splits() from splits.py to prepare the rows for fitting or predicting.
    """
    table, feature_cols = build_features(df, n_lags, n_rain_lags)
    table = assign_splits(table, feature_cols, train_end, val_end)
    return table, feature_cols


def fit_model(model_name: str, cfg: dict, final: bool = False) -> dict:
    """Train one model and return a bundle that can predict later without retraining.
    """
    if model_name not in MODELS:
        raise ValueError(f"unknown model '{model_name}'; add it to MODELS in pipeline.py")

    # Same seed for every run, so results are repeatable
    set_seed(cfg["seed"])

    # Each model has specific settings in the cfg input parameter.
    # During tuning, the cfg is built from default.yaml overlaid with one trial's values from tuning.yaml.
    # During final training, the cfg is built from default.yaml overlaid with best_params.yaml.
    params = dict(cfg["models"][model_name])

    # Settings are copied into the bundle. That way, predicting later uses exactly what training used even if default.yaml has changed since.
    # These settings, unlike the rest of the params coming from cfg, describe how the input rows are built and are NOT parameters of the model itself.
    # So, they are popped from params and stored separately in the bundle.
    settings = {
        "n_lags": params.pop("n_lags"),
        "n_rain_lags": params.pop("n_rain_lags"),
        "train_end": cfg["splits"]["train_end"],
        "val_end": cfg["splits"]["val_end"],
    }

    df = load_combined() #called from features.py, which reads data/processed/combined.csv
    table, feature_cols = _prepare_rows(df, **settings) #adding features and assigning splits to the table

    # Decide which rows to fit on. val_rows is only for early stopping during tuning.
    if final:
        fit_rows = select(table, "train", "val")
        val_rows = None                      # no held-out set left after refitting on train + val
    else:
        fit_rows = select(table, "train")
        val_rows = select(table, "val")

    # Rainfall scaling learned from the fitting rows only. 
    # Like settings, the scaler is also copied into the bundle so it can be used later to transform the predict rows in exactly the same way.
    scaler = RainScaler().fit(fit_rows)
    fit_rows = scaler.transform(fit_rows)
    if val_rows is not None:
        val_rows = scaler.transform(val_rows)

    # Create the model. Every model receives both the processed series and the features processed above, then uses only what it needs.
    # the **params unpacks the model-specific hyperparameters from the cfg input parameter
    model = MODELS[model_name](feature_cols=feature_cols, series=build_series(df), **params)

    with Timer() as fit_timer:
        model.fit(fit_rows, val_rows)

    return {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "final": final,
        **settings,
        "fit_seconds": fit_timer.seconds,
        "n_parameters": model.n_parameters(),
    }


def predict_model(bundle: dict, split: str) -> tuple[pd.DataFrame, float]:
    """Predict one split with an already-fitted bundle. No training happens here.
    """
    model_name = bundle["model_name"]

    # Rebuild the rows with the bundle's own settings, then keep the requested split.
    df = load_combined()
    table, _ = _prepare_rows(df, bundle["n_lags"], bundle["n_rain_lags"],
                             bundle["train_end"], bundle["val_end"])
    rows = bundle["scaler"].transform(select(table, split))

    with Timer() as predict_timer:
        pred_frac = np.asarray(bundle["model"].predict(rows), dtype=float)

    # Check for one prediction per row
    if pred_frac.shape != (len(rows),):
        raise ValueError(f"{model_name}: expected {len(rows)} predictions, got shape {pred_frac.shape}")
    if not np.isfinite(pred_frac).all():
        raise ValueError(f"{model_name}: predictions contain NaN or infinite values")

    # Back to megalitres for scoring
    predictions = pd.DataFrame({
        "reservoir": rows["reservoir"].to_numpy(),
        "pred_target_month": rows["pred_target_month"].to_numpy(),
        "y_true_ml": frac_to_ml(rows["pred_target_frac"], rows["capacity_ml"]),
        "y_pred_ml": frac_to_ml(pred_frac, rows["capacity_ml"]),
    })
    return predictions, predict_timer.seconds


def run_model(model_name: str, cfg: dict, final: bool = False) -> tuple[pd.DataFrame, dict, dict]:
    """Fit then predict in one call
    """
    bundle = fit_model(model_name, cfg, final)
    predictions, predict_seconds = predict_model(bundle, "test" if final else "val")

    cost = {
        "model": model_name,
        "fit_seconds": bundle["fit_seconds"],
        "predict_seconds": predict_seconds,
        "n_parameters": bundle["n_parameters"],
    }
    return predictions, cost, bundle