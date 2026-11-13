"""Project paths, data sources, reservoir-station mapping, excluded reservoirs, maximum lags, and configuration loading.
"""

from pathlib import Path
import yaml

# Paths are anchored to this file's location
# That way, they work whether code is run from the repo root, a notebook, or the VS Code run button.

ROOT = Path(__file__).resolve().parents[1]   # forecast/config.py -> repo root

CONFIGS = ROOT / "configs"
BEST_PARAMS_PATH = CONFIGS / "best_params.yaml"


DATA_RAW = ROOT / "data" / "raw"
DATA_RAW_SAWATER = DATA_RAW / "sawater"
DATA_RAW_BOM = DATA_RAW / "bom"

DATA_INTERIM = ROOT / "data" / "interim"
SAWATER_COMBINED = DATA_INTERIM / "sawater_combined.csv"     
RAINFALL_COMBINED = DATA_INTERIM / "rainfall_combined.csv"   

DATA_PROCESSED = ROOT / "data" / "processed"
SAWATER_CLEAN = DATA_PROCESSED / "sawater_clean.csv"
RAINFALL_CLEAN = DATA_PROCESSED / "rainfall_clean.csv"
COMBINED = DATA_PROCESSED / "combined.csv"                      # cleaned reservoir volumes + rainfall


RESULTS = ROOT / "results"

RESULTS_TUNING = RESULTS / "tuning"
TUNING_MODELS = RESULTS_TUNING / "models"
TUNING_PREDICTIONS = RESULTS_TUNING / "predictions"
TUNING_METRICS = RESULTS_TUNING / "metrics"
TUNING_FIGURES = RESULTS_TUNING / "figures"
TRIALS_SUFFIX = "_trials.csv"

RESULTS_FINAL = RESULTS / "final"
FINAL_MODELS = RESULTS_FINAL / "models"
FINAL_PREDICTIONS = RESULTS_FINAL / "predictions"
FINAL_METRICS = RESULTS_FINAL / "metrics"
FINAL_FIGURES = RESULTS_FINAL / "figures"


RESULTS_DEMO = RESULTS / "demo"
DEMO_METRICS = RESULTS_DEMO / "metrics"
DEMO_FIGURES = RESULTS_DEMO / "figures"


# Data sources
SAWATER_API_URL = "https://data.sa.gov.au/data/api/3/action/package_show"
SAWATER_DATASET_ID = "reservoir-volumes-2018" # data.sa.gov.au/data/dataset/reservoir-volumes-2018

#Reservoir-Station Mapping
RAIN_STATIONS = {
    "Mt Bold Reservoir": "023734",
    "Happy Valley Reservoir": "023721",
    "Millbrook Reservoir": "023731",
    "Kangaroo Creek Reservoir": "023892",
    "Hope Valley Reservoir": "023096",
    "Myponga Reservoir": "023783",
    "Warren Reservoir": "023878",
    "South Para Reservoir": "023820",
    "Barossa Reservoir": "023752",
    "Little Para Reservoir": "023915",
    "Middle River Reservoir": "022835",
}

# Reservoirs left out of the study. They are dropped in clean.py. 
EXCLUDED_RESERVOIRS = {"Middle River Reservoir"} # Middle River is on Kangaroo Island, not part of the Adelaide metropolitan network, and only has records from March 2022 (see EDA_interim.ipynb).
assert EXCLUDED_RESERVOIRS <= set(RAIN_STATIONS), "excluded reservoir is not in RAIN_STATIONS"

# The reservoir-station pairs that are actually modelled
MODEL_RAIN_STATIONS = {r: s for r, s in RAIN_STATIONS.items() if r not in EXCLUDED_RESERVOIRS}

# Largest number of lags any run may use (volume lags, rainfall lags, LSTM sequence length).
# Raise it if, in tuning.yaml, the search space expands to more lags.
MAX_LAGS = 3

#configurations live in configs/default.yaml and configs/best_params.yaml
def load_config(use_best_params: bool = True) -> dict:
    """Load default.yaml, then overlay tuned values from best_params.yaml if it exists."""
    with open(CONFIGS / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    best_path = CONFIGS / "best_params.yaml"
    if use_best_params and best_path.exists():
        with open(best_path) as f:
            best = yaml.safe_load(f) or {}
        # best_params.yaml only holds models.<name>.<param>; update each model's
        # defaults key by key so untuned defaults are kept.
        for model_name, params in best.get("models", {}).items():
            cfg["models"][model_name].update(params)
    return cfg
