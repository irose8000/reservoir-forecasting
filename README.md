# Forecasting the Flow
Forecasting next-month volumes for South Australia's metropolitan reservoirs, comparing neural networks (FFNN, LSTM) against SVR, MLR, SARIMA and a seasonal naive baseline.

## Setup
Requires Python 3.10–3.14.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
```

## Run

Run the scripts in order from the repository root:

| Script | What it does |
|---|---|
| `scripts/00_download_data.py` | Downloads the SA Water reservoir CSVs into `data/raw/sawater/`. BoM rainfall CSVs are downloaded manually from Climate Data Online into `data/raw/bom/`. |
| `scripts/01_build_data.py` | Combines and cleans both datasets into `data/processed/combined.csv`. |
| `scripts/02_tuning.py` | Grid search on validation; saves the best settings to `configs/best_params.yaml`. |
| `scripts/03_execute_models.py` | Refits on train + validation, scores on test and saves the models. |
| `scripts/04_demo.py` | Loads the saved models and runs inference only. |

Fitted models are not committed, so run `03_execute_models.py` before `04_demo.py`.

## Layout

```
configs/     default settings, tuning search spaces, tuned settings
forecast/    package: data processing, models, evaluation
notebooks/   exploratory data analysis
results/     tuning, final and demo outputs (predictions, metrics, figures)
scripts/     the five steps above
```