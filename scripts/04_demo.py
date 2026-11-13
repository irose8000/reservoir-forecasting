"""Demo: load the final saved models and predict the test period. No training happens here.

Run from the repo root:
    python scripts/04_demo.py                          # network view, every saved model
    python scripts/04_demo.py --reservoir "Mt Bold"    # operator view (partial, case-insensitive names work)
    python scripts/04_demo.py --models svr lstm        # only the models named
"""

import argparse
import re

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast import configure
from forecast.data_processing.features import load_combined
from forecast.evaluate.metrics import evaluate
from forecast.evaluate.plots import (MODEL_COLOURS, OBSERVED_COLOUR, TOTAL, _colour, _style,
                                     plot_cost_vs_accuracy, save_figure)
from forecast.pipeline import predict_model

SPLIT = "test"   # the final models were refitted on train + val, so test is the only fair split to score

# metric column (from metrics.evaluate) -> (axis label, file name, is lower better?)
METRICS = {
    "relative_rmse_pct": ("Relative RMSE (%)", "relative_rmse", True),
    "mae_ml":            ("MAE (ML)",          "mae",           True),
    "r":                 ("Pearson R",         "r",             False),
}


def saved_models(requested: list[str] | None) -> list[str]:
    """Models with a saved final bundle, in the fixed colour order (so figures always list them the same way)."""
    saved = {p.stem for p in configure.FINAL_MODELS.glob("*.joblib")}
    if not saved:
        raise SystemExit(f"no saved models in {configure.FINAL_MODELS}; run 03_execute_models.py first")
    names = [m for m in MODEL_COLOURS if m in saved] + sorted(saved - set(MODEL_COLOURS))
    if requested:
        missing = [m for m in requested if m not in saved]
        if missing:
            raise SystemExit(f"no saved final model for {missing}; saved models are {names}")
        names = [m for m in names if m in requested]
    return names


def resolve_reservoir(query: str, reservoirs: list[str]) -> str:
    """Match a full or partial reservoir name, case-insensitively ("mt bold" -> "Mt Bold Reservoir")."""
    q = query.strip().lower()
    exact = [r for r in reservoirs if r.lower() in (q, f"{q} reservoir")]
    matches = exact or [r for r in reservoirs if q in r.lower()]
    if len(matches) != 1:
        problem = "no reservoir matches" if not matches else "more than one reservoir matches"
        raise SystemExit(f"{problem} '{query}'. Choose from: {', '.join(sorted(reservoirs))}")
    return matches[0]


def predict_all(model_names: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Load each saved bundle and predict the test split (inference only)."""
    predictions, predict_seconds = {}, {}
    for name in model_names:
        bundle = joblib.load(configure.FINAL_MODELS / f"{name}.joblib")
        if not bundle.get("final", False):
            print(f"  warning: {name}.joblib was not refitted on train + val (final=False)")
        predictions[name], predict_seconds[name] = predict_model(bundle, SPLIT)
        print(f"  {name}: predicted {len(predictions[name])} rows in {predict_seconds[name]:.4f}s", flush=True)
    return predictions, predict_seconds


def score(predictions: dict[str, pd.DataFrame], reservoir: str | None) -> pd.DataFrame:
    """One row per model: the named reservoir's scores, or the mean over reservoirs (each weighted equally)."""
    rows = {}
    for name, preds in predictions.items():
        table = evaluate(preds).drop(index=TOTAL)          # per-reservoir scores only
        rows[name] = table.loc[reservoir] if reservoir else table.mean(numeric_only=True)
    out = pd.DataFrame(rows).T[list(METRICS)]
    out.index.name = "model"
    return out


def percent_full(predictions: dict[str, pd.DataFrame], reservoir: str | None) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Observed and predicted volume as % of capacity per target month.    """
    df = load_combined()
    capacity = (df[["reservoir", "month", "capacity_ml"]]
                .rename(columns={"month": "pred_target_month"}))

    def to_pct(preds: pd.DataFrame) -> pd.DataFrame:
        p = preds.merge(capacity, on=["reservoir", "pred_target_month"], how="left", validate="one_to_one")
        if p["capacity_ml"].isna().any():
            raise ValueError("capacity missing for some predicted months")
        if reservoir:
            p = p[p["reservoir"] == reservoir]
        p = p.assign(true_pct=p["y_true_ml"] / p["capacity_ml"] * 100,
                     pred_pct=p["y_pred_ml"] / p["capacity_ml"] * 100)
        # Only months where every reservoir has a value, so the mean never changes composition month to month
        counts = p.groupby("pred_target_month")["reservoir"].transform("size")
        return p[counts == p["reservoir"].nunique()]

    pct = {name: to_pct(preds) for name, preds in predictions.items()}
    first = next(iter(pct.values()))
    observed = first.groupby("pred_target_month")["true_pct"].mean()
    predicted = {name: p.groupby("pred_target_month")["pred_pct"].mean() for name, p in pct.items()}
    return observed, predicted



def plot_metric(scores: pd.DataFrame, metric: str, title: str) -> plt.Figure:
    """One vertical bar per model, value labelled on each bar; the best model's label is bold."""
    label, _, lower_is_better = METRICS[metric]
    values = scores[metric]
    best = values.idxmin() if lower_is_better else values.idxmax()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(values))
    ax.bar(x, values.to_numpy(), width=0.6, color=[_colour(m) for m in values.index])

    # Value labels: above positive bars, below negative ones (R can be negative for a single reservoir)
    fmt = "{:.3f}" if metric == "r" else ("{:,.0f}" if metric == "mae_ml" else "{:.2f}")
    span = np.nanmax(np.abs(values.to_numpy())) or 1.0
    for xi, (model, v) in zip(x, values.items()):
        if np.isnan(v):
            ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=8, color="#555555")
            continue
        offset = 0.015 * span if v >= 0 else -0.015 * span
        ax.text(xi, v + offset, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, fontweight="bold" if model == best else "normal", color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(values.index, fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.margins(y=0.12)
    if metric == "mae_ml":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _style(ax)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_percent_full(observed: pd.Series, predicted: dict[str, pd.Series], title: str) -> plt.Figure:
    """Observed vs predicted % full over the test period: one coloured line per model."""
    # Target months are stored as month-ends; plot them at the start of the month so each point sits on its tick
    def month_start(s: pd.Series) -> pd.DatetimeIndex:
        return s.index.to_period("M").to_timestamp()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(month_start(observed), observed.values, color=OBSERVED_COLOUR, linewidth=2.5,
            label="Observed", zorder=3)
    for name, series in predicted.items():
        ax.plot(month_start(series), series.values, color=_colour(name), linewidth=2, linestyle="--",
                marker="o", markersize=4, label=name)
    ax.set_ylabel("Volume (% of capacity)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(title, fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_cost(costs: pd.DataFrame, title: str, y_label: str) -> plt.Figure:
    """Prevents near-tied models (e.g. sarima and ffnn) from printing their names on top of each other.
    """
    fig = plot_cost_vs_accuracy(costs, title=title, x_col="fit_predict_seconds",
                                x_label="Fit + predict time on train+val, network-wide (s, log scale)",
                                y_label=y_label)
    ax = fig.axes[0]
    for text in list(ax.texts):              # drop the default labels, then redraw them
        text.remove()

    x = np.log10(costs["fit_predict_seconds"].to_numpy())
    y = costs["mean_relative_rmse_pct"].to_numpy()
    x_span = (x.max() - x.min()) or 1.0
    y_span = (y.max() - y.min()) or 1.0
    for i, model in enumerate(costs.index):
        # "Close" = within 8% of the plotted range on both axes; the lower of the pair gets its label below
        close = [j for j in range(len(x)) if j != i
                 and abs(x[j] - x[i]) / x_span < 0.08 and abs(y[j] - y[i]) / y_span < 0.08]
        below = any(y[i] < y[j] or (y[i] == y[j] and i > j) for j in close)
        ax.annotate(model, (costs["fit_predict_seconds"].iloc[i], y[i]),
                    xytext=(6, -11 if below else 4), textcoords="offset points", fontsize=8)
    return fig


def cost_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Network-wide fit + predict time from the final run, paired with this view's relative RMSE."""
    path = configure.FINAL_METRICS / "costs.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run 03_execute_models.py first")
    costs = pd.read_csv(path, index_col=0)
    missing = [m for m in scores.index if m not in costs.index]
    if missing:
        raise SystemExit(f"no cost row for {missing} in {path}")
    # plot_cost_vs_accuracy() reads its y-value from mean_relative_rmse_pct
    return pd.DataFrame({"fit_predict_seconds": costs.loc[scores.index, "fit_predict_seconds"],
                         "mean_relative_rmse_pct": scores["relative_rmse_pct"]})


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().replace(" reservoir", "")).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only demo with the saved final models.")
    parser.add_argument("--reservoir", help="one reservoir for the operator view (default: mean over all reservoirs)")
    parser.add_argument("--models", nargs="+", help="model names to include (default: every saved final model)")
    args = parser.parse_args()

    model_names = saved_models(args.models)
    reservoirs = sorted(load_combined()["reservoir"].unique())
    reservoir = resolve_reservoir(args.reservoir, reservoirs) if args.reservoir else None

    view = reservoir.replace(" Reservoir", "") if reservoir else f"mean of {len(reservoirs)} reservoirs"
    out_dir = configure.DEMO_FIGURES / (slug(reservoir) if reservoir else "all_reservoirs")
    print(f"Demo ({view}), models: {', '.join(model_names)}", flush=True)

    predictions, predict_seconds = predict_all(model_names)
    months = next(iter(predictions.values()))["pred_target_month"]
    period = f"{months.min():%b %Y} – {months.max():%b %Y}"

    scores = score(predictions, reservoir)
    costs = cost_table(scores)

    # Save the numbers behind the figures (handy for filling slides) next to them
    summary = scores.join(costs["fit_predict_seconds"]).assign(live_predict_seconds=pd.Series(predict_seconds))
    configure.DEMO_METRICS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(configure.DEMO_METRICS / f"{out_dir.name}.csv")

    for metric, (label, file_stem, _) in METRICS.items():
        save_figure(plot_metric(scores, metric, f"Test {label.split(' (')[0]}: {view}\n({period})"),
                    out_dir / f"{file_stem}.png")

    observed, predicted = percent_full(predictions, reservoir)
    save_figure(plot_percent_full(observed, predicted, f"Test predictions: {view} ({period})"),
                out_dir / "predictions.png")

    save_figure(plot_cost(costs, title=f"Cost vs test accuracy: {view}",
                          y_label="Relative RMSE, test (%)" if reservoir else "Mean relative RMSE, test (%)"),
                out_dir / "cost_vs_accuracy.png")

    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 200,
                           "display.max_columns", None):
        print(f"\nTest scores ({view}, {period}):\n{summary}")
    print(f"\nfigures saved to {out_dir}")


if __name__ == "__main__":
    main()