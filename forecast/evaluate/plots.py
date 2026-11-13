"""Figures for tuning results and final results"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

TOTAL = "Reservoirs Total"   # matches the total row in metrics.evaluate()

# One fixed colour per model, so a model looks the same in every figure.
MODEL_COLOURS = {
    "naive":  "#2a78d6",   # blue
    "sarima": "#eb6834",   # orange
    "mlr":    "#1baf7a",   # aqua
    "svr":    "#eda100",   # yellow
    "ffnn":   "#e87ba4",   # magenta
    "lstm":   "#6250d6",   # violet
}
OBSERVED_COLOUR = "#222222"   # observed values: black so they never look like a model
GRID_COLOUR = "#e5e5e5"


def _colour(model: str) -> str:
    return MODEL_COLOURS.get(model, "#888888")


# Axis numbers with thousands separators (e.g. 12,500 rather than 12500)
THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")


def _style(ax) -> None:
    """Quiet axes: light horizontal grid, no top/right frame, so the data stands out."""
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", labelsize=8)


def save_figure(fig, path) -> None:
    """Save a figure as PNG (creating the folder if needed) and free its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(predictions_by_model: dict[str, pd.DataFrame], title: str = "") -> plt.Figure:
    """Observed vs predicted volume over time: one panel per reservoir, plus the network total.
    """
    first = next(iter(predictions_by_model.values()))
    reservoirs = sorted(first["reservoir"].unique())
    panels = reservoirs + [TOTAL]

    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.6 * n_rows), sharex=True)
    axes = axes.ravel()

    for ax, panel in zip(axes, panels):
        # Observed values are the same for every model, so they are drawn once.
        if panel == TOTAL:
            observed = first.groupby("pred_target_month")["y_true_ml"].sum()
        else:
            observed = first[first["reservoir"] == panel].set_index("pred_target_month")["y_true_ml"]
        ax.plot(observed.index, observed.values, color=OBSERVED_COLOUR, linewidth=2, label="Observed")

        for model, preds in predictions_by_model.items():
            if panel == TOTAL:
                predicted = preds.groupby("pred_target_month")["y_pred_ml"].sum()
            else:
                predicted = preds[preds["reservoir"] == panel].set_index("pred_target_month")["y_pred_ml"]
            ax.plot(predicted.index, predicted.values, color=_colour(model), linewidth=2,
                    linestyle="--", label=model)

        ax.set_title(panel.replace(" Reservoir", ""), fontsize=9,
                     fontweight="bold" if panel == TOTAL else "normal")
        ax.yaxis.set_major_formatter(THOUSANDS)
        _style(ax)

    for ax in axes[len(panels):]:
        ax.axis("off")                          # hide unused panels

    fig.supylabel("Volume (ML)", fontsize=9)
    fig.autofmt_xdate()

    # Legend: in the spare panel if there is one, otherwise below the grid, so it never covers a title.
    handles, labels = axes[0].get_legend_handles_labels()
    if len(axes) > len(panels):
        axes[len(panels)].legend(handles, labels, loc="center", frameon=False, fontsize=9)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, -0.03))
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_metric_bars(metrics_by_model: dict[str, pd.DataFrame], metric: str = "relative_rmse_pct",
                     label: str = "Relative RMSE (%)", title: str = "") -> plt.Figure:
    """One metric per reservoir (and the total), with one horizontal bar per model.
    """
    models = list(metrics_by_model)
    rows = list(next(iter(metrics_by_model.values())).index)      # reservoirs, then the total
    height = 0.8 / len(models)                                     # bars of one reservoir share 0.8 of a row

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) * max(1, len(models) / 2) + 1))
    y = np.arange(len(rows))
    for i, model in enumerate(models):
        values = metrics_by_model[model].loc[rows, metric]
        ax.barh(y + i * height, values, height=height * 0.9, color=_colour(model), label=model)

    ax.set_yticks(y + height * (len(models) - 1) / 2)
    ax.set_yticklabels([r.replace(" Reservoir", "") for r in rows], fontsize=8)
    ax.invert_yaxis()                                              # first reservoir at the top
    ax.set_xlabel(label, fontsize=9)
    ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if len(models) > 1:
        ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_cost_vs_accuracy(costs: pd.DataFrame, title: str = "",
                          x_col: str = "tuning_seconds",
                          x_label: str = "Total tuning time (s, log scale)",
                          y_label: str = "Mean relative RMSE, validation (%)",
                          yerr_col: str | None = None) -> plt.Figure:
    """Cost (log scale) against mean relative RMSE, one labelled point per model.

    yerr_col: optional column with the spread across seeds, drawn as a vertical error bar.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, row in costs.iterrows():
        x, y = row[x_col], row["mean_relative_rmse_pct"]
        if yerr_col and yerr_col in costs.columns and row[yerr_col] > 0:
            ax.errorbar(x, y, yerr=row[yerr_col], fmt="none", ecolor=_colour(model),
                        capsize=3, linewidth=1.2, zorder=2)
        ax.scatter(x, y, s=60, color=_colour(model), edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(model, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.2, y=0.15)

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    _style(ax)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig