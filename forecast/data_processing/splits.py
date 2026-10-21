"""Label each forecast row train / val / test by its TARGET month.

"""
import pandas as pd

from forecast import configure

SPLIT_ORDER = ["train", "val", "test"]


def assign_splits(table: pd.DataFrame, feature_cols: list[str],
                  train_end, val_end) -> pd.DataFrame:
    """Returns a copy with only usable rows and a 'split' column, sorted by reservoir and month.

    """

    # Every lag setting is trained and scored on exactly the same rows, so comparisons are fair.
    max_lags = configure.MAX_LAGS

    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if not train_end < val_end:
        raise ValueError("train_end must be earlier than val_end")

    # A target at month index i needs volumes back to index i - max_lags
    months = pd.DatetimeIndex(sorted(table["month"].unique()))
    first_target = months[max_lags]

    # Drop the last month (no next month to predict) and targets before first_target.
    usable = table["pred_target_month"].notna() & (table["pred_target_month"] >= first_target)
    out = table[usable].copy()

    # Every input must be present on every usable row
    missing = out[feature_cols].isna().any()
    if missing.any():
        raise ValueError(f"inputs with missing values on usable rows: {missing[missing].index.tolist()}. "
                         f"Is a lag setting larger than {max_lags}?")

    out["split"] = "test"
    out.loc[out["pred_target_month"] <= val_end, "split"] = "val"
    out.loc[out["pred_target_month"] <= train_end, "split"] = "train"

    for name in SPLIT_ORDER:
        if not (out["split"] == name).any():
            raise ValueError(f"the {name} split is empty; check train_end / val_end")

    return out.sort_values(["reservoir", "month"]).reset_index(drop=True)


def select(table: pd.DataFrame, *names: str) -> pd.DataFrame:
    """Rows belonging to the given splits, e.g. select(t, "train", "val")"""
    unknown = set(names) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"unknown split name(s): {unknown}")
    return table[table["split"].isin(names)]