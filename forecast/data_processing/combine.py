"""Stack the raw data files into single tables. Combine the two tables together.

combine_sawater()   98 SA Water monthly CSVs   -> data/interim/sawater_combined.csv
combine_rainfall()  11 BoM monthly rainfall CSVs -> data/interim/rainfall_combined.csv
combine_datasets()  merge the two tables on month and reservoir/station -> data/interim/combined.csv

These functions only make the raw files stackable:
- consistent column names
- one row per record
- a common month key

They do NOT fix values.
Missing values, numeric conversion, capacity checks, and gap filling happen in clean.py

Every decision here comes from notebooks/eda/EDA_raw.ipynb.
"""
import re

import pandas as pd

from forecast import configure


# SA Water

# Column names differ between files (7 different headers across the 98 files).
SAWATER_COLUMN_MAP = {
    "date": "date",
    "reservoirname": "reservoir",
    "reservoirname1": "reservoir_alt",   # duplicate name column in 2 newer files
    "roreportlabel": "reservoir",        # name column in the Aug & Sep 2026 files
    "capacityml": "capacity_ml",
    "volumetodayml": "volume_ml",
    "volumethisdaylastyearml": "volume_last_year_ml",
    "averagevolumethismonth": "avg_volume_month_ml",
    "latitude": "latitude",
    "longitude": "longitude",
}

# Columns that exist in only some files and aren't needed for forecasting.
SAWATER_DROP_COLUMNS = {"roid", "rainfallthisweekmm"}

# Files that are duplicates of a previous month's data, re-published under a new month's name. 
# New data for their filename months (Dec 2023, Apr 2024, Jul 2024) is missing.
SAWATER_DROP_FILES = {
    "reservoirvolumesfordecember2023.csv",    # copy of November 2023
    "reservoir-volumes-for-april-2024.csv",   # copy of March 2024
    "reservoirvolumesforjuly2024.csv",        # copy of May 2024 (not June)
}

# A reading dated in the first half of a month reports the previous month (files are often published a few days after month-end, up to the 12th).
REPORT_LAG_DAYS = 15


def _key(column: str) -> str:
    """Reduce a column name to lowercase letters and digits."""
    return re.sub(r"[^a-z0-9]", "", column.lower())


def _detect_sep(path) -> str:
    """Return the file's separator.

    Two raw files are tab-separated rather than comma-separated.
    """
    # utf-8-sig so a byte-order mark doesn't end up in first_line
    with open(path, encoding="utf-8-sig") as f:
        first_line = f.readline()
    return "\t" if "\t" in first_line and "," not in first_line else ","


def _read_sawater_file(path) -> pd.DataFrame:
    """Read one SA Water CSV and give it the standard column names."""
    #  dtype=str: read every value as text so pandas can't guess types yet;
    #  encoding="utf-8-sig": strips the invisible byte-order mark some files start with.
    # na_values="nan": SA Water writes missing capacity/latitude/longitude as the word "nan".
    df = pd.read_csv(path, dtype=str, sep=_detect_sep(path), encoding="utf-8-sig", na_values="nan")

    keys = {col: _key(col) for col in df.columns}

    # Fail on a column that hasn't been mapped or dropped.
    unknown = {col for col, k in keys.items()
               if k not in SAWATER_COLUMN_MAP and k not in SAWATER_DROP_COLUMNS}
    if unknown:
        raise ValueError(f"{path.name}: unmapped columns {unknown}")

    df = df.rename(columns={col: SAWATER_COLUMN_MAP[k]
                            for col, k in keys.items() if k in SAWATER_COLUMN_MAP})
    df = df.drop(columns=[col for col, k in keys.items() if k in SAWATER_DROP_COLUMNS])

    if df.columns.duplicated().any():
        raise ValueError(f"{path.name}: duplicate columns after rename "
                         f"{df.columns[df.columns.duplicated()].tolist()}")

    # Keep the source filename on every row so any odd value can be traced back.
    return df.assign(source_file=path.name)


def combine_sawater() -> pd.DataFrame:
    """Stack all SA Water CSVs into one table with one row per reservoir per month.

    Saves the result to data/interim/sawater_combined.csv and also returns it.
    """
    files = sorted(configure.DATA_RAW_SAWATER.glob("*.csv"))
    print(f"SA Water: reading {len(files)} files")

    # 1. Read every file with standard column names and stack them 
    df = pd.concat([_read_sawater_file(f) for f in files], ignore_index=True)

    # All columns are text, so strip stray spaces from every value.
    for col in df.columns:
        df[col] = df[col].str.strip()

    # 2. Remove the duplicate name column
    both = df["reservoir_alt"].notna()
    mismatched = df.loc[both & (df["reservoir"] != df["reservoir_alt"])]
    if not mismatched.empty:
        raise ValueError("reservoir and reservoir_alt disagree in: "
                         f"{mismatched['source_file'].unique().tolist()}")
    df = df.drop(columns="reservoir_alt")

    # 3. Drop the duplicate files
    n_before = len(df)
    df = df[~df["source_file"].isin(SAWATER_DROP_FILES)]
    print(f"  dropped {n_before - len(df)} rows from {len(SAWATER_DROP_FILES)} duplicate files")

    # 4. Keep only real reservoir rows
    # The June 2018 file has a footer line ("Adelaide's consumption today is") that reads in as a 12th reservoir.
    # This catches it and any similar junk row in future files.
    is_reservoir = df["reservoir"].isin(configure.RAIN_STATIONS)
    if (~is_reservoir).any():
        print(f"  dropped non-reservoir rows: {df.loc[~is_reservoir, 'reservoir'].unique().tolist()}")
    df = df[is_reservoir]

    # 5. Add the month each reading represents
    # "month" is the month-end the reading belongs to, since the raw date isn't always a month-end
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")
    df["month"] = (df["date"] - pd.Timedelta(days=REPORT_LAG_DAYS)) + pd.offsets.MonthEnd(0)

    # 6. Sanity check: exactly one row per reservoir per month 
    duplicated = df.duplicated(["reservoir", "month"], keep=False)
    if duplicated.any():
        raise ValueError("more than one row for the same reservoir and month:\n"
                         f"{df.loc[duplicated, ['reservoir', 'month', 'source_file']]}")

    # 7. Tidy column order and row order, then save
    df = df[["month", "reservoir", "date", "capacity_ml", "volume_ml",
             "volume_last_year_ml", "avg_volume_month_ml",
             "latitude", "longitude", "source_file"]]
    df = df.sort_values(["reservoir", "month"]).reset_index(drop=True)

    configure.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(configure.SAWATER_COMBINED, index=False)
    print(f"  saved {len(df)} rows, {df['reservoir'].nunique()} reservoirs, "
          f"{df['month'].nunique()} months -> {configure.SAWATER_COMBINED.name}")
    return df


# BoM rainfall

# Standard names for the BoM monthly columns ("Product code" is dropped).
RAINFALL_COLUMN_MAP = {
    "Station number": "station",
    "Year": "year",
    "Month": "month_num",
    "Monthly Precipitation Total (millimetres)": "rain_mm",
    "Quality": "quality",           # Y = quality-controlled, N = not yet
}


def _read_rainfall_file(path) -> pd.DataFrame:
    """Read one BoM *_Data1.csv file and give it the standard column names."""
    # keep station number as text so its leading zero survives
    # na_values="null": BoM writes missing rainfall as the word "null".
    df = pd.read_csv(path, dtype={"Station number": str, "Month": str},
                     na_values="null", encoding="utf-8-sig")
    return df.rename(columns=RAINFALL_COLUMN_MAP)[list(RAINFALL_COLUMN_MAP.values())]


def combine_rainfall() -> pd.DataFrame:
    """Stack all BoM monthly rainfall files into one table, one row per station per month.

    Saves the result to data/interim/rainfall_combined.csv and also returns it.
    """
    files = sorted(configure.DATA_RAW_BOM.glob("*_Data1.csv"))
    print(f"Rainfall: reading {len(files)} files")

    # 1. Check the files match the station mapping
    # The station number is the second part of the filename: IDCJAC0001_<station>_Data1.csv
    stations_in_files = {f.name.split("_")[1] for f in files}
    stations_in_map = set(configure.RAIN_STATIONS.values())
    if stations_in_files != stations_in_map:
        raise ValueError(f"missing files for: {stations_in_map - stations_in_files}; "
                         f"unexpected files for: {stations_in_files - stations_in_map}")

    # 2. Read and stack
    df = pd.concat([_read_rainfall_file(f) for f in files], ignore_index=True)

    # 3. Add the month-end date to match SA Water's "month" field
    df["month"] = (pd.to_datetime(df["year"].astype(str) + "-" + df["month_num"] + "-01")
                   + pd.offsets.MonthEnd(0))

    # 4. Sanity check: exactly one row per station per month
    duplicated = df.duplicated(["station", "month"], keep=False)
    if duplicated.any():
        raise ValueError(f"more than one row for the same station and month:\n"
                         f"{df.loc[duplicated, ['station', 'month']]}")

    # 5. Tidy and save
    df = df[["month", "station", "rain_mm", "quality"]]
    df = df.sort_values(["station", "month"]).reset_index(drop=True)

    configure.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(configure.RAINFALL_COMBINED, index=False)
    print(f"  saved {len(df)} rows, {df['station'].nunique()} stations, "
          f"{df['month'].min().date()} to {df['month'].max().date()} -> {configure.RAINFALL_COMBINED.name}")
    return df


# Merge
def combine_datasets() -> pd.DataFrame:
    """Merge the cleaned SA Water and rainfall tables into one row per reservoir per month.

    Saves the result to data/processed/combined.csv and also returns it.
    """
    sawater = pd.read_csv(configure.SAWATER_CLEAN, parse_dates=["month"])
    rainfall = pd.read_csv(configure.RAINFALL_CLEAN, parse_dates=["month"], dtype={"station": str})

    # 1. Attach each reservoir's rainfall station
    sawater["station"] = sawater["reservoir"].map(configure.RAIN_STATIONS)
    if sawater["station"].isna().any():
        raise ValueError("reservoirs with no rainfall station: "
                         f"{sawater.loc[sawater['station'].isna(), 'reservoir'].unique().tolist()}")

    # 2. Join on station + month. Each reservoir matches exactly one station-month.
    rain_columns = rainfall[["station", "month", "rain_mm", "rain_source", "quality"]]
    rain_columns = rain_columns.rename(columns={"quality": "rain_quality"})
    df = sawater.merge(rain_columns, on=["station", "month"], how="left", validate="one_to_one")

    # 3. Sanity checks: no rows gained or lost, no unmatched rainfall
    if len(df) != len(sawater):
        raise ValueError(f"row count changed in the merge: {len(sawater)} -> {len(df)}")
    if df["rain_mm"].isna().any():
        raise ValueError("rows with no rainfall after the merge:\n"
                         f"{df.loc[df['rain_mm'].isna(), ['reservoir', 'month']]}")

    # 4. Tidy column order and row order, then save
    df = df[["month", "reservoir", "station", "capacity_ml", "volume_ml", "volume_source",
             "yoy_change_ml", "rain_mm", "rain_source", "rain_quality"]]
    df = df.sort_values(["reservoir", "month"]).reset_index(drop=True)

    configure.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(configure.COMBINED, index=False)
    print(f"Combined: saved {len(df)} rows, {df['reservoir'].nunique()} reservoirs, "
          f"{df['month'].nunique()} months -> {configure.COMBINED.name}")
    return df