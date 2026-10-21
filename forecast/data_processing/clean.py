"""Fix values and fill gaps in the stacked tables. Every fill is flagged so it can be traced.

clean_sawater()   data/interim/sawater_combined.csv  -> data/interim/sawater_clean.csv
clean_rainfall()  data/interim/rainfall_combined.csv -> data/interim/rainfall_clean.csv

Every decision here comes from notebooks/exploratory_data_analysis/EDA_interim.ipynb
"""
import pandas as pd

from forecast import configure


# SA Water

# A reservoir's capacity was re-surveyed once (March 2019). 
# The latest capacity value is used throughout, so the re-survey must stay small for that to be safe.
MAX_CAPACITY_CHANGE = 0.01      # as a fraction of capacity (largest seen: 0.9%)

# Volume at month m is also reported in the file for month m + 12 ("volume this day last year").
YEAR_LAG_MONTHS = 12


def clean_sawater() -> pd.DataFrame:
    """Clean the stacked SA Water table and save it to data/interim/sawater_clean.csv.

    Volumes above capacity (Warren, Kangaroo Creek) are real and are NOT clipped.
    """
    df = pd.read_csv(configure.SAWATER_COMBINED, parse_dates=["month"])
    print(f"SA Water: read {len(df)} rows from {configure.SAWATER_COMBINED.name}")

    # 1. Drop excluded reservoirs
    excluded = df["reservoir"].isin(configure.EXCLUDED_RESERVOIRS)
    df = df[~excluded]
    print(f"  dropped {excluded.sum()} rows from {sorted(configure.EXCLUDED_RESERVOIRS)}")

    missing_reservoirs = set(configure.MODEL_RAIN_STATIONS) - set(df["reservoir"])
    if missing_reservoirs:
        raise ValueError(f"no data for reservoirs {sorted(missing_reservoirs)}")

    # Numeric columns must already be numbers (combine.py leaves unreadable values as blanks)
    for col in ["capacity_ml", "volume_ml", "volume_last_year_ml"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} is not numeric")
    if df["volume_ml"].isna().any() or (df["volume_ml"] < 0).any():
        raise ValueError("volume_ml has blank or negative readings")

    # 2. One capacity per reservoir: the latest non-blank value
    capacity = df.sort_values("month").groupby("reservoir")["capacity_ml"].last()
    spread = df.groupby("reservoir")["capacity_ml"].agg(["min", "max"])
    change = (spread["max"] - spread["min"]) / spread["max"]
    if capacity.isna().any() or (capacity <= 0).any():
        raise ValueError(f"capacity is blank or not positive for: {capacity[~(capacity > 0)].index.tolist()}")
    if (change > MAX_CAPACITY_CHANGE).any():
        raise ValueError("capacity changed by more than "
                         f"{MAX_CAPACITY_CHANGE:.0%} for: {change[change > MAX_CAPACITY_CHANGE].index.tolist()}")

    # 3. Put every reservoir on a complete monthly grid; absent months appear as blank rows.
    months = pd.date_range(df["month"].min(), df["month"].max(), freq="ME")
    grid = pd.MultiIndex.from_product([sorted(df["reservoir"].unique()), months],
                                      names=["reservoir", "month"])
    vol = df.set_index(["reservoir", "month"])[["volume_ml", "volume_last_year_ml"]].reindex(grid)
    vol["volume_source"] = "observed"
    absent = vol["volume_ml"].isna()

    # 3. For non-existent/duplicated files, recover from the file 12 months later
    recovered = vol.groupby(level="reservoir")["volume_last_year_ml"].shift(-YEAR_LAG_MONTHS)
    use_recovered = absent & recovered.notna()
    vol.loc[use_recovered, "volume_ml"] = recovered[use_recovered]
    vol.loc[use_recovered, "volume_source"] = "recovered"

    # Anything still blank is interpolated
    still_absent = vol["volume_ml"].isna()
    vol["volume_ml"] = vol.groupby(level="reservoir")["volume_ml"].transform(
        # limit=1 leaves the second month of a longer gap blank, which stops the build.
        lambda s: s.interpolate(limit=1, limit_area="inside"))
    vol.loc[still_absent, "volume_source"] = "interpolated"
    if vol["volume_ml"].isna().any():
        raise ValueError("a gap longer than one month could not be filled:\n"
                         f"{vol.loc[vol['volume_ml'].isna()].index.tolist()}")

    # 4. Year-on-year change: volume now minus volume 12 months earlier in the cleaned series
    earlier = vol.groupby(level="reservoir")["volume_ml"].shift(YEAR_LAG_MONTHS)
    reported = vol["volume_ml"] - vol["volume_last_year_ml"]

    # falling back to the reported value where there are no earlier months.
    vol["yoy_change_ml"] = (vol["volume_ml"] - earlier).fillna(reported)
    if vol["yoy_change_ml"].isna().any():
        raise ValueError("year-on-year change could not be computed for some rows")

    # 5. Tidy and save
    out = vol.reset_index()
    out["capacity_ml"] = out["reservoir"].map(capacity)
    out = out[["month", "reservoir", "capacity_ml", "volume_ml", "volume_source", "yoy_change_ml"]]
    out = out.sort_values(["reservoir", "month"]).reset_index(drop=True)

    configure.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(configure.SAWATER_CLEAN, index=False)
    print(f"  volume_source: {out['volume_source'].value_counts().to_dict()}")
    print(f"  {(out['volume_ml'] > out['capacity_ml']).sum()} readings above capacity (kept, not clipped)")
    print(f"  saved {len(out)} rows, {out['reservoir'].nunique()} reservoirs, "
          f"{out['month'].nunique()} months -> {configure.SAWATER_CLEAN.name}")
    return out


# BoM rainfall

# Correlations and scaling ratios use observed months from this date onwards.
DONOR_FIT_START = "2000-01-01" #enough overlap to be stable, without the older records.
DONOR_MIN_OVERLAP = 60          # months two stations must share to be compared (5 years)


def clean_rainfall() -> pd.DataFrame:
    """Clean the stacked BoM table and save it to data/interim/rainfall_clean.csv.

    rain_source says "observed" or "filled" (donor_station says which). 
    The BoM quality flag is kept as it came; it is blank for filled rows.
    """

    rain = pd.read_csv(configure.RAINFALL_COMBINED, parse_dates=["month"], dtype={"station": str})
    print(f"Rainfall: read {len(rain)} rows from {configure.RAINFALL_COMBINED.name}")

    # 1. Stations of the retained reservoirs only
    stations = sorted(configure.MODEL_RAIN_STATIONS.values())
    rain = rain[rain["station"].isin(stations)]

    # 2. Study period = the SA Water period, and a complete grid within it
    sawater_months = pd.read_csv(configure.SAWATER_COMBINED, usecols=["month"], parse_dates=["month"])["month"]
    window = pd.date_range(sawater_months.min(), sawater_months.max(), freq="ME")
    grid = pd.MultiIndex.from_product([stations, window], names=["station", "month"])
    r = rain.set_index(["station", "month"])[["rain_mm", "quality"]].reindex(grid)
    r["rain_source"] = "observed"
    r["donor_station"] = pd.Series(pd.NA, index=r.index, dtype="object")

    # 3. Fill gaps.

    # Station-by-station correlation of monthly rainfall. 
    observed = rain.pivot(index="month", columns="station", values="rain_mm")     # wide table of real BoM readings for the whole history (not just the study period)
    fit = observed[observed.index >= DONOR_FIT_START]
    corr = fit.corr(min_periods=DONOR_MIN_OVERLAP) 

    # Work out, once per station with gaps, who its donors are and how to scale them.
    plan = {}   # station -> ([donors, best first], {donor: scaling ratio})
    for station in sorted({s for s, _ in r.index[r["rain_mm"].isna()]}):   # stations with >= 1 gap
        # Rank the other stations by correlation with this one, best first.
        # drop(station): a station is perfectly correlated with itself.
        # dropna(): removes stations without enough overlapping months.
        donors = corr[station].drop(station).dropna().sort_values(ascending=False).index.tolist()

        #figure out by how much to scale each donor's reading to match this station's typical rainfall. 
        ratios = {}
        for donor in donors:
            # Only months where BOTH stations have a real reading
            both = fit[[station, donor]].dropna()

            # How much rain this station gets per unit of the donor's rain (e.g. 1.39 = 39% more).
            # Ratio of totals rather than an average of monthly ratios, so near-zero months can't inflate the average.
            ratios[donor] = both[station].sum() / both[donor].sum()
        plan[station] = (donors, ratios)

    # Fill every blank station-month in the study-period grid `r`.
    n_filled = 0   # only used for the summary print at the end
    for station, month in r.index[r["rain_mm"].isna()]:
        donors, ratios = plan[station]
        # Take the best-correlated donor that has a real reading in this particular month.
        donor = next((d for d in donors if pd.notna(observed.at[month, d])), None)
        if donor is None:
            raise ValueError(f"no donor station has a reading for {station} in {month.date()}")
        
        # Donor's reading scaled to this station's typical rainfall, plus flags for traceability.
        r.at[(station, month), "rain_mm"] = observed.at[month, donor] * ratios[donor]
        r.at[(station, month), "rain_source"] = "filled"
        r.at[(station, month), "donor_station"] = donor
        n_filled += 1


    for station, (donors, ratios) in plan.items():
        used = r.loc[station].query("rain_source == 'filled'")
        print(f"  {station}: {len(used)} months filled, donors used "
              f"{used['donor_station'].value_counts().to_dict()}, "
              + ", ".join(f"ratio vs {d} = {ratios[d]:.2f}" for d in used["donor_station"].unique()))

    if r["rain_mm"].isna().any() or (r["rain_mm"] < 0).any():
        raise ValueError("rain_mm still has blank or negative values after filling")

    # 4. Tidy and save
    out = r.reset_index()[["month", "station", "rain_mm", "rain_source", "donor_station", "quality"]]
    out = out.sort_values(["station", "month"]).reset_index(drop=True)

    configure.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(configure.RAINFALL_CLEAN, index=False)
    print(f"  saved {len(out)} rows ({n_filled} filled), {out['station'].nunique()} stations, "
          f"{out['month'].nunique()} months -> {configure.RAINFALL_CLEAN.name}")
    return out