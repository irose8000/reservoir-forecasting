# reservoir-forecasting


All BoM data downloaded October 17th.

| Reservoir | BoM Station No. | Station Name | Distance from Reservoir |
| --- | --- | --- | --- |
| Mt Bold Reservoir | 023734 | Mount Bold Reservoir | 2.4 km |
| Happy Valley Reservoir | 023721 | Happy Valley Reservoir | 1.3 km |
| Millbrook Reservoir | 023731 | Cudlee Creek (Millbrook Reservoir) | 1.8 km |
| Kangaroo Creek Reservoir | 023892 | Montacute | 3.6 km |
| Hope Valley Reservoir | 023096 | Adelaide (Hope Valley Reservoir) | 0.8 km |
| Myponga Reservoir | 023783 | Myponga Reservoir | 1.3 km |
| Warren Reservoir | 023878 | Mount Crawford AWS | 2.9 km |
| South Para Reservoir | 023820 | Williamstown (South Para Reservoir) | 2.2 km |
| Barossa Reservoir | 023752 | Williamstown | 2.5 km |
| Little Para Reservoir | 023915 | Gould Creek (Little Para Reservoir) | 1.5 km |
| Middle River Reservoir | 022835 | Parndana (Turkey Lane) | 3.9 km |


Results from the cleaning process:
| Stage | Files | Rows | Reservoirs | Months |
|---|---|---|---|---|
| Raw CSVs | 98 | 1,034 | 11 + footer text | 95 (3 repeated) |
| Drop 3 duplicate files (`combine_sawater`) | 95 | 1,001 | 11 + footer text | 95 |
| Drop footer row → `sawater_combined.csv` | 95 | 1,000 | 11 | 95 |
| Drop Middle River (`clean_sawater`) | 95 | 950 | 10 | 95 |
| Fill the 6 missing months → `sawater_clean.csv` | 95 | 1,010 | 10 | 101 |
