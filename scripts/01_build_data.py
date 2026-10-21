"""Build the modelling dataset from the raw files.

Each step saves its output, so later steps can read those files without re-running everything.
"""
from forecast.data_processing.combine import combine_sawater, combine_rainfall
from forecast.data_processing.clean import clean_sawater, clean_rainfall
from forecast.data_processing.combine import combine_datasets


def main() -> None:
    # Step 1: stack the raw files independently
    combine_sawater()
    combine_rainfall()

    # Step 2: Clean each dataset independently
    clean_sawater()
    clean_rainfall()

    # Step 3: merge reservoir volumes with rainfall
    combine_datasets()


if __name__ == "__main__":
    main()