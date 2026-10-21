"""Download up-to-date SA Water reservoir record files and save to data/raw/sawater.

BoM rainfall files are downloaded manually from the BoM website and saved to data/raw/bom.
"""

from forecast.data_processing.download_sawater import download_sawater

if __name__ == "__main__":
    download_sawater() #dataset ID saved in configs/default.yaml