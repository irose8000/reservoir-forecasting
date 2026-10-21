"""Download the SA Water reservoir-volume CSVs from data.sa.gov.au."""
import requests
from forecast import configure

def download_sawater() -> None:
    """Download every CSV resource in the dataset into data/raw/sawater/."""
    out_dir = configure.DATA_RAW_SAWATER
    out_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(configure.SAWATER_API_URL, params={"id": configure.SAWATER_DATASET_ID}, timeout=30)
    response.raise_for_status()
    resources = response.json()["result"]["resources"]

    csvs = [r for r in resources if r["url"].lower().endswith(".csv")]
    print(f"Found {len(csvs)} CSVs ({len(resources)} resources total)")

    for resource in csvs:
        filename = resource["url"].split("/")[-1]
        path = out_dir / filename
        if path.exists():
            continue  # already downloaded
        file_response = requests.get(resource["url"], timeout=60) #timeout prevents a stalled connection from hanging forever.
        file_response.raise_for_status() # prevents a failed download (404, server error) from being silently saved as an HTML error page
        path.write_bytes(file_response.content)
        print("Downloaded", filename)