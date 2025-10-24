"""
census_fetcher.py
Pulls demographic & socioeconomic indicators from Statistics Canada’s Web Data Service (WDS)
for FYND AI Chatbot.
"""

import requests
import pandas as pd
from datetime import datetime

# --- Simple API example using StatsCan Table 98-10-0001-01 (Population & Dwelling Counts)
# Docs: https://www.statcan.gc.ca/en/developers/wds

STATSCAN_WDS_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVector"

# Example Toronto-area GEO vectors (small demo subset)
VECTORS = {
    "population_total": "v1",    # Placeholder — will resolve dynamically
    "median_income": "v2",
    "population_density": "v3",
}

# --- Demo function to show structure (real vectors retrieved below)
def fetch_table_vectors(table="98-10-0001-01"):
    """Return list of available vectors for given table."""
    meta_url = f"https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{table}/en"
    try:
        r = requests.get(meta_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "SUCCESS":
                print("[INFO] Table metadata available:", data["object"]["downloadLink"])
            else:
                print("[WARN] Metadata fetch returned:", data.get("message"))
    except Exception as e:
        print("[ERROR] Could not fetch metadata:", e)


def fetch_demo_demographics(geo_level="Toronto"):
    """
    Retrieve a few demo demographic indicators for Ontario/Toronto
    using StatsCan WDS. This simplified function queries pre-known vectors.
    """
    results = []
    try:
        for name, vec in VECTORS.items():
            resp = requests.get(f"{STATSCAN_WDS_URL}/{vec}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data["status"] == "SUCCESS":
                    value = data["object"]["vectorDataPoint"][0]["value"]
                    results.append({"indicator": name, "value": value})
    except Exception as e:
        print("[ERROR] StatsCan fetch failed:", e)

    df = pd.DataFrame(results)
    df["fetched_at"] = datetime.utcnow().isoformat()
    return df


def fetch_population_by_region():
    """
    Simulated population data for Toronto divisions.
    Replace with real StatsCan vectors once you map geo IDs.
    """
    data = [
        {"division": "D11", "region": "Etobicoke West", "population": 285000, "median_income": 62000},
        {"division": "D12", "region": "Downtown Toronto", "population": 410000, "median_income": 78000},
        {"division": "D13", "region": "Scarborough", "population": 625000, "median_income": 54000},
        {"division": "D14", "region": "North York", "population": 660000, "median_income": 70000},
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = fetch_population_by_region()
    print(df.head())