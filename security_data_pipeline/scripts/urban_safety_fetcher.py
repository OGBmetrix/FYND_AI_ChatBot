"""
urban_safety_fetcher.py
Fetches broader City of Toronto urban safety intelligence datasets:
 - Fire response times
 - Community crisis data
 - Neighborhood wellbeing
"""

import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"

DATASETS = {
    "fire_response": "c3c8f9f9-80c5-4ad4-8222-06e9ee9f3d68",
    "crisis_response": "706a9e5e-46f3-47b9-8fa7-8b8b7b9a7f15",
    "neighborhood_wellbeing": "c0c2e66e-7a46-4a7a-8e5d-27b9d8a0a6c1",
    "shelter_occupancy": "f48bdb84-08cd-4c7b-8f3e-74f5f3ce31e2",
}

def fetch_dataset(resource_id, limit=5000):
    try:
        resp = requests.get(BASE_URL, params={"resource_id": resource_id, "limit": limit})
        resp.raise_for_status()
        data = resp.json()["result"]["records"]
        return pd.DataFrame(data)
    except Exception as e:
        print(f"[ERROR] Could not fetch dataset ({resource_id}): {e}")
        return pd.DataFrame()

def fetch_urban_safety():
    print("[INFO] Fetching Urban Safety datasets...")
    datasets = {name: fetch_dataset(rid) for name, rid in DATASETS.items()}

    for name, df in datasets.items():
        print(f"[DATA] {name}: {len(df)} records, columns: {list(df.columns)[:5]}")

    return datasets

if __name__ == "__main__":
    all_data = fetch_urban_safety()
    for key, df in all_data.items():
        print(f"\n=== {key.upper()} SAMPLE ===")
        print(df.head())