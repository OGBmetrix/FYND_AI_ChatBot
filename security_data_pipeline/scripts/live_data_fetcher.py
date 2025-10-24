"""
live_data_fetcher.py
Auto-discovers and fetches Toronto Police Major Crime Indicators (MCI)
data from the City of Toronto Open Data API for FYND AI.

Features
--------
✅ Automatically resolves the newest resource_id
✅ Falls back to cached parquet if API fails
✅ Provides summarize_recent(df, days) helper
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# -------------------------------
# 1️⃣  Auto-discover latest dataset
# -------------------------------
def get_latest_resource_id(dataset_name="major-crime-indicators") -> str | None:
    """Query CKAN for the newest resource ID by dataset name."""
    try:
        base = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show"
        resp = requests.get(f"{base}?id={dataset_name}", timeout=15)
        resp.raise_for_status()
        pkg = resp.json().get("result", {})
        for res in pkg.get("resources", []):
            if res.get("format", "").lower() == "csv":
                return res["id"]
        # fallback: first resource
        if pkg.get("resources"):
            return pkg["resources"][0]["id"]
    except Exception as e:
        print(f"[WARN] Could not auto-discover resource_id: {e}")
    return None


# -------------------------------
# 2️⃣  Fetch latest data (with fallback)
# -------------------------------
def fetch_live_mci(limit: int = 5000) -> pd.DataFrame:
    """Download MCI data or use cached copy."""
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"
    resource_id = get_latest_resource_id() or "fd1c82b6-74f0-48ac-bc3a-08ad69ea60a4"

    try:
        url = f"{base_url}?resource_id={resource_id}&limit={limit}"
        print(f"[INFO] Fetching live MCI data from: {url}")
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        data = r.json()["result"]["records"]
        df = pd.DataFrame(data)
        cache_path = Path(__file__).resolve().parents[1] / "data" / "mci_cached.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[OK] Cached latest MCI data → {cache_path}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch live MCI: {e}")
        cache_path = Path(__file__).resolve().parents[1] / "data" / "mci_cached.parquet"
        if cache_path.exists():
            print("[INFO] Using cached MCI data instead.")
            return pd.read_parquet(cache_path)
        return pd.DataFrame()


# -------------------------------
# 3️⃣  Summarization helper
# -------------------------------
def summarize_recent(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Aggregate incidents for the last N days by category."""
    if df.empty:
        return pd.DataFrame()
    date_col = next((c for c in df.columns if "occurrence" in c.lower() or "date" in c.lower()), None)
    cat_col = next((c for c in df.columns if "category" in c.lower()), None)
    if not date_col or not cat_col:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days)
    recent = df[df[date_col] >= cutoff]
    summary = (
        recent.groupby(cat_col)
        .size()
        .reset_index(name="Incidents")
        .sort_values("Incidents", ascending=False)
    )
    summary.rename(columns={cat_col: "MCI Category"}, inplace=True)
    return summary


# -------------------------------
# 4️⃣  Manual test mode
# -------------------------------
if __name__ == "__main__":
    df = fetch_live_mci()
    print(f"[INFO] Retrieved {len(df):,} rows")
    if not df.empty:
        print(summarize_recent(df, 7).head())
