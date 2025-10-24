"""
alert_fetcher.py
Fetches real-time emergency and weather alerts for Ontario / Toronto.

Sources:
 - Environment Canada (CAP/ATOM feed)
 - Alert Ready Canada (RSS)
"""

import feedparser
import pandas as pd
from datetime import datetime
import requests

# Feeds
ENV_CANADA_FEED = "https://dd.weather.gc.ca/alerts/cap/ON.atom"
ALERT_READY_FEED = "https://www.alertready.ca/rss-en.xml"


def fetch_environment_canada_alerts():
    """Fetch active Environment Canada alerts (Ontario)."""
    feed = feedparser.parse(ENV_CANADA_FEED)
    records = []
    for e in feed.entries:
        records.append({
            "source": "Environment Canada",
            "title": e.title,
            "summary": getattr(e, "summary", ""),
            "updated": getattr(e, "updated", None),
            "link": e.link,
            "region": e.get("cap_areaDesc", "Ontario")
        })
    return pd.DataFrame(records)


def fetch_alert_ready():
    """Fetch national Alert Ready RSS feed."""
    try:
        feed = feedparser.parse(ALERT_READY_FEED)
        records = []
        for e in feed.entries:
            records.append({
                "source": "Alert Ready Canada",
                "title": e.title,
                "summary": getattr(e, "summary", ""),
                "updated": getattr(e, "updated", None),
                "link": e.link
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[ERROR] Alert Ready fetch failed: {e}")
        return pd.DataFrame()


def fetch_all_alerts():
    """Merge all alert sources into a single DataFrame."""
    env_df = fetch_environment_canada_alerts()
    ar_df = fetch_alert_ready()

    frames = [df for df in [env_df, ar_df] if not df.empty]
    if not frames:
        print("[INFO] No active alerts at this time.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["fetched_at"] = datetime.utcnow().isoformat()
    return combined


if __name__ == "__main__":
    df = fetch_all_alerts()
    if df.empty:
        print("✅ No active alerts found.")
    else:
        print(f"⚠️ Found {len(df)} active alerts:")
        print(df[["source", "title", "region", "updated"]].head())