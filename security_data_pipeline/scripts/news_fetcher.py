"""
news_fetcher.py
Fetches live Canadian crime & safety news for FYND AI Chatbot.
Sources: NewsData.io or GNews (choose based on your API key).
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional

# === CONFIG ===
# You can get a free NewsData.io API key from https://newsdata.io/
NEWSDATA_API_KEY = "YOUR_NEWSDATA_API_KEY"
GNEWS_API_KEY = "YOUR_GNEWS_API_KEY"  # optional backup

# --- Option 1: NewsData.io (preferred) ---
def fetch_newsdata(query="crime Toronto", country="ca", max_results=10) -> pd.DataFrame:
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&country={country}&language=en"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("results", [])
        records = []
        for a in articles[:max_results]:
            records.append({
                "source": "NewsData.io",
                "title": a.get("title"),
                "description": a.get("description"),
                "pubDate": a.get("pubDate"),
                "link": a.get("link"),
                "category": a.get("category"),
                "country": a.get("country"),
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[ERROR] NewsData fetch failed: {e}")
        return pd.DataFrame()

# --- Option 2: GNews (backup) ---
def fetch_gnews(query="crime Toronto", max_results=10) -> pd.DataFrame:
    url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country=ca&max={max_results}&token={GNEWS_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        records = []
        for a in articles:
            records.append({
                "source": "GNews",
                "title": a.get("title"),
                "description": a.get("description"),
                "pubDate": a.get("publishedAt"),
                "link": a.get("url"),
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[ERROR] GNews fetch failed: {e}")
        return pd.DataFrame()


def fetch_latest_news(topic: Optional[str] = "crime Toronto") -> pd.DataFrame:
    """Try NewsData first, fallback to GNews."""
    df = fetch_newsdata(topic)
    if df.empty:
        df = fetch_gnews(topic)
    df["fetched_at"] = datetime.utcnow().isoformat()
    return df


if __name__ == "__main__":
    df = fetch_latest_news("crime Toronto")
    if df.empty:
        print("❌ No news found.")
    else:
        print(f"✅ Retrieved {len(df)} articles.")
        print(df[["title", "pubDate"]].head())