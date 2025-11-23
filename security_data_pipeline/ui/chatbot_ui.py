import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# --- FYND Intelligence Fetchers ---
# changed imports to relative so module resolution works when the package is imported/run
from .live_data_fetcher import fetch_live_mci, summarize_recent
from .urban_safety_fetcher import fetch_urban_safety
from .alert_fetcher import fetch_all_alerts
from .news_fetcher import fetch_latest_news
from .census_fetcher import fetch_population_by_region

# --- FYND Knowledge Base (Semantic Reasoning Layer) ---
try:
    from query_knowledge import search as kb_search
    KB_READY = True
except Exception as e:
    KB_READY = False
    st.warning(f"⚠️ Knowledge base not ready: {e}")

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="FYND AI Security Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 FYND AI — Security Intelligence Chatbot")
st.caption("Integrating Toronto Police Data + Urban Safety + Alerts + News + Census + Semantic Reasoning")

# -----------------------------
# LOAD LOCAL MASTER DATA
# -----------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "security_master_geo.parquet"

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# -----------------------------
# CHATBOT SECTION
# -----------------------------
st.subheader("💬 Ask FYND AI about Toronto’s safety and security")
query = st.text_input("Type your question below:", placeholder="e.g. What happened this week in Toronto?")

# --- Sidebar Diagnostics ---
with st.sidebar:
    st.subheader("System Status")
    st.write("📂 Master Dataset:", f"{len(df)} rows" if not df.empty else "Not Loaded")
    st.write("🧠 Knowledge Base:", "✅ Ready" if KB_READY else "❌ Missing")
    st.write("🌦️ APIs Linked:", "Live MCI | Urban | Alerts | News | Census")

# --- Core Query Logic ---
if query:
    query_lower = query.lower().strip()

    # === 1️⃣ Car Theft Hotspots ===
    if "car" in query_lower and "theft" in query_lower:
        metric_cols = [c for c in df.columns if "auto_theft" in c or "theft" in c]
        if metric_cols:
            result = df.groupby("location_name")[metric_cols].sum(numeric_only=True)
            top_location = result.sum(axis=1).idxmax()
            st.success(f"🚗 The highest car theft activity is in **{top_location}**.")
            st.bar_chart(result.sum(axis=1).sort_values(ascending=False).head(10))
        else:
            st.warning("No auto theft data found in your dataset.")

    # === 2️⃣ Robbery Hotspots ===
    elif "robbery" in query_lower:
        metric_cols = [c for c in df.columns if "robbery" in c]
        if metric_cols:
            result = df.groupby("location_name")[metric_cols].sum(numeric_only=True)
            top_location = result.sum(axis=1).idxmax()
            st.success(f"💰 The highest robbery incidents are in **{top_location}**.")
            st.bar_chart(result.sum(axis=1).sort_values(ascending=False).head(10))
        else:
            st.warning("No robbery data found in your dataset.")

    # === 3️⃣ Yearly Crime Trend ===
    elif "crime" in query_lower and "trend" in query_lower:
        if "year" in df.columns and "total_crime_count" in df.columns:
            yearly = df.groupby("year")["total_crime_count"].sum().reset_index()
            st.info("📈 Overall Crime Trend in Toronto")
            fig = px.line(yearly, x="year", y="total_crime_count", title="Year-over-Year Crime Trend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No yearly crime trend data available.")

    # === 4️⃣ Real-Time Crime Updates ===
    elif "latest" in query_lower or "recent" in query_lower or "this week" in query_lower:
        st.info("🔎 Fetching live Toronto Police incident data...")
        live_df = fetch_live_mci()
        if live_df.empty:
            st.error("No live data available right now. Try again later.")
        else:
            summary = summarize_recent(live_df, 7)
            st.success("📅 Major Crimes in the Last 7 Days")
            st.dataframe(summary)
            if not summary.empty:
                fig = px.bar(summary, x="MCI Category", y="Incidents",
                             title="Recent Major Crimes (7 days)", color="MCI Category")
                st.plotly_chart(fig, use_container_width=True)

    # === 5️⃣ Map Visualization ===
    elif "map" in query_lower or "show" in query_lower:
        if {"latitude", "longitude"}.issubset(df.columns):
            st.info("🗺️ Toronto Division Map")
            map_df = df.groupby(["location_name", "latitude", "longitude"]).size().reset_index(name="incidents")
            st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
        else:
            st.warning("No geographic data available for map display.")

    # === 6️⃣ Urban Safety Intelligence ===
    elif any(word in query_lower for word in ["urban", "safety", "response", "fire", "community", "wellbeing"]):
        st.info("🌆 Fetching City of Toronto Urban Safety data...")
        datasets = fetch_urban_safety()
        if not datasets:
            st.error("⚠️ Could not load any safety intelligence datasets.")
        else:
            for name, df_city in datasets.items():
                if not df_city.empty:
                    st.success(f"📊 {name.replace('_', ' ').title()} — showing latest data")
                    st.dataframe(df_city.head(10))
                    numeric_cols = df_city.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        fig = px.histogram(df_city, x=numeric_cols[0],
                                           title=f"{name.title()} — {numeric_cols[0]} Distribution")
                        st.plotly_chart(fig, use_container_width=True)

    # === 7️⃣ Real-Time Alerts (Weather & Emergencies) ===
    elif any(word in query_lower for word in ["alert", "emergency", "weather", "warning", "storm"]):
        st.info("🚨 Checking Environment Canada & Alert Ready feeds…")
        alerts_df = fetch_all_alerts()
        if alerts_df.empty:
            st.success("✅ No active alerts in Ontario right now.")
        else:
            st.warning(f"⚠️ {len(alerts_df)} active alerts detected.")
            for _, row in alerts_df.head(5).iterrows():
                st.markdown(f"**{row['source']}** — {row['title']}")
                st.caption(f"{row.get('region','')} | {row.get('updated','')}")
                st.write(row.get('summary', '')[:400] + "…")
                st.markdown(f"[🔗 Read more]({row['link']})")

    # === 8️⃣ News Intelligence (Crime Headlines) ===
    elif any(word in query_lower for word in ["news", "headline", "update", "incident", "media", "report"]):
        st.info("🗞️ Fetching live Canadian crime & safety headlines…")
        topic = "Toronto crime" if "toronto" in query_lower else "canada crime"
        news_df = fetch_latest_news(topic)
        if news_df.empty:
            st.warning("No recent safety news found.")
        else:
            st.success(f"📰 Showing {len(news_df)} latest headlines:")
            for _, row in news_df.head(5).iterrows():
                st.markdown(f"**{row['title']}**")
                st.caption(f"{row.get('pubDate','')} | {row.get('source','')}")
                st.write(row.get('description', '')[:400] + "…")
                st.markdown(f"[🔗 Read more]({row['link']})")

    # === 9️⃣ Census / Demographic Insights ===
    elif any(word in query_lower for word in ["population", "income", "density", "demographic", "risk", "region compare"]):
        st.info("📊 Fetching StatsCan demographic context for Toronto divisions…")
        census_df = fetch_population_by_region()
        if census_df.empty:
            st.warning("No census data available right now.")
        else:
            st.dataframe(census_df)
            fig = px.bar(census_df, x="region", y="population",
                         color="median_income", title="Population & Income by Region")
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ FYND AI can now relate crime risk to demographics.")

    # === 🔟 Semantic Knowledge Base (Reasoning Layer) ===
    elif KB_READY:
        st.info("🧠 Engaging FYND AI Knowledge Base for deeper insights...")
        results = kb_search(query, top_k=5)
        if not results:
            st.warning("No close matches found in the knowledge base.")
        else:
            for r in results:
                st.markdown(f"**📘 Source:** {r['source']} — Confidence: {r['confidence']*100:.0f}%")
                st.caption(r['text'][:350] + "…")
            avg_conf = np.mean([r["confidence"] for r in results])
            if avg_conf >= 0.85:
                st.success("Answer quality: **High confidence** ✅")
            elif avg_conf >= 0.6:
                st.warning("Answer quality: **Moderate confidence** ⚠️")
            else:
                st.error("Answer quality: **Low confidence** ❗ FYND AI is estimating based on patterns.")

    # === 🧩 Fallback ===
    else:
        st.warning("🤖 I don’t understand that yet. Try asking about *car thefts*, *alerts*, *news*, *income*, or *urban safety*.")

else:
    st.info("💡 Try asking: *Show latest Toronto crime news* or *Are there any active weather alerts?*")

# small helper: intent router and safe fetch wrappers
from typing import Callable, Any
import time
import logging
logging.basicConfig(level=logging.INFO)

def safe_fetch(fn: Callable[..., Any], *args, timeout: float = 10.0, **kwargs) -> Any:
    """Call external fetcher with basic timeout and exception handling."""
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        return result
    except Exception as e:
        logging.exception("Fetcher failed: %s", fn.__name__)
        st.error(f"Error fetching {fn.__name__}: {e}")
        return pd.DataFrame() if 'pandas' in str(type(e)).lower() else {}
    finally:
        elapsed = time.time() - start
        logging.info("%s finished in %.2fs", fn.__name__, elapsed)

# simple keyword-based intent detection (replaceable with a model later)
INTENTS = {
    "car_theft": {"must": ["car", "auto", "vehicle"], "any": ["theft", "stolen"]},
    "robbery": {"any": ["robbery", "robbed"]},
    "trend": {"any": ["trend", "year", "yearly", "over time", "year-over-year"]},
    "latest": {"any": ["latest", "recent", "this week", "today"]},
    "map": {"any": ["map", "show", "where", "location"]},
    "urban_safety": {"any": ["urban", "safety", "response", "community", "wellbeing", "fire"]},
    "alerts": {"any": ["alert", "emergency", "weather", "warning", "storm"]},
    "news": {"any": ["news", "headline", "update", "media", "report", "incident"]},
    "census": {"any": ["population", "income", "density", "demographic", "region", "compare"]},
}

def detect_intent(query: str) -> str:
    q = query.lower()
    for intent, rules in INTENTS.items():
        if "must" in rules and not all(w in q for w in rules["must"]):
            continue
        if "any" in rules and any(w in q for w in rules["any"]):
            return intent
    return "kb_or_fallback"