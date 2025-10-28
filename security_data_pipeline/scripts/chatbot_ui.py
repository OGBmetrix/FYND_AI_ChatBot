import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# --- FYND Intelligence Fetchers ---
from live_data_fetcher import fetch_live_mci, summarize_recent
from urban_safety_fetcher import fetch_urban_safety
from alert_fetcher import fetch_all_alerts
from news_fetcher import fetch_latest_news
from census_fetcher import fetch_population_by_region
from intent_parser import parse_intent
from reasoning_engine import reason_about_data

# --- FYND Knowledge Base (Semantic Reasoning Layer) ---
try:
    from query_knowledge import kb_search
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
    if results:
        st.info("📘 Knowledge Base Answers:")
        for r in results:
            conf = r.get("confidence", 0)
            source = r.get("source", "Knowledge Base")
            st.markdown(f"**📘 Source:** {source} — Confidence: {conf*100:.0f}%")
            st.write(r.get("text", "")[:400] + "…")
    else:
        st.warning("🤖 FYND AI is working tirelessly to improve our datasets and knowledge base. Please check back soon for more intelligent insights!")
else:
    st.info("🤖 FYND AI is working tirelessly to improve our datasets and understanding. Try asking about *car thefts*, *alerts*, *news*, or *urban safety* for now!")
    intent_data = parse_intent(query)

# -----------------------------
# INTENT DETECTION & REASONING
# -----------------------------
intent_data = {"intent": "other", "confidence": 0.0}  # default fallback

# Provide a safe default context snippet so parse_intent has something to format.
# This can be enriched later with conversation history or relevant dataset summaries.
context_snippet = ""

try:
    intent_data = parse_intent(f"{context_snippet}\nUser: {query}")
except Exception as e:
    st.warning(f"⚠️ Intent parsing failed: {e}")
    intent_data = {"intent": "other", "confidence": 0.0}

intent = intent_data.get("intent", "other")
confidence = intent_data.get("confidence", 0.0)

st.caption(f"🤔 Detected intent: *{intent}* (confidence {confidence*100:.0f}%)")

# If FYND AI is uncertain, it politely asks for clarification
if confidence < 0.6:
    st.info("🧠 FYND AI wants to be sure — are you asking about **crime stats**, **alerts**, **demographics**, or **news**?")
    st.stop()

# Define a simple get_recent_context function
def get_recent_context():
    return "No recent context available."

st.write(f"🧩 FYND AI Context: {get_recent_context()}")

# Generate reasoning output
insight = reason_about_data(intent, query, df)
if insight:
    st.success(insight)
else:
    st.warning("🤖 FYND AI is working tirelessly to improve our datasets and reasoning.")