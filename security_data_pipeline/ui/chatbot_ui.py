#!/usr/bin/env python3
"""
FYND AI Chatbot – Semantic Reasoning Prototype
----------------------------------------------
Offline chatbot that answers data-driven security queries
using embeddings from data/security_ai_index.parquet
and data/ai_embeddings.npy.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_parquet("data/security_ai_index.parquet")
    emb = np.load("data/ai_embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, emb, model

df, embeddings, model = load_data()

# === UI Layout ===
st.set_page_config(page_title="FYND AI Chatbot", layout="wide", page_icon="🧠")
st.title("🧠 FYND AI – Security Intelligence Chatbot")
st.markdown("Ask me about **crime, collisions, or police activity** by division or year.")

query = st.text_input("💬 Type your question:", placeholder="e.g. Which division had the highest crime rate in 2022?")

# === Query Processing ===
if query:
    q_emb = model.encode([query])
    sims = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = sims.argsort()[-5:][::-1]
    top_df = df.iloc[top_idx].copy()

    st.subheader("🔎 Top Matches")
    st.dataframe(top_df[["location_name", "year", "total_crime_count", "police_presence_index", "security_score"]])

    # Simple template-based reasoning
    best = top_df.iloc[0]
    response = (
        f"In **{best.location_name} ({int(best.year)})**, "
        f"there were **{int(best.total_crime_count or 0)} total crimes**, "
        f"and the police presence index was **{round(best.police_presence_index or 0, 2)}**. "
        f"The overall security score was **{round(best.security_score or 0, 2)}**."
    )

    st.subheader("🤖 Chatbot Response")
    st.write(response)

    # Optional visualization
    st.subheader("📊 Context Overview")
    st.bar_chart(top_df.set_index("year")[["total_crime_count", "police_presence_index"]])