#!/usr/bin/env python3
"""
FYND AI Chatbot – v2 Conversational Reasoning
---------------------------------------------
Hybrid chatbot that:
  • Retrieves top semantic matches from security_ai_index.parquet
  • Summarizes insights using OpenAI GPT reasoning
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Safely handle a missing 'openai' package so editors/CI won't flag an import error.
# If OpenAI isn't installed, we provide a minimal dummy object with the same attributes
# the rest of the code expects (api_key and chat.completions.create) so runtime checks
# and exception handling continue to work.
try:
    import importlib
    openai = importlib.import_module("openai")
except Exception:
    class _DummyOpenAI:
        api_key = None

        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    raise RuntimeError(
                        "OpenAI SDK is not installed. Install the 'openai' package to enable GPT reasoning."
                    )

    openai = _DummyOpenAI()

# =============================================
# 1️⃣  PAGE SETUP
# =============================================
st.set_page_config(page_title="FYND AI Chatbot v2", layout="wide", page_icon="🧠")
st.title("🧠 FYND AI – Conversational Security Intelligence")
st.caption("Semantic retrieval + AI reasoning over security data.")

# =============================================
# 2️⃣  API KEY SETUP
# =============================================
openai.api_key = os.getenv("OPENAI_API_KEY")

with st.expander("🔑 Configure OpenAI (Optional)"):
    user_key = st.text_input("Enter your OpenAI API Key", type="password")
    if user_key:
        openai.api_key = user_key
        st.success("✅ API key loaded successfully.")

# =============================================
# 3️⃣  LOAD DATA + MODEL
# =============================================
@st.cache_data
def load_resources():
    df = pd.read_parquet("data/security_ai_index.parquet")
    emb = np.load("data/ai_embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, emb, model

df, embeddings, model = load_resources()

# =============================================
# 4️⃣  CHAT INTERFACE
# =============================================
query = st.text_input(
    "💬 Ask a security-related question:",
    placeholder="e.g. Which division had the highest police activity in 2023?",
)

if query:
    q_emb = model.encode([query])
    sims = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = sims.argsort()[-5:][::-1]
    top_df = df.iloc[top_idx].copy()

    st.subheader("🔎 Top Matching Records")
    st.dataframe(
        top_df[["location_name", "year", "total_crime_count", "police_presence_index", "security_score"]]
    )

    # -----------------------------------------
    # 5️⃣ LOCAL SUMMARY (fallback if no OpenAI)
    # -----------------------------------------
    best = top_df.iloc[0]
    base_summary = (
        f"In **{best.location_name} ({int(best.year)})**, "
        f"there were **{int(best.total_crime_count or 0)} total crimes**, "
        f"and the police presence index was **{round(best.police_presence_index or 0, 2)}**. "
        f"The overall security score was **{round(best.security_score or 0, 2)}**."
    )

    # -----------------------------------------
    # 6️⃣ GPT REASONING (optional)
    # -----------------------------------------
    if openai.api_key:
        try:
            with st.spinner("🤖 Generating AI summary..."):
                summary = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful Canadian public safety analyst. "
                                "Explain key security insights clearly and factually. "
                                "Avoid speculation; summarize patterns and implications."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {query}\n\n"
                                f"Relevant records:\n{top_df[['location_name','year','total_crime_count','police_presence_index','security_score']].to_markdown(index=False)}"
                            ),
                        },
                    ],
                    max_tokens=400,
                )
                ai_response = summary.choices[0].message.content.strip()
                st.subheader("🤖 FYND AI Reasoning")
                st.write(ai_response)
        except Exception as e:
            st.warning(f"⚠️ GPT reasoning unavailable ({e}). Showing local summary.")
            st.subheader("🧩 Local Summary")
            st.write(base_summary)
    else:
        st.subheader("🧩 Local Summary")
        st.write(base_summary)

    # -----------------------------------------
    # 7️⃣ OPTIONAL CHART
    # -----------------------------------------
    st.subheader("📊 Trend Overview")
    st.bar_chart(top_df.set_index("year")[["total_crime_count", "police_presence_index"]])
