#!/usr/bin/env python3
"""
FYND AI Chatbot – v4 Persistent Memory
--------------------------------------
Adds local JSON-based memory persistence.
Session history is saved and reloaded automatically between runs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional OpenAI SDK (provide a lightweight stub if package is missing)
try:
    import importlib
    openai = importlib.import_module("openai")
except Exception:
    # openai not installed; provide a minimal stub so the app can still run without GPT features.
    class _Completions:
        @staticmethod
        def create(*args, **kwargs):
            raise RuntimeError("openai package not installed; install with 'pip install openai' to enable GPT features.")

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _OpenAIStub:
        def __init__(self):
            self.api_key = None
            self.chat = _Chat()

    openai = _OpenAIStub()

import os, json, datetime
from pathlib import Path

# =============================================
# 1️⃣  PAGE SETUP
# =============================================
st.set_page_config(page_title="FYND AI Chatbot v4", layout="wide", page_icon="🧠")
st.title("🧠 FYND AI – Persistent Memory Security Analyst")
st.caption("Remembers previous sessions and insights across restarts.")

MEMORY_FILE = Path("data/fynd_memory.json")

# =============================================
# 2️⃣  MEMORY UTILITIES
# =============================================
def load_memory():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"sessions": []}

def save_memory(history):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

memory = load_memory()

# Initialize current session
if "history" not in st.session_state:
    st.session_state.history = []

# =============================================
# 3️⃣  API KEY
# =============================================
openai.api_key = os.getenv("OPENAI_API_KEY")
with st.expander("🔑 Configure OpenAI (Optional)"):
    user_key = st.text_input("Enter your OpenAI API Key", type="password")
    if user_key:
        openai.api_key = user_key
        st.success("✅ API key loaded successfully.")

# =============================================
# 4️⃣  LOAD DATA
# =============================================
@st.cache_data
def load_resources():
    df = pd.read_parquet("data/security_ai_index.parquet")
    emb = np.load("data/ai_embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, emb, model

df, embeddings, model = load_resources()

# =============================================
# 5️⃣  DISPLAY SAVED SESSIONS
# =============================================
if memory["sessions"]:
    with st.expander("📜 View Saved Sessions"):
        for s in memory["sessions"]:
            st.markdown(f"**🗓 {s['timestamp']} — {len(s['conversation'])} exchanges**")
            for entry in s["conversation"]:
                st.markdown(f"- **{entry['role'].title()}:** {entry['message']}")

# =============================================
# 6️⃣  CHAT DISPLAY (current session)
# =============================================
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# =============================================
# 7️⃣  CHAT INPUT
# =============================================
query = st.chat_input("Ask about security data or trends...")

if query:
    st.chat_message("user").write(query)
    st.session_state.history.append(("user", query))

    # Semantic retrieval
    q_emb = model.encode([query])
    sims = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = sims.argsort()[-5:][::-1]
    top_df = df.iloc[top_idx].copy()

    # Local fallback summary
    best = top_df.iloc[0]
    base_summary = (
        f"In **{best.location_name} ({int(best.year)})**, total crimes were "
        f"**{int(best.total_crime_count or 0)}**, police presence index "
        f"**{round(best.police_presence_index or 0,2)}**, security score "
        f"**{round(best.security_score or 0,2)}**."
    )

    # GPT reasoning with persistent context
    if openai.api_key:
        full_context = []
        for session in memory["sessions"][-2:]:
            for item in session["conversation"]:
                full_context.append(f"{item['role'].title()}: {item['message']}")
        for role, msg in st.session_state.history[-5:]:
            full_context.append(f"{role.title()}: {msg}")

        try:
            with st.spinner("🤖 Analyzing with context..."):
                completion = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are FYND_AI, a Canadian public safety intelligence assistant. "
                                "Use prior discussion context and provide concise, factual insights."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{chr(10).join(full_context)}\n\n"
                                f"New question: {query}\n\n"
                                f"Relevant data:\n"
                                f"{top_df[['location_name','year','total_crime_count','police_presence_index','security_score']].to_markdown(index=False)}"
                            ),
                        },
                    ],
                    max_tokens=500,
                )
                answer = completion.choices[0].message.content.strip()
        except Exception as e:
            answer = f"⚠️ GPT reasoning unavailable ({e}).\n\n{base_summary}"
    else:
        answer = base_summary

    # Output and store
    st.chat_message("assistant").write(answer)
    st.session_state.history.append(("assistant", answer))

    # Save to persistent memory
    memory["sessions"].append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "conversation": [{"role": r, "message": m} for r, m in st.session_state.history],
    })
    save_memory(memory)

    # Optional data context
    with st.expander("🔎 Data Context"):
        st.dataframe(
            top_df[
                ["location_name", "year", "total_crime_count", "police_presence_index", "security_score"]
            ]
        )
