#!/usr/bin/env python3
"""
FYND AI Chatbot – v3 Contextual Memory
--------------------------------------
Conversational agent that remembers previous questions
and builds contextual answers using local embeddings + GPT reasoning.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Optional OpenAI SDK; if not installed, continue without GPT features
openai = None
try:
    import importlib
    import importlib.util
    if importlib.util.find_spec("openai") is not None:
        openai = importlib.import_module("openai")
except Exception:
    openai = None

# =============================================
# 1️⃣ PAGE SETUP
# =============================================
# 2️⃣ API KEY
# =============================================
if openai:
    openai.api_key = os.getenv("OPENAI_API_KEY")
with st.expander("🔑 Configure OpenAI (Optional)"):
    user_key = st.text_input("Enter your OpenAI API Key", type="password")
    if user_key:
        if openai:
            openai.api_key = user_key
            st.success("✅ API key loaded successfully.")
        else:
            st.warning("OpenAI SDK not installed; install the 'openai' package to enable GPT reasoning.")
with st.expander("🔑 Configure OpenAI (Optional)"):
    user_key = st.text_input("Enter your OpenAI API Key", type="password")
    if user_key:
        openai.api_key = user_key
        st.success("✅ API key loaded successfully.")

# =============================================
# 3️⃣ LOAD DATA
# =============================================
@st.cache_data
def load_resources():
    df = pd.read_parquet("data/security_ai_index.parquet")
    emb = np.load("data/ai_embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, emb, model

df, embeddings, model = load_resources()

# =============================================
# 4️⃣ SESSION MEMORY
# =============================================
if "history" not in st.session_state:
    st.session_state.history = []  # [(user, bot), ...]

# helper: show chat history
def show_history():
    for role, msg in st.session_state.history:
        if role == "user":
            st.chat_message("user").markdown(f"**You:** {msg}")
        else:
            st.chat_message("assistant").write(msg)

show_history()

# =============================================
# 5️⃣ CHAT INPUT
# =============================================
query = st.chat_input("Ask about security, crime, or police presence...")

if query:
    st.chat_message("user").markdown(f"**You:** {query}")
    st.session_state.history.append(("user", query))

    # ---  semantic retrieval
    q_emb = model.encode([query])
    sims = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = sims.argsort()[-5:][::-1]
    top_df = df.iloc[top_idx].copy()

    # ---  GPT reasoning with conversation memory
    if openai and getattr(openai, "api_key", None):
        conversation = "\n".join(
            [f"User: {u}\nAssistant: {a}" for u, a in st.session_state.history if a]
        )
        # safe fallback summary if GPT is unavailable
        base_summary = (
            "Relevant records:\n"
            + top_df[['location_name','year','total_crime_count','police_presence_index','security_score']].to_markdown(index=False)
        )
        try:
            with st.spinner("🤖 Thinking..."):
                completion = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are FYND_AI, a Canadian public safety analyst assistant. "
                                "Use factual insights, recall context from earlier discussion, "
                                "and keep answers concise but informative."
                            ),
                        },
                        {"role": "user", "content": f"{conversation}\n\nNew Question: {query}"},
                        {
                            "role": "user",
                            "content": (
                                f"Relevant records:\n"
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
        # no OpenAI key/SDK available — provide the fallback summary
        base_summary = (
            "Relevant records:\n"
            + top_df[['location_name','year','total_crime_count','police_presence_index','security_score']].to_markdown(index=False)
        )
        answer = base_summary
    
    # ---  display + store response
    st.chat_message("assistant").write(answer)
    st.session_state.history.append(("assistant", answer))

    # ---  optional quick view
    with st.expander("🔎 Top Matches (Data Context)"):
        st.dataframe(
            top_df[
                ["location_name", "year", "total_crime_count", "police_presence_index", "security_score"]
            ]
        )
