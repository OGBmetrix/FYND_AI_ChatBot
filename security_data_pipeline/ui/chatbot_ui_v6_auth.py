#!/usr/bin/env python3
"""
FYND AI – v6 Auth & Audit
-------------------------
Adds:
  • API-key sign-in with roles: admin / analyst / viewer
  • Feature gating (what each role can do)
  • Append-only audit logs (CSV)
Keeps: persistent memory + semantic retrieval + GPT reasoning.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os, json, datetime, hashlib, time, io
from pathlib import Path
from typing import Dict, Any, List

# =========================
# SETTINGS
# =========================
st.set_page_config(page_title="FYND AI v6 – Auth & Audit", layout="wide", page_icon="🔐")

MEMORY_FILE = Path("data/fynd_memory.json")
AUDIT_LOG = Path("logs/audit_log.csv")
INDEX_PARQUET = Path("data/security_ai_index.parquet")
INDEX_EMB = Path("data/ai_embeddings.npy")

# ---- Role policy (edit as you wish)
ROLE_PERMS = {
    "admin": {
        "view_memory", "export_memory", "clear_memory",
        "use_gpt", "view_data", "export_data",
    },
    "analyst": {
        "view_memory", "export_memory",
        "use_gpt", "view_data", "export_data",
    },
    "viewer": {
        "view_data"
    },
}

# ---- API keys -> roles (minimal PoC; swap to DB/Secrets in prod)
# You can set these via env vars (comma-separated "KEY:ROLE")
# e.g. FYND_USERS="abc123:admin, xyz789:analyst"
def load_user_keys() -> Dict[str, str]:
    users = {}
    env_cfg = os.getenv("FYND_USERS", "").strip()
    if env_cfg:
        for pair in env_cfg.split(","):
            pair = pair.strip()
            if not pair: 
                continue
            if ":" in pair:
                key, role = pair.split(":", 1)
                users[key.strip()] = role.strip().lower()
    # Fallback dev keys (remove in prod)
    if not users:
        users = {
            "admin-key-demo": "admin",
            "analyst-key-demo": "analyst",
            "viewer-key-demo": "viewer",
        }
    return users

USERS = load_user_keys()

# =========================
# Helpers
# =========================
def load_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"sessions": []}

def save_memory(mem):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(mem, indent=2, ensure_ascii=False), encoding="utf-8")

def load_resources():
    df = pd.read_parquet(INDEX_PARQUET)
    emb = np.load(INDEX_EMB)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, emb, model

def user_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]

def can(perm: str) -> bool:
    role = st.session_state.get("role")
    return bool(role and perm in ROLE_PERMS.get(role, set()))

def audit(action: str, detail: str = "", ok: bool = True, meta: Dict[str, Any] = None):
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "user": st.session_state.get("user_id", "anon"),
        "role": st.session_state.get("role", "none"),
        "action": action,
        "ok": ok,
        "detail": detail,
    }
    if meta:
        row.update(meta)
    hdr = not AUDIT_LOG.exists()
    pd.DataFrame([row]).to_csv(AUDIT_LOG, mode="a", index=False, header=hdr)

# =========================
# Auth UI
# =========================
st.title("🔐 FYND AI – Auth & Audit")

if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.role = None
    st.session_state.user_id = "anon"
    st.session_state.history = []

with st.sidebar:
    st.subheader("Sign in")
    api_key_in = st.text_input("Enter your FYND API Key", type="password")
    if st.button("Sign in"):
        role = USERS.get(api_key_in)
        if role:
            st.session_state.auth = True
            st.session_state.role = role
            st.session_state.user_id = user_fingerprint(api_key_in)
            st.success(f"Signed in as **{role}**")
            audit("signin", ok=True)
        else:
            st.error("Invalid key")
            audit("signin", ok=False)

    if st.session_state.auth:
        st.caption(f"User: `{st.session_state.user_id}` • Role: **{st.session_state.role}**")
        if st.button("Sign out"):
            audit("signout", ok=True)
            for k in ["auth", "role", "user_id", "history"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

# Block if not authed
if not st.session_state.auth:
    st.info("Enter a valid FYND API key to continue. (Configure via env var FYND_USERS)")
    st.stop()

# =========================
# OpenAI key (optional)
# =========================
openai.api_key = os.getenv("OPENAI_API_KEY")
with st.expander("🔑 Configure OpenAI (Optional)"):
    user_key = st.text_input("OpenAI API Key (for GPT & Whisper)", type="password")
    if user_key:
        openai.api_key = user_key
        st.success("✅ OpenAI key set")
        audit("set_openai_key", ok=True)

# =========================
# Load Data/Embeddings
# =========================
@st.cache_data
def _load_all():
    return load_resources(), load_memory()

((df, embeddings, sbert), memory) = _load_all()

# =========================
# Memory controls (gated)
# =========================
with st.expander("🧠 Memory"):
    if can("view_memory"):
        if memory["sessions"]:
            for s in memory["sessions"][-3:]:
                st.markdown(f"**🗓 {s['timestamp']} – {len(s['conversation'])} turns**")
        else:
            st.caption("No stored sessions.")
        if can("export_memory"):
            if st.button("Export memory (JSON)"):
                audit("export_memory", ok=True)
                st.download_button("Download", data=json.dumps(memory, indent=2), file_name="fynd_memory.json", mime="application/json")
        if can("clear_memory"):
            if st.button("Clear ALL memory"):
                save_memory({"sessions": []})
                audit("clear_memory", ok=True)
                st.warning("Memory cleared. Reload page.")
    else:
        st.warning("Your role cannot view memory.")

# =========================
# Chat display
# =========================
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# =========================
# Retrieval + Reasoning
# =========================
def retrieve(query: str) -> pd.DataFrame:
    q_emb = sbert.encode([query])
    sims = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = sims.argsort()[-5:][::-1]
    return df.iloc[top_idx].copy()

def answer_with_gpt(query: str, top_df: pd.DataFrame) -> str:
    # Compose base summary
    best = top_df.iloc[0]
    base = (
        f"In **{best.location_name} ({int(best.year)})**, total crimes: "
        f"**{int(best.total_crime_count or 0)}**, police presence index "
        f"**{round(best.police_presence_index or 0,2)}**, security score "
        f"**{round(best.security_score or 0,2)}**."
    )
    if not can("use_gpt") or not openai.api_key:
        return base

    # Build minimal cross-session context (last 1 session + last 5 turns)
    ctx_lines = []
    for sess in memory["sessions"][-1:]:
        for item in sess["conversation"][-6:]:
            ctx_lines.append(f"{item['role'].title()}: {item['message']}")
    for role, msg in st.session_state.history[-5:]:
        ctx_lines.append(f"{role.title()}: {msg}")

    table_md = top_df[
        ["location_name", "year", "total_crime_count", "police_presence_index", "security_score"]
    ].to_markdown(index=False)

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are FYND_AI, a public-safety analyst. Be factual, concise, and cite numbers from the table when possible."},
                {"role": "user",
                 "content": f"Context:\n" + "\n".join(ctx_lines) + f"\n\nQuestion: {query}\n\nRelevant data:\n{table_md}"}
            ],
            max_tokens=450,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        audit("gpt_error", ok=False, detail=str(e))
        return base

# =========================
# Main chat input
# =========================
user_q = st.chat_input("Ask about safety trends, police presence, collisions…")
if user_q:
    st.chat_message("user").write(user_q)
    st.session_state.history.append(("user", user_q))
    audit("query", ok=True, detail=f"len={len(user_q)}")

    top_df = retrieve(user_q)
    ans = answer_with_gpt(user_q, top_df)
    st.chat_message("assistant").write(ans)
    st.session_state.history.append(("assistant", ans))

    # Save a session snapshot (analyst/admin only to avoid sensitive viewer logging)
    if can("view_memory"):
        memory["sessions"].append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "conversation": [{"role": r, "message": m} for r, m in st.session_state.history]
        })
        save_memory(memory)
        audit("save_session", ok=True)

    # Data context (viewer can at least see data slice)
    if can("view_data"):
        with st.expander("🔎 Data Context"):
            st.dataframe(
                top_df[
                    ["location_name", "year", "total_crime_count", "police_presence_index", "security_score"]
                ]
            )
            if can("export_data"):
                csv = top_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="fynd_result.csv", mime="text/csv")
                audit("export_data", ok=True)