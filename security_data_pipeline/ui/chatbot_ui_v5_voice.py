#!/usr/bin/env python3
"""
FYND AI Chatbot – v5 Voice I/O
------------------------------
Adds:
  • 🎙️ Voice input (mic or audio upload)
  • 🔊 Text-to-speech responses (offline pyttsx3 or online gTTS)
  • Keeps v4 features: persistent memory + embeddings + GPT reasoning
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os, io, json, datetime, tempfile
from pathlib import Path

# Optional speech libs (we'll import lazily)
# pip install faster-whisper  (local STT, optional)
# pip install gTTS            (online TTS)
# pip install pyttsx3         (offline TTS)

MEMORY_FILE = Path("data/fynd_memory.json")

st.set_page_config(page_title="FYND AI v5 – Voice", layout="wide", page_icon="🗣️")
st.title("🗣️ FYND AI – Voice-Enabled Security Analyst")
st.caption("Talk to your data: mic/upload → semantic retrieval → GPT reasoning → spoken answer.")

# -----------------------------
# API KEY
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
with st.expander("🔑 Configure OpenAI (Optional)"):
    key = st.text_input("OpenAI API Key (for Whisper & GPT)", type="password")
    if key:
        openai.api_key = key
        st.success("✅ API key loaded")

# -----------------------------
# Memory utils
# -----------------------------
def load_memory():
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"sessions": []}

def save_memory(mem):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(mem, indent=2, ensure_ascii=False), encoding="utf-8")

memory = load_memory()
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, message)]

# -----------------------------
# Load data + model
# -----------------------------
@st.cache_data
def load_resources():
    df = pd.read_parquet("data/security_ai_index.parquet")
    emb = np.load("data/ai_embeddings.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, emb, model

df, embeddings, sbert = load_resources()

# -----------------------------
# UI: show saved sessions
# -----------------------------
with st.expander("📜 Saved Sessions"):
    if memory["sessions"]:
        for s in memory["sessions"]:
            st.markdown(f"**🗓 {s['timestamp']} – {len(s['conversation'])} turns**")
            for turn in s["conversation"]:
                st.markdown(f"- **{turn['role'].title()}:** {turn['message']}")
    else:
        st.info("No saved sessions yet.")

# -----------------------------
# Helper: display chat history
# -----------------------------
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# -----------------------------
# 🔊 TTS (text → audio bytes)
# -----------------------------
def speak_tts(text: str) -> bytes:
    """
    Try offline pyttsx3 first; fallback to gTTS (requires internet).
    Returns WAV/MP3 bytes to feed st.audio.
    """
    # Try pyttsx3 (WAV)
    try:
        import pyttsx3
        import wave
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_wav = tf.name
        engine.save_to_file(text, out_wav)
        engine.runAndWait()
        data = Path(out_wav).read_bytes()
        Path(out_wav).unlink(missing_ok=True)
        return data  # WAV
    except Exception:
        pass

    # Fallback: gTTS (MP3)
    try:
        from gtts import gTTS
        mp3_obj = io.BytesIO()
        gTTS(text).write_to_fp(mp3_obj)
        mp3_obj.seek(0)
        return mp3_obj.read()  # MP3
    except Exception:
        # Last fallback: return nothing; UI will just show text
        return b""

# -----------------------------
# 🎙️ STT (audio → text)
# -----------------------------
def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """
    Prefer OpenAI Whisper API if key is set; else try faster-whisper local.
    """
    # 1) OpenAI Whisper API
    if openai.api_key:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file_bytes)
                tmp_path = tf.name
            with open(tmp_path, "rb") as audio_file:
                # Whisper v1 (chat-style)
                # Newer API style: client = OpenAI(); client.audio.transcriptions.create(...)
                # Here we use legacy openai for compatibility with many envs.
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            Path(tmp_path).unlink(missing_ok=True)
            return transcript.text.strip()
        except Exception as e:
            st.warning(f"OpenAI Whisper failed ({e}). Trying local model…")

    # 2) Local faster-whisper (tiny/int8 recommended for CPU)
    try:
        from faster_whisper import WhisperModel
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix or ".wav") as tf:
            tf.write(file_bytes)
            audio_path = tf.name
        model = WhisperModel("tiny", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        Path(audio_path).unlink(missing_ok=True)
        return text
    except Exception as e:
        st.error(f"Local transcription failed: {e}")
        return ""

# -----------------------------
# 🔎 Retrieval + Reasoning
# -----------------------------
def retrieve_and_answer(query: str) -> str:
    # Semantic retrieval
    q_emb = sbert.encode([query])
    sims = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = sims.argsort()[-5:][::-1]
    top_df = df.iloc[top_idx].copy()

    # Local fallback summary
    best = top_df.iloc[0]
    base_summary = (
        f"In **{best.location_name} ({int(best.year)})**, total crimes were **{int(best.total_crime_count or 0)}**, "
        f"police presence index **{round(best.police_presence_index or 0,2)}**, "
        f"security score **{round(best.security_score or 0,2)}**."
    )

    # GPT reasoning with cross-session context
    if openai.api_key:
        # take last two saved sessions + last five live turns
        ctx = []
        for sess in memory["sessions"][-2:]:
            for item in sess["conversation"]:
                ctx.append(f"{item['role'].title()}: {item['message']}")
        for role, msg in st.session_state.history[-5:]:
            ctx.append(f"{role.title()}: {msg}")

        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are FYND_AI, a Canadian public safety intelligence assistant. "
                            "Be factual, concise, and use the provided data context. "
                            "If data is missing, say so and suggest the nearest proxy."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{chr(10).join(ctx)}\n\n"
                            f"User Question: {query}\n\n"
                            f"Relevant Data:\n"
                            f"{top_df[['location_name','year','total_crime_count','police_presence_index','security_score']].to_markdown(index=False)}"
                        ),
                    },
                ],
                max_tokens=500,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"GPT reasoning unavailable ({e}). Falling back to local summary.")
            return base_summary
    else:
        return base_summary

# -----------------------------
# INPUT MODES
# -----------------------------
tabs = st.tabs(["💬 Text", "🎙️ Voice (Mic/Upload)"])

# ---- Text tab ----
with tabs[0]:
    user_q = st.chat_input("Type your question…")
    if user_q:
        st.chat_message("user").write(user_q)
        st.session_state.history.append(("user", user_q))
        answer = retrieve_and_answer(user_q)
        st.chat_message("assistant").write(answer)
        st.session_state.history.append(("assistant", answer))

        # Speak it
        audio_bytes = speak_tts(answer)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")  # pyttsx3 is WAV; gTTS MP3 also works

        # Save session snapshot
        memory["sessions"].append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "conversation": [{"role": r, "message": m} for r, m in st.session_state.history]
        })
        save_memory(memory)

# ---- Voice tab ----
with tabs[1]:
    st.markdown("**Option A:** Upload an audio file (WAV/MP3/M4A), or\n**Option B:** Use your system recorder and upload the file here.")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        st.info("Transcribing…")
        text = transcribe_audio(audio_file.read(), audio_file.name)
        if not text:
            st.error("Could not transcribe audio.")
        else:
            st.success(f"📝 Transcribed text: {text}")
            st.chat_message("user").write(text)
            st.session_state.history.append(("user", text))

            answer = retrieve_and_answer(text)
            st.chat_message("assistant").write(answer)
            st.session_state.history.append(("assistant", answer))

            # Speak it
            audio_bytes = speak_tts(answer)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

            # Save session snapshot
            memory["sessions"].append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "conversation": [{"role": r, "message": m} for r, m in st.session_state.history]
            })
            save_memory(memory)