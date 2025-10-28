import os
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")
OUT_INDEX = os.path.join(DATA_DIR, "vector_index.faiss")
OUT_TEXTS = os.path.join(DATA_DIR, "vector_texts.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# === LOAD MODEL ===
print(f"[INFO] Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# === COLLECT TEXTS ===
corpus = []

# 1️⃣ Load master parquet
master_path = os.path.join(DATA_DIR, "security_master_geo.parquet")
if os.path.exists(master_path):
    print(f"[INFO] Loading master dataset: {master_path}")
    df = pd.read_parquet(master_path)
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    for col in text_cols:
        corpus.extend(df[col].dropna().astype(str).tolist())

# 2️⃣ Collect text from other CSVs or knowledge folder
for file in os.listdir(KNOWLEDGE_DIR):
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(KNOWLEDGE_DIR, file))
            for col in df.select_dtypes(include=["object"]).columns:
                corpus.extend(df[col].dropna().astype(str).tolist())
            print(f"[INFO] Added knowledge from {file} ({len(df)} rows)")
        except Exception as e:
            print(f"[WARN] Skipping {file}: {e}")

if not corpus:
    raise RuntimeError("No text data found in master or knowledge folder. Add CSVs or ensure the parquet exists.")

print(f"[INFO] Building embeddings for {len(corpus)} text entries...")

# === ENCODE AND BUILD INDEX ===
embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === SAVE INDEX AND TEXT MAP ===
faiss.write_index(index, OUT_INDEX)
with open(OUT_TEXTS, "wb") as f:
    pickle.dump(corpus, f)

print(f"✅ Knowledge base built successfully!")
print(f"📦 Saved FAISS index → {OUT_INDEX}")
print(f"📘 Saved text map → {OUT_TEXTS}")
