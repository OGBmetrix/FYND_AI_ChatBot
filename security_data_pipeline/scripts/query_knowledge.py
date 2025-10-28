import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
INDEX_PATH = os.path.join(DATA_DIR, "vector_index.faiss")
TEXTS_PATH = os.path.join(DATA_DIR, "vector_texts.pkl")

# === MODEL ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def kb_search(query: str, top_k: int = 5):
    """Semantic search through FYND AI knowledge base with confidence scoring"""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index file: {INDEX_PATH}")
    if not os.path.exists(TEXTS_PATH):
        raise FileNotFoundError(f"Missing text mapping: {TEXTS_PATH}")

    # Load index + texts
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        corpus = pickle.load(f)

    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True)

    # Perform FAISS search
    distances, indices = index.search(query_vec, top_k)

    # Normalize distances → confidence (0–1)
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    confs = 1 - (distances - min_dist) / (max_dist - min_dist + 1e-9)

    results = []
    for dist, conf, idx in zip(distances[0], confs[0], indices[0]):
        if idx < len(corpus):
            results.append({
                "text": corpus[idx],
                "confidence": float(conf),
                "source": "FYND Knowledge Base"
            })

    return results