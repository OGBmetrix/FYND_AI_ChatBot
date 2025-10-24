from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH, META_PATH = "data/vector_index.faiss", "data/vector_meta.pkl"

def search(query, top_k=5):
    index = faiss.read_index(INDEX_PATH)
    meta = pickle.load(open(META_PATH, "rb"))
    q_emb = MODEL.encode([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        confidence = round(1 - dist / np.max(D), 2)
        results.append({**meta[idx], "similarity": confidence})
    return results
