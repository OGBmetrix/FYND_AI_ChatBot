from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np, faiss, os, pickle

DATA_DIR = "data/knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "data/vector_index.faiss"
META_PATH = "data/vector_meta.pkl"

def build_index():
    model = SentenceTransformer(MODEL_NAME)
    texts, meta = [], []

    # Ingest CSVs
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            for _, row in df.iterrows():
                text = " ".join(map(str, row.values))
                texts.append(text)
                meta.append({"source": file})

    # Ingest API caches (JSON or CSV)
    for file in os.listdir("data/api_cache"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join("data/api_cache", file))
            for _, row in df.iterrows():
                text = " ".join(map(str, row.values))
                texts.append(text)
                meta.append({"source": file})

    # Build embeddings
    emb = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb, dtype="float32"))

    # Save artifacts
    faiss.write_index(index, INDEX_PATH)
    pickle.dump(meta, open(META_PATH, "wb"))
    print(f"[OK] Trained {len(texts)} records into vector index")

if __name__ == "__main__":
    build_index()
