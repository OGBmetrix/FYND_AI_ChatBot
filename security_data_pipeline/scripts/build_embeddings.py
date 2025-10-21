#!/usr/bin/env python3
"""
Build embeddings for FYND AI Chatbot
------------------------------------
This script takes `data/security_ai_base.parquet`,
turns each row into a natural-language sentence,
and generates embeddings for semantic search.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# === Load dataset ===
df = pd.read_parquet("data/security_ai_base.parquet")
print(f"[INFO] Loaded {len(df)} rows for embedding generation.")

# === Create text representation per row ===
def row_to_text(x):
    return (
        f"{x.location_name}, {x.year}: "
        f"Total crimes {x.total_crime_count}, "
        f"Violent {x.violent_crime_count}, "
        f"Property {x.property_crime_count}, "
        f"Collisions {x.total_collisions}, "
        f"Police presence {x.police_presence_index}, "
        f"Security score {x.security_score:.2f}."
    )

df["text"] = df.apply(row_to_text, axis=1)

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Encoding rows...")

embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# === Save artifacts ===
np.save("data/ai_embeddings.npy", embeddings)
df.to_parquet("data/security_ai_index.parquet", index=False)
print(f"[OK] Saved embeddings → data/ai_embeddings.npy ({embeddings.shape})")
print("[OK] Saved indexed parquet → data/security_ai_index.parquet")