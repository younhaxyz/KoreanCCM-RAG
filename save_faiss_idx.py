import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

MODEL_NAME = "Bossssss/ccm-retriever"
DATASET_NAME = "Bossssss/CCM-list"

INDEX_PATH = "./faiss.index"
META_PATH = "./metadata.pkl"

model = SentenceTransformer(MODEL_NAME)
dataset = load_dataset(DATASET_NAME)["train"]

embeddings = []
metadata = []

for row in dataset:
    lyric = row["lyrics"]
    title = row["title"]

    emb = model.encode(lyric, normalize_embeddings=True)
    embeddings.append(emb)

    metadata.append({
        "title": title,
        "lyrics": lyric,
        "artist": row.get("artist", "")
    })

embeddings = np.array(embeddings).astype("float32")

# FAISS index (cosine → inner product)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)

index.add(embeddings)

# save
faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("[SAVE] FAISS index")