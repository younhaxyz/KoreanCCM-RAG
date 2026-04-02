from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import faiss
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

MODEL_NAME = "Bossssss/ccm-retriever"
INDEX_PATH = "./faiss.index"
META_PATH = "./metadata.pkl"

TOP_K = 5

# 의미없는 쿼리 차단 목록
BLOCKED_QUERIES = {"none", "null", "없음", "아니", "아니야", "아니요", "응아니야", "ㄴ", "no", "nope"}

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)


def is_valid_query(query: str) -> bool:
    q = query.strip().lower()
    if len(q) <= 1:
        return False
    if q in BLOCKED_QUERIES:
        return False
    return True


def search(query):
    if not is_valid_query(query):
        return []

    query_vec = model.encode(query, normalize_embeddings=True)
    query_vec = np.array([query_vec]).astype("float32")

    scores, indices = index.search(query_vec, TOP_K * 5)

    # top-1이 0.4 미만이면 결과 없음
    if scores[0][0] < 0.4:
        return []

    # top-5(인덱스 4)의 스코어 확인
    top5_idx = min(TOP_K - 1, len(scores[0]) - 1)
    top5_score = scores[0][top5_idx]

    # top-5가 0.4 이상이면 → 0.4 이상인 것 모두 출력 (no cap)
    # top-5가 0.4 미만이면 → top-5까지만 출력
    no_cap = top5_score >= 0.4

    seen_titles = set()
    seen_first_lines = set()
    results = []

    for rank, idx in enumerate(indices[0]):
        score = scores[0][rank]

        if no_cap:
            # 0.4 미만이면 중단 (이상인 것만 모두 수집)
            if score < 0.4:
                break
        else:
            # top-5까지만 수집
            if len(results) >= TOP_K:
                break

        item = metadata[idx]

        title = item.get("title", "Unknown")
        lyrics = item.get("lyrics", "")
        artist = item.get("artist", "")

        if not isinstance(lyrics, str) or len(lyrics.strip()) == 0 or lyrics.strip().lower() == "none":
            continue

        clean_title = re.sub(r"\([^)]*\)", "", title).strip()
        first_line = lyrics.strip().split("\n")[0].strip()

        if clean_title in seen_titles or first_line in seen_first_lines:
            continue

        if isinstance(artist, str) and len(artist.strip()) > 0:
            artist = artist.strip().replace("\n", "").replace("\r", "")
            if len(artist) % 2 == 0:
                half = len(artist) // 2
                if artist[:half] == artist[half:]:
                    artist = artist[:half]
        else:
            artist = "Unknown"

        seen_titles.add(clean_title)
        seen_first_lines.add(first_line)

        results.append({
            "title": title,
            "artist": artist,
            "lyrics": lyrics if isinstance(lyrics, str) else "",
            "score": float(score)
        })

    return results

@app.get("/search")
def search_api(query: str):
    results = search(query)
    return {"results": results}