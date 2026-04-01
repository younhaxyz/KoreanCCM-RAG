import re
import random
from datasets import load_dataset, Dataset
from huggingface_hub import login
from tqdm import tqdm

HF_TOKEN = "your_HF_token"
DATASET_NAME = "Bossssss/CCM-list"
OUTPUT_DATASET_NAME = "Bossssss/ccm-retrieval-dataset"

USE_CHUNKING = False   # True: chunk-level / False: full-lyricss
CHUNK_SIZE = 256       # chunk size (character-level)
SEED = 42

random.seed(SEED)

login(token=HF_TOKEN)

# remove parentheses
def clean_title(title):
    return re.sub(r"\([^)]*\)", "", title).strip()

# chunking function
def chunk_text(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    return chunks

dataset = load_dataset(DATASET_NAME)
data = dataset["train"]

titles = data["title"]
lyrics = data["lyrics"]

N = len(data)

anchors = []
positives = []
negatives = []

for i in tqdm(range(N)):
    title = titles[i]
    lyric = lyrics[i]

    if not isinstance(title, str) or not isinstance(lyric, str):
        continue

    # anchor
    anchor = clean_title(title)

    if len(anchor) == 0 or len(lyric.strip()) == 0:
        continue

    # negative 샘플 선택
    neg_idx = i
    while neg_idx == i:
        neg_idx = random.randint(0, N - 1)

    negative_lyric = lyrics[neg_idx]

    if not isinstance(negative_lyric, str) or len(negative_lyric.strip()) == 0:
        continue

    if USE_CHUNKING:
        pos_chunks = chunk_text(lyric, CHUNK_SIZE)
        neg_chunks = chunk_text(negative_lyric, CHUNK_SIZE)

        if len(pos_chunks) == 0 or len(neg_chunks) == 0:
            continue

        # 같은 개수 맞추기
        min_len = min(len(pos_chunks), len(neg_chunks))

        for j in range(min_len):
            anchors.append(anchor)
            positives.append(pos_chunks[j])
            negatives.append(neg_chunks[j])

    else:
        anchors.append(anchor)
        positives.append(lyric)
        negatives.append(negative_lyric)

dataset_dict = {
    "anchor": anchors,
    "positive": positives,
    "negative": negatives,
}

train_dataset = Dataset.from_dict(dataset_dict)

train_dataset.push_to_hub(OUTPUT_DATASET_NAME)

print(f"Dataset uploaded to: {OUTPUT_DATASET_NAME}")