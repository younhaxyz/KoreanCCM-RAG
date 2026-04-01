import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from huggingface_hub import login
from tqdm import tqdm
import os

HF_TOKEN = "your_HF_token"

DATASET_NAME = "Bossssss/ccm-retrieval-dataset"
MODEL_NAME = "BM-K/KoSimCSE-roberta"
OUTPUT_DIR = "./ccm_retriever_model"
HF_MODEL_NAME = "Bossssss/ccm-retriever"

BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5
MAX_LENGTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

login(token=HF_TOKEN)

dataset = load_dataset(DATASET_NAME)
data = dataset["train"]

# InputExample 변환
train_examples = []

for row in tqdm(data):
    anchor = row["anchor"]
    positive = row["positive"]

    if not isinstance(anchor, str) or not isinstance(positive, str):
        continue

    if len(anchor.strip()) == 0 or len(positive.strip()) == 0:
        continue

    train_examples.append(
        InputExample(texts=[anchor, positive])
    )

print(f"Total training samples: {len(train_examples)}")

model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# max length
model.max_seq_length = MAX_LENGTH

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)

train_loss = losses.MultipleNegativesRankingLoss(model)

# train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    optimizer_params={"lr": LR},
    show_progress_bar=True
)

model.save(OUTPUT_DIR)
model.push_to_hub(HF_MODEL_NAME)

print(f"Model uploaded to: {HF_MODEL_NAME}")