FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    faiss-cpu \
    numpy \
    onnxruntime \
    optimum \
    sentence-transformers --no-deps \
    huggingface-hub \
    tokenizers \
    tqdm

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]