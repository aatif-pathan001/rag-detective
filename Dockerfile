FROM python:3.11-slim

# Create non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Environment settings for Python and Hugging Face cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    EMBEDDING_DEVICE=cpu \
    PORT=7860

# Install dependencies (uses CPU-only PyTorch from requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the ONNX BAAI/bge-small-en-v1.5 model weights at build time for instant startup
RUN mkdir -p /home/user/.cache/huggingface && \
    python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" && \
    chown -R user:user /home/user

# Copy application source and frontend
COPY --chown=user:user rag_detective ./rag_detective
COPY --chown=user:user frontend ./frontend

USER user

EXPOSE 7860

CMD ["uvicorn", "rag_detective.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
