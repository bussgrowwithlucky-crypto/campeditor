# campeditor Dockerfile — fully local, zero cloud.
#
# Bundles: ffmpeg + Ollama + faster-whisper + the campeditor Python app.
# Result: any Render instance (or any host) gets a working video-render
# pipeline with no external API keys.

FROM python:3.11-slim AS base

# System deps: ffmpeg for video processing, curl for health checks,
# ca-certificates so Ollama can pull models from the HF registry.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Ollama ----
# Pin a known-stable version. Ollama publishes Linux builds at
# https://ollama.com/download/linux. For Render free tier we use the
# Linux x86_64 binary; for ARM (Oracle free tier), the Dockerfile
# picks up the right one via the download script.
ARG OLLAMA_VERSION=0.5.7
RUN curl -fsSL https://ollama.com/install.sh | sh

# ---- Install Python deps ----
WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
COPY broll_intelligence ./broll_intelligence
COPY static ./static
COPY start.sh ./
RUN chmod +x start.sh

# faster-whisper is the local Whisper implementation. It downloads
# model weights on first use, so we don't bake them into the image.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---- Persistent data dir for cache + library index ----
RUN mkdir -p /app/data/cache /app/data/jobs

# Render injects $PORT; ollama serves on 11434 in-container.
EXPOSE 11434
EXPOSE 8000

# ---- Entrypoint ----
# Two daemons need to be live for the app to work:
#   1. ollama serve        (vision + chat + embeddings)
#   2. uvicorn app.main:app (the web service)
# start.sh runs ollama in the background, waits for the /api/tags
# endpoint, then execs uvicorn in the foreground so Render's health
# check sees the right process.
CMD ["./start.sh"]