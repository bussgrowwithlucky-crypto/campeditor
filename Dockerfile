# campeditor Dockerfile — fully cloud, no local models.
#
# Bundles: ffmpeg + the campeditor Python app.
# Vision + text + transcription all use cloud APIs (byNara Router + Groq).
# Result: small image, fast cold-start, runs on Render free tier.

FROM python:3.11-slim AS base

# System deps:
#   ffmpeg           — video processing (transcoding, trimming, muxing)
#   curl             — health checks + yt-dlp fallback
#   ca-certificates  — outbound HTTPS to byNara / Groq / yt-dlp
#   wget             — used by audio-separator's model downloads
#   build-essential  — gcc + make + headers; required because audio-separator
#                      → diffq ships a C extension (bitpack.c) with no
#                      prebuilt wheel for linux-x86_64 + Python 3.11, so
#                      pip has to compile it from source.
#   python3-dev      — Python.h headers needed by the same C extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
        wget \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps ----
WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
COPY broll_intelligence ./broll_intelligence
COPY static ./static
COPY start.sh ./
RUN chmod +x start.sh

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---- Persistent data dir for cache + library index ----
RUN mkdir -p /app/data/cache /app/data/jobs

# Render injects $PORT; the app binds there directly.
EXPOSE 8000

# ---- Entrypoint ----
# One process: uvicorn. Render's health check hits /api/health on $PORT.
CMD ["./start.sh"]