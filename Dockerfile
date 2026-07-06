# campeditor Dockerfile — fully cloud, no local models.
#
# Bundles: ffmpeg + the campeditor Python app.
# Vision + text + transcription all use cloud APIs (byNara Router + Groq).
# Result: small image, fast cold-start, runs on Render free tier.

FROM python:3.11-slim AS base

# System deps: ffmpeg for video processing, curl for health checks,
# ca-certificates for outbound HTTPS to byNara / Groq / yt-dlp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
        wget \
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