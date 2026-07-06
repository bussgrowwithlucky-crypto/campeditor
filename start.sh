#!/usr/bin/env bash
# start.sh — local-only entrypoint.
#
# Boots Ollama in the background, waits for it to come up, then execs
# uvicorn so Render's health check sees the web process. On exit, kills
# the ollama daemon so the container shuts down cleanly.

set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

echo "[start.sh] starting Ollama on :${OLLAMA_PORT}"
# `ollama serve` is the daemon entry point. We background it so the
# container's main process can be uvicorn (Render tracks the PID 1
# process for health checks).
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
trap "kill -TERM $OLLAMA_PID 2>/dev/null || true" EXIT

# Wait for Ollama to come up. We retry /api/tags up to 60 times
# (~60s) before giving up; in practice it starts in 2-5s.
echo "[start.sh] waiting for Ollama at http://localhost:${OLLAMA_PORT}/api/tags"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
        echo "[start.sh] Ollama is up after ${i} attempts"
        break
    fi
    sleep 1
done

# Pull the configured models so the first real request doesn't pay the
# download cost. This is a no-op on subsequent deploys if the persistent
# disk is mounted at /root/.ollama (the default model cache path).
OLLAMA_VISION_MODEL="${OLLAMA_VISION_MODEL:-llava:13b}"
OLLAMA_TEXT_MODEL="${OLLAMA_TEXT_MODEL:-llama3.1:8b}"
echo "[start.sh] ensuring models are pulled: $OLLAMA_VISION_MODEL, $OLLAMA_TEXT_MODEL"
ollama pull "$OLLAMA_VISION_MODEL" 2>/dev/null || echo "[start.sh] warning: failed to pull $OLLAMA_VISION_MODEL"
ollama pull "$OLLAMA_TEXT_MODEL" 2>/dev/null || echo "[start.sh] warning: failed to pull $OLLAMA_TEXT_MODEL"

echo "[start.sh] launching campeditor on ${HOST}:${PORT} with ${WORKERS} worker(s)"
exec uvicorn app.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --proxy-headers \
    --forwarded-allow-ips="*"