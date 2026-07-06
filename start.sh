#!/usr/bin/env bash
# start.sh — entrypoint.
#
# Vision + text + transcription are all cloud (byNara Router + Groq).
# No local model daemon to boot — just exec uvicorn so Render's health
# check sees the right process.

set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"

echo "[start.sh] launching campeditor on ${HOST}:${PORT} with ${WORKERS} worker(s)"
exec uvicorn app.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --proxy-headers \
    --forwarded-allow-ips="*"