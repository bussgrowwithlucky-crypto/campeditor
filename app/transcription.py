"""Cloud Whisper transcription via the Groq API.

Calls https://api.groq.com/openai/v1/audio/transcriptions with the
configured GROQ_TRANSCRIPTION_MODEL (default: whisper-large-v3-turbo).
No local model weights — the trimmed WAV is sent to Groq over HTTPS and
the response (segments + word-level timestamps) is mapped into the
campeditor Transcript schema.

The `model_size` kwarg is accepted for backward compatibility with the
historical faster-whisper signature; it's ignored — the Groq model is
configured via the `GROQ_TRANSCRIPTION_MODEL` env var (or the
`groq_transcription_model` setting).

Never raises: returns an empty Transcript on failure so the pipeline can
still render (without captions) instead of dying.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import httpx

from app.config import Settings
from app.models import Transcript, TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)


DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "whisper-large-v3-turbo"
# Groq's transcription endpoint timeout. The actual call usually completes
# in 1-3 seconds for a 12-15s clip; 120s leaves plenty of headroom for slow
# uploads / cold model loads on Groq's side.
GROQ_TIMEOUT_SECONDS = 120.0


def transcribe(
    source_path: Path,
    start: float,
    end: float,
    work_dir: Path,
    settings: Settings,
    model_size: str | None = None,  # accepted for backward compat; ignored
) -> Transcript:
    """Transcribe `source_path[start:end]` via the Groq API. Returns a
    `Transcript` populated with segments + (when available) word-level
    timestamps from Groq's verbose_json response.

    Rotates across the configured Groq API keys (GROQ_API_KEY and
    GROQ_API_KEY_2) on auth (401/403) or rate-limit (429) responses so a
    single exhausted key doesn't take the pipeline down.

    `model_size` is silently ignored — model selection is via the
    `GROQ_TRANSCRIPTION_MODEL` env var.
    """
    keys = settings.groq_api_keys()
    if not keys:
        logger.warning(
            "Groq transcription skipped: no GROQ_API_KEY / GROQ_API_KEY_2 set "
            "(rendering without captions)"
        )
        return Transcript()
    try:
        audio_path = _extract_audio(source_path, start, end, work_dir, settings)
    except Exception:
        logger.exception("Audio extraction failed")
        return Transcript()
    try:
        return _transcribe_with_groq(audio_path, settings, keys)
    except Exception:
        logger.exception("Groq transcription failed (all keys exhausted)")
        return Transcript()


def _extract_audio(
    source_path: Path,
    start: float,
    end: float,
    work_dir: Path,
    settings: Settings,
) -> Path:
    """Slice `source_path[start:end]` to a 16 kHz mono WAV with ffmpeg.

    All returned timestamps from the Groq response are interpreted as
    relative to the trimmed clip — which matches how the render timeline
    starts (ffmpeg -ss <start>). No chunking: clips are 12-15s.
    """
    audio_path = work_dir / "audio.wav"
    command = [
        settings.ffmpeg_path,
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source_path),
        "-t", f"{end - start:.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed: {result.stderr[-500:]}"
        )
    return audio_path


def _transcribe_with_groq(audio_path: Path, settings: Settings, keys: list[str]) -> Transcript:
    """POST the WAV to Groq's transcription endpoint, return a Transcript.

    Tries each key in order on auth (401/403) or rate-limit (429) responses
    so a single exhausted/banned key doesn't take the pipeline down.
    Non-rotating errors (5xx, 4xx other than auth/rate) are raised
    immediately — retrying those with another key won't help.

    Uses `response_format=verbose_json` and requests both segment- and
    word-level timestamp granularities. Word timestamps are only emitted
    when Groq's model supports them; when they're missing (or empty) we
    fall back to deriving word timings from segment timings via
    `_words_from_segments` so the caption renderer always has per-word
    coordinates to work with.
    """
    base_url = (settings.groq_base_url or DEFAULT_GROQ_BASE_URL).rstrip("/")
    url = f"{base_url}/audio/transcriptions"
    model = settings.groq_transcription_model or DEFAULT_GROQ_MODEL

    # Status codes that mean "this key is no good, try the next one".
    # 401/403 = auth/revoked; 429 = per-key quota exhausted.
    ROTATABLE_STATUSES = {401, 403, 429}

    last_error: Exception | None = None
    for key_index, api_key in enumerate(keys):
        logger.info(
            "Groq transcription: model=%s, file=%s, key=%d/%d",
            model, audio_path.name, key_index + 1, len(keys),
        )
        with open(audio_path, "rb") as audio_file:
            try:
                response = httpx.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (audio_path.name, audio_file, "audio/wav")},
                    data={
                        "model": model,
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "segment",
                        "timestamp_granularities[]": "word",
                    },
                    timeout=GROQ_TIMEOUT_SECONDS,
                )
            except httpx.HTTPError as exc:
                # Network-level failure — try the next key. The next call
                # is independent so a transient connect failure shouldn't
                # cascade into a full pipeline stall.
                logger.warning(
                    "Groq transcription key %d/%d network error: %s",
                    key_index + 1, len(keys), exc,
                )
                last_error = exc
                continue

        if response.status_code in ROTATABLE_STATUSES:
            # Surface the Groq error message so failures are diagnosable.
            try:
                err = response.json()
            except Exception:
                err = {"error": response.text[:500]}
            logger.warning(
                "Groq transcription key %d/%d returned HTTP %d: %s — "
                "rotating to next key",
                key_index + 1, len(keys), response.status_code, err,
            )
            last_error = RuntimeError(
                f"Groq transcription HTTP {response.status_code}: {err}"
            )
            continue

        if response.status_code >= 400:
            # Non-rotating error (500, 400, etc.) — bail out; another key
            # won't help and the caller surfaces this to the user.
            try:
                err = response.json()
            except Exception:
                err = {"error": response.text[:500]}
            raise RuntimeError(
                f"Groq transcription HTTP {response.status_code}: {err}"
            )

        payload = response.json()
        break  # success — fall through to parsing
    else:
        # Every key exhausted; raise the last error so the outer
        # try/except can log it and return an empty Transcript.
        assert last_error is not None
        raise last_error

    # verbose_json shape:
    #   { "task": "transcribe", "language": "en", "duration": ...,
    #     "text": "...",
    #     "segments": [{"id","seek","start","end","text","tokens",
    #                   "temperature","avg_logprob","compression_ratio",
    #                   "no_speech_prob"}, ...],
    #     "words": [{"word","start","end"}, ...] }
    segments_raw = payload.get("segments") or []
    words_raw = payload.get("words") or []

    segments = [
        TranscriptSegment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=str(seg.get("text", "")).strip(),
        )
        for seg in segments_raw
        if str(seg.get("text", "")).strip()
    ]

    words: list[TranscriptWord] = []
    for w in words_raw:
        token = str(w.get("word", "")).strip()
        if not token:
            continue
        words.append(
            TranscriptWord(
                word=token,
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
            )
        )

    if not words and segments:
        words = _words_from_segments(segments)
    return Transcript(words=words, segments=segments)


def _words_from_segments(segments: list[TranscriptSegment]) -> list[TranscriptWord]:
    """Fallback when Groq didn't emit per-word timestamps.

    Splits each segment's time range across its words proportionally to
    character length. Used by the caption renderer so every clip has
    per-word coordinates to render against, even when the upstream model
    only returns segment-level timings.
    """
    words: list[TranscriptWord] = []
    for segment in segments:
        tokens = segment.text.split()
        if not tokens:
            continue
        total_chars = sum(len(token) for token in tokens) or 1
        duration = max(0.1, segment.end - segment.start)
        cursor = segment.start
        for token in tokens:
            share = duration * len(token) / total_chars
            words.append(
                TranscriptWord(word=token, start=cursor, end=cursor + share)
            )
            cursor += share
    return words