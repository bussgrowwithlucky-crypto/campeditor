"""Local Whisper transcription with word-level timestamps.

Uses faster-whisper (CTranslate2 + Whisper) running on the local CPU/GPU.
No API keys. Model is downloaded on first use from Hugging Face and cached
under the standard HF cache (~/.cache/huggingface/).

The audio slice (start..end) is extracted locally with FFmpeg first, so all
returned timestamps are relative to the trimmed clip — which is also how the
render timeline starts (ffmpeg -ss <start>). No chunking: clips are 12-15s.

Never raises: returns an empty Transcript on failure so the pipeline can
still render (without captions) instead of dying.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.config import Settings
from app.models import Transcript, TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)


# Model name → faster-whisper size. The default "large-v3" gives the best
# quality at the cost of speed; "medium" or "small" are faster on CPU-only
# boxes. Override via WHISPER_MODEL env var.
DEFAULT_WHISPER_MODEL = "large-v3"
_FASTER_WHISPER_MODEL_CACHE: dict[tuple[str, str], object] = {}


def transcribe(
    source_path: Path,
    start: float,
    end: float,
    work_dir: Path,
    settings: Settings,
    model_size: str | None = None,
) -> Transcript:
    try:
        audio_path = _extract_audio(source_path, start, end, work_dir, settings)
    except Exception:
        logger.exception("Audio extraction failed")
        return Transcript()
    try:
        return _transcribe_with_whisper(audio_path, model_size=model_size)
    except Exception:
        logger.exception("Local Whisper transcription failed")
        return Transcript()


def _extract_audio(source_path: Path, start: float, end: float, work_dir: Path, settings: Settings) -> Path:
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
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr[-500:]}")
    return audio_path


def _transcribe_with_whisper(audio_path: Path, model_size: str | None = None) -> Transcript:
    """Run faster-whisper on a 16 kHz mono WAV. Returns word + segment
    timestamps matching the campeditor Transcript schema.

    The faster-whisper model is loaded lazily and cached in
    _FASTER_WHISPER_MODEL_CACHE so subsequent calls in the same process
    don't pay the model-load cost (~3-15s depending on size and disk).

    `model_size` overrides the default (large-v3) when supplied — used by
    analyze_reference to call the much smaller `medium` model, since the
    reference's transcript only feeds gap detection and doesn't need
    large-v3 word accuracy. On a 21-second reel that drops reference
    transcription from ~60s to ~20s.
    """
    import os
    if model_size is None:
        model_size = os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    # device="auto" picks CUDA when available, CPU otherwise.
    # compute_type="auto" picks int8 on CPU, float16 on GPU.
    try:
        model = _get_whisper_model(model_size)
    except Exception:
        # 3-tier fallback chain in _get_whisper_model has already logged the
        # intermediate failures; if we get here, every tier failed (e.g. an
        # OOM on `small` too, or non-memory errors at every level). Rather
        # than bubble up and fail the whole job, return an empty transcript
        # so the pipeline can still render the clip without captions.
        logger.exception(
            "All Whisper model load attempts failed for '%s'; "
            "returning empty transcript",
            model_size,
        )
        return Transcript()
    # `word_timestamps=True` is what gives us the per-word start/end that
    # the caption renderer needs.
    segments_iter, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    words: list[TranscriptWord] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text)
        )
        if seg.words:
            for w in seg.words:
                token = (w.word or "").strip()
                if not token:
                    continue
                words.append(
                    TranscriptWord(
                        word=token,
                        start=float(w.start) if w.start is not None else float(seg.start),
                        end=float(w.end) if w.end is not None else float(seg.end),
                    )
                )

    if not words and segments:
        words = _words_from_segments(segments)
    return Transcript(words=words, segments=segments)


def _is_memory_error(exc: BaseException) -> bool:
    """True if `exc` looks like a CTranslate2/MKL memory-allocation failure."""
    msg = str(exc).lower()
    return "mkl_malloc" in msg or "memory" in msg


def _get_whisper_model(model_size: str):
    """Lazy-load + cache the faster-whisper model with a 3-tier fallback chain.

    On Windows, the larger models (medium / large-v3) can fail at startup with
    `RuntimeError: mkl_malloc: failed to allocate memory` because the ctranslate2
    arena allocator + Windows memory fragmentation leaves no contiguous block
    big enough — even when RAM nominally has plenty free. To keep the pipeline
    alive we degrade in three steps:

      1. Requested model (e.g. medium) with compute_type='auto' (default).
      2. Requested model with compute_type='int8' (more memory-efficient).
      3. 'small' model with compute_type='int8' (last resort).

    Each step only triggers when the previous raised an mkl_malloc / memory
    RuntimeError. Non-memory errors (missing file, corrupt download, etc.)
    propagate immediately.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. Run `pip install faster-whisper` "
            "in your venv. On first run, the model weights download from "
            "Hugging Face (~3 GB for large-v3)."
        )

    # Tier 1: requested model, default compute_type.
    cache_key = (model_size, "auto")
    if cache_key in _FASTER_WHISPER_MODEL_CACHE:
        return _FASTER_WHISPER_MODEL_CACHE[cache_key]
    logger.info("Loading Whisper model '%s' (one-time cost)...", model_size)
    try:
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        _FASTER_WHISPER_MODEL_CACHE[cache_key] = model
        return model
    except RuntimeError as exc:
        if not _is_memory_error(exc):
            raise
        logger.warning(
            "Whisper model '%s' hit mkl_malloc / memory error (%s); "
            "retrying with compute_type='int8'.",
            model_size, str(exc)[:200],
        )

    # Tier 2: requested model, int8 (less arena memory).
    cache_key_int8 = (model_size, "int8")
    if cache_key_int8 in _FASTER_WHISPER_MODEL_CACHE:
        return _FASTER_WHISPER_MODEL_CACHE[cache_key_int8]
    try:
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        _FASTER_WHISPER_MODEL_CACHE[cache_key_int8] = model
        return model
    except RuntimeError as exc:
        if not _is_memory_error(exc):
            raise
        logger.warning(
            "Whisper model '%s' (int8) hit mkl_malloc / memory error (%s); "
            "falling back to 'small' model with int8.",
            model_size, str(exc)[:200],
        )

    # Tier 3: 'small' model, int8. Final fallback — if this raises,
    # _transcribe_with_whisper will catch it and return empty Transcript.
    cache_key_small = ("small", "int8")
    if cache_key_small in _FASTER_WHISPER_MODEL_CACHE:
        return _FASTER_WHISPER_MODEL_CACHE[cache_key_small]
    logger.info("Loading fallback Whisper model 'small' (int8)...")
    model = WhisperModel("small", device="auto", compute_type="int8")
    _FASTER_WHISPER_MODEL_CACHE[cache_key_small] = model
    return model


def _words_from_segments(segments: list[TranscriptSegment]) -> list[TranscriptWord]:
    """fallback if faster-whisper didn't emit per-word timestamps on a
    particular run (rare; happens with very short segments or pure
    music/noise). Split each segment's time range across its words
    proportionally to character length."""
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
            words.append(TranscriptWord(word=token, start=cursor, end=cursor + share))
            cursor += share
    return words