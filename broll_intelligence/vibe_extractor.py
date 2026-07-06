"""Vibe extraction — vision tagging + OpenCV-derived quantitative features.

`extract_from_video(path, settings)` returns a FeatureVector populated from
two independent passes on the same clip:

  1. **Vision pass** — ffmpeg extracts the middle frame at 50% of duration,
     `vision_ladder.call()` runs the JSON-only prompt, and the response is
     parsed (with a forgiving fence-strip + brace-span fallback) into the
     categorical fields (subjects, setting, action, category, mood, energy,
     lighting, color_palette, shot_type, camera_motion, depth_of_field,
     query).

  2. **CV pass** — ffmpeg extracts frames at 25/50/75% (or 0/50/100% when the
     clip is short enough that those collapse to the same instant). For each
     frame we compute palette warmth / saturation / brightness, contrast, edge
     density; across the 3-frame sample we also compute motion_intensity as
     the mean grayscale absdiff between consecutive frames.

Confidence is the blend defined in the spec:

    confidence = (vision_fields_nonempty / vision_fields_total) * 0.5
               + (1.0 if all numerics else 0.5) * 0.5

When the vision pass returns nothing parseable (every provider down, model
hallucinated non-JSON, etc.) we return `empty_feature_vector()` so callers
can keep going on CV-only signal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import Settings
from .feature_vector import (
    FeatureVector,
    empty_feature_vector,
    feature_vector_to_dict,
)
from .vision_ladder import call as vision_call

logger = logging.getLogger(__name__)

# Fields counted as "vision fields" for the confidence numerator. The set is
# kept here (rather than derived from the dataclass) so we don't accidentally
# count provenance / quantitative fields whose absence means something else.
_VISION_FIELDS: tuple[str, ...] = (
    "subjects",
    "setting",
    "action",
    "category",
    "query",
    "mood",
    "energy",
    "lighting",
    "color_palette",
    "shot_type",
    "camera_motion",
    "depth_of_field",
)

# JSON-shape the model is asked to emit. Keep in lock-step with
# broll_intelligence/CONTRACT.md — that doc is the public source of truth.
VISION_PROMPT = (
    "You are analyzing a video frame for an intelligent B-roll matching system. "
    "Reply with ONLY a JSON object (no markdown, no prose) with these exact keys:\n"
    "{\n"
    '  "subjects": list of 1-3 concrete nouns (objects/people, lowercase),\n'
    '  "setting": list of 1-2 location descriptors,\n'
    '  "action": list of 0-2 verbs,\n'
    '  "category": one of [movie, sports, tech, lifestyle, money, other],\n'
    '  "query": a short stock-footage search phrase of 3-8 words,\n'
    '  "mood": list of 1-3 words from [tense, uplifting, mysterious, epic,\n'
    "              melancholic, energetic, calm, aggressive, romantic, nostalgic,\n"
    "              ominous, joyful, neutral, dramatic, playful, sinister],\n"
    '  "energy": one of [low, medium, high],\n'
    '  "lighting": one of [low-key, high-key, natural, neon, golden-hour, mixed],\n'
    '  "color_palette": list of 2-3 dominant colors (e.g. ["deep blue", "amber"]),\n'
    '  "shot_type": one of [wide, medium, close-up, extreme-close-up, aerial, overhead, two-shot],\n'
    '  "camera_motion": one of [static, pan, tilt, dolly, handheld, tracking, zoom],\n'
    '  "depth_of_field": one of [deep, shallow]\n'
    "}\n"
    "Ignore any burned-in text/captions. Use empty lists/empty strings when nothing fits."
)


# ---------------------------------------------------------------------------
# ffmpeg plumbing
# ---------------------------------------------------------------------------


def _ffmpeg(settings: Settings) -> str:
    return settings.resolved_ffmpeg()


def _ffprobe(settings: Settings) -> str:
    return settings.resolved_ffprobe()


def probe_duration(path: Path, settings: Settings) -> float:
    """ffprobe duration in seconds. Returns 0.0 on any failure (test parity
    with app.rendering.probe_duration)."""
    try:
        result = subprocess.run(
            [
                _ffprobe(settings),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("probe_duration: %s for %s", type(exc).__name__, path)
        return 0.0
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _frame_cache_root(settings: Settings) -> Path:
    """Where extracted frames live for the duration of one extract_from_video
    call. We use a per-extract tempdir so concurrent indexers can't trip
    over each other's filenames, but anchor it under the configured data_dir
    so a debugger can find them after the fact."""
    base = settings.data_dir / "cache" / "broll_intelligence_frames"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _extract_frame(
    source: Path,
    at_seconds: float,
    output_path: Path,
    settings: Settings,
    *,
    timeout: float = 30.0,
) -> bool:
    """Single-frame ffmpeg grab. Returns True iff the JPEG landed on disk
    with a sane size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                _ffmpeg(settings),
                "-y",
                "-ss", f"{max(0.0, at_seconds):.3f}",
                "-i", str(source),
                "-frames:v", "1",
                "-vf", "scale=640:-1",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.debug("frame extract timed out for %s @ %.2fs", source, at_seconds)
        return False
    except OSError as exc:
        logger.debug("frame extract OSError for %s: %s", source, exc)
        return False
    if result.returncode != 0:
        logger.debug(
            "frame extract failed (%s): %s",
            source,
            (result.stderr or "")[-200:],
        )
        return False
    if not output_path.exists():
        return False
    return output_path.stat().st_size > 500


def _sample_times(duration: float) -> list[float]:
    """25/50/75% by default. For very short clips (≤ 1.5s) collapse to
    0/50/100% so the three samples aren't all the same frame."""
    if duration <= 0:
        return [0.0]
    if duration <= 1.5:
        return [
            max(0.0, duration * 0.0),
            duration * 0.5,
            max(0.0, duration - 0.05),
        ]
    return [duration * 0.25, duration * 0.5, duration * 0.75]


def _sample_frame_paths(
    source: Path, duration: float, settings: Settings
) -> list[Path]:
    """Extract the sampled frames into a stable subdir keyed by absolute
    source path. Returns the list of paths that landed on disk (may be
    shorter than 3 when ffmpeg failed for some samples — the caller treats
    shorter lists as "fewer frames averaged, confidence lower")."""
    base = _frame_cache_root(settings)
    key = hashlib.md5(str(source.resolve()).encode()).hexdigest()[:16]
    sub = base / key
    sub.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, at in enumerate(_sample_times(duration)):
        out = sub / f"frame-{idx:02d}.jpg"
        if _extract_frame(source, max(0.0, min(duration - 0.05, at)), out, settings):
            paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# CV feature passes
# ---------------------------------------------------------------------------


def _safe_import_cv2() -> Any:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        return cv2, np
    except ImportError:
        return None, None


def _central_crop(frame: Any) -> Any:
    """Central 64% (18% trimmed from each side) of the frame, matching the
    spec. Falls back to the full frame when the crop would be empty."""
    h, w = frame.shape[:2]
    top = int(h * 0.18)
    bot = int(h * 0.82)
    left = int(w * 0.18)
    right = int(w * 0.82)
    crop = frame[top:bot, left:right]
    if crop.size == 0:
        crop = frame
    return crop


def _cv_features_for_frame(frame_path: Path) -> dict[str, float]:
    """Compute the 5 single-frame numerics. Returns a dict with whichever
    features computed successfully; missing keys default to 0.0 in the
    caller."""
    cv2, _np = _safe_import_cv2()
    if cv2 is None:
        return {}
    try:
        frame = cv2.imread(str(frame_path))
    except Exception:
        return {}
    if frame is None or frame.size == 0:
        return {}

    out: dict[str, float] = {}

    crop = _central_crop(frame)

    # palette_warmth: mean((R-B)/255) over the central 64% crop. cv2 loads
    # BGR, so warmth = mean(R-B) after extracting channels.
    try:
        b, g, r = cv2.split(crop)
        warmth = float((r.astype("float32") - b.astype("float32")).mean() / 255.0)
        out["palette_warmth"] = _clip01(warmth)
    except Exception:
        out.setdefault("palette_warmth", 0.0)

    # saturation / brightness: HSV channel means / 255
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = float(hsv[..., 1].mean() / 255.0)
        bri = float(hsv[..., 2].mean() / 255.0)
        out["palette_saturation"] = _clip01(sat)
        out["palette_brightness"] = _clip01(bri)
    except Exception:
        out.setdefault("palette_saturation", 0.0)
        out.setdefault("palette_brightness", 0.0)

    # contrast: std(grayscale) / 128, clipped to 0..1
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std() / 128.0)
        out["contrast"] = _clip01(contrast)
    except Exception:
        out.setdefault("contrast", 0.0)

    # edge_density: mean(Canny) / 255
    try:
        edges = cv2.Canny(crop, 80, 180)
        ed = float(edges.mean() / 255.0)
        out["edge_density"] = _clip01(ed)
    except Exception:
        out.setdefault("edge_density", 0.0)

    return out


def _cv_motion_intensity(frame_paths: list[Path]) -> float:
    """Mean grayscale absdiff between consecutive sampled frames, normalised
    to 0..1. With only 0 or 1 frames, motion is undefined → 0.0."""
    if len(frame_paths) < 2:
        return 0.0
    cv2, _np = _safe_import_cv2()
    if cv2 is None:
        return 0.0
    grays: list[Any] = []
    for p in frame_paths:
        try:
            frame = cv2.imread(str(p))
        except Exception:
            continue
        if frame is None:
            continue
        try:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        except Exception:
            continue
    if len(grays) < 2:
        return 0.0
    diffs: list[float] = []
    for i in range(1, len(grays)):
        if grays[i].shape != grays[i - 1].shape:
            grays[i] = cv2.resize(grays[i], (grays[i - 1].shape[1], grays[i - 1].shape[0]))
        try:
            d = float(cv2.absdiff(grays[i], grays[i - 1]).mean() / 255.0)
            diffs.append(d)
        except Exception:
            continue
    if not diffs:
        return 0.0
    return _clip01(sum(diffs) / len(diffs))


def _aggregate_cv(frame_paths: list[Path]) -> dict[str, float]:
    """Run per-frame CV passes, then average across frames. Missing frames
    contribute nothing to the mean (the per-frame default of 0.0 is NOT
    included in the average so a failed extract doesn't pull the value to
    zero). motion_intensity is computed across the full frame set."""
    if not frame_paths:
        return {}
    per_frame: list[dict[str, float]] = []
    for p in frame_paths:
        feats = _cv_features_for_frame(p)
        if feats:
            per_frame.append(feats)
    if not per_frame:
        return {}
    keys = (
        "palette_warmth",
        "palette_saturation",
        "palette_brightness",
        "contrast",
        "edge_density",
    )
    averaged: dict[str, float] = {}
    for key in keys:
        values = [f[key] for f in per_frame if key in f]
        if values:
            averaged[key] = _clip01(sum(values) / len(values))
        else:
            averaged[key] = 0.0
    averaged["motion_intensity"] = _cv_motion_intensity(frame_paths)
    return averaged


def _clip01(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


# ---------------------------------------------------------------------------
# Vision JSON parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_vision_json(raw: str) -> dict[str, Any]:
    """Forgiving JSON parser for vision responses.

    Accepts: bare JSON, ```json ... ``` fenced JSON, prose-prefixed JSON,
    prose-suffixed JSON, or anything with a single {...} span somewhere
    inside. Returns {} on any failure.
    """
    if not raw:
        return {}
    text = raw.strip()
    # Strip leading ``` fence (single-line or with language hint).
    text = _FENCE_RE.sub("", text, count=1).lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    # Find the outermost {...} span; this is what app/broll.py does.
    match = _BRACE_RE.search(text)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # Last-ditch: try json.loads on the whole text after fence strip.
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if not isinstance(obj, dict):
        return {}
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_feature_vector(
    raw_vision: dict[str, Any] | None,
    cv_features: dict[str, float],
    media_path: Path,
    *,
    source: str = "library",
    vision_field_completeness: float = 0.0,
    cv_field_completeness: float = 0.0,
) -> FeatureVector:
    """Merge the vision response + CV features into a normalised
    FeatureVector. Uses the existing `feature_vector_from_dict` for the
    enum-safe construction."""
    from .feature_vector import feature_vector_from_dict

    payload: dict[str, Any] = dict(raw_vision or {})
    payload["media_path"] = str(media_path.resolve()) if media_path else ""
    payload["source"] = source
    # CV values land on top; vision JSON never carries these fields so this
    # is non-destructive.
    for key, value in cv_features.items():
        payload.setdefault(key, float(value))
    fv = feature_vector_from_dict(payload)
    # Confidence blend: vision completeness * 0.5 + CV completeness * 0.5.
    vision_part = max(0.0, min(1.0, vision_field_completeness))
    cv_part = 1.0 if cv_field_completeness >= 1.0 else 0.5
    fv.confidence = _clip01(vision_part * 0.5 + cv_part * 0.5)
    return fv


def _vision_field_completeness(parsed: dict[str, Any]) -> float:
    """Fraction of VISION_FIELDS that are non-empty in the parsed JSON."""
    if not parsed:
        return 0.0
    hits = 0
    for key in _VISION_FIELDS:
        value = parsed.get(key)
        if isinstance(value, list) and len(value) > 0:
            hits += 1
        elif isinstance(value, str) and value.strip():
            hits += 1
    return hits / len(_VISION_FIELDS)


def _cv_field_completeness(features: dict[str, float]) -> float:
    """Was every CV field computed (1.0) or did some fall through to 0 (≤1.0).
    For confidence purposes any nonzero computed feature counts as OK; the
    caller flips a boolean when ALL six numerics made it."""
    if not features:
        return 0.0
    target = {
        "palette_warmth",
        "palette_saturation",
        "palette_brightness",
        "contrast",
        "edge_density",
        "motion_intensity",
    }
    return 1.0 if target.issubset(features.keys()) else 0.0


def extract_from_video(
    path: Path | str,
    settings: Settings,
    *,
    source: str = "library",
    _vision_ladder_call=None,
) -> FeatureVector:
    """The canonical entry point. Returns a normalised FeatureVector for the
    clip at `path`.

    `source` is one of {"library","reference","youtube"} and lands on the
    FeatureVector's provenance field — downstream matchers weight this.

    `_vision_ladder_call` is an optional monkeypatch hook for tests: when
    supplied, it replaces the real `vision_ladder.call`. Must accept
    (image_path, prompt, settings) and return str.
    """
    vision_call_override = _vision_ladder_call or vision_call
    src = Path(path)
    media_path_str = str(src)

    duration = 0.0
    try:
        if src.exists():
            duration = probe_duration(src, settings)
    except Exception:
        duration = 0.0

    if duration <= 0:
        logger.debug("extract_from_video: zero/unknown duration for %s", src)
        return _empty_with_provenance(src, source)

    frame_paths = _sample_frame_paths(src, duration, settings)
    if not frame_paths:
        logger.debug("extract_from_video: ffmpeg failed for %s", src)
        return _empty_with_provenance(src, source)

    # The middle frame is the canonical "tag" frame.
    middle_frame = frame_paths[len(frame_paths) // 2]

    raw = ""
    try:
        raw = vision_call_override(middle_frame, VISION_PROMPT, settings) or ""
    except Exception as exc:
        logger.warning("vision_ladder raised for %s (%s); using empty tags", src, type(exc).__name__)
        raw = ""

    parsed = _parse_vision_json(raw)
    cv_features = _aggregate_cv(frame_paths)

    vision_completeness = _vision_field_completeness(parsed)
    cv_completeness = _cv_field_completeness(cv_features)

    fv = _build_feature_vector(
        parsed,
        cv_features,
        src,
        source=source,
        vision_field_completeness=vision_completeness,
        cv_field_completeness=cv_completeness,
    )
    return fv


def _empty_with_provenance(media_path: Path, source: str) -> FeatureVector:
    """Empty FeatureVector with the right provenance fields filled in."""
    fv = empty_feature_vector(str(media_path), source=source)
    return fv


# ---------------------------------------------------------------------------
# Convenience helpers (used by library_indexer + ad-hoc scripts)
# ---------------------------------------------------------------------------


def write_frame_for_debug(
    source: Path,
    settings: Settings,
    at_fraction: float = 0.5,
) -> Path | None:
    """Helper for debug scripts / REPL: extract one frame and return the
    path. Returns None if extraction failed. NOT used by extract_from_video
    itself (which writes its own frames internally)."""
    duration = probe_duration(source, settings)
    if duration <= 0:
        return None
    at = max(0.0, min(duration - 0.05, duration * at_fraction))
    base = _frame_cache_root(settings)
    key = hashlib.md5(str(source.resolve()).encode()).hexdigest()[:16]
    out = base / key / "debug.jpg"
    if _extract_frame(source, at, out, settings):
        return out
    return None


def frame_cache_key(source: Path) -> str:
    """Deterministic per-source key for the extracted-frame cache directory."""
    return hashlib.md5(str(source.resolve()).encode()).hexdigest()[:16]


def reset_frame_cache_for(source: Path, settings: Settings) -> None:
    """Remove the extracted-frame directory for a single source. Useful in
    tests that want to force a re-extract without touching the library
    index."""
    base = _frame_cache_root(settings)
    sub = base / frame_cache_key(source)
    if sub.exists():
        import shutil
        shutil.rmtree(sub, ignore_errors=True)


__all__ = [
    "extract_from_video",
    "probe_duration",
    "VISION_PROMPT",
    "write_frame_for_debug",
    "frame_cache_key",
    "reset_frame_cache_for",
    "feature_vector_to_dict",
]


# Silence the "imported but unused" lint on os / tempfile (kept available for
# future ad-hoc debug helpers in this module).
_ = os, tempfile