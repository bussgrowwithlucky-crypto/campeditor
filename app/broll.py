"""B-roll pipeline for Replicate mode.

Seven stages, run once per job:
  1. detect_broll_spans   — find every cutaway in the reference timeline (frame
                             clustering + hard-cut splitting on 10fps frames).
  2. describe_spans       — vision-tag each span (subjects/setting/action/
                             category + a short search query), 3 sampled frames.
  3. build_library_index  — scan the local B-roll library once, vision-tag each
                             clip, cache the result forever (incremental).
  4. match_local           — cheap category/subject/setting scoring over the
                             index, LLM tie-break over the top 5.
  5. search_youtube        — YouTube Data API search (key rotation on
                             403/429), falling back to yt-dlp ytsearch; vision-
                             compare the top 2 candidates' previews, pick the
                             closer one.
  6. crop_reference_cutaway — last resort: a caption-dodging crop of the
                              reference's own cutaway. Always succeeds.
  7. fetch_broll_cuts / fetch_broll_cut_variations / fetch_learned_broll_cuts
                             — place cuts at the reference's exact timestamps.

Every span for a replicate (reference-driven) job produces a cut: Local ->
YouTube -> reference-crop is a guaranteed ladder, so span count always matches
the reference's B-roll cutaway count.
"""

import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.models import BrollCut, BrollPackItem, BrollRecoveryDiagnostic, ReferenceAnalysis
from app.rendering import probe_duration

logger = logging.getLogger(__name__)

# NOTE: ``app.selector_helpers`` is imported below (after the data model
# section) to avoid a circular import: selector_helpers needs LibraryClip
# / SpanProfile, which are defined after these top-of-file imports.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_FPS = 10.0  # 0.1s timing resolution for reference cut detection

# Cloud vision calls per span in describe_spans. 1 = mid-span frame only.
# Raising this back to 3 restores the old wider coverage; the cache is keyed
# per-frame so the bump does not invalidate existing tags.
FRAMES_PER_SPAN_DESCRIBE = 1
MIN_BROLL_SPAN = 0.08
MIN_OUTPUT_BROLL_SPAN = 0.08
MAX_BROLL_SPANS = 24
SHOT_CUT_DISTANCE = 0.85
PRIMARY_SHOT_DISTANCE = 0.78

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
VALID_CATEGORIES = {"movie", "sports", "tech", "lifestyle", "money", "other"}

# Closed vocabs — verbatim from broll_intelligence/CONTRACT.md §1.1.
# Used by _parse_tags to validate vision-model output. Empty list / empty
# string is the "no data" sentinel — the scorer treats empty as "absent"
# (not "unknown") and falls back to the house style or zero credit.
_MOOD_VOCAB: frozenset[str] = frozenset({
    "tense", "uplifting", "mysterious", "epic", "melancholic", "energetic",
    "calm", "aggressive", "romantic", "nostalgic", "ominous", "joyful",
    "neutral", "dramatic", "playful", "sinister",
})
_ENERGY_VOCAB: frozenset[str] = frozenset({"low", "medium", "high"})
_LIGHTING_VOCAB: frozenset[str] = frozenset({
    "low-key", "high-key", "natural", "neon", "golden-hour", "mixed",
})
_SHOT_TYPE_VOCAB: frozenset[str] = frozenset({
    "wide", "medium", "close-up", "extreme-close-up", "aerial", "overhead",
    "two-shot",
})
_CAMERA_MOTION_VOCAB: frozenset[str] = frozenset({
    "static", "pan", "tilt", "dolly", "handheld", "tracking", "zoom",
})
_DEPTH_OF_FIELD_VOCAB: frozenset[str] = frozenset({"deep", "shallow"})

# Field caps. Clip-level moods cap at 3, span-level at 3, color_palette at 3.
# Anything beyond the cap is silently truncated (never raised).
_MOOD_CAP = 3
_COLOR_PALETTE_CAP = 3

VISION_TIMEOUT = 25.0
OLLAMA_TIMEOUT = 30.0
SEARCH_TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 120.0
PREVIEW_SECONDS = 20.0

YT_MAX_SOURCE_SECONDS = 150
YT_MIN_SOURCE_SECONDS = 3.0
YT_RESULTS_PER_QUERY = 10
# Titles that signal talking-head / commentary uploads rather than clean stock
# footage. Cheap substring reject before any download or vision call.
YT_JUNK_TITLE_TERMS = (
    "podcast", "reaction", "react", "explained", "interview", "tutorial",
    "how to", "review", "vlog", "story time", "storytime", "q&a", "commentary",
    "tier list", "ranking", "compilation of", "top 10", "top ten",
)

# Bumped from 1 -> 2 in the V2 selector. The frame-tag prompt and library
# clip tag shape gained the seven vibe/cinematography keys; an old version-1
# cache is treated as empty by build_library_index and re-tagged on the
# next build.
_INDEX_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SpanProfile:
    """Vision-derived description of one B-roll cutaway.

    Vibe fields (`mood`, `energy`, `lighting`, `shot_type`,
    `camera_motion`, `depth_of_field`, `color_palette`) are populated from
    `ReferenceAnalysis.broll_span_tags[i]` when available. The historical
    `_describe_broll_span` indexer may not produce all of them — in which
    case the intelligent selector recognises the gap and falls back to the
    legacy keyword-only score.
    """

    start: float
    end: float
    subjects: list[str] = field(default_factory=list)
    setting: list[str] = field(default_factory=list)
    action: list[str] = field(default_factory=list)
    category: str = "other"
    query: str = ""
    # Optional extended vibe tags, populated by _span_profile_for. Empty
    # values are the "no data" sentinel — the scorer checks for these.
    mood: list[str] = field(default_factory=list)
    energy: str = ""
    lighting: str = ""
    shot_type: str = ""
    camera_motion: str = ""
    depth_of_field: str = ""
    color_palette: list[str] = field(default_factory=list)


@dataclass
class LibraryClip:
    """One local B-roll library file, tagged and cached forever.

    Vibe fields (`mood`, `lighting`, `shot_type`, `energy`,
    `camera_motion`, `depth_of_field`) are populated when the library is
    indexed by `broll_intelligence/library_indexer.py` (which produces the
    `broll_intelligence_index.json` cache). Legacy indexes produced by the
    older `app/broll.py:_build_library_index` leave them as empty defaults —
    the intelligent-selector scorer detects that and falls back to the
    historical keyword-only score so a mixed library doesn't penalise
    un-tagged clips.
    """

    path: Path
    mtime: float
    size: int
    subjects: list[str]
    setting: list[str]
    category: str
    folder: str
    query: str = ""
    # Optional extended vibe tags. Empty when not indexed by
    # broll_intelligence; the scorer checks `hasattr`/truthiness before use.
    mood: list[str] = field(default_factory=list)
    energy: str = ""
    lighting: str = ""
    shot_type: str = ""
    camera_motion: str = ""
    depth_of_field: str = ""
    color_palette: list[str] = field(default_factory=list)


# V2 intelligent-selector helpers live in their own module so the new
# logic (cinema_match, continuity ledger, house-style resolver) keeps
# app/broll.py readable. Imported here (after the data model section) to
# avoid a circular import: selector_helpers needs LibraryClip/SpanProfile
# from this file.
from app.selector_helpers import (  # noqa: E402
    ContinuityLedger as _ContinuityLedger,
    build_reference_house_style as _build_reference_house_style,
    cinema_match as _cinema_match,
    cosine_similarity_6d as _cosine_similarity_6d,
    feature_vector_for_clip as _feature_vector_for_clip,
    resolve_span_vibe as _resolve_span_vibe,
    _vibe_score_for_resolved as _vibe_score_for_resolved,
    _CINEMA_FLOOR,
    _CINEMA_LIFT_TERM,
)


# Re-export the public helpers at the package surface so existing
# ``from app.broll import build_reference_house_style`` etc. keeps working
# alongside the canonical selectors path.
build_reference_house_style = _build_reference_house_style
cinema_match = _cinema_match
ContinuityLedger = _ContinuityLedger


# ---------------------------------------------------------------------------
# Vision + LLM ladder
# ---------------------------------------------------------------------------


class _VisionBudget:
    """Optional per-call-sequence time cap. None = uncapped.

    Scoped locally to one caller's loop (e.g. one analyze_reference run) rather
    than shared global state, so a budget spent tagging reference spans can
    never starve an unrelated later stage (library indexing, YouTube preview
    tagging) that happens to run in the same process.
    """

    __slots__ = ("remaining",)

    def __init__(self, seconds: float | None):
        self.remaining = seconds

    def spend(self, seconds: float) -> None:
        if self.remaining is None or seconds <= 0:
            return
        self.remaining = max(0.0, self.remaining - seconds)

    def exhausted(self) -> bool:
        return self.remaining is not None and self.remaining <= 0.0


# Vision ladder. Two rungs, cloud-first, local-fallback:
#
#   1. LLM cloud rung — Settings.llm_api_key + Settings.llm_vision_model
#      on Settings.llm_base_url (OpenAI-compatible). Auth header added by
#      OllamaClient when api_key is set.
#   2. Ollama local rung — Settings.ollama_vision_model on
#      Settings.ollama_base_url. Used only when the cloud rung is empty
#      (no key / no model) or when the cloud call raises.
#
# Either rung empty disables it. If both are empty, _vision returns ""
# and the caller falls back to empty FeatureVector (same degradation as
# before the cloud rung existed).


def _cloud_vision_call(
    image_path: Path,
    prompt: str,
    settings: Settings,
    timeout: float,
) -> str:
    """Single-frame call against the LLM cloud vision rung. Returns the
    raw model text or "" on any failure. Never raises.
    """
    try:
        from broll_intelligence.vision_ladder import OllamaClient
    except ImportError as exc:
        logger.warning("OllamaClient import failed: %s", exc)
        return ""
    try:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    except Exception as exc:
        logger.warning("Cloud vision: cannot read %s (%s)", image_path, exc)
        return ""
    client = OllamaClient(
        settings.llm_base_url,
        timeout=timeout,
        api_key=settings.llm_api_key,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }
    ]
    try:
        response = client.chat_sync(
            settings.llm_vision_model,
            messages,
            temperature=0.2,
            max_tokens=350,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Cloud vision failed (%s): %s", type(exc).__name__, exc)
        return ""
    if not response:
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def _vision(image_path: Path, prompt: str, settings: Settings, budget: "_VisionBudget | None" = None) -> str:
    """Single-frame vision call.

    Tries the LLM cloud rung first (when LLM_API_KEY and LLM_VISION_MODEL
    are both set), then falls back to the local Ollama rung (when
    OLLAMA_VISION_MODEL is set). Returns the raw model text, or "" if
    both rungs are absent or every attempt fails.

    `budget` is a soft cap on cloud spend — preserved for backward
    compatibility with the pipeline watchdog.
    """
    started_at = time.monotonic()
    cloud_ok = bool(
        settings.llm_api_key.strip()
        and settings.llm_vision_model.strip()
        and settings.llm_base_url.strip()
    )
    if cloud_ok:
        try:
            result = _cloud_vision_call(
                image_path, prompt, settings, timeout=settings.ollama_timeout
            )
        except Exception as exc:
            logger.warning("Cloud vision rung crashed (%s): %s", type(exc).__name__, exc)
            result = ""
        if result:
            if budget is not None:
                budget.spend(time.monotonic() - started_at)
            return result
        # cloud rung configured but returned empty — don't fall through
        # to local; the user opted in to cloud-only by clearing
        # ollama_vision_model. Surface the failure.
        if budget is not None:
            budget.spend(time.monotonic() - started_at)
        return ""
    if not settings.ollama_vision_model:
        logger.debug("_vision skipped: no vision rung configured (set LLM_VISION_MODEL or OLLAMA_VISION_MODEL)")
        return ""
    try:
        from broll_intelligence.vision_ladder import call as vl_call
    except ImportError as exc:
        logger.warning("broll_intelligence.vision_ladder import failed: %s", exc)
        return ""
    try:
        result = vl_call(image_path, prompt, settings, timeout=settings.ollama_timeout)
    except Exception as exc:
        logger.warning("Local Ollama vision failed (%s): %s", type(exc).__name__, exc)
        return ""
    if budget is not None:
        budget.spend(time.monotonic() - started_at)
    return (result or "").strip()


def _chat(prompt: str, settings: Settings, timeout: float = 15.0) -> str:
    """Single-shot chat completion. Used for the local-library tie-break.

    Tries the LLM cloud rung first (when LLM_API_KEY and LLM_MODEL are
    both set), then the local Ollama rung (when OLLAMA_TEXT_MODEL is
    set). Returns "" if no rung is configured or every rung fails.
    """
    cloud_ok = bool(
        settings.llm_api_key.strip()
        and settings.llm_model.strip()
        and settings.llm_base_url.strip()
    )
    if cloud_ok:
        try:
            from broll_intelligence.vision_ladder import OllamaClient
            client = OllamaClient(
                settings.llm_base_url,
                timeout=timeout,
                api_key=settings.llm_api_key,
            )
            response = client.chat_sync(
                settings.llm_model,
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
                timeout=timeout,
            )
            if response:
                choices = response.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    content = (message.get("content") or "").strip()
                    if content:
                        return content
        except Exception as exc:
            logger.warning("Cloud chat failed (%s): %s", type(exc).__name__, exc)
        # cloud rung empty-result: don't fall through if user disabled
        # Ollama (their choice — surface the failure).
        if not settings.ollama_text_model:
            return ""
    if not settings.ollama_text_model:
        return ""
    try:
        from broll_intelligence.vision_ladder import OllamaClient
    except ImportError:
        return ""
    client = OllamaClient(settings.ollama_base_url, timeout=timeout)
    try:
        response = client.chat_sync(
            settings.ollama_text_model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Local Ollama chat failed (%s): %s", type(exc).__name__, exc)
        return ""
    if not response:
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


# ---------------------------------------------------------------------------
# Stage 1: cutaway detection
# ---------------------------------------------------------------------------


def extract_reference_frames(source: Path, duration: float, frames_dir: Path, settings: Settings) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame-*.jpg"):
        old_frame.unlink(missing_ok=True)
    result = subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", f"fps={FRAME_FPS},scale=400:-1",
            str(frames_dir / "frame-%04d.jpg"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.warning("Reference frame extraction failed: %s", result.stderr[-300:])
        return []
    return sorted(frames_dir.glob("frame-*.jpg"))


def _reference_frame_feature(frame_path: Path, cv2, np):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None
    height = frame.shape[0]
    crop = frame[int(height * 0.18):int(height * 0.82), :]
    if crop.size == 0:
        crop = frame
    small = cv2.resize(crop, (32, 32)).astype("float32") / 255.0
    hsv = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten().astype("float32")
    hist = hist / (hist.sum() or 1)
    feat = np.concatenate([small.flatten() * 0.55, hist * 2.0])
    feat = feat - feat.mean()
    norm = np.linalg.norm(feat)
    return feat / (norm or 1)


def _primary_shot_centroid(features, np):
    clusters: list[dict] = []
    for index, feature in enumerate(features):
        best: tuple[float, dict] | None = None
        for cluster in clusters:
            distance = float(np.linalg.norm(feature - cluster["centroid"]))
            if best is None or distance < best[0]:
                best = (distance, cluster)
        if best and best[0] < PRIMARY_SHOT_DISTANCE:
            cluster = best[1]
            cluster["indices"].append(index)
            centroid = features[cluster["indices"]].mean(axis=0)
            cluster["centroid"] = centroid / (np.linalg.norm(centroid) or 1)
        else:
            clusters.append({"indices": [index], "centroid": feature.copy()})
    if not clusters:
        return None
    primary = max(clusters, key=lambda c: len(c["indices"]))
    return primary["centroid"]


def _shot_cut_indices(features, np) -> list[int]:
    cuts: list[int] = []
    for index in range(1, len(features)):
        if float(np.linalg.norm(features[index] - features[index - 1])) >= SHOT_CUT_DISTANCE:
            cuts.append(index)
    return cuts


def _has_cutaway_visual_energy(frame_path: Path) -> bool:
    try:
        import cv2
    except ImportError:
        return True
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    band = gray[int(height * 0.18):int(height * 0.82), :]
    if band.size == 0:
        band = gray
    lit_ratio = float((band > 24).mean())
    mean = float(band.mean())
    contrast = float(band.std())
    edges = cv2.Canny(band, 80, 180)
    edge_density = float(edges.mean() / 255)
    bright_image = lit_ratio > 0.28 and mean > 20 and (contrast > 12 or edge_density > 0.01)
    dark_textured_image = lit_ratio > 0.10 and contrast > 14 and edge_density > 0.018
    return bright_image or dark_textured_image


def _spans_from_shot_flags(
    primary_flags: list[bool],
    visual_flags: list[bool],
    cut_indices: list[int],
) -> list[tuple[float, float]]:
    boundaries = [0]
    boundaries.extend(i for i in sorted(set(cut_indices)) if 0 < i < len(primary_flags))
    boundaries.append(len(primary_flags))

    spans: list[tuple[float, float]] = []
    for start_index, end_index in zip(boundaries, boundaries[1:]):
        length = end_index - start_index
        if length <= 0:
            continue
        primary_ratio = sum(primary_flags[start_index:end_index]) / length
        visual_ratio = sum(visual_flags[start_index:end_index]) / length
        if primary_ratio <= 0.45 and visual_ratio >= 0.5:
            start_t = start_index / FRAME_FPS
            end_t = end_index / FRAME_FPS
            if end_t - start_t >= MIN_BROLL_SPAN:
                spans.append((start_t, end_t))
    return spans[:MAX_BROLL_SPANS]


def _shot_based_spans(frames: list[Path]) -> list[tuple[float, float]]:
    """Learn the recurring talking-head shot and treat visually different
    shots as B-roll, split on hard cuts so each cutaway keeps its own timing."""
    if len(frames) < 2:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    features = [_reference_frame_feature(frame, cv2, np) for frame in frames]
    if any(f is None for f in features):
        return []
    feature_array = np.array(features, dtype="float32")
    primary_centroid = _primary_shot_centroid(feature_array, np)
    if primary_centroid is None:
        return []

    primary_flags = [
        float(np.linalg.norm(feature - primary_centroid)) <= PRIMARY_SHOT_DISTANCE
        for feature in feature_array
    ]
    visual_flags = [_has_cutaway_visual_energy(frame) for frame in frames]
    cut_indices = _shot_cut_indices(feature_array, np)
    return _spans_from_shot_flags(primary_flags, visual_flags, cut_indices)


def _face_flags(frames: list[Path]) -> list[bool]:
    try:
        import cv2
    except ImportError:
        return [True] * len(frames)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        return [True] * len(frames)
    flags: list[bool] = []
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            flags.append(True)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
        flags.append(len(faces) > 0)
    return flags


def _fill_single_frame_gaps(flags: list[bool]) -> list[bool]:
    if len(flags) < 3:
        return flags
    smoothed = list(flags)
    for index in range(1, len(flags) - 1):
        if not flags[index] and flags[index - 1] and flags[index + 1]:
            smoothed[index] = True
    return smoothed


def _spans_from_flags(flags: list[bool]) -> list[tuple[float, float]]:
    """Maximal runs of consecutive True frames, as (start, end) seconds."""
    flags = _fill_single_frame_gaps(flags)
    spans: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, is_broll in enumerate(flags + [False]):
        if is_broll and run_start is None:
            run_start = index
        elif not is_broll and run_start is not None:
            start_t = run_start / FRAME_FPS
            end_t = index / FRAME_FPS
            if end_t - start_t >= MIN_BROLL_SPAN:
                spans.append((start_t, end_t))
            run_start = None
    return spans[:MAX_BROLL_SPANS]


def detect_broll_spans(frames: list[Path]) -> list[tuple[float, float]]:
    """Detect every B-roll cutaway in the reference timeline. Primary method
    is shot clustering (works even when B-roll contains faces); falls back to
    a no-face-frames heuristic when OpenCV/features are unavailable."""
    spans = _shot_based_spans(frames)
    if spans:
        return spans
    face_flags = _face_flags(frames)
    return _spans_from_flags([not has_face for has_face in face_flags])


# ---------------------------------------------------------------------------
# Stage 2: span description (vision tagging)
# ---------------------------------------------------------------------------

_TAG_PROMPT = (
    "You are analyzing a single video frame for a B-roll matching system. "
    "Reply with ONLY a JSON object (no markdown, no prose) with these exact keys:\n"
    '{"subjects": list of 1-3 concrete nouns (objects/people, lowercase), '
    '"setting": list of 1-2 location descriptors (e.g. ["office","indoors"]), '
    '"action": list of 0-2 verbs (e.g. ["typing","running"]), '
    '"category": one of [movie, sports, tech, lifestyle, money, other], '
    '"query": a short stock-footage search phrase of 3-8 words describing the shot, '
    '"mood": list of 0-3 words from [tense, uplifting, mysterious, epic, melancholic, '
    'energetic, calm, aggressive, romantic, nostalgic, ominous, joyful, neutral, '
    'dramatic, playful, sinister], '
    '"energy": one of [low, medium, high], '
    '"lighting": one of [low-key, high-key, natural, neon, golden-hour, mixed], '
    '"shot_type": one of [wide, medium, close-up, extreme-close-up, aerial, overhead, two-shot], '
    '"camera_motion": one of [static, pan, tilt, dolly, handheld, tracking, zoom], '
    '"depth_of_field": one of [deep, shallow], '
    '"color_palette": list of 0-3 dominant colors (e.g. ["deep blue", "amber"]) }.\n'
    "Use empty lists / empty strings when nothing fits. Unknown enum values are "
    "tolerated (we drop them silently), so prefer leaving a field empty over "
    "guessing a value that is not in the allowed list."
)


def _frame_content_hash(frame_path: Path) -> str:
    h = hashlib.sha256()
    try:
        with frame_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception:
        try:
            stat = frame_path.stat()
            h.update(f"{frame_path}:{stat.st_size}:{int(stat.st_mtime)}".encode())
        except Exception:
            h.update(str(frame_path).encode())
    return h.hexdigest()


def _tags_cache_dir(settings: Settings) -> Path:
    return settings.data_dir / "cache" / "broll_tags"


def _frame_tag_prompt_stamp_path(settings: Settings) -> Path:
    """Version-stamp file that lives OUTSIDE `broll_tags/` so a future purge
    doesn't delete its own stamp. Stores the integer prompt version that was
    last applied to the frame-tag cache."""
    return settings.data_dir / "cache" / "_frame_tag_prompt_version"


def _maybe_purge_frame_tag_cache(settings: Settings) -> bool:
    """One-shot purge of `data/cache/broll_tags/*.json` when the on-disk prompt
    version stamp is older than the live `intelligent_frame_tag_prompt_version`
    setting. Triggered at the top of every `build_library_index` call so old
    frame-tag cache entries (produced by an older prompt shape) are wiped
    exactly once when the prompt version is bumped.

    Idempotent: a second call with the same live version is a no-op.
    Returns True when a purge happened (useful for tests / diagnostics).
    Never raises — purge failure is logged and the live version stamp is NOT
    updated, so the next call will retry.
    """
    live_version = int(getattr(settings, "intelligent_frame_tag_prompt_version", 1) or 1)
    stamp_path = _frame_tag_prompt_stamp_path(settings)
    stamp_value = "0"
    if stamp_path.exists():
        try:
            stamp_value = stamp_path.read_text(encoding="utf-8").strip() or "0"
        except Exception:
            stamp_value = "0"
    try:
        stamp_int = int(stamp_value)
    except ValueError:
        stamp_int = 0
    if stamp_int >= live_version:
        return False
    cache_dir = _tags_cache_dir(settings)
    purged = 0
    try:
        if cache_dir.exists():
            for old in cache_dir.glob("*.json"):
                try:
                    old.unlink()
                    purged += 1
                except Exception:
                    logger.debug("Could not delete stale frame-tag cache file %s", old)
    except Exception:
        logger.exception("Frame-tag cache purge failed for %s", cache_dir)
        return False
    try:
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(str(live_version), encoding="utf-8")
    except Exception:
        logger.exception("Could not persist frame-tag prompt version stamp at %s", stamp_path)
        return False
    logger.info(
        "Purged %d stale frame-tag cache files (prompt version %d -> %d)",
        purged, stamp_int, live_version,
    )
    return True


def _parse_tags(raw: str) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if "\n" in text:
            _, _, rest = text.partition("\n")
            text = rest
        if text.endswith("```"):
            text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}

    def _as_list(v, cap: int) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip().lower() for x in v if str(x).strip()][:cap]
        if isinstance(v, str) and v.strip():
            return [v.strip().lower()]
        return []

    def _enum(value: object, vocab: frozenset[str]) -> str:
        s = str(value).strip().lower() if value is not None else ""
        return s if s in vocab else ""

    def _enum_list(value: object, vocab: frozenset[str], cap: int) -> list[str]:
        out: list[str] = []
        if isinstance(value, list):
            for x in value:
                s = str(x).strip().lower()
                if s in vocab and s not in out:
                    out.append(s)
                    if len(out) >= cap:
                        break
        elif isinstance(value, str):
            s = value.strip().lower()
            if s in vocab:
                out.append(s)
        return out

    def _palette(value: object, cap: int) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip().lower() for x in value if str(x).strip()][:cap]
        if isinstance(value, str) and value.strip():
            return [value.strip().lower()][:cap]
        return []

    category = str(obj.get("category", "")).strip().lower()
    if category not in VALID_CATEGORIES:
        category = ""
    query = " ".join(str(obj.get("query", "")).split())

    return {
        "subjects": _as_list(obj.get("subjects"), 3),
        "setting": _as_list(obj.get("setting"), 2),
        "action": _as_list(obj.get("action"), 2),
        "category": category,
        "query": query,
        "mood": _enum_list(obj.get("mood"), _MOOD_VOCAB, _MOOD_CAP),
        "energy": _enum(obj.get("energy"), _ENERGY_VOCAB),
        "lighting": _enum(obj.get("lighting"), _LIGHTING_VOCAB),
        "shot_type": _enum(obj.get("shot_type"), _SHOT_TYPE_VOCAB),
        "camera_motion": _enum(obj.get("camera_motion"), _CAMERA_MOTION_VOCAB),
        "depth_of_field": _enum(obj.get("depth_of_field"), _DEPTH_OF_FIELD_VOCAB),
        "color_palette": _palette(obj.get("color_palette"), _COLOR_PALETTE_CAP),
    }


def _frame_tags(frame_path: Path | None, settings: Settings, budget: "_VisionBudget | None") -> dict:
    """Structured scene tags for a single frame, disk-cached by frame hash so
    re-tagging across jobs/variations/library rebuilds is free."""
    if frame_path is None or not frame_path.exists():
        return {}
    cache_dir = _tags_cache_dir(settings)
    frame_hash = _frame_content_hash(frame_path)
    cache_file = cache_dir / f"{frame_hash}.json"
    try:
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        raw = _vision(frame_path, _TAG_PROMPT, settings, budget)
    except Exception:
        logger.exception("B-roll frame-tag vision failed for %s", frame_path)
        return {}
    tags = _parse_tags(raw)
    if tags:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(tags, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.debug("broll tag cache write failed for %s", frame_hash)
    return tags


def _merge_tags(frame_tag_list: list[dict]) -> dict:
    """Union a list of per-frame tag dicts into one span/clip-level dict."""
    if not frame_tag_list:
        return {}

    def _union(key: str, cap: int) -> list[str]:
        seen: list[str] = []
        for tags in frame_tag_list:
            raw = tags.get(key, [])
            # Tolerate non-list callers: a bare string becomes a one-element
            # list, anything else contributes nothing.
            if isinstance(raw, str):
                candidates: list[object] = [raw]
            elif isinstance(raw, list):
                candidates = raw
            else:
                continue
            for v in candidates:
                if not isinstance(v, str):
                    continue
                if v and v not in seen:
                    seen.append(v)
                if len(seen) >= cap:
                    return seen
        return seen

    def _mode(key: str) -> str:
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for i, tags in enumerate(frame_tag_list):
            v = tags.get(key, "")
            if isinstance(v, str) and v:
                counts[v] = counts.get(v, 0) + 1
                first_seen.setdefault(v, i)
        if not counts:
            return ""
        # Highest count, then earliest first-seen index for tie-break.
        best = max(counts.items(), key=lambda kv: (kv[1], -first_seen[kv[0]]))
        return best[0]

    counts: dict[str, int] = {}
    for tags in frame_tag_list:
        c = tags.get("category", "")
        if c:
            counts[c] = counts.get(c, 0) + 1
    category = max(counts, key=lambda k: counts[k]) if counts else ""

    # Prefer the middle sample's query (most representative of the span);
    # fall back to the first non-empty one found.
    query = ""
    mid = len(frame_tag_list) // 2
    order = [mid] + [i for i in range(len(frame_tag_list)) if i != mid]
    for i in order:
        q = frame_tag_list[i].get("query", "")
        if q:
            query = q
            break

    return {
        "subjects": _union("subjects", 4),
        "setting": _union("setting", 2),
        "action": _union("action", 2),
        "category": category,
        "query": query,
        "mood": _union("mood", _MOOD_CAP),
        "energy": _mode("energy"),
        "lighting": _mode("lighting"),
        "shot_type": _mode("shot_type"),
        "camera_motion": _mode("camera_motion"),
        "depth_of_field": _mode("depth_of_field"),
        "color_palette": _union("color_palette", _COLOR_PALETTE_CAP),
    }


def _sample_span_frames(start: float, end: float, frames: list[Path]) -> list[Path]:
    """Sample up to 3 frames at 25/50/75% of the span's frame range.

    When FRAMES_PER_SPAN_DESCRIBE is set to 1 (default) only the middle frame
    is returned. Three samples gave marginally better tag coverage on long
    b-roll cuts, but for the 1-3s cutaways the analyzer detects from a
    short reference, a single mid-span frame carries the same subjects /
    setting / action information at 1/3 the cloud-vision cost. The cache
    keys are still per-frame so switching back to 3 for a debug pass does
    not invalidate existing tags.
    """
    if not frames:
        return []
    lo = max(0, min(len(frames) - 1, int(start * FRAME_FPS)))
    hi = max(lo, min(len(frames) - 1, int(end * FRAME_FPS)))
    span_frames = frames[lo:hi + 1] or [frames[lo]]
    if len(span_frames) <= 1:
        return span_frames
    if FRAMES_PER_SPAN_DESCRIBE <= 1:
        return [span_frames[len(span_frames) // 2]]
    if len(span_frames) <= FRAMES_PER_SPAN_DESCRIBE:
        return span_frames
    return [span_frames[min(len(span_frames) - 1, int(frac * len(span_frames)))] for frac in (0.25, 0.5, 0.75)[:FRAMES_PER_SPAN_DESCRIBE]]


def _truncate_query(query: str, settings: Settings) -> str:
    words = (query or "").split()
    limit = max(1, settings.broll_query_max_words)
    return " ".join(words[:limit])


def describe_spans(spans: list[tuple[float, float]], frames: list[Path], settings: Settings) -> list[SpanProfile]:
    """Vision-tag every detected span. Budget scoped to this call only, sized
    by span count, so a job with many spans can't hang forever on flaky keys.

    Frame-tag calls run in parallel. For an 8-span reference with 3 sampled
    frames each = 24 cloud calls. At ~15s/call on a third-party LLM router
    (see job cef566622990's `analyzing` stage = 357s on byNara router),
    sequential execution is the difference between 6 min and ~3 min wall-
    clock. Per-span budget accounting is guarded by a lock so concurrent
    workers can't race the read-modify-write on `budget.remaining`.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    budget = _VisionBudget(min(480.0, max(120.0, 40.0 * max(1, len(spans)))))
    budget_lock = threading.Lock()
    budget_wall_start = time.monotonic()

    def _guarded_tags(f: Path) -> dict:
        # Refuse to call the cloud rung once the budget is exhausted so a
        # stalled upstream can't drag the parallel pool past its cap.
        with budget_lock:
            if budget.exhausted():
                return {}
            remaining_before = budget.remaining
        tags = _frame_tags(f, settings, budget)
        return tags

    # Collect every (span_index, frame) pair up-front so parallel workers
    # can pick from a flat queue and we preserve the original span order
    # in the returned profiles.
    work: list[tuple[int, Path]] = []
    span_frames: list[list[Path]] = []
    for index, (start, end) in enumerate(spans):
        sampled = _sample_span_frames(start, end, frames)
        span_frames.append(sampled)
        for f in sampled:
            work.append((index, f))

    # Cap workers at ollama_max_concurrency (default 2): the vision ladder
    # already gates upstream concurrency at this number, so spawning more
    # threads here just queues them up while still respecting that cap.
    # Bound by len(work) so an empty span list doesn't spin up a pool.
    max_workers = max(1, min(settings.ollama_max_concurrency, len(work)))

    tags_by_span: list[list[dict]] = [[] for _ in spans]
    if max_workers == 1 or len(work) <= 1:
        for index, frame in work:
            tags_by_span[index].append(_guarded_tags(frame))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_guarded_tags, frame): index
                for index, frame in work
            }
            for future, index in futures.items():
                tags_by_span[index].append(future.result())

    if time.monotonic() - budget_wall_start > 30.0:
        logger.info(
            "describe_spans: %d cloud frame tags across %d spans in %.1fs (workers=%d)",
            len(work), len(spans), time.monotonic() - budget_wall_start, max_workers,
        )

    profiles: list[SpanProfile] = []
    for index, _ in enumerate(spans):
        merged = _merge_tags(tags_by_span[index])
        profiles.append(SpanProfile(
            start=spans[index][0],
            end=spans[index][1],
            subjects=merged.get("subjects", []),
            setting=merged.get("setting", []),
            action=merged.get("action", []),
            category=merged.get("category") or "other",
            query=_truncate_query(merged.get("query", ""), settings),
        ))
    return profiles


# ---------------------------------------------------------------------------
# Stage 3: local library index
# ---------------------------------------------------------------------------


def _folder_category(folder: str) -> str:
    """Seed a default category from the library's folder-name conventions."""
    key = folder.strip().lower().replace("_", " ")
    if "good stuff" in key:
        return "movie"
    if "nba" in key:
        return "sports"
    if "ronaldo" in key:
        return "sports"
    return "other"


def _extract_single_frame(source: Path, at_seconds: float, output_path: Path, settings: Settings) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                settings.ffmpeg_path,
                "-y",
                "-ss", f"{max(0.0, at_seconds):.3f}",
                "-i", str(source),
                "-frames:v", "1",
                "-vf", "scale=320:-1",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 500


def _tag_library_clip(path: Path, settings: Settings, budget: "_VisionBudget | None") -> dict:
    try:
        duration = probe_duration(path, settings)
    except Exception:
        duration = 0.0
    if duration <= 0:
        return {}
    sample_dir = settings.data_dir / "cache" / "broll_library_frames"
    key = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:16]
    frame_tags: list[dict] = []
    for frac in (0.33, 0.66):
        at = max(0.0, min(duration - 0.05, duration * frac))
        frame_path = sample_dir / f"{key}_{int(frac * 100)}.jpg"
        if not frame_path.exists():
            _extract_single_frame(path, at, frame_path, settings)
        if frame_path.exists():
            frame_tags.append(_frame_tags(frame_path, settings, budget))
    return _merge_tags(frame_tags)


def _index_cache_path(settings: Settings) -> Path:
    return settings.data_dir / "cache" / "broll_index.json"


def build_library_index(settings: Settings) -> list[LibraryClip]:
    """Scan the local B-roll library once, tag every new/changed file with
    vision, and cache the result forever at data/cache/broll_index.json.
    Incremental: unchanged files (same mtime+size) are read straight from
    cache, so only new library additions pay the vision cost."""
    # One-shot purge of the frame-tag cache when the prompt version stamp is
    # older than the live setting. Runs BEFORE any clip re-tag so newly-tagged
    # frames always come from the current prompt. Idempotent — second call
    # with the same live version is a no-op.
    _maybe_purge_frame_tag_cache(settings)
    library_dir = settings.broll_library_dir
    if not library_dir.exists():
        return []
    cache_path = _index_cache_path(settings)
    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if raw.get("version") == _INDEX_CACHE_VERSION:
                cached = raw.get("clips", {})
        except Exception:
            cached = {}

    files = sorted(p for p in library_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    clips: list[LibraryClip] = []
    changed = False
    budget = _VisionBudget(None)  # uncapped: one-time build, deterministic call count, disk-cached forever
    # When every vision provider is down (rate-limited / quota-exhausted),
    # tagging 290 clips would waste ~2 failed ladder walks per clip. After a
    # few consecutive all-frames-failed clips, stop calling vision for the
    # rest of this run: those clips get folder-seeded categories NOW but are
    # NOT persisted, so the next run (with providers back) tags them properly.
    consecutive_vision_failures = 0
    vision_disabled = False

    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        key = str(path.resolve())
        entry = cached.get(key)
        if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
            clips.append(LibraryClip(
                path=path, mtime=stat.st_mtime, size=stat.st_size,
                subjects=entry.get("subjects", []), setting=entry.get("setting", []),
                category=entry.get("category", "other"), folder=entry.get("folder", ""),
                query=entry.get("query", ""),
                mood=entry.get("mood", []),
                energy=entry.get("energy", ""),
                lighting=entry.get("lighting", ""),
                shot_type=entry.get("shot_type", ""),
                camera_motion=entry.get("camera_motion", ""),
                depth_of_field=entry.get("depth_of_field", ""),
                color_palette=entry.get("color_palette", []),
            ))
            continue

        folder = path.parent.name
        tagged = {} if vision_disabled else _tag_library_clip(path, settings, budget)
        if not vision_disabled:
            if tagged.get("subjects") or tagged.get("query"):
                consecutive_vision_failures = 0
            else:
                consecutive_vision_failures += 1
                if consecutive_vision_failures >= 4:
                    vision_disabled = True
                    logger.warning(
                        "B-roll library tagging: vision failing repeatedly; "
                        "using folder categories for remaining untagged clips this run"
                    )
        category = tagged.get("category") or _folder_category(folder)
        clip = LibraryClip(
            path=path, mtime=stat.st_mtime, size=stat.st_size,
            subjects=tagged.get("subjects", []), setting=tagged.get("setting", []),
            category=category, folder=folder, query=tagged.get("query", ""),
            mood=tagged.get("mood", []),
            energy=tagged.get("energy", ""),
            lighting=tagged.get("lighting", ""),
            shot_type=tagged.get("shot_type", ""),
            camera_motion=tagged.get("camera_motion", ""),
            depth_of_field=tagged.get("depth_of_field", ""),
            color_palette=tagged.get("color_palette", []),
        )
        clips.append(clip)
        # Persist only successfully-tagged entries: an empty-tag entry written
        # to the cache would never be retried once providers recover.
        if tagged.get("subjects") or tagged.get("query"):
            cached[key] = {
                "mtime": stat.st_mtime, "size": stat.st_size,
                "subjects": clip.subjects, "setting": clip.setting,
                "category": clip.category, "folder": clip.folder, "query": clip.query,
                "mood": clip.mood,
                "energy": clip.energy,
                "lighting": clip.lighting,
                "shot_type": clip.shot_type,
                "camera_motion": clip.camera_motion,
                "depth_of_field": clip.depth_of_field,
                "color_palette": clip.color_palette,
            }
            changed = True

    if changed:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"version": _INDEX_CACHE_VERSION, "clips": cached}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(cache_path)
        except Exception:
            logger.exception("Failed to persist B-roll library index cache")
    return clips


# ---------------------------------------------------------------------------
# Stage 4: local matching
# ---------------------------------------------------------------------------


_LOCAL_SCORE_MAX = 8.0  # 3.0 (category) + 3.0 (subjects, cap 3) + 2.0 (setting, cap 2)

# How much the vibe/cinematography "intelligent selector" bonus can lift the
# raw keyword score when both sides carry extended vibe tags. 0.35 = at most
# ~a third of the way from 0.0 to 1.0 in the worst case (already ~0.7 base +
# vibe boost brings a candidate from "B-tier" to "A-tier"). The bonus is a
# multiplier on the (1.0 - base) gap so a base of 1.0 cannot exceed 1.0 and a
# base of 0.0 needs a real vibe match to come back into play. Tunable in
# one place.
_VIBE_BONUS_WEIGHT = 0.35

# Vibe-match score ranges in [0, 1] for the closed-vocab enums. Unknown /
# empty values degrade gracefully via the `available` checks below: a clip
# with no `lighting` field doesn't penalise (and isn't penalised) for not
# matching.
_VIBE_FIELDS: tuple[tuple[str, float, float], ...] = (
    # (field_name, weight_within_vibe, exact_match_value)
    ("mood",        0.40, 1.00),
    ("lighting",    0.20, 1.00),
    ("energy",      0.15, 1.00),
    ("shot_type",   0.15, 1.00),
    ("depth_of_field", 0.10, 1.00),
)

# Continuity ledger: one entry per (job_id) so parallel jobs don't pollute
# each other's diversity nudge. Empty when no job has recorded a pick yet.
# Reset between variations inside fetch_broll_cut_variations.
_CONTINUITY_LEDGER: dict[str, "_ContinuityLedger"] = {}


def _clip_vibe_tags(clip: LibraryClip) -> dict[str, object]:
    """Return the clip's vibe fields as a dict, OR an empty dict when the
    clip wasn't indexed by `broll_intelligence/library_indexer`. Used to
    detect whether the intelligent selector has anything to chew on —
    legacy-clipped libraries fall through to the keyword-only score."""
    out: dict[str, object] = {}
    for name, _, _ in _VIBE_FIELDS:
        v = getattr(clip, name, None)
        if not v:
            continue
        if isinstance(v, list) and not any(str(x).strip() for x in v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[name] = v
    return out


def _span_vibe_tags(profile: SpanProfile) -> dict[str, object]:
    """Mirror of `_clip_vibe_tags` for the reference-span profile. The
    SpanProfile currently doesn't carry vibe fields directly — they live
    on `ReferenceAnalysis.broll_span_tags[i]` — so we attach them at the
    call site (see `_vibe_score_for`). This helper just makes the contract
    explicit and test-friendly."""
    out: dict[str, object] = {}
    for name, _, _ in _VIBE_FIELDS:
        v = getattr(profile, name, None)
        if not v:
            continue
        if isinstance(v, list) and not any(str(x).strip() for x in v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[name] = v
    return out


def _vibe_subscore(a: object, b: object) -> float:
    """Compare two vibe values. Both lists -> Jaccard (lower-case, trimmed).
    Both strings -> 1.0 on exact match, 0.5 on a small synonym bucketing,
    else 0.0. Mixed types -> 0.0 (irrelevant comparison)."""
    if isinstance(a, list) and isinstance(b, list):
        sa = {str(x).strip().lower() for x in a if str(x).strip()}
        sb = {str(x).strip().lower() for x in b if str(x).strip()}
        if not sa and not sb:
            return 0.0
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)
    if isinstance(a, str) and isinstance(b, str):
        a_s, b_s = a.strip().lower(), b.strip().lower()
        if not a_s or not b_s:
            return 0.0
        if a_s == b_s:
            return 1.0
        # Lighting/energy get a small "adjacent" allowance so a span asking
        # for "low-key" doesn't completely zero out a clip tagged "mixed".
        adj = {
            "low-key": {"mixed"},
            "high-key": {"mixed"},
            "low": {"medium"},
            "high": {"medium"},
            "wide": {"medium", "aerial"},
            "medium": {"wide", "close-up"},
            "close-up": {"medium", "extreme-close-up"},
            "extreme-close-up": {"close-up"},
            "aerial": {"overhead", "wide"},
            "overhead": {"aerial"},
            "shallow": {"deep"},
            "deep": {"shallow"},
        }
        if b_s in adj.get(a_s, set()) or a_s in adj.get(b_s, set()):
            return 0.5
        return 0.0
    return 0.0


def _vibe_score_for(profile: SpanProfile, clip: LibraryClip) -> float:
    """0..1 vibe-match score (LEGACY path, kept for back-compat). Returns
    0.0 when either side lacks vibe fields.

    The SPEC §8 path lives in ``app.selector_helpers._vibe_score_for_resolved``
    and goes through ``_resolve_span_vibe`` first so empty span fields
    back-fill from the reference house style. ``_local_score`` uses the
    new path; this helper is preserved for any caller that wants the
    non-resolved, profile-only match.
    """
    profile_tags = _span_vibe_tags(profile)
    clip_tags = _clip_vibe_tags(clip)
    if not profile_tags or not clip_tags:
        return 0.0
    total_weight = 0.0
    weighted = 0.0
    for name, weight, _exact_max in _VIBE_FIELDS:
        if name in profile_tags and name in clip_tags:
            weighted += weight * _vibe_subscore(profile_tags[name], clip_tags[name])
            total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return max(0.0, min(1.0, weighted / total_weight))


def _local_score(
    profile: SpanProfile,
    clip: LibraryClip,
    *,
    intelligent: bool = False,
    reference_house: dict | None = None,
    continuity_penalty: float = 0.0,
    job_id: str | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> tuple[float, float, float, float]:
    """0..1 normalized match score, plus three diagnostic components.

    Returns ``(total, vibe, cinema, continuity_penalty)``.

    When ``intelligent=False``, ``vibe`` and ``cinema`` are both 0.0 (callers
    should ignore them); ``continuity_penalty`` is always returned because
    the diagnostic surface wants it either way.

    When ``intelligent=True`` AND the clip + span carry vibe fields, the
    score is multiplied through:

      * ``vibe`` is the resolved vibe match (uses ``reference_house`` as
        fallback when the span's own field is empty — SPEC §8);
      * ``cinema`` is the cinema-match score from `_cinema_match`, floored
        at ``_CINEMA_FLOOR`` so a true cinematographic mismatch cannot
        escape the bottom (SPEC §7);
      * ``cinema_lift = _CINEMA_LIFT_TERM * (cinema - 0.6)`` when
        ``cinema < 0.6`` (else 0.0) is added on top of the legacy
        keyword + vibe math.

    Cinema-aware math shape (SPEC §7):
      ``boosted = base + _VIBE_BONUS_WEIGHT * vibe * (1 - base) + cinema_lift``

    ``continuity_penalty`` (non-positive) is added once at the end so a
    candidate visually identical to the previous pick can never escape
    a -0.08 tax via a perfect content match.
    """
    score = 0.0
    if profile.category and clip.category and profile.category == clip.category:
        score += 3.0
    score += len(set(profile.subjects) & set(clip.subjects)) * 1.0
    score += len(set(profile.setting) & set(clip.setting)) * 1.0
    base = min(1.0, score / _LOCAL_SCORE_MAX)

    if not intelligent:
        total = max(0.0, min(1.0, base + continuity_penalty))
        return total, 0.0, 0.0, continuity_penalty

    resolved_span_vibe = _resolve_span_vibe(profile, reference_house)
    vibe = _vibe_score_for_resolved(resolved_span_vibe, clip)
    cinema = _cinema_match(profile, clip)

    cinema_lift = _CINEMA_LIFT_TERM * (cinema - 0.6) if cinema < 0.6 else 0.0
    boosted = base + _VIBE_BONUS_WEIGHT * vibe * (1.0 - base) + cinema_lift
    total = max(0.0, min(1.0, boosted + continuity_penalty))
    return total, vibe, cinema, continuity_penalty


def _rank_local(
    profile: SpanProfile,
    index: list[LibraryClip],
    used_clips: set[Path],
    *,
    intelligent: bool = False,
    reference_house: dict | None = None,
    job_id: str | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> list[tuple[LibraryClip, float, float, float, float]]:
    """Like before, but each row now carries
    ``(clip, total_score, vibe_score, cinema_score, continuity_penalty)``.

    ``cinema_score`` and ``continuity_penalty`` are plumbed through
    unchanged from ``_local_score`` so the caller can surface them on
    the diagnostic row.
    """
    scored: list[tuple[LibraryClip, float, float, float, float]] = []
    for clip in index:
        try:
            if clip.path.resolve() in used_clips:
                continue
        except OSError:
            continue
        # Pre-compute the continuity penalty for this candidate against
        # the current ledger state. Threaded in so consecutive picks
        # pay the diversity tax (SPEC §9).
        pending_cont = 0.0
        if ledger is not None:
            pending_cont = ledger.penalty_for(clip)
        total, vibe, cinema, cont = _local_score(
            profile, clip, intelligent=intelligent,
            reference_house=reference_house,
            continuity_penalty=pending_cont,
            job_id=job_id, ledger=ledger,
        )
        scored.append((clip, total, vibe, cinema, cont))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def _profile_description(profile: SpanProfile) -> str:
    parts = list(profile.subjects) + list(profile.setting)
    if profile.action:
        parts.append(", ".join(profile.action))
    text = ", ".join(p for p in parts if p) or profile.query or profile.category
    return f"{text} (category: {profile.category})"


def _llm_pick_local(profile: SpanProfile, ranked_top: list[tuple[LibraryClip, float, float, float, float]], settings: Settings) -> LibraryClip | None:
    if not ranked_top:
        return None
    # Disk cache: skip the cloud chat round-trip when we've already judged
    # this exact (query, top-clip set, model) triple. Stored under a
    # per-model subdir so changing llm_vision_model auto-invalidates the
    # whole rung without a sweep. Cached value is the index string ("0",
    # "1", ...) or "" for the "no match" sentinel.
    cache_dir = settings.data_dir / "cache" / "llm_picks" / settings.llm_vision_model
    sorted_paths = sorted(str(clip.path) for clip, _s, _v, _c, _co in ranked_top)
    hash_input = "|".join([
        profile.query or "",
        *sorted_paths,
        settings.llm_vision_model or "",
    ])
    digest = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{digest}.txt"
    try:
        if cache_file.exists():
            cached_value = cache_file.read_text(encoding="utf-8").strip()
            if cached_value == "":
                return None
            try:
                cached_idx = int(cached_value)
            except ValueError:
                cached_idx = -1
            if 0 <= cached_idx < len(ranked_top):
                return ranked_top[cached_idx][0]
            return None
    except Exception:
        pass

    lines = []
    for i, (clip, _score, _vibe, _cinema, _cont) in enumerate(ranked_top):
        desc = ", ".join(clip.subjects + clip.setting) or clip.category
        lines.append(f"{i}: {clip.category} — {desc} (folder: {clip.folder})")
    prompt = (
        f"Reference cutaway shows: {_profile_description(profile)}.\n"
        "Which of these candidate library clips shows the same kind of "
        "objects/scene/location category? Same category in a different exact "
        "place is fine (e.g. laptop-at-home matches PC-at-home). Reply with "
        "ONLY the number, or NONE if none are a reasonable match.\n" + "\n".join(lines)
    )
    try:
        raw = _chat(prompt, settings, timeout=15.0)
    except Exception:
        return None

    cached_value = ""
    result_clip: LibraryClip | None = None
    if not raw or "none" in raw.strip().lower()[:12]:
        result_clip = None
    else:
        match = re.search(r"\d+", raw)
        if match:
            idx = int(match.group())
            if 0 <= idx < len(ranked_top):
                cached_value = str(idx)
                result_clip = ranked_top[idx][0]
            else:
                result_clip = None
        else:
            result_clip = None

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(cached_value, encoding="utf-8")
    except Exception:
        logger.debug("LLM pick cache write failed for %s", cache_file)
    return result_clip


def match_local(
    profile: SpanProfile,
    index: list[LibraryClip],
    used_clips: set[Path],
    settings: Settings,
    *,
    intelligent: bool = False,
    reference_house: dict | None = None,
    job_id: str | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> tuple[LibraryClip, float, float, float] | None:
    """Cheap category/subject/setting scoring over the whole index, with
    an optional vibe/lighting/shot-type bonus when `intelligent=True` AND
    the clip library was indexed by `broll_intelligence/library_indexer`
    (i.e. carries `mood` / `lighting` / `shot_type` fields). LLM tie-break
    still runs over the top 5.

    Returns ``(clip, vibe_score, continuity_penalty, total_score)`` per
    SPEC §9 so the caller can record the vibe component + the diversity
    tax + the overall match confidence on the diagnostic row.
    ``total_score`` is the picked clip's full normalized score
    (``_local_score``'s `total` field, after vibe + cinema lift +
    continuity penalty) and lets `_gather_span_pool` decide whether the
    local match is confident enough to skip the YouTube rung even when
    `pool_size > 1`. `vibe_score == 0.0` means the intelligent bonus was
    inactive (either off, or the data wasn't there). `continuity_penalty`
    is non-positive (0.0 when no previous pick is in the ledger, else
    ``settings.intelligent_continuity_penalty_max`` when the cosine
    similarity crosses the threshold).

    Returns `None` when the threshold isn't met — search moves on to
    YouTube.

    LLM-pick short-circuit: when the ranker's top-1 clip has a clear lead
    over the runner-up AND its score is well above the threshold, skip the
    per-span cloud LLM call (~15s on a third-party router). Across an
    8-span reference this saves up to 8 cloud round-trips ≈ 120s of wall-
    clock on broll_recovery. Falls through to the existing LLM pick when
    the ranker is too close to call.
    """
    ranked = _rank_local(
        profile, index, used_clips, intelligent=intelligent,
        reference_house=reference_house, job_id=job_id, ledger=ledger,
    )
    if not ranked:
        return None
    top = ranked[:5]
    best_clip, best_score, best_vibe, _best_cinema, best_cont = top[0]
    runner_up_score = top[1][1] if len(top) > 1 else 0.0
    skip_llm = (
        best_score >= settings.broll_local_match_threshold
        and best_score >= runner_up_score + 0.15
    )
    if not skip_llm:
        picked = _llm_pick_local(profile, top, settings)
        if picked is not None:
            # Look the picked clip back up to recover its vibe + continuity
            # + total score (so _gather_span_pool can decide whether to skip
            # the YouTube rung).
            for clip, total, vibe, _cinema, cont in ranked:
                if clip is picked:
                    return clip, vibe, cont, total
            return picked, 0.0, 0.0, 0.0
    if best_score >= settings.broll_local_match_threshold:
        return best_clip, best_vibe, best_cont, best_score
    return None


# ---------------------------------------------------------------------------
# Stage 5: YouTube fallback
# ---------------------------------------------------------------------------

_BROWSER_COOKIES_BROKEN = False


def _ytdlp_cookie_args(settings: Settings) -> list[str]:
    global _BROWSER_COOKIES_BROKEN
    cookies_file = settings.ytdlp_cookies_file
    if cookies_file and cookies_file.exists():
        return ["--cookies", str(cookies_file)]
    if settings.ytdlp_cookies_from_browser.strip() and not _BROWSER_COOKIES_BROKEN:
        return ["--cookies-from-browser", settings.ytdlp_cookies_from_browser.strip()]
    return []


def _is_browser_cookie_error(stderr: str) -> bool:
    lowered = stderr.lower()
    if "could not copy" in lowered and "cookie" in lowered:
        return True
    if "failed to decrypt" in lowered and "dpapi" in lowered:
        return True
    if "extract cookies" in lowered and "browser" in lowered:
        return True
    return False


def _iso8601_duration_to_seconds(value: str) -> float:
    """Parse a YouTube contentDetails.duration (ISO-8601, e.g. 'PT1M30S')."""
    if not value:
        return 0.0
    m = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not m:
        return 0.0
    days, hours, mins, secs = (int(g) if g else 0 for g in m.groups())
    return days * 86400 + hours * 3600 + mins * 60 + secs


def _is_junk_youtube_entry(url: str, title: str) -> bool:
    if "/shorts/" in url.lower():
        return True
    title_lower = title.lower()
    return any(term in title_lower for term in YT_JUNK_TITLE_TERMS)


def _youtube_data_api_search(query: str, per_query: int, settings: Settings) -> list[dict] | None:
    """Search via the Data API v3. Returns None (not []) when every key
    failed/quota-exceeded, so the caller falls back to yt-dlp ytsearch."""
    keys = settings.youtube_data_api_keys
    if not keys:
        return None
    for key in keys:
        try:
            search_url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode({
                "part": "snippet", "q": query, "type": "video",
                "maxResults": min(50, max(1, per_query)),
                "videoEmbeddable": "true", "key": key,
            })
            with urllib.request.urlopen(search_url, timeout=SEARCH_TIMEOUT) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                logger.warning("YouTube Data API key rotated (HTTP %s)", e.code)
                continue
            logger.warning("YouTube Data API search failed (HTTP %s)", e.code)
            return None
        except Exception as e:
            logger.warning("YouTube Data API search error (%s): %s", type(e).__name__, e)
            continue

        ids = [item.get("id", {}).get("videoId") for item in data.get("items", []) if item.get("id", {}).get("videoId")]
        if not ids:
            return []
        durations: dict[str, float] = {}
        try:
            videos_url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode({
                "part": "contentDetails", "id": ",".join(ids), "key": key,
            })
            with urllib.request.urlopen(videos_url, timeout=SEARCH_TIMEOUT) as response:
                vdata = json.loads(response.read())
            for item in vdata.get("items", []):
                durations[item.get("id")] = _iso8601_duration_to_seconds(
                    item.get("contentDetails", {}).get("duration", "")
                )
        except Exception:
            pass
        by_id = {item.get("id", {}).get("videoId"): item.get("snippet", {}) for item in data.get("items", [])}
        entries = []
        for video_id in ids:
            snippet = by_id.get(video_id, {})
            entries.append({
                "id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "duration": durations.get(video_id, 0.0),
                "title": snippet.get("title") or "",
            })
        return entries
    return None


def _yt_dlp_search(query: str, per_query: int, settings: Settings) -> list[dict]:
    command = [
        sys.executable, "-m", "yt_dlp", "--dump-json", "--flat-playlist",
        "--no-download", "--ignore-errors", f"ytsearch{per_query}:{query}",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=SEARCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return []
    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            info = json.loads(line)
        except json.JSONDecodeError:
            continue
        video_id = info.get("id")
        info["url"] = (
            info.get("url") or info.get("webpage_url") or info.get("original_url")
            or (f"https://www.youtube.com/watch?v={video_id}" if video_id else None)
        )
        rows.append(info)
    return rows


_YT_SEARCH_CACHE_VERSION = 1
# Cache hits older than this are re-fetched: trending videos go stale and a
# 7-day-old hit could be a removed / region-locked clip by now.
_YT_SEARCH_CACHE_TTL_SECONDS = 7 * 24 * 3600


def _yt_search_cache_dir(settings: Settings) -> Path:
    return settings.data_dir / "cache" / "youtube_search"


def _yt_search_cache_key(query: str, settings: Settings) -> Path:
    """Filename hash for a (query, settings) cache entry. Settings fingerprints
    YT_RESULTS_PER_QUERY + the API key set so a quota-rotated key still produces
    a fresh entry."""
    fingerprint = "|".join([
        query.strip().lower(),
        str(YT_RESULTS_PER_QUERY),
        ",".join(settings.youtube_data_api_keys) or "-",
    ])
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    return _yt_search_cache_dir(settings) / f"{digest}.json"


def _yt_search_cache_read(query: str, settings: Settings) -> list[dict] | None:
    cache_file = _yt_search_cache_key(query, settings)
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("version") != _YT_SEARCH_CACHE_VERSION:
        return None
    cached_at = float(payload.get("cached_at", 0))
    if cached_at <= 0 or (time.time() - cached_at) > _YT_SEARCH_CACHE_TTL_SECONDS:
        return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else None


def _yt_search_cache_write(query: str, settings: Settings, rows: list[dict]) -> None:
    try:
        cache_dir = _yt_search_cache_dir(settings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _yt_search_cache_key(query, settings)
        cache_file.write_text(
            json.dumps(
                {"version": _YT_SEARCH_CACHE_VERSION, "cached_at": time.time(), "rows": rows},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("YouTube search cache write failed for query=%s", query)


def _youtube_search_entries(queries: list[str], settings: Settings) -> list[dict]:
    """Aggregate results across query variants, Data API first (falling back
    to yt-dlp ytsearch per query on failure), deduped and junk-filtered.

    Each query is cached on disk for 7 days: re-running the same analysis
    (variations, retries, similar jobs) reuses the search and skips the
    YouTube API quota burn + the yt-dlp round-trip entirely.
    """
    seen_ids: set[str] = set()
    entries: list[dict] = []
    for query in queries:
        api_rows: list[dict] | None
        cached = _yt_search_cache_read(query, settings)
        if cached is not None:
            api_rows = cached
        else:
            fresh = _youtube_data_api_search(query, YT_RESULTS_PER_QUERY, settings)
            api_rows = fresh if fresh is not None else _yt_dlp_search(query, YT_RESULTS_PER_QUERY, settings)
            # Cache even the yt-dlp fallback so the next run skips it.
            if api_rows:
                _yt_search_cache_write(query, settings, api_rows)
        rows = api_rows or []
        for info in rows:
            video_id = info.get("id")
            if not video_id or video_id in seen_ids:
                continue
            duration = float(info.get("duration") or 0)
            if duration and not (YT_MIN_SOURCE_SECONDS <= duration <= YT_MAX_SOURCE_SECONDS):
                continue
            url = info.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            title = info.get("title") or ""
            if _is_junk_youtube_entry(url, title):
                continue
            seen_ids.add(video_id)
            entries.append({"id": video_id, "url": url, "duration": duration, "title": title})
    return entries


def _download_youtube_preview(url: str, output_path: Path, settings: Settings, preview_seconds: float = PREVIEW_SECONDS) -> bool:
    """Short, low-res preview download for vision comparison — not the final
    B-roll source necessarily, but cheap enough to try on every candidate."""
    global _BROWSER_COOKIES_BROKEN
    if output_path.exists() and output_path.stat().st_size > 10_000:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_command = [
        sys.executable, "-m", "yt_dlp", "--no-playlist",
        "-f", "worstvideo[height>=360][ext=mp4]/worst[ext=mp4]/worst",
        "--download-sections", f"*0-{max(3.0, preview_seconds):.0f}",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
    ]
    cookie_args = _ytdlp_cookie_args(settings)
    try:
        result = subprocess.run(base_command + cookie_args + [url], capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT)
    except subprocess.TimeoutExpired:
        logger.info("YouTube preview download timed out for %s", url)
        output_path.unlink(missing_ok=True)
        return False
    if result.returncode != 0 and cookie_args and "cookie" in result.stderr.lower():
        if _is_browser_cookie_error(result.stderr):
            _BROWSER_COOKIES_BROKEN = True
        try:
            result = subprocess.run(base_command + [url], capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 10_000


def _profile_similarity(profile: SpanProfile, tags: dict) -> float:
    score = 0.0
    if profile.category and tags.get("category") and profile.category == tags["category"]:
        score += 3.0
    score += len(set(profile.subjects) & set(tags.get("subjects", []))) * 1.0
    score += len(set(profile.setting) & set(tags.get("setting", []))) * 1.0
    return score


def search_youtube_candidates(
    profile: SpanProfile,
    cache_dir: Path,
    settings: Settings,
    count: int = 2,
    *,
    reference_house: dict | None = None,
    intelligent: bool = True,
) -> list[Path]:
    """Download + score up to `count` YouTube preview candidates for a profile.

    When ``intelligent=True`` (default): scores each preview with
    ``_local_score(intelligent=True)`` so a downloaded preview also benefits
    from the cinema/vibe lift. Per-frame tagging uses the EXTENDED
    ``_TAG_PROMPT`` (mood/lighting/shot_type/camera_motion/depth_of_field/
    color_palette) — see SPEC §3. When the new fields are missing on a
    preview (vision-model miss), falls back to the legacy ``_profile_similarity``
    (subject / setting / category overlap) divided by 6 to put it in the
    same 0..1 space.

    The continuity ledger does NOT participate in YouTube scoring
    (``continuity_penalty=0.0``) — YouTube is the safety-net rung, not the
    visual-variety driver.

    Caller owns the `cache_dir`; this function writes preview_0..N-1.mp4 into
    it and may safely be called from the pack path so multiple spans share a
    single YouTube cache directory without colliding.
    """
    query = _truncate_query(profile.query, settings)
    if not query:
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)
    queries = [query]
    wide = f"{query} b roll"
    if wide.lower() != query.lower():
        queries.append(wide)

    entries = _youtube_search_entries(queries, settings)
    if not entries:
        return []

    scored: list[tuple[Path, float, float, float]] = []
    for i, entry in enumerate(entries[: max(count, 2)]):
        preview_path = cache_dir / f"preview_{i}.mp4"
        if not _download_youtube_preview(entry["url"], preview_path, settings):
            continue
        try:
            duration = probe_duration(preview_path, settings)
        except Exception:
            duration = 0.0
        if duration <= 0:
            continue
        frame_path = cache_dir / f"preview_{i}.jpg"
        if not _extract_single_frame(preview_path, duration / 2, frame_path, settings):
            continue
        tags = _frame_tags(frame_path, settings, budget=None)
        if not tags:
            continue
        if intelligent:
            # Wrap the frame tags into a LibraryClip-shaped stub so the
            # vibe-aware scorer can read its cinema / mood / lighting fields.
            # A stub (not a real LibraryClip) means continuity / ranking
            # bookkeeping is NOT engaged for YouTube — see the docstring.
            stub = LibraryClip(
                path=preview_path, mtime=0.0, size=0,
                subjects=list(tags.get("subjects", [])),
                setting=list(tags.get("setting", [])),
                category=str(tags.get("category") or "other"),
                folder="youtube",
                query=str(tags.get("query") or ""),
                mood=list(tags.get("mood", [])),
                energy=str(tags.get("energy") or ""),
                lighting=str(tags.get("lighting") or ""),
                shot_type=str(tags.get("shot_type") or ""),
                camera_motion=str(tags.get("camera_motion") or ""),
                depth_of_field=str(tags.get("depth_of_field") or ""),
                color_palette=list(tags.get("color_palette", [])),
            )
            total, _vibe, _cinema, _cont = _local_score(
                profile, stub, intelligent=True,
                reference_house=reference_house, continuity_penalty=0.0,
            )
            if total > 0:
                scored.append((preview_path, total, 0.0, 0.0))
                continue
        # Graceful fallback: legacy scoring when new fields are missing or
        # the new scorer returned 0. Divide by 6 to land in 0..1 space.
        legacy = _profile_similarity(profile, tags)
        if legacy > 0:
            scored.append((preview_path, legacy / 6.0, 0.0, 0.0))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [path for path, _score, _vibe, _cont in scored[:count]]


def search_youtube(
    profile: SpanProfile,
    cache_dir: Path,
    settings: Settings,
    *,
    reference_house: dict | None = None,
    intelligent: bool = True,
) -> Path | None:
    """Data-API-first search (key rotation on 403/429) with an automatic
    yt-dlp ytsearch fallback. Thin shim over `search_youtube_candidates` that
    returns the single best preview (or None) — kept so the variation /
    single-clip call sites don't churn.

    New keyword-only kwargs are threaded through to
    ``search_youtube_candidates`` so the YouTube rung can also benefit from
    the intelligent selector (cinema + vibe matching on the preview's
    single-frame tag) without breaking old call sites that omit them."""
    candidates = search_youtube_candidates(
        profile, cache_dir, settings, count=1,
        reference_house=reference_house, intelligent=intelligent,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Stage 6: last resort — crop the reference's own cutaway
# ---------------------------------------------------------------------------


def _detect_content_band(reference_path: Path, at_seconds: float, settings: Settings) -> tuple[float, float] | None:
    """(top, height) of the non-black content band, as fractions of frame
    height. None when the reference is full-bleed (no letterbox bars)."""
    try:
        import cv2
    except ImportError:
        return None
    probe_dir = settings.data_dir / "cache" / "band_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_key = hashlib.md5(f"{reference_path.resolve()}|{at_seconds:.3f}".encode()).hexdigest()[:16]
    probe_frame = probe_dir / f"{probe_key}.jpg"
    if not probe_frame.exists() and not _extract_single_frame(reference_path, at_seconds, probe_frame, settings):
        return None
    frame = cv2.imread(str(probe_frame))
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    height = gray.shape[0]
    best_start, best_length = 0, 0
    run_start: int | None = None
    for index, value in enumerate([*row_means, 0.0]):
        if value > 20 and run_start is None:
            run_start = index
        elif value <= 20 and run_start is not None:
            if index - run_start > best_length:
                best_start, best_length = run_start, index - run_start
            run_start = None
    band_height = best_length / height
    if band_height < 0.2:
        return None
    return best_start / height, band_height


def crop_reference_cutaway(reference_path: Path, span: tuple[float, float], output_path: Path, settings: Settings) -> Path | None:
    """Caption-dodging crop of the reference's own cutaway footage. Always
    succeeds (falls back to an uncropped re-encode) so a span is never left
    empty — this is the guaranteed last rung of the sourcing ladder."""
    start, end = span
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 500:
        return output_path

    band = _detect_content_band(reference_path, (start + end) / 2, settings)
    if band is not None:
        band_top, band_height = band
        window_top = band_top + band_height * 0.14
        window_height = band_height * 0.40
        crop = f"crop=iw:ih*{window_height:.4f}:0:ih*{window_top:.4f}"
    else:
        # No detectable letterbox band (full-bleed reference) — dodge the
        # usual title/caption zones by keeping the middle 70% vertically.
        crop = "crop=iw:ih*0.70:0:ih*0.15"

    def _run(vf: str) -> bool:
        try:
            result = subprocess.run(
                [
                    settings.ffmpeg_path, "-y",
                    "-ss", f"{start:.3f}",
                    "-i", str(reference_path),
                    "-t", f"{max(0.05, end - start):.3f}",
                    "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    str(output_path),
                ],
                capture_output=True, text=True, timeout=90,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 500

    if _run(crop):
        return output_path
    # Absolute last-ditch fallback: re-encode uncropped so this rung can never
    # fail. Captions may be visible, but a span is never left empty.
    return output_path if _run("null") else None


# ---------------------------------------------------------------------------
# Stage 7: placement
# ---------------------------------------------------------------------------


def _align_span_to_clip(ref_start: float, ref_end: float, ref_duration: float, clip_duration: float) -> tuple[float, float]:
    """Map a reference B-roll span onto OUR clip's timeline.

    Default rule: identical absolute timestamps. Only when the reference
    span's end would land past our clip's duration do we scale the span
    proportionally so the ratio of (span / total reference) is preserved.
    """
    if clip_duration <= 0:
        return (ref_start, max(ref_start, ref_end))
    if ref_end <= clip_duration:
        return (ref_start, ref_end)
    span = max(0.0, ref_end - ref_start)
    if ref_duration <= 0 or span <= 0:
        return (ref_start, clip_duration)
    span_frac = min(1.0, span / ref_duration)
    scaled_end = clip_duration
    scaled_start = max(0.0, scaled_end - span_frac * clip_duration)
    if scaled_end - scaled_start < MIN_OUTPUT_BROLL_SPAN:
        scaled_start = max(0.0, scaled_end - MIN_OUTPUT_BROLL_SPAN)
    return (scaled_start, scaled_end)


def _extract_segment(source: Path, duration: float, output_path: Path, settings: Settings) -> Path | None:
    """Simple ffmpeg trim of `source` to `duration` seconds (centered), or a
    looped extraction when the source itself is shorter than needed."""
    if output_path.exists() and output_path.stat().st_size > 500:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_duration = probe_duration(source, settings)
    except Exception:
        source_duration = 0.0

    if source_duration >= duration > 0:
        start = max(0.0, (source_duration - duration) / 2)
        command = [
            settings.ffmpeg_path, "-y",
            "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{duration:.3f}", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(output_path),
        ]
    else:
        command = [
            settings.ffmpeg_path, "-y",
            "-stream_loop", "-1", "-i", str(source),
            "-t", f"{max(duration, 0.1):.3f}", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(output_path),
        ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 500:
        return None
    return output_path


def _span_profile_for(analysis: ReferenceAnalysis, index: int) -> SpanProfile:
    start, end, query = analysis.broll_spans[index]
    tags = analysis.broll_span_tags[index] if index < len(analysis.broll_span_tags) else {}
    if not isinstance(tags, dict):
        tags = {}
    return SpanProfile(
        start=start, end=end,
        subjects=list(tags.get("subjects", [])),
        setting=list(tags.get("setting", [])),
        action=list(tags.get("action", [])),
        category=tags.get("category") or "other",
        query=query or "",
        # Extended vibe tags feed the intelligent selector. Tolerantly
        # copied — _vibe_score_for will degrade to the legacy score when
        # any are missing.
        mood=list(tags.get("mood", [])),
        energy=str(tags.get("energy") or ""),
        lighting=str(tags.get("lighting") or ""),
        shot_type=str(tags.get("shot_type") or ""),
        camera_motion=str(tags.get("camera_motion") or ""),
        depth_of_field=str(tags.get("depth_of_field") or ""),
        color_palette=list(tags.get("color_palette", [])),
    )


def _gather_span_pool(
    profile: SpanProfile,
    reference_path: Path,
    cache_dir: Path,
    library_index: list[LibraryClip],
    used_clips: set[Path],
    settings: Settings,
    pool_size: int,
    used_clips_lock: "threading.Lock | None" = None,
    intelligent: bool = False,
    reference_house: dict | None = None,
    job_id: str | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> list[tuple[Path, str, str, float, float, float]]:
    """Per-span candidate pool for variation rotation: Local -> YouTube ->
    reference-crop. Always returns at least one entry (guaranteed by the
    reference-crop rung), padded to `pool_size` by repeating the last entry.

    `used_clips_lock` is required when multiple `_gather_span_pool` calls run
    concurrently (the BROLL_RECOVERY per-span parallelization). Local library
    matching marks `used_clips` BEFORE releasing the lock so a concurrent
    span's `_rank_local` call sees the updated set; otherwise two spans could
    race-pick the same clip. YouTube / crop work runs after the lock is
    released so they overlap across spans.

    `intelligent=True` enables the vibe/cinematography bonus on top of the
    keyword-only local score. The bonus is a no-op when no local clips
    carry extended vibe tags (i.e. when the library wasn't indexed by
    `broll_intelligence/library_indexer`) — the score reverts to the
    legacy keyword-only behaviour.

    Returns ``(path, provider, reason, vibe_score, cinema_score,
    continuity_penalty)`` per row (SPEC §11) so callers can record the
    vibe + cinema + continuity components on the diagnostic row.
    """
    import threading  # local import keeps this module's import graph tidy
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool: list[tuple[Path, str, str, float, float, float]] = []
    span_duration = profile.end - profile.start
    # Very short spans are not worth a YouTube search: a 0.4s cutaway pays a
    # 20s preview download + frame extract + vision-score round-trip just to
    # produce something the renderer will only show for 400ms. Reference crop
    # is instant and visually fine for sub-second cutaways.
    skip_youtube = span_duration < MIN_YOUTUBE_SPAN_SECONDS
    # The confidence of the local match (0.0 until match_local finds one).
    # Used by the strict short-circuit below to skip the slow YouTube rung
    # even when `pool_size > 1` (variations mode) — a confident local match
    # is good enough; don't waste 30-60s on a YouTube preview download.
    picked_score = 0.0

    if used_clips_lock is None:
        used_clips_lock = threading.Lock()

    with used_clips_lock:
        ranked_local = _rank_local(
            profile, library_index, used_clips, intelligent=intelligent,
            reference_house=reference_house, job_id=job_id, ledger=ledger,
        )
        if ranked_local:
            picked_result = match_local(
                profile, library_index, used_clips, settings,
                intelligent=intelligent, reference_house=reference_house,
                job_id=job_id, ledger=ledger,
            )
            if picked_result is not None:
                picked, picked_vibe, picked_cont, picked_score = picked_result
                # Look up the cinema score for the picked clip from the
                # ranking output (we already computed it once).
                picked_cinema = 0.0
                for clip, _score, _vibe, cinema, _cont in ranked_local:
                    if clip is picked:
                        picked_cinema = cinema
                        break
                pool.append((
                    picked.path, "local",
                    f"library match: {picked.path.name}",
                    picked_vibe, picked_cinema, picked_cont,
                ))
                used_clips.add(picked.path.resolve())
                # Record the pick in the continuity ledger so the next
                # span's candidates pay the diversity tax (SPEC §9).
                if ledger is not None:
                    try:
                        ledger.note_picked(picked)
                    except Exception:
                        logger.debug("Continuity ledger note_picked failed", exc_info=True)
                for clip, score, vibe, cinema, cont in ranked_local:
                    if len(pool) >= pool_size or score <= 0:
                        break
                    if clip.path.resolve() == picked.path.resolve():
                        continue
                    # Reserve alternate inside the lock too — otherwise a
                    # concurrent span could pick the same alternate.
                    resolved = clip.path.resolve()
                    used_clips.add(resolved)
                    pool.append((
                        clip.path, "local",
                        f"library alternate: {clip.path.name}",
                        vibe, cinema, cont,
                    ))

    if not skip_youtube and len(pool) < pool_size:
        # Strict short-circuit: when the local match's score is well above
        # the threshold we already trusted for the LLM-pick gate, treat
        # the local rung as sufficient and skip YouTube entirely — even
        # in variations mode (pool_size > 1). The `1.5x` cushion means
        # we're not just barely past the threshold; we're confidently past
        # it, so the extra candidates aren't worth a 30-60s preview
        # download per span.
        confident_local_threshold = settings.broll_local_match_threshold * 1.5
        if picked_score >= confident_local_threshold and pool:
            logger.debug(
                "Skipping YouTube rung: confident local match "
                "(score=%.3f >= %.3f, threshold=%.3f)",
                picked_score, confident_local_threshold,
                settings.broll_local_match_threshold,
            )
        else:
            # YouTube work is the slow network-heavy rung; it runs OUTSIDE the
            # local-library lock so multiple spans can download in parallel.
            # Now re-ranks with the same vibe/cinema logic as the local rung
            # (SPEC §10) — a downloaded preview also gets the extended
            # prompt + _local_score(intelligent=True) treatment.
            yt_path = search_youtube(
                profile, cache_dir / "youtube", settings,
                reference_house=reference_house, intelligent=intelligent,
            )
            if yt_path is not None:
                # YouTube candidates don't carry the cinema-aware feature
                # vector (preview only — single frame, no per-frame tag
                # history), so cinema and continuity stay 0.0.
                pool.append((yt_path, "youtube", "matched YouTube clip", 0.0, 0.0, 0.0))

    if not pool:
        crop_path = cache_dir / "reference_crop.mp4"
        cropped = crop_reference_cutaway(reference_path, (profile.start, profile.end), crop_path, settings)
        if cropped is not None:
            pool.append((
                cropped, "reference_crop",
                "no local/YouTube match; cropped reference cutaway",
                0.0, 0.0, 0.0,
            ))

    while pool and len(pool) < pool_size:
        pool.append(pool[-1])
    return pool


# B-roll spans shorter than this skip the YouTube search rung and go straight
# to the local library / reference-crop rungs. The cost of a 20s preview
# download + vision-score round-trip dwarfs the on-screen gain when the
# cutaway is sub-second. 1.0s keeps the existing 1s-span behavior intact
# while skipping the 0.4-0.7s fragments that show up in noisy references.
MIN_YOUTUBE_SPAN_SECONDS = 1.0


def fetch_broll_cut_variations(
    analysis: ReferenceAnalysis,
    reference_path: Path,
    clip_duration: float,
    cache_dir: Path,
    settings: Settings,
    variations: int = 1,
    diagnostics: list[BrollRecoveryDiagnostic] | None = None,
    *,
    intelligent: bool = False,
    job_id: str | None = None,
    reference_house: dict | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> list[list[BrollCut]]:
    """Produce `variations` B-roll cut lists sharing timeline placement but
    drawing from different source clips per span where alternatives exist.

    Timing is absolute, not proportional: a reference cut at 0.1s-2.0s becomes
    an output cut at 0.1s-2.0s, scaled only when the output is shorter.

    Per-span sourcing runs in parallel (max 4 workers) so an 8-span job's
    YouTube downloads + vision scorings overlap instead of running serially.
    Library matching is serialized through a lock (cheap section, no contention
    concern). YouTube/crop work runs concurrently outside the lock.

    Backwards-compatible kwargs: when ``reference_house`` is None the house
    style is computed internally from ``analysis`` (SPEC §8); when ``ledger``
    is None the per-job ``_CONTINUITY_LEDGER`` dict is used (legacy keying
    on ``job_id``). Passing both from the caller lets a single ``Pipeline``
    construct them ONCE per job and share them across the broll_pack /
    broll_cut_variations ladder without paying for two house-style walks.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    variations = max(1, variations)
    if not analysis.broll_spans or analysis.duration <= 0:
        return [[] for _ in range(variations)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics is not None:
        diagnostics.clear()
    library_index = build_library_index(settings)

    # House style (SPEC §8): computed ONCE per job from every span's tags
    # so per-span scoring can back-fill empty span vibe fields from a
    # stable per-job aggregate. Falls back to internal computation when
    # the caller (e.g. an older jobs.py) did not pass one in.
    if reference_house is None:
        reference_house = _build_reference_house_style(analysis)

    aligned: list[tuple[int, float, float, SpanProfile]] = []
    for index in range(len(analysis.broll_spans)):
        ref_start, ref_end, _query = analysis.broll_spans[index]
        out_start, out_end = _align_span_to_clip(ref_start, ref_end, analysis.duration, clip_duration)
        if out_end - out_start < MIN_OUTPUT_BROLL_SPAN:
            continue
        aligned.append((index, out_start, out_end, _span_profile_for(analysis, index)))

    used_clips: set[Path] = set()
    used_clips_lock = threading.Lock()
    pools: list[list[tuple[Path, str, str, float, float, float]]] = []
    # Cap workers at 4: more just hammers YouTube's per-IP rate limit and the
    # CPU ffmpeg pool, without meaningfully shrinking the longest span's
    # wall-clock cost.
    max_workers = min(4, max(1, len(aligned)))
    if max_workers == 1:
        for index, out_start, out_end, profile in aligned:
            pools.append(_gather_span_pool(
                profile, reference_path, cache_dir / f"span_{index}",
                library_index, used_clips, settings, pool_size=variations,
                used_clips_lock=used_clips_lock,
                intelligent=intelligent,
                reference_house=reference_house,
                job_id=job_id, ledger=ledger,
            ))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _gather_span_pool,
                    profile, reference_path, cache_dir / f"span_{index}",
                    library_index, used_clips, settings, pool_size=variations,
                    used_clips_lock=used_clips_lock,
                    intelligent=intelligent,
                    reference_house=reference_house,
                    job_id=job_id, ledger=ledger,
                ): index
                for index, _out_start, _out_end, profile in aligned
            }
            results_by_index: dict[int, list[tuple[Path, str, str, float, float, float]]] = {}
            for future, index in futures.items():
                try:
                    results_by_index[index] = future.result()
                except Exception:
                    logger.exception("B-roll span %d pool gather failed", index)
                    results_by_index[index] = []
            # Preserve original alignment order so per-span diagnostics line up.
            pools = [results_by_index[index] for index, _, _, _ in aligned]

    cut_lists: list[list[BrollCut]] = []
    for v in range(variations):
        # Continuity ledger (SPEC §9): RESET per variation so variation 0's
        # picks don't influence variation 1. Two paths:
        #   1. Caller (jobs.py) supplied an explicit `ledger` — reuse it,
        #      reset its history in-place so the same ContinuityLedger
        #      instance drives every variation but starts each one fresh.
        #   2. Caller omitted `ledger` — fall back to the module-level
        #      `_CONTINUITY_LEDGER` keyed on `job_id` (or an ephemeral
        #      key when `job_id` is None) so old call sites keep working.
        if ledger is not None:
            variation_ledger = ledger
        else:
            ledger_key = job_id if job_id else f"_anon_{id(analysis)}"
            variation_ledger = _CONTINUITY_LEDGER.setdefault(ledger_key, _ContinuityLedger(max_history=2))
        # Reset in-place so cross-variation state doesn't leak.
        if hasattr(variation_ledger, "_history"):
            variation_ledger._history.clear()  # type: ignore[attr-defined]
        cuts: list[BrollCut] = []
        for (index, out_start, out_end, profile), pool in zip(aligned, pools):
            if not pool:
                continue
            path, provider, reason, vibe_score, cinema_score, cont = pool[min(v, len(pool) - 1)]
            segment = _extract_segment(path, out_end - out_start, cache_dir / f"span_{index}" / f"cut_v{v}.mp4", settings)
            if segment is None:
                continue
            cuts.append(BrollCut(start=out_start, end=out_end, clip_path=segment, query=profile.query))
            if diagnostics is not None and v == 0:
                diagnostics.append(BrollRecoveryDiagnostic(
                    start=out_start, end=out_end, query=profile.query,
                    provider=provider, source=str(path), match_type=provider,
                    selected=True,
                    reason=(
                        f"{reason}; cinema={cinema_score:.2f} "
                        f"continuity_penalty={cont:.2f}"
                    ),
                    vibe_score=float(vibe_score),
                    cinema=float(cinema_score),
                    continuity_penalty=float(cont),
                    intelligent_active=bool(intelligent),
                ))
        cut_lists.append(cuts)
    return cut_lists


def fetch_broll_cuts(
    analysis: ReferenceAnalysis,
    reference_path: Path,
    clip_duration: float,
    cache_dir: Path,
    settings: Settings,
    diagnostics: list[BrollRecoveryDiagnostic] | None = None,
    *,
    intelligent: bool = False,
    job_id: str | None = None,
    reference_house: dict | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> list[BrollCut]:
    return fetch_broll_cut_variations(
        analysis, reference_path, clip_duration, cache_dir, settings,
        variations=1, diagnostics=diagnostics,
        intelligent=intelligent,
        job_id=job_id,
        reference_house=reference_house,
        ledger=ledger,
    )[0]


# ---------------------------------------------------------------------------
# B-roll PACK — gather 1-2 trimmed clips per span WITHOUT inserting them
# ---------------------------------------------------------------------------


def _gather_pack_sources_for_span(
    span_index: int,
    out_start: float,
    out_end: float,
    profile: SpanProfile,
    reference_path: Path,
    cache_dir: Path,
    library_index: list[LibraryClip],
    used_clips: set[Path],
    used_clips_lock: "threading.Lock",
    settings: Settings,
    per_span: int,
    *,
    intelligent: bool = False,
    reference_house: dict | None = None,
    job_id: str | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> list[BrollPackItem]:
    """Per-span source gathering for the pack path, extracted so the parent
    loop can run it concurrently across spans. Same ladder (Local -> YouTube
    -> reference-crop) and same dedupe rules as the original inline loop.

    `intelligent=True` enables the vibe/cinematography bonus during the
    local-library rung — see `match_local`. Has no effect on YouTube
    previews (those are reranked by their own profile-similarity score).

    The local-library rung reserves each accepted clip into `used_clips`
    INSIDE the lock so a concurrent span's `_rank_local` sees the updated set
    and can't race-pick the same source. Without this, the parallelization
    would let span N re-pick what span 0 already locked in.
    """
    sources: list[tuple[Path, str]] = []

    # Rung 1 — local library (top ranked), up to per_span entries.
    with used_clips_lock:
        ranked_local = _rank_local(
            profile, library_index, used_clips, intelligent=intelligent,
            reference_house=reference_house, job_id=job_id, ledger=ledger,
        )
        for clip, score, _vibe, _cinema, _cont in ranked_local:
            if score <= 0:
                break
            resolved = clip.path.resolve()
            if any(resolved == s.resolve() for s, _ in sources):
                continue
            used_clips.add(resolved)  # reserve BEFORE releasing the lock
            sources.append((clip.path, "local"))
            if len(sources) >= per_span:
                break

    # Rung 2 — YouTube previews. Skipped for sub-MIN_YOUTUBE_SPAN_SECONDS spans
    # where the preview-download + vision-score cost dwarfs the on-screen gain.
    if len(sources) < per_span and (out_end - out_start) >= MIN_YOUTUBE_SPAN_SECONDS:
        yt_remaining = per_span - len(sources)
        yt_dir = cache_dir / f"span_{span_index}" / "youtube"
        yt_candidates = search_youtube_candidates(
            profile, yt_dir, settings, count=yt_remaining,
            reference_house=reference_house, intelligent=intelligent,
        )
        for yt_path in yt_candidates:
            resolved = yt_path.resolve()
            if any(resolved == s.resolve() for s, _ in sources):
                continue
            sources.append((yt_path, "youtube"))
            if len(sources) >= per_span:
                break

    # Rung 3 — guaranteed-last-rung reference crop. Always fills so a span is
    # never empty; dedupe keeps it from shadowing rank-1 when the ladder
    # already produced something better. Reference crops ARE per-span, so they
    # don't need cross-span dedupe bookkeeping.
    if len(sources) < per_span:
        crop_path = cache_dir / f"span_{span_index}" / "reference_crop.mp4"
        cropped = crop_reference_cutaway(
            reference_path,
            (profile.start, profile.end),
            crop_path,
            settings,
        )
        if cropped is not None:
            resolved = cropped.resolve()
            if not any(resolved == s.resolve() for s, _ in sources):
                sources.append((cropped, "reference_crop"))

    # Emit one BrollPackItem per source, in rank order. Library clips are
    # already reserved in used_clips from rung 1; the reference crop dedupes
    # against same-crop-from-other-spans in a follow-up pass.
    items: list[BrollPackItem] = []
    for rank, (source_path, provider) in enumerate(sources, start=1):
        output_segment = _extract_segment(
            source_path,
            out_end - out_start,
            cache_dir / f"span_{span_index}" / f"pack_opt{rank}.mp4",
            settings,
        )
        if output_segment is None:
            continue
        items.append(BrollPackItem(
            span_index=span_index,
            rank=rank,
            start=out_start,
            end=out_end,
            query=profile.query,
            provider=provider,
            clip_path=output_segment,
        ))
    return items


def gather_broll_pack(
    analysis: ReferenceAnalysis,
    reference_path: Path,
    clip_duration: float,
    cache_dir: Path,
    settings: Settings,
    per_span: int = 2,
    *,
    intelligent: bool = False,
    job_id: str | None = None,
    reference_house: dict | None = None,
    ledger: "_ContinuityLedger | None" = None,
) -> list[BrollPackItem]:
    """Build a downloadable B-roll pack for the reference's detected spans.

    Walk every output-timeline span (same alignment / skip rules as
    `fetch_broll_cut_variations`) and collect up to `per_span` distinct
    candidates in ladder order:

        1. Top local library ranked clips with score > 0 (reuses _rank_local
           + match_local with the shared used_clips set so adjacent spans
           don't double-pick the same clip).
        2. Top YouTube previews via `search_youtube_candidates`.
        3. `crop_reference_cutaway` as the always-succeeds last filler.

    Each accepted candidate is trimmed to the span's output duration with
    `_extract_segment` and saved to `cache_dir / span_{i} / pack_opt{rank}.mp4`.

    Same source video is NEVER emitted twice in one span's pack; if only 1
    distinct candidate exists the span gets exactly 1 entry (degraded, never
    padded by repeating — that would be useless duplicates for the user).
    The main rendered video gets NO B-roll cuts when this path is used; the
    pack IS the deliverable.

    Per-span work runs in parallel (max 4 workers) — same shape as
    `fetch_broll_cut_variations` for the same wall-clock reason.

    Backwards-compatible: ``reference_house`` and ``ledger`` default to
    None and are computed internally when omitted (legacy callers stay
    working). When passed by ``app/jobs.py`` they are reused across the
    pack + variation ladder so a single per-job object is paid for.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    per_span = max(1, per_span)
    if not analysis.broll_spans or analysis.duration <= 0:
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    library_index = build_library_index(settings)

    # House-style back-fill (SPEC §8): one computation per job, passed
    # down to every per-span scoring call so empty span vibe fields can
    # back-fill from a stable per-job aggregate.
    if reference_house is None:
        reference_house = _build_reference_house_style(analysis)

    aligned: list[tuple[int, float, float, SpanProfile]] = []
    for index in range(len(analysis.broll_spans)):
        ref_start, ref_end, _query = analysis.broll_spans[index]
        out_start, out_end = _align_span_to_clip(ref_start, ref_end, analysis.duration, clip_duration)
        if out_end - out_start < MIN_OUTPUT_BROLL_SPAN:
            continue
        aligned.append((index, out_start, out_end, _span_profile_for(analysis, index)))

    used_clips: set[Path] = set()
    used_clips_lock = threading.Lock()

    max_workers = min(4, max(1, len(aligned)))
    if max_workers == 1:
        per_span_items = [
            _gather_pack_sources_for_span(
                span_index, out_start, out_end, profile,
                reference_path, cache_dir, library_index, used_clips,
                used_clips_lock, settings, per_span,
                intelligent=intelligent,
                reference_house=reference_house,
                job_id=job_id, ledger=ledger,
            )
            for span_index, out_start, out_end, profile in aligned
        ]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _gather_pack_sources_for_span,
                    span_index, out_start, out_end, profile,
                    reference_path, cache_dir, library_index, used_clips,
                    used_clips_lock, settings, per_span,
                    intelligent=intelligent,
                    reference_house=reference_house,
                    job_id=job_id, ledger=ledger,
                ): span_index
                for span_index, out_start, out_end, profile in aligned
            }
            per_span_items_by_index: dict[int, list[BrollPackItem]] = {}
            for future, span_index in futures.items():
                try:
                    per_span_items_by_index[span_index] = future.result()
                except Exception:
                    logger.exception("B-roll pack span %d gather failed", span_index)
                    per_span_items_by_index[span_index] = []
            per_span_items = [per_span_items_by_index[index] for index, _, _, _ in aligned]

    items: list[BrollPackItem] = []
    for sublist in per_span_items:
        items.extend(sublist)
    return items


def fetch_learned_broll_cuts(
    source_path: Path,
    broll_spans: list[tuple[float, float, str]],
    clip_duration: float,
    cache_dir: Path,
    settings: Settings,
    *,
    intelligent: bool = False,
    ledger: "_ContinuityLedger | None" = None,
) -> list[BrollCut]:
    """Source B-roll for a raw clip (no reference) at learned placements.
    There is no reference cutaway to fall back on, so a span with no
    local/YouTube match is left empty (A-roll stays).

    No reference analysis is available in learned mode so ``reference_house``
    is unconditionally ``None`` (the spec: "When fetch_learned_broll_cuts
    runs (no-reference mode), there is no house style — fall back to no-house
    and the legacy-rung YouTube search; keep the continuity ledger on.").

    The continuity ledger IS still threaded through so consecutive
    learned-span picks pay the diversity tax — same penalty shape as the
    reference-driven ladder."""
    if not broll_spans or clip_duration <= 0:
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)
    library_index = build_library_index(settings)
    used_clips: set[Path] = set()
    cuts: list[BrollCut] = []
    for index, (start, end, query) in enumerate(broll_spans):
        out_end = min(end, clip_duration)
        if out_end - start < MIN_OUTPUT_BROLL_SPAN:
            continue
        profile = SpanProfile(start=start, end=out_end, category="other", query=query or "")
        span_cache = cache_dir / f"span_{index}"
        span_cache.mkdir(parents=True, exist_ok=True)

        # Learned mode has no reference analysis so house-style is None;
        # continuity ledger (when supplied) still participates so consecutive
        # learned spans pay the diversity tax.
        picked_result = match_local(
            profile, library_index, used_clips, settings,
            intelligent=intelligent, reference_house=None,
            job_id=None, ledger=ledger,
        )
        source: Path | None = None
        if picked_result is not None:
            picked, _vibe, _cont, _score = picked_result
            source = picked.path
            used_clips.add(picked.path.resolve())
            # Record the pick so the next span's candidates pay the tax
            # when a ledger was supplied.
            if ledger is not None:
                try:
                    ledger.note_picked(picked)
                except Exception:
                    logger.debug("Continuity ledger note_picked failed", exc_info=True)
        else:
            # Legacy-rung YouTube search (the no-reference fallback):
            # same shape as before — no intelligent kwargs because the
            # caller (jobs.py) already opted into legacy for learned mode.
            source = search_youtube(
                profile, span_cache / "youtube", settings,
                reference_house=None, intelligent=False,
            )
        if source is None:
            continue
        segment = _extract_segment(source, out_end - start, span_cache / "cut.mp4", settings)
        if segment is None:
            continue
        cuts.append(BrollCut(start=start, end=out_end, clip_path=segment, query=profile.query))
    return cuts
