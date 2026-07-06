"""Pipeline orchestrator for the broll_intelligence system.

Ladder: LIBRARY (matcher) -> YOUTUBE (search) -> REFERENCE_CROP (placeholder).
Each rung is gated by a minimum composite score threshold; if no rung
produces a pick above its threshold, we fall through to the next rung.

This module is the single entry-point for the rest of campeditor if /
when the new system is wired into the existing pipeline. Until then,
``demo.py`` and ``compare.py`` are the consumers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .feature_vector import FeatureVector, feature_vector_from_dict, feature_vector_to_dict
from .vibe_extractor import extract_from_video
from .library_indexer import IndexedClip, build_library_index, load_index_as_clips
from .matcher import rank_candidates
from .search import search_broll

logger = logging.getLogger(__name__)

# Minimum composite scores the pipeline requires to accept a pick.
# Tuned to be similar to the spec; expose via Settings if you want knobs.
LIBRARY_ACCEPT_THRESHOLD = 0.55
YOUTUBE_ACCEPT_THRESHOLD = 0.50

REFERENCE_FEATURE_CACHE_DIRNAME = "broll_intelligence_ref"
YOUTUBE_PREVIEW_CACHE_DIRNAME = "broll_intelligence_yt"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BrollItem:
    source: str                  # "library" | "youtube" | "reference_crop"
    path: Path | None
    url: str | None
    score: float
    features: FeatureVector | None = None
    notes: str = ""


@dataclass
class BrollPack:
    reference: FeatureVector
    items: list[BrollItem] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reference": feature_vector_to_dict(self.reference),
            "items": [
                {
                    "source": it.source,
                    "path": str(it.path) if it.path else None,
                    "url": it.url,
                    "score": it.score,
                    "notes": it.notes,
                    "features": feature_vector_to_dict(it.features) if it.features else None,
                }
                for it in self.items
            ],
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Reference analysis (with caching)
# ---------------------------------------------------------------------------


def _hash_path(path: Path) -> str:
    try:
        stat = path.stat()
        return hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:24]
    except OSError:
        return hashlib.sha256(str(path).encode()).hexdigest()[:24]


def _ref_cache_path(settings: Settings, video_path: Path) -> Path:
    return settings.data_dir / "cache" / REFERENCE_FEATURE_CACHE_DIRNAME / f"{_hash_path(video_path)}.json"


def analyze_reference(
    video_path: Path,
    settings: Settings | None = None,
) -> FeatureVector:
    """Extract a FeatureVector from a reference video.

    Caches the result to ``data/cache/broll_intelligence_ref/`` so a
    second call with the same video (same size + mtime) is free.
    """
    settings = settings or get_settings()
    cache_path = _ref_cache_path(settings, video_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            return feature_vector_from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
        except Exception:
            # Corrupt cache — fall through and re-extract.
            logger.debug("reference cache unreadable; re-extracting (%s)", cache_path)

    fv = extract_from_video(video_path, settings)
    try:
        cache_path.write_text(json.dumps(feature_vector_to_dict(fv), ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.debug("reference cache write failed (%s)", cache_path)
    return fv


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


def _try_library(
    reference: FeatureVector,
    settings: Settings,
    top_k: int,
    used_paths: set[Path],
) -> list[BrollItem]:
    """Run the library rung of the ladder."""
    if not settings.library_dir.exists():
        return []
    # Build/refresh the index (incremental).
    try:
        build_library_index(settings)
    except Exception as exc:
        logger.warning("library index build failed: %s", type(exc).__name__)
    clips: list[IndexedClip] = load_index_as_clips(settings)
    if not clips:
        return []
    ranked = rank_candidates(
        reference,
        [(c.path, c.features) for c in clips],
        top_k=top_k,
        used_clips=used_paths,
    )
    return [
        BrollItem(
            source="library",
            path=path,
            url=None,
            score=score,
            features=features,
            notes=f"library index: {features.category or 'unknown'} / {features.mood[:2]}",
        )
        for path, features, score in ranked
        if score >= LIBRARY_ACCEPT_THRESHOLD
    ]


def _try_youtube(
    reference: FeatureVector,
    settings: Settings,
    top_k: int,
    cache_dir: Path,
) -> list[BrollItem]:
    """Run the YouTube rung of the ladder."""
    # No LLM available — skip YouTube rung. Keeps offline tests fast and
    # avoids burning real API quota during a compare run that just wants
    # to see library picks.
    if not (settings.groq_api_key or settings.nvidia_keys() or settings.gemini_api_key):
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        candidates = search_broll(
            reference,
            top_k=top_k,
            cache_dir=cache_dir,
            settings=settings,
        )
    except Exception as exc:
        logger.warning("YouTube rung failed (%s): %s", type(exc).__name__, exc)
        return []
    return [
        BrollItem(
            source="youtube",
            path=c.preview_path,
            url=c.url,
            score=c.score,
            features=c.features,
            notes=f"query: {c.source_query[:60]!r}",
        )
        for c in candidates
        if c.score >= YOUTUBE_ACCEPT_THRESHOLD
    ]


def select_broll(
    ref_video: Path,
    top_k: int = 5,
    cache_dir: Path | None = None,
    settings: Settings | None = None,
    *,
    enable_youtube: bool = True,
) -> BrollPack:
    """Run the full ladder for ``ref_video`` and return a BrollPack.

    Ladder order: LIBRARY (threshold 0.55) -> YOUTUBE (threshold 0.50) ->
    REFERENCE_CROP (placeholder). If a rung produces at least one pick
    above its threshold, those picks are returned without invoking the
    next rung. If a rung returns nothing, the next rung is tried.

    If every rung returns nothing, the pack's ``items`` is a single
    REFERENCE_CROP placeholder so the caller always gets a usable
    structure.

    Pass ``enable_youtube=False`` to skip the YouTube rung entirely
    (useful for fast tests / offline comparison runs that want to focus
    on the library pick).
    """
    settings = settings or get_settings()
    started = time.monotonic()
    reference = analyze_reference(ref_video, settings)
    cache_dir = cache_dir or (settings.data_dir / "cache" / YOUTUBE_PREVIEW_CACHE_DIRNAME)
    used_paths: set[Path] = set()

    diagnostics: dict[str, Any] = {
        "library_threshold": LIBRARY_ACCEPT_THRESHOLD,
        "youtube_threshold": YOUTUBE_ACCEPT_THRESHOLD,
        "reference_path": str(ref_video),
        "library_dir": str(settings.library_dir),
    }

    items: list[BrollItem] = []
    rungs_fired: list[str] = []

    # Rung 1: library
    items = _try_library(reference, settings, top_k, used_paths)
    rungs_fired.append("library")
    diagnostics["library_picks"] = len(items)

    # Rung 2: YouTube
    if not items and enable_youtube:
        items = _try_youtube(reference, settings, top_k, cache_dir)
        rungs_fired.append("youtube")
        diagnostics["youtube_picks"] = len(items)

    # Rung 3: reference crop placeholder
    if not items:
        items = [
            BrollItem(
                source="reference_crop",
                path=None,
                url=None,
                score=0.0,
                features=reference,
                notes="Ladder fell through to reference-crop placeholder. "
                "Implement reference-crop in a follow-up if you need this rung.",
            )
        ]
        rungs_fired.append("reference_crop")
        diagnostics["reference_crop_picks"] = 1

    # Mark used paths (so future calls don't re-pick the same clip).
    for it in items:
        if it.path:
            used_paths.add(it.path)

    diagnostics["rungs_fired"] = rungs_fired
    diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 3)

    return BrollPack(reference=reference, items=items, diagnostics=diagnostics)