"""Incremental B-roll library index.

Walks the configured library directory once, extracts a FeatureVector per
video file, and persists the result to `data/cache/broll_intelligence_index.json`.
Re-runs are cheap: a clip is only re-tagged when its mtime OR size changes
from the cached entry.

Cache schema (must match CONTRACT.md):

    {
      "version": 1,
      "clips": {
        "<absolute_clip_path>": {
          "mtime": <float seconds>,
          "size": <int bytes>,
          "features": <FeatureVector-as-dict>
        }
      }
    }

Failure modes handled gracefully:
  * Corrupt / partial cache JSON -> treated as empty cache (never raised).
  * Unreadable / zero-duration clip -> skipped (logged at debug, not added
    to clips dict).
  * Permission errors / OSError on stat -> skipped.

Atomic persistence: write to <index_path>.tmp then rename. A crash mid-write
leaves the previous good copy in place.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import Settings
from .feature_vector import (
    FeatureVector,
    feature_vector_from_dict,
    feature_vector_to_dict,
)
from .vibe_extractor import extract_from_video

logger = logging.getLogger(__name__)

INDEX_VERSION = 1
VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
)


@dataclass
class IndexedClip:
    """In-memory representation of one cached row."""

    path: Path
    mtime: float
    size: int
    features: FeatureVector


@dataclass
class IndexReport:
    """What happened during a build run. Returned by `build_library_index`."""

    total: int = 0
    indexed: int = 0
    re_extracted: int = 0
    skipped: int = 0
    failed: int = 0
    clips: list[IndexedClip] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _load_cache(index_path: Path) -> dict[str, dict]:
    """Read the cache JSON. Tolerant of missing / corrupt files. Only entries
    with the right schema version are accepted (forward-compat for v2+)."""
    if not index_path.exists():
        return {}
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Index cache unreadable; treating as empty (%s)", exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("Index cache top-level not a dict; treating as empty")
        return {}
    if raw.get("version") != INDEX_VERSION:
        logger.info(
            "Index cache version %s != %s; rebuilding from scratch",
            raw.get("version"),
            INDEX_VERSION,
        )
        return {}
    clips = raw.get("clips")
    if not isinstance(clips, dict):
        return {}
    return clips


def _atomic_write(index_path: Path, payload: dict) -> None:
    """Write payload to <path>.tmp then rename. Survives mid-write crashes:
    the previous good file (if any) stays in place until the rename."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".broll_intelligence_index.", suffix=".tmp", dir=str(index_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_name, index_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _persist(clips_by_path: dict[str, dict]) -> None:
    """Placeholder hook for the persister so the indexer is testable in
    isolation. The actual write happens via _atomic_write inside
    `build_library_index` (so callers can swap it in tests)."""
    raise NotImplementedError("callers should use build_library_index directly")


# ---------------------------------------------------------------------------
# Library walk
# ---------------------------------------------------------------------------


def iter_library_files(library_dir: Path) -> Iterable[Path]:
    """Yield every file under `library_dir` whose suffix is in VIDEO_EXTS,
    sorted for deterministic runs."""
    if not library_dir.exists():
        return
    for p in sorted(library_dir.rglob("*")):
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        if p.suffix.lower() in VIDEO_EXTS:
            yield p


def _entry_matches(entry: dict, mtime: float, size: int) -> bool:
    """True iff the cached entry is fresh for this file."""
    try:
        cached_mtime = float(entry.get("mtime", -1.0))
        cached_size = int(entry.get("size", -1))
    except (TypeError, ValueError):
        return False
    return cached_mtime == mtime and cached_size == size


def _deserialize_features(entry: dict) -> FeatureVector | None:
    feats = entry.get("features")
    if not isinstance(feats, dict):
        return None
    try:
        return feature_vector_from_dict(feats)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_library_index(
    settings: Settings,
    *,
    force: bool = False,
    _extractor=None,
) -> IndexReport:
    """Walk the library, build / refresh the index cache, return a report.

    `force=True` re-extracts every clip regardless of mtime/size (used by
    tests + the "I'm updating the schema" admin path).

    `_extractor` is an optional monkeypatch hook for tests: when supplied, it
    replaces `vibe_extractor.extract_from_video` and is expected to accept
    `(path, settings)` and return a FeatureVector. Tests use this to assert
    the indexer only calls the extractor for stale / new clips.
    """
    extractor = _extractor or extract_from_video
    library_dir = settings.library_dir
    index_path = settings.index_path

    cached = {} if force else _load_cache(index_path)

    report = IndexReport()
    fresh_cache: dict[str, dict] = {}
    files = list(iter_library_files(library_dir))
    report.total = len(files)

    for path in files:
        try:
            stat = path.stat()
        except OSError as exc:
            logger.debug("Skip %s: stat failed (%s)", path, exc)
            report.skipped += 1
            continue

        key = str(path.resolve())
        mtime = float(stat.st_mtime)
        size = int(stat.st_size)

        if not force:
            entry = cached.get(key)
            if entry and _entry_matches(entry, mtime, size):
                fv = _deserialize_features(entry)
                if fv is not None:
                    report.clips.append(IndexedClip(path=path, mtime=mtime, size=size, features=fv))
                    fresh_cache[key] = entry
                    report.indexed += 1
                    continue

        # Stale or missing — call the extractor.
        try:
            fv = extractor(path, settings)
        except Exception as exc:
            logger.warning("Extractor failed for %s (%s); skipping", path, type(exc).__name__)
            report.failed += 1
            continue

        if fv is None or fv.media_path == "" or fv.confidence <= 0.0:
            # Extractor gave us nothing usable (None, missing media_path, or
            # confidence==0.0 sentinel from empty_feature_vector) — skip so
            # we don't pollute the cache with a row that has no signal.
            # The next run with a healthy extractor will pick it up.
            report.skipped += 1
            continue

        # Make sure provenance reflects the actual file we just indexed.
        fv.media_path = key
        fv.source = fv.source or "library"

        entry = {
            "mtime": mtime,
            "size": size,
            "features": feature_vector_to_dict(fv),
        }
        fresh_cache[key] = entry
        report.clips.append(IndexedClip(path=path, mtime=mtime, size=size, features=fv))
        report.indexed += 1
        report.re_extracted += 1

    # Persist atomically. We persist even if nothing changed because tests
    # expect the file to exist after a build.
    payload = {"version": INDEX_VERSION, "clips": fresh_cache}
    try:
        _atomic_write(index_path, payload)
    except OSError as exc:
        logger.error("Failed to persist broll intelligence index: %s", exc)

    return report


def load_index(settings: Settings) -> dict[str, dict]:
    """Read the on-disk index into a {abs_path: cached_entry} dict. Used by
    downstream tasks (matcher, YouTube comparator) that want raw access
    without re-running the build. Returns {} on missing / corrupt cache."""
    return _load_cache(settings.index_path)


def load_index_as_clips(settings: Settings) -> list[IndexedClip]:
    """Same as load_index but as a list of typed IndexedClip. Corrupt feature
    entries are silently dropped (logged at debug)."""
    raw = _load_cache(settings.index_path)
    out: list[IndexedClip] = []
    for key, entry in raw.items():
        try:
            mtime = float(entry.get("mtime", 0.0))
            size = int(entry.get("size", 0))
        except (TypeError, ValueError):
            continue
        fv = _deserialize_features(entry)
        if fv is None:
            continue
        out.append(IndexedClip(path=Path(key), mtime=mtime, size=size, features=fv))
    return out


def invalidate_clip(path: Path, settings: Settings) -> bool:
    """Drop one clip from the on-disk cache so the next build re-extracts
    it. Returns True iff the entry was actually present. Useful for tests +
    the "tag was wrong, re-do it" admin path."""
    cache = _load_cache(settings.index_path)
    key = str(path.resolve())
    if key not in cache:
        return False
    cache.pop(key, None)
    try:
        _atomic_write(settings.index_path, {"version": INDEX_VERSION, "clips": cache})
    except OSError as exc:
        logger.error("invalidate_clip persist failed: %s", exc)
        return False
    return True


__all__ = [
    "build_library_index",
    "load_index",
    "load_index_as_clips",
    "invalidate_clip",
    "iter_library_files",
    "IndexedClip",
    "IndexReport",
    "INDEX_VERSION",
    "VIDEO_EXTS",
]