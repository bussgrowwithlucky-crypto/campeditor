"""Smart YouTube fallback for broll_intelligence.

`search_broll(reference, top_k, cache_dir, settings)` runs the full
vibe-aware YouTube discovery pipeline:

  1. Build 5 stock-footage search queries from the reference FeatureVector
     via the LLM ladder (subject / mood / lighting / scene / aesthetic).
  2. yt-dlp ytsearch each query; aggregate, dedup, junk-filter.
  3. For each surviving candidate, download a short preview with yt-dlp,
     extract a single representative frame with ffmpeg, run
     vibe_extractor.extract_from_frame() to get a FeatureVector, score it
     against the reference using the SAME composite formula as the
     matcher (multi-signal content + vibe + cinema + motion, weighted).
  4. Return the top_k candidates ordered by score desc.

Design constraints (CONTRACT.md + project scope):
  * No `app.*` imports. The yt-dlp call is re-implemented inline from the
    pattern in `app/broll.py::_yt_dlp_search` (which is never imported).
  * No `selector.*` imports either — the composite formula lives here as a
    standalone function. Re-using the exact same math is the whole point;
    a sibling copy in this module is the cheapest way to guarantee
    single-source-of-truth drift doesn't bite us later.
  * All subprocess calls are wrapped to fail soft (download errors ->
    score=0, preview_path=None).
  * Fully offline-testable: every external side-effect (LLM, yt-dlp,
    ffmpeg, vibe_extractor) is injected through a `_chat_fn`,
    `_ytdlp_search_fn`, `_download_fn`, `_vibe_extractor_fn` keyword so
    tests can monkeypatch them without touching the filesystem or the
    network.
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .feature_vector import FeatureVector
from .vibe_extractor import _extract_frame, probe_duration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must mirror app/broll.py so the two paths agree on
# junk / duration / preview budgets.
# ---------------------------------------------------------------------------

YT_RESULTS_PER_QUERY = 10  # yt-dlp ytsearch depth per query
PREVIEW_SECONDS = 20.0     # how much of each candidate to grab
DOWNLOAD_TIMEOUT = 120.0    # yt-dlp per-candidate wall-clock cap
FRAME_EXTRACT_TIMEOUT = 30.0
MIN_PREVIEW_BYTES = 10_000

# Titles that signal talking-head / commentary uploads rather than clean stock
# footage. Cheap substring reject BEFORE any download or vision call.
YT_JUNK_TITLE_TERMS: tuple[str, ...] = (
    "podcast", "reaction", "react", "explained", "interview", "tutorial",
    "how to", "review", "vlog", "story time", "storytime", "q&a", "commentary",
    "tier list", "ranking", "compilation of", "top 10", "top ten",
)

# Max words per generated query (LLM is asked for <= 8; this is the enforced cap
# regardless of whether the prompt came from the model or the fallback).
QUERY_MAX_WORDS = 8

# Composite weights — MUST match broll_intelligence.selector.selector_config
# defaults so the search score and the matcher score live on the same scale.
# The diversity bucket (0.15 in the selector) is folded into motion
# (0.10 -> 0.25 effective) so the score stays on the same 0..1 scale
# without needing a diversity ledger for a single YouTube candidate.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "content": 0.25,
    "vibe": 0.30,
    "cinema": 0.20,
    "motion": 0.25,
}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class BrollCandidate:
    """One scored YouTube result.

    `preview_path` is the downloaded short mp4 when the download succeeded,
    None otherwise. `features` is the FeatureVector extracted from a
    representative frame of the preview (None when the preview failed).
    `source_query` is the query that surfaced this candidate — useful for
    audit rows ("found via 'mood: mysterious'").
    """

    video_id: str
    url: str
    preview_path: Path | None
    score: float
    source_query: str
    features: FeatureVector | None


# ---------------------------------------------------------------------------
# Query generation prompt — appended verbatim to CONTRACT.md. Kept here
# alongside the parsing so the prompt and its expected shape don't drift.
# ---------------------------------------------------------------------------

QUERY_GENERATION_PROMPT = (
    "You generate stock-footage search queries that capture the VIBE of a B-roll scene. "
    "Given this scene description:\n"
    "  subjects: {subjects}\n"
    "  setting: {setting}\n"
    "  mood: {mood}\n"
    "  lighting: {lighting}\n"
    "  shot_type: {shot_type}\n"
    "  color_palette: {color_palette}\n"
    'Reply with ONLY a JSON object: {{"queries": [q1, q2, q3, q4, q5]}}\n'
    "Each query is <= 8 words. Each captures a DIFFERENT angle:\n"
    "  q1 = subject-focused (main object/action)\n"
    "  q2 = mood-focused (emotional tone)\n"
    "  q3 = lighting-focused (visual lighting)\n"
    "  q4 = scene-focused (location + action)\n"
    "  q5 = aesthetic-focused (overall visual feel)"
)


def _build_query_prompt(reference: FeatureVector) -> str:
    return QUERY_GENERATION_PROMPT.format(
        subjects=", ".join(reference.subjects) or "(unspecified)",
        setting=", ".join(reference.setting) or "(unspecified)",
        mood=", ".join(reference.mood) or "(unspecified)",
        lighting=reference.lighting or "(unspecified)",
        shot_type=reference.shot_type or "(unspecified)",
        color_palette=", ".join(reference.color_palette) or "(unspecified)",
    )


# ---------------------------------------------------------------------------
# Deterministic fallback query set
# ---------------------------------------------------------------------------


def _fallback_queries(reference: FeatureVector) -> list[str]:
    """When the LLM fails (no key, 429 storm, junk output) build a
    deterministic query set from the reference's own fields. We aim for
    5 queries covering the same 5 angles as the LLM prompt."""
    subjects = reference.subjects or ["broll"]
    setting = reference.setting or ["scene"]
    primary_subject = subjects[0]

    q1_parts = [primary_subject]
    if reference.action:
        q1_parts.append(reference.action[0])
    q1 = " ".join(q1_parts[:3])

    mood = (reference.mood[0] if reference.mood else reference.energy or "cinematic")
    q2 = f"{primary_subject} {mood}"

    lighting = reference.lighting or "natural"
    q3 = f"{primary_subject} {lighting}"

    setting_phrase = " ".join(setting[:2])
    q4 = f"{primary_subject} {setting_phrase}".strip()

    palette = reference.color_palette[0] if reference.color_palette else "cinematic"
    q5 = f"{primary_subject} {palette}"

    return [q1, q2, q3, q4, q5]


def _truncate_query(q: str) -> str:
    """Trim to QUERY_MAX_WORDS, collapse whitespace, strip junk chars. Returns
    "" for queries that end up empty."""
    cleaned = re.sub(r"[\"']", "", (q or "").strip())
    words = cleaned.split()
    if not words:
        return ""
    return " ".join(words[:QUERY_MAX_WORDS])


# ---------------------------------------------------------------------------
# yt-dlp search + download (re-implemented inline; NOT from app.*)
# ---------------------------------------------------------------------------


def _ytdlp_search(query: str, per_query: int) -> list[dict[str, Any]]:
    """`yt-dlp --dump-json --flat-playlist --no-download ytsearchN:QUERY`.

    Mirrors `app/broll.py::_yt_dlp_search` but inlines the subprocess call
    so this package stays standalone. Returns a list of dicts with at
    least `id` + `url` + `title`. Returns [] on any failure."""
    import sys

    command = [
        sys.executable, "-m", "yt_dlp", "--dump-json", "--flat-playlist",
        "--no-download", "--ignore-errors", f"ytsearch{per_query}:{query}",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        logger.warning("yt-dlp search timed out for query=%s", query)
        return []
    except OSError as exc:
        logger.warning("yt-dlp search OSError for query=%s (%s)", query, exc)
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            info = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(info, dict):
            continue
        video_id = info.get("id")
        info["url"] = (
            info.get("url") or info.get("webpage_url") or info.get("original_url")
            or (f"https://www.youtube.com/watch?v={video_id}" if video_id else None)
        )
        if video_id and info.get("url"):
            rows.append(info)
    return rows


def _is_junk_title(url: str, title: str) -> bool:
    """Mirrors `app/broll.py::_is_junk_youtube_entry`. YouTube Shorts are
    junk (vertical orientation, not landscape stock); junk substrings
    signal talking-head / commentary uploads."""
    if "/shorts/" in (url or "").lower():
        return True
    title_lower = (title or "").lower()
    return any(term in title_lower for term in YT_JUNK_TITLE_TERMS)


def _download_preview(url: str, output_path: Path, ffmpeg_path: str) -> bool:
    """Short, low-res preview download. Mirrors
    `app/broll.py::_download_youtube_preview` but inline. Returns True iff
    the mp4 landed on disk with a sane size."""
    if output_path.exists() and output_path.stat().st_size > MIN_PREVIEW_BYTES:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import sys

    base_command = [
        sys.executable, "-m", "yt_dlp", "--no-playlist",
        "-f", "worstvideo[height>=360][ext=mp4]/worst[ext=mp4]/worst",
        "--download-sections", f"*0-{max(3.0, PREVIEW_SECONDS):.0f}",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
    ]
    try:
        result = subprocess.run(
            base_command + [url],
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logger.info("yt-dlp preview download timed out for %s", url)
        try:
            output_path.unlink()
        except OSError:
            pass
        return False
    except OSError as exc:
        logger.warning("yt-dlp download OSError for %s (%s)", url, exc)
        return False
    return (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size > MIN_PREVIEW_BYTES
    )


# ---------------------------------------------------------------------------
# Vibe extractor (single frame) — wraps the frame helper from vibe_extractor
# so callers can monkeypatch it without touching the cache layout.
# ---------------------------------------------------------------------------


def extract_from_frame(frame_path: Path, settings: Settings, *, source: str = "youtube") -> FeatureVector:
    """Single-frame FeatureVector pass. Re-uses the public extract_from_video
    machinery in a minimal way: we already have a JPEG on disk, so we just
    need to (a) read it, (b) build the CV features, (c) build the vector.

    Vision tagging on a single still isn't useful here — the model would
    see the same content twice. So we deliberately skip the vision ladder
    and rely on the CV pass alone; `confidence` therefore reflects CV-only
    signal (0.5)."""
    from .feature_vector import feature_vector_from_dict
    from .vibe_extractor import _aggregate_cv, _clip01

    duration = 0.0  # fake; we use single-frame sampling
    cv_features = _aggregate_cv([frame_path])
    payload: dict[str, Any] = {
        "media_path": str(frame_path.resolve()),
        "source": source,
    }
    for k, v in cv_features.items():
        payload.setdefault(k, float(v))
    fv = feature_vector_from_dict(payload)
    fv.confidence = _clip01(0.5)  # CV-only signal
    return fv


# ---------------------------------------------------------------------------
# Composite score (matches broll_intelligence.selector.scoring without the
# diversity / continuity / establishing terms — single candidate, no ledger).
# ---------------------------------------------------------------------------


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = {x.strip().lower() for x in a if x}
    sb = {x.strip().lower() for x in b if x}
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _energy_compat(a: str, b: str) -> float:
    if a == b:
        return 1.0
    order = {"low": 0, "medium": 1, "high": 2}
    ai, bi = order.get(a, 1), order.get(b, 1)
    if abs(ai - bi) == 1:
        return 0.5
    return 0.0


def _lighting_compat(a: str, b: str) -> float:
    """Lighting compatibility per CONTRACT.md §Compatibility matrices.

      * 1.0 — exact match
      * 0.4 — declared compatible pair (low-key<->natural, high-key<->natural,
                neon<->mixed, golden-hour<->natural)
      * 0.0 — otherwise
    """
    if a == b:
        return 1.0
    compatible_pairs = {
        frozenset({"low-key", "natural"}),
        frozenset({"high-key", "natural"}),
        frozenset({"neon", "mixed"}),
        frozenset({"golden-hour", "natural"}),
    }
    if frozenset({a, b}) in compatible_pairs:
        return 0.4
    return 0.0


def _hue_distance(a: list[str], b: list[str]) -> float:
    vocab = ["warm", "cool", "neutral", "earthy", "monochrome", "neon", "pastel"]
    va = set(x.strip().lower() for x in a[:3]) & set(vocab)
    vb = set(x.strip().lower() for x in b[:3]) & set(vocab)
    if not va and not vb:
        return 0.5
    diffs = sum(1 for x in vocab if (x in va) != (x in vb))
    return 1.0 - diffs / len(vocab)


def _cauchy(a: float, b: float, *, gamma: float) -> float:
    if gamma <= 0.0:
        return 1.0
    d = (float(a) - float(b)) / gamma
    return 1.0 / (1.0 + d * d)


def _shot_type_compat(a: str, b: str) -> float:
    if a == b:
        return 1.0
    adjacents = {
        "wide": {"medium"},
        "medium": {"wide", "close-up"},
        "close-up": {"medium", "extreme-close-up", "two-shot"},
        "extreme-close-up": {"close-up"},
        "aerial": {"overhead"},
        "overhead": {"aerial"},
        "two-shot": {"close-up"},
    }
    if b in adjacents.get(a, set()):
        return 0.7
    return 0.3


def _camera_motion_compat(a: str, b: str) -> float:
    """Camera-motion compatibility per CONTRACT.md.

      * 1.0 — exact match
      * 0.4 — one of the two is "static" (a soft preference for a static
               reference paired with deliberate motion, or vice versa)
      * 0.0 — otherwise (the two motions are unrelated and might clash)

    Note: this intentionally does NOT score (static, dolly) higher than
    (static, static). "Dolly and pan complement each other" is a separate
    concern from "the clip's motion matches the reference's motion" — the
    pipeline is matching *vibe*, not constructing an editing sequence.
    """
    if a == b:
        return 1.0
    if a == "static" or b == "static":
        return 0.4
    return 0.0


# Backwards-compat alias for any older internal callers.
_camera_motion_complement = _camera_motion_compat


def _content_signal(taste: FeatureVector, cand: FeatureVector) -> float:
    """Same weights + sparse-span redistribution as the matcher's
    _content_match. Returned in [0, 1]."""
    subj = _jaccard(taste.subjects, cand.subjects)
    sett = _jaccard(taste.setting, cand.setting)
    cat = 1.0 if taste.category and taste.category == cand.category else 0.0
    act = _jaccard(taste.action, cand.action)
    if not taste.subjects:
        return (
            (0.25 + 0.40 * (0.25 / 0.75)) * sett
            + (0.20 + 0.40 * (0.20 / 0.75)) * cat
            + (0.15 + 0.40 * (0.15 / 0.75)) * act
        ) / (0.25 + 0.20 + 0.15 + 0.40)
    return 0.40 * subj + 0.25 * sett + 0.20 * cat + 0.15 * act


def _vibe_signal(taste: FeatureVector, cand: FeatureVector) -> float:
    mood = _jaccard(taste.mood, cand.mood)
    energy = _energy_compat(taste.energy, cand.energy)
    lighting = _lighting_compat(taste.lighting, cand.lighting)
    hue = _hue_distance(taste.color_palette, cand.color_palette)
    warmth = _cauchy(taste.palette_warmth, cand.palette_warmth, gamma=0.20)
    saturation = _cauchy(taste.palette_saturation, cand.palette_saturation, gamma=0.20)
    brightness = _cauchy(taste.palette_brightness, cand.palette_brightness, gamma=0.20)
    return (
        0.35 * mood
        + 0.15 * energy
        + 0.15 * lighting
        + 0.15 * hue
        + 0.10 * warmth
        + 0.05 * saturation
        + 0.05 * brightness
    )


def _cinema_signal(taste: FeatureVector, cand: FeatureVector) -> float:
    shot = _shot_type_compat(taste.shot_type, cand.shot_type)
    motion = _camera_motion_complement(taste.camera_motion, cand.camera_motion)
    dof = 1.0 if taste.depth_of_field == cand.depth_of_field else 0.5
    return 0.45 * shot + 0.35 * motion + 0.20 * dof


def _motion_signal(taste: FeatureVector, cand: FeatureVector) -> float:
    return (
        0.55 * _cauchy(taste.motion_intensity, cand.motion_intensity, gamma=0.25)
        + 0.25 * _cauchy(taste.contrast, cand.contrast, gamma=0.30)
        + 0.20 * _cauchy(taste.edge_density, cand.edge_density, gamma=0.30)
    )


def composite_score(reference: FeatureVector, candidate: FeatureVector) -> float:
    """The composite score, in [0, 1], that the matcher would assign a
    single clip against the reference if there were no diversity / ledger
    considerations.

    Weights match `broll_intelligence.selector.selector_config.SelectorConfig`
    defaults exactly; the only adaptation is that the diversity bucket
    (0.15) is folded into motion (0.10 -> 0.25 effective) so the score
    stays on the same 0..1 scale without needing a ledger.

    Both inputs are normalised via `.normalised()` to neutralise
    case / whitespace drift before the per-signal math runs.
    """
    taste = reference.normalised()
    cand = candidate.normalised()

    content = _content_signal(taste, cand)
    vibe = _vibe_signal(taste, cand)
    cinema = _cinema_signal(taste, cand)
    motion = _motion_signal(taste, cand)

    weights = COMPOSITE_WEIGHTS
    total = (
        weights["content"] * content
        + weights["vibe"] * vibe
        + weights["cinema"] * cinema
        + weights["motion"] * motion
    )
    return max(0.0, min(1.0, total))


# ---------------------------------------------------------------------------
# Aggregation + dedup + scoring pipeline
# ---------------------------------------------------------------------------


def _aggregate_search_results(
    queries: list[str],
    *,
    max_per_query: int,
    max_total: int,
    _search_fn: Callable[[str, int], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Run yt-dlp per query, dedup by video_id, junk-filter, cap at
    max_total. Each surviving entry has at minimum `id`, `url`, `title`."""
    search_fn = _search_fn or _ytdlp_search
    seen_ids: set[str] = set()
    entries: list[dict[str, Any]] = []
    for query in queries:
        try:
            rows = search_fn(query, max_per_query)
        except Exception as exc:
            logger.warning("yt-dlp search raised for query=%s (%s)", query, type(exc).__name__)
            continue
        for info in rows:
            video_id = info.get("id")
            if not video_id or video_id in seen_ids:
                continue
            url = info.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            title = info.get("title") or ""
            if _is_junk_title(url, title):
                continue
            seen_ids.add(video_id)
            entries.append({"id": video_id, "url": url, "title": title, "query": query})
            if len(entries) >= max_total:
                return entries
    return entries


def _score_candidate(
    reference: FeatureVector,
    entry: dict[str, Any],
    cache_dir: Path,
    settings: Settings,
    *,
    _download_fn: Callable[[str, Path, str], bool] | None = None,
    _frame_extractor_fn: Callable[[Path, float, Path, Settings], bool] | None = None,
    _vibe_fn: Callable[[Path, Settings], FeatureVector] | None = None,
) -> BrollCandidate:
    """Download + frame-extract + vibe-score one YouTube entry. On any
    failure: returns score=0.0, preview_path=None, features=None."""
    video_id = entry["id"]
    url = entry["url"]
    source_query = entry.get("query", "")

    download_fn = _download_fn or _download_preview
    extract_fn = _frame_extractor_fn or _extract_frame
    vibe_fn = _vibe_fn or extract_from_frame

    preview_path = cache_dir / f"preview_{video_id}.mp4"
    frame_path = cache_dir / f"preview_{video_id}.jpg"
    ffmpeg_path = settings.resolved_ffmpeg()

    try:
        if not download_fn(url, preview_path, ffmpeg_path):
            return BrollCandidate(
                video_id=video_id, url=url, preview_path=None,
                score=0.0, source_query=source_query, features=None,
            )
        duration = probe_duration(preview_path, settings) or 0.0
        at_seconds = max(0.0, duration / 2.0)
        if not extract_fn(preview_path, at_seconds, frame_path, settings):
            return BrollCandidate(
                video_id=video_id, url=url, preview_path=preview_path,
                score=0.0, source_query=source_query, features=None,
            )
        clip_features = vibe_fn(frame_path, settings)
        score = composite_score(reference, clip_features)
    except Exception as exc:
        logger.warning(
            "search: candidate pipeline failed for %s (%s)",
            video_id, type(exc).__name__,
        )
        return BrollCandidate(
            video_id=video_id, url=url, preview_path=preview_path if preview_path.exists() else None,
            score=0.0, source_query=source_query, features=None,
        )

    return BrollCandidate(
        video_id=video_id, url=url, preview_path=preview_path,
        score=score, source_query=source_query, features=clip_features,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def search_broll(
    reference: FeatureVector,
    top_k: int,
    cache_dir: Path,
    settings: Settings,
    max_candidates_per_query: int = 4,
    max_total_candidates: int = 12,
    *,
    _chat_fn: Callable[[str, Settings], list[str]] | None = None,
    _search_fn: Callable[[str, int], list[dict[str, Any]]] | None = None,
    _download_fn: Callable[[str, Path, str], bool] | None = None,
    _frame_extractor_fn: Callable[[Path, float, Path, Settings], bool] | None = None,
    _vibe_fn: Callable[[Path, Settings], FeatureVector] | None = None,
) -> list[BrollCandidate]:
    """Run the vibe-aware YouTube fallback for one reference FeatureVector.

    Returns up to `top_k` candidates, sorted by composite_score desc.
    Falls back gracefully when any pipeline stage fails (LLM down, yt-dlp
    timeout, ffmpeg crash) — at minimum you get the URLs that survived
    junk-filtering, with score=0.0 if we couldn't preview them.

    All subprocess side-effects are injectable via `_chat_fn` /
    `_search_fn` / `_download_fn` / `_frame_extractor_fn` / `_vibe_fn`
    keyword args so tests can fully mock them.
    """
    if top_k <= 0:
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Queries — LLM first, deterministic fallback on any failure.
    chat_fn = _chat_fn
    if chat_fn is None:
        from .llm_ladder import generate_search_queries

        chat_fn = lambda prompt, settings: generate_search_queries(prompt, settings)  # noqa: E731

    prompt = _build_query_prompt(reference)
    raw_queries: list[str] = []
    try:
        raw_queries = chat_fn(prompt, settings) or []
    except Exception as exc:
        logger.warning("LLM query gen raised (%s); falling back", type(exc).__name__)
        raw_queries = []

    queries: list[str] = []
    for q in raw_queries:
        t = _truncate_query(q)
        if t and t not in queries:
            queries.append(t)
    if len(queries) < 5:
        for fallback in _fallback_queries(reference):
            t = _truncate_query(fallback)
            if t and t not in queries:
                queries.append(t)
            if len(queries) >= 5:
                break

    # 2. Search — aggregate, dedup, junk-filter, cap.
    entries = _aggregate_search_results(
        queries,
        max_per_query=max_candidates_per_query,
        max_total=max_total_candidates,
        _search_fn=_search_fn,
    )

    # 3. Preview + score — for each surviving entry.
    scored: list[BrollCandidate] = []
    for entry in entries:
        scored.append(_score_candidate(
            reference, entry, cache_dir, settings,
            _download_fn=_download_fn,
            _frame_extractor_fn=_frame_extractor_fn,
            _vibe_fn=_vibe_fn,
        ))

    # 4. Rank — score desc, then stable on insertion order.
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


__all__ = [
    "BrollCandidate",
    "search_broll",
    "composite_score",
    "extract_from_frame",
    "QUERY_GENERATION_PROMPT",
    "YT_JUNK_TITLE_TERMS",
    "COMPOSITE_WEIGHTS",
    "QUERY_MAX_WORDS",
]