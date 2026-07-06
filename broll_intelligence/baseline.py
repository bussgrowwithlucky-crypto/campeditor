"""Baseline re-implementation of the legacy campeditor B-roll scoring.

Used only by :mod:`broll_intelligence.compare` to produce a side-by-side
report: the new system vs the old heuristic. This module deliberately
re-implements the legacy logic inline so the comparison report is honest
— it must NOT silently pick up fixes from the new system.

Mirrors ``app/broll.py::_local_score`` weights:
  * category match: 3.0 points
  * subject overlap: 1.0 per shared subject
  * setting overlap: 1.0 per shared setting
  * max 8.0 -> normalized to [0, 1]
And ``app/broll.py::_truncate_query`` (8-word cap on search queries).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .feature_vector import FeatureVector

# Cap matches the legacy app/broll.py:8 (broll_query_max_words default).
TRUNCATE_WORDS = 8

# Weights match the legacy _local_score function: 3.0 + 1.0*N + 1.0*M,
# normalised by _LOCAL_SCORE_MAX = 8.0.
CATEGORY_POINTS = 3.0
SUBJECT_POINTS = 1.0
SETTING_POINTS = 1.0
LOCAL_SCORE_MAX = 8.0


@dataclass
class BaselinePick:
    """One library pick from the legacy heuristic."""

    path: Path
    subject_score: float         # 0..1
    subjects: list[str]
    setting: list[str]
    category: str


def truncate_query(query: str) -> str:
    """Truncate a search query to ``TRUNCATE_WORDS`` words (legacy behaviour)."""
    words = (query or "").split()
    return " ".join(words[:TRUNCATE_WORDS])


def local_score(profile: FeatureVector, clip_features: FeatureVector) -> float:
    """The legacy local-library match score in [0, 1]."""
    score = 0.0
    if profile.category and clip_features.category and profile.category == clip_features.category:
        score += CATEGORY_POINTS
    score += len(set(profile.subjects) & set(clip_features.subjects)) * SUBJECT_POINTS
    score += len(set(profile.setting) & set(clip_features.setting)) * SETTING_POINTS
    return min(1.0, score / LOCAL_SCORE_MAX)


def rank_library_legacy(
    reference: FeatureVector,
    index: list[tuple[Path, FeatureVector]],
    top_k: int = 5,
    threshold: float = 0.35,
) -> list[BaselinePick]:
    """Rank library clips with the legacy scoring formula.

    Same threshold default as ``app/broll.py::broll_local_match_threshold``
    (0.35). Returns the top-K above the threshold, descending.
    """
    scored: list[BaselinePick] = []
    for path, fv in index:
        s = local_score(reference, fv)
        if s < threshold:
            continue
        scored.append(
            BaselinePick(
                path=path,
                subject_score=s,
                subjects=list(fv.subjects),
                setting=list(fv.setting),
                category=fv.category,
            )
        )
    scored.sort(key=lambda p: p.subject_score, reverse=True)
    return scored[:top_k]


def youtube_query_legacy(reference: FeatureVector) -> str:
    """Build a single search query the legacy way: subject + category + raw query, truncated to 8 words."""
    parts: list[str] = []
    if reference.subjects:
        parts.append(" ".join(reference.subjects[:3]))
    if reference.setting:
        parts.append(" ".join(reference.setting[:2]))
    if reference.query:
        parts.append(reference.query)
    if not parts:
        # Fall back to category + action if there's nothing else.
        parts.append(reference.category or "b roll")
        if reference.action:
            parts.append(" ".join(reference.action))
    return truncate_query(" ".join(parts))