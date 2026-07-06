"""Multi-dimensional B-roll candidate ranking.

Ranks library clips against a reference ``FeatureVector`` using a weighted
composite score that balances legacy subject/category overlap with the new
vibe dimensions (mood, lighting, color palette, motion, cinematography,
energy). The final score lives in ``[0, 1]`` and is orderable.

Public API
----------

``rank_candidates(reference, index, top_k, used_clips)`` returns a list of
``(Path, FeatureVector, score)`` tuples ordered by score descending,
skipping any clip whose path is in ``used_clips``. Clips with
``confidence == 0.0`` (the ``empty_feature_vector()`` sentinel) are pushed
to the back of the ranking but never silently dropped — they're useful
last-resort picks when no signal-bearing clip is available.

Scoring formula (final composite in ``[0, 1]``)
-----------------------------------------------

Each component produces a value in ``[0, 1]``. Sub-component weights inside
a component sum to 1.0; top-level component weights also sum to 1.0
(0.30 subject + 0.45 vibe + 0.15 cinematography + 0.10 energy). When a
sub-component cannot be evaluated (both sides missing the required data),
its contribution drops to 0 and the remaining sub-components are
renormalised to sum to 1.0 — graceful degradation so missing fields don't
collapse the whole ranking to zero.

Subject component (weight 0.30):
    ``0.4 * category_score + 0.4 * Jaccard(subjects) + 0.2 * Jaccard(setting)``
    ``category_score``: 1.0 exact, 0.5 same family, 0.0 different.
    Families (intersection): ``sports <-> lifestyle``, ``money <-> lifestyle``.

Vibe component (weight 0.45):
    ``0.30 * mood_sim + 0.20 * (1 - palette_dist/3) + 0.15 * lighting
     + 0.20 * (1 - |motion_diff|) + 0.15 * (1 - |contrast_diff|)``
    ``mood_sim`` is Jaccard on the expanded mood sets (each mood expanded
    to its synonym group; see ``MOOD_SYNONYMS``). Synonyms match at full
    strength.
    ``palette_dist`` = mean absolute diff across (warmth, saturation,
    brightness), each in ``[0, 1]``, so the term lives in ``[0, 1]``.
    ``lighting_score``: 1.0 exact, 0.4 compatible per
    ``LIGHTING_COMPATIBLE``, 0.0 otherwise.

Cinematography component (weight 0.15):
    ``0.4 * shot_type_score + 0.3 * camera_motion_score + 0.3 * dof_score``
    ``shot_type_score``: 1.0 exact, 0.5 compatible per
    ``SHOT_TYPE_COMPATIBLE``, 0.0 otherwise.
    ``camera_motion_score``: 1.0 exact, 0.4 compatible per
    ``CAMERA_MOTION_COMPATIBLE``, 0.0 otherwise.
    ``dof_score``: 1.0 exact, 0.0 otherwise (no compatibility pairs).

Energy component (weight 0.10):
    ``1.0 - |energy_score_diff|`` where low=0.0, medium=0.5, high=1.0.

If a major component evaluates to 0.0 (every sub-term missing on both
sides), the top-level weights are renormalised so the remaining active
components still span ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .feature_vector import FeatureVector

# ---------------------------------------------------------------------------
# Compatibility tables (mirrored in CONTRACT.md "Compatibility matrices"
# addendum). Symmetric: if (a, b) is here, (b, a) is also compatible.
# ---------------------------------------------------------------------------

LIGHTING_COMPATIBLE: frozenset[tuple[str, str]] = frozenset(
    {
        ("low-key", "natural"),
        ("high-key", "natural"),
        ("neon", "mixed"),
        ("golden-hour", "natural"),
    }
)

SHOT_TYPE_COMPATIBLE: frozenset[tuple[str, str]] = frozenset(
    {
        ("close-up", "extreme-close-up"),
        ("wide", "medium"),
        ("aerial", "overhead"),
    }
)

# Camera-motion compatibility — pairs that read as "same family" even if not
# identical. Static / handheld / zoom stand alone; pan<->tilt (rotational);
# dolly<->tracking (positional push/pull).
CAMERA_MOTION_COMPATIBLE: frozenset[tuple[str, str]] = frozenset(
    {
        ("pan", "tilt"),
        ("dolly", "tracking"),
    }
)

# Subject-category families. (a, b) implies b is in a's family. Used for
# the 0.5 "same family" partial credit in the subject component.
CATEGORY_FAMILIES: frozenset[tuple[str, str]] = frozenset(
    {
        ("sports", "lifestyle"),
        ("money", "lifestyle"),
    }
)

# Mood synonyms — connected components of the equivalence graph. A mood
# expanded to its synonym group matches any peer mood that lands in the
# same group. "mysterious" <-> "ominous", etc.
MOOD_SYNONYMS: dict[str, frozenset[str]] = {
    "mysterious": frozenset({"mysterious", "ominous"}),
    "ominous": frozenset({"mysterious", "ominous"}),
    "energetic": frozenset({"energetic", "aggressive"}),
    "aggressive": frozenset({"energetic", "aggressive"}),
    "calm": frozenset({"calm", "melancholic"}),
    "melancholic": frozenset({"calm", "melancholic"}),
    "joyful": frozenset({"joyful", "uplifting"}),
    "uplifting": frozenset({"joyful", "uplifting"}),
    "epic": frozenset({"epic", "dramatic"}),
    "dramatic": frozenset({"epic", "dramatic"}),
    # Standalone moods (no synonym neighbour) — identity expansion.
    "tense": frozenset({"tense"}),
    "romantic": frozenset({"romantic"}),
    "nostalgic": frozenset({"nostalgic"}),
    "neutral": frozenset({"neutral"}),
    "playful": frozenset({"playful"}),
    "sinister": frozenset({"sinister"}),
}

# Top-level component weights. Sum = 1.0.
W_SUBJECT = 0.30
W_VIBE = 0.45
W_CINEMA = 0.15
W_ENERGY = 0.10

# Subject sub-component weights. Sum = 1.0.
W_SUBJ_CATEGORY = 0.40
W_SUBJ_SUBJECTS = 0.40
W_SUBJ_SETTING = 0.20

# Vibe sub-component weights. Sum = 1.0.
W_VIBE_MOOD = 0.30
W_VIBE_PALETTE = 0.20
W_VIBE_LIGHTING = 0.15
W_VIBE_MOTION = 0.20
W_VIBE_CONTRAST = 0.15

# Cinematography sub-component weights. Sum = 1.0.
W_CINE_SHOT = 0.40
W_CINE_MOTION = 0.30
W_CINE_DOF = 0.30

# Energy numeric scale: low=0.0, medium=0.5, high=1.0. Anything else
# collapses to medium so a typo doesn't brick the diff.
_ENERGY_VALUE: dict[str, float] = {"low": 0.0, "medium": 0.5, "high": 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalised(fv: FeatureVector) -> FeatureVector:
    """Use the FeatureVector's own normalisation (lowercasing, caps)."""
    return fv.normalised()


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity over two string iterables (case-insensitive,
    whitespace-trimmed). 0.0 when both inputs are empty."""
    sa = {x.strip().lower() for x in a if x and x.strip()}
    sb = {x.strip().lower() for x in b if x and x.strip()}
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0


def _category_score(a: str, b: str) -> float:
    """1.0 exact match, 0.5 same family, 0.0 different."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if (a, b) in CATEGORY_FAMILIES or (b, a) in CATEGORY_FAMILIES:
        return 0.5
    return 0.0


def _compatibility_score(a: str, b: str, table: frozenset[tuple[str, str]]) -> float:
    """1.0 exact, 0.4 compatible, 0.0 incompatible. Both empty/missing → 0."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if (a, b) in table or (b, a) in table:
        return 0.4
    return 0.0


def _shot_type_score(a: str, b: str) -> float:
    """1.0 exact, 0.5 compatible, 0.0 else. Slightly higher than the
    generic _compatibility_score because shot type is a strong style cue."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if (a, b) in SHOT_TYPE_COMPATIBLE or (b, a) in SHOT_TYPE_COMPATIBLE:
        return 0.5
    return 0.0


def _mood_synonym_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard with synonym expansion. Each mood on either side is mapped
    to its synonym group; Jaccard is computed on those expanded sets so
    ``mysterious`` ~ ``ominous`` scores 1.0 instead of 0.0."""
    expanded_a: set[str] = set()
    expanded_b: set[str] = set()
    for m in a:
        if not m:
            continue
        key = m.strip().lower()
        if key:
            expanded_a.update(MOOD_SYNONYMS.get(key, frozenset({key})))
    for m in b:
        if not m:
            continue
        key = m.strip().lower()
        if key:
            expanded_b.update(MOOD_SYNONYMS.get(key, frozenset({key})))
    if not expanded_a and not expanded_b:
        return 0.0
    inter = expanded_a & expanded_b
    union = expanded_a | expanded_b
    return len(inter) / len(union) if union else 0.0


def _palette_distance(a: FeatureVector, b: FeatureVector) -> float | None:
    """Mean absolute diff across (warmth, saturation, brightness), or
    ``None`` when any side is missing all three numerics (so the vibe
    component can renormalise without inventing a fake distance)."""
    vals_a = (a.palette_warmth, a.palette_saturation, a.palette_brightness)
    vals_b = (b.palette_warmth, b.palette_saturation, b.palette_brightness)
    # All-zero is treated as "not computed" (CV failure sentinel) so we
    # don't penalise a perfectly-grey clip vs a perfectly-warm clip.
    if all(v == 0.0 for v in vals_a) or all(v == 0.0 for v in vals_b):
        return None
    return sum(abs(x - y) for x, y in zip(vals_a, vals_b)) / 3.0


def _diff_or_none(a: float, b: float) -> float | None:
    """Absolute difference, or None when either side is the 0.0 sentinel."""
    if a == 0.0 or b == 0.0:
        return None
    return abs(a - b)


def _energy_value(e: str) -> float:
    return _ENERGY_VALUE.get((e or "").strip().lower(), 0.5)


def _renorm(parts: list[tuple[float, float, bool]]) -> float:
    """Weighted sum with per-part renormalisation, distinguishing
    "scored 0 because data was incompatible" from "scored 0 because data
    was missing on both sides".

    ``parts`` is a list of ``(score, weight, had_data)`` tuples. Score in
    [0, 1]; weight > 0; ``had_data`` indicates whether the underlying
    fields were populated on either side. When ``had_data`` is False the
    sub-term contributes 0 AND drops its weight (graceful degradation).
    When ``had_data`` is True the sub-term keeps its full weight even if
    the score is 0 — that's real signal saying "different", not "missing".

    Returns the renormalised score in [0, 1]. All-zero returns 0.0.
    """
    total_w = sum(w for _, w, had in parts if w > 0 and had)
    if total_w <= 0:
        return 0.0
    return sum(s * w for s, w, had in parts if w > 0 and had) / total_w


# ---------------------------------------------------------------------------
# Per-component scorers (each returns (score, had_data))
# ---------------------------------------------------------------------------


def _score_subject(ref: FeatureVector, cand: FeatureVector) -> tuple[float, bool]:
    cat = _category_score(ref.category, cand.category)
    subj = _jaccard(ref.subjects, cand.subjects)
    setting = _jaccard(ref.setting, cand.setting)
    parts: list[tuple[float, float, bool]] = [
        (cat, W_SUBJ_CATEGORY, bool(ref.category or cand.category)),
        (subj, W_SUBJ_SUBJECTS, bool(ref.subjects or cand.subjects)),
        (setting, W_SUBJ_SETTING, bool(ref.setting or cand.setting)),
    ]
    return _renorm(parts), any(had for _, _, had in parts)


def _score_vibe(ref: FeatureVector, cand: FeatureVector) -> tuple[float, bool]:
    mood = _mood_synonym_jaccard(ref.mood, cand.mood)
    palette = _palette_distance(ref, cand)
    lighting = _compatibility_score(
        ref.lighting, cand.lighting, LIGHTING_COMPATIBLE
    )
    motion_diff = _diff_or_none(ref.motion_intensity, cand.motion_intensity)
    contrast_diff = _diff_or_none(ref.contrast, cand.contrast)
    palette_score = 1.0 - palette if palette is not None else 0.0
    motion_score = 1.0 - motion_diff if motion_diff is not None else 0.0
    contrast_score = 1.0 - contrast_diff if contrast_diff is not None else 0.0
    parts: list[tuple[float, float, bool]] = [
        (mood, W_VIBE_MOOD, bool(ref.mood or cand.mood)),
        (palette_score, W_VIBE_PALETTE, palette is not None),
        (lighting, W_VIBE_LIGHTING, bool(ref.lighting and cand.lighting)),
        (motion_score, W_VIBE_MOTION, motion_diff is not None),
        (contrast_score, W_VIBE_CONTRAST, contrast_diff is not None),
    ]
    return _renorm(parts), any(had for _, _, had in parts)


def _score_cinema(ref: FeatureVector, cand: FeatureVector) -> tuple[float, bool]:
    shot = _shot_type_score(ref.shot_type, cand.shot_type)
    motion = _compatibility_score(
        ref.camera_motion, cand.camera_motion, CAMERA_MOTION_COMPATIBLE
    )
    dof = 1.0 if ref.depth_of_field and cand.depth_of_field and ref.depth_of_field == cand.depth_of_field else 0.0
    parts: list[tuple[float, float, bool]] = [
        (shot, W_CINE_SHOT, bool(ref.shot_type and cand.shot_type)),
        (motion, W_CINE_MOTION, bool(ref.camera_motion and cand.camera_motion)),
        (dof, W_CINE_DOF, bool(ref.depth_of_field and cand.depth_of_field)),
    ]
    return _renorm(parts), any(had for _, _, had in parts)


def _score_energy(ref: FeatureVector, cand: FeatureVector) -> tuple[float, bool]:
    """Energy is a single closed enum — always present (defaults to
    'medium' on empty FeatureVector), so we always have data."""
    diff = abs(_energy_value(ref.energy) - _energy_value(cand.energy))
    return 1.0 - diff, True


def score_clip(ref: FeatureVector, cand: FeatureVector) -> float:
    """Top-level composite score in ``[0, 1]``. Exposed so tests and other
    callers can score a single (reference, candidate) pair without going
    through the full index."""
    ref = _normalised(ref)
    cand = _normalised(cand)
    s_subject, d_subject = _score_subject(ref, cand)
    s_vibe, d_vibe = _score_vibe(ref, cand)
    s_cinema, d_cinema = _score_cinema(ref, cand)
    s_energy, d_energy = _score_energy(ref, cand)
    # Top-level renormalisation: a component contributes 0 only when its
    # underlying fields were ALL missing — that's "graceful degradation".
    # A component that scored 0 because data was PRESENT but incompatible
    # keeps its full weight (the 0 is real signal saying "different").
    parts: list[tuple[float, float, bool]] = [
        (s_subject, W_SUBJECT, d_subject),
        (s_vibe, W_VIBE, d_vibe),
        (s_cinema, W_CINEMA, d_cinema),
        (s_energy, W_ENERGY, d_energy),
    ]
    return _renorm(parts)


# ---------------------------------------------------------------------------
# Public ranking API
# ---------------------------------------------------------------------------


def rank_candidates(
    reference: FeatureVector,
    index: list[tuple[Path, FeatureVector]],
    top_k: int = 5,
    used_clips: set[Path] | None = None,
) -> list[tuple[Path, FeatureVector, float]]:
    """Rank library clips against ``reference`` by composite similarity.

    Parameters
    ----------
    reference : FeatureVector
        The query clip / span we're trying to match. Use ``empty_feature_vector``
        to ask for "anything with signal" — it'll be scored against every
        candidate on whatever fields ARE populated.
    index : list[tuple[Path, FeatureVector]]
        Candidate clips. Order is irrelevant; the function sorts internally.
    top_k : int
        Number of results to return. Default 5.
    used_clips : set[Path] | None
        Paths to skip entirely (already-shown clips, diversity dedupe, etc.).

    Returns
    -------
    list[tuple[Path, FeatureVector, float]]
        Top-``top_k`` clips ordered by score descending. The float is the
        composite score in ``[0, 1]``. Clips with ``confidence == 0.0``
        (empty-sentinel vectors) are kept but pushed behind any signal-
        bearing clip — they're the last-resort picks when nothing better
        is available.

    Notes
    -----
    Scoring is symmetric: ``rank_candidates(ref, index)`` and
    ``rank_candidates(swap)`` produce the same relative ordering of the
    same pair, since the formula depends on the pair, not on which side
    is "reference".
    """
    if top_k <= 0:
        return []
    skip = used_clips or set()
    scored: list[tuple[Path, FeatureVector, float]] = []
    for path, fv in index:
        if path in skip:
            continue
        score = score_clip(reference, fv)
        scored.append((path, fv, score))
    # Sort by score desc, but push confidence==0 (empty sentinel) to the
    # back regardless of raw score so real signal always wins ties.
    scored.sort(
        key=lambda item: (item[2], 0 if item[1].confidence > 0.0 else 1),
        reverse=True,
    )
    return scored[:top_k]