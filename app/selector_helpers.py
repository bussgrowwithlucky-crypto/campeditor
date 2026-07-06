"""Helpers for the V2 intelligent B-roll selector.

Spec: ``C:/campeditor/INTELLIGENT_SELECTOR_SPEC.md`` sections 7, 8, 9.

Scope (agent B):
  * Strengthened ``_cinema_match`` with sub-scores for shot_type,
    camera_motion, depth_of_field and a floor at ``_CINEMA_FLOOR``
    (SPEC §7).
  * Reference house-style back-fill and the per-field vibe resolver that
    ``_vibe_score_for`` consumes (SPEC §8).
  * The continuity penalty: a six-dim cosine similarity against a moving
    average of the last 2 picks (SPEC §9).

This module is the implementation surface for agent B. ``app/broll.py``
re-exports the public helpers via ``__all__`` so call sites in
``app/jobs.py`` and friends keep importing from ``app.broll`` unchanged.

Hard rule:
  * The closed vocab constants in ``app/broll.py`` (added by agent A)
    remain the single source of truth. We import them here, do NOT
    redefine them.
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reuse the closed vocabs Agent A dropped into app/broll.py. Keep this
# import lazy to avoid a circular import between app.broll and
# app.selector_helpers when SELECTOR_HELPERS_LOADED gets wired.
# ---------------------------------------------------------------------------
from app.broll import (  # noqa: E402  (import-order kept stable intentionally)
    LibraryClip,
    SpanProfile,
    _MOOD_CAP,
    _COLOR_PALETTE_CAP,
    _MOOD_VOCAB,
    _SHOT_TYPE_VOCAB,
    _CAMERA_MOTION_VOCAB,
    _DEPTH_OF_FIELD_VOCAB,
)


# ---------------------------------------------------------------------------
# Constants (SPEC §2, §7, §9)
# ---------------------------------------------------------------------------

#: Hard floor on ``_cinema_match`` so a true cinematographic mismatch cannot
#: score above ~20% even on perfect content overlap. Mirrors
#: ``Settings.intelligent_cinema_floor`` (SPEC §12) — kept as a local
#: constant so the math is testable without settings plumbing.
_CINEMA_FLOOR: float = 0.18

#: Multiplier on the ``(cinema - 0.6)`` gap when cinema < 0.6. Higher =
#: harsher cinema mismatch penalty. SPEC §7.
_CINEMA_LIFT_TERM: float = 0.5

#: Shot-type broad classes. Wide / aerial / overhead form the WIDE group;
#: close-up / extreme-close-up / two-shot form the TIGHT group. ``medium``
#: stands alone in its own group.
_SHOT_TYPE_GROUP_WIDE: frozenset[str] = frozenset({"wide", "aerial", "overhead"})
_SHOT_TYPE_GROUP_TIGHT: frozenset[str] = frozenset({"close-up", "extreme-close-up", "two-shot"})

#: Camera-motion compatibility table split by intent strength. SPEC §2.
#: HIGH pairs read as deliberately complementary (static + handheld = static
#: ref wants a dynamic B-roll); LOW pairs are exact matches that the spec
#: treats as deliberately low-credit (handheld + handheld for gritty refs,
#: pan + tilt as a rotation pair).
_CAMERA_MOTION_INTENT_HIGH: frozenset[tuple[str, str]] = frozenset({
    ("static", "dolly"),
    ("static", "handheld"),
    ("static", "tracking"),
    ("static", "pan"),
    ("static", "tilt"),
    ("static", "zoom"),
    ("dolly", "tracking"),
})
_CAMERA_MOTION_INTENT_LOW: frozenset[tuple[str, str]] = frozenset({
    ("handheld", "handheld"),
    ("pan", "tilt"),
})


# ---------------------------------------------------------------------------
# Vibe resolver: per-field fallback (profile -> house style -> empty)
# ---------------------------------------------------------------------------

_VIBE_FIELD_NAMES: tuple[str, ...] = (
    "mood",
    "energy",
    "lighting",
    "shot_type",
    "camera_motion",
    "depth_of_field",
    "color_palette",
)


def _profile_vibe_value(profile: SpanProfile, name: str):
    """Pull a single vibe field off a SpanProfile, normalizing empty
    containers/strings to the absence sentinel."""
    raw = getattr(profile, name, None)
    if raw is None:
        return _absent_sentinel_for(name)
    if isinstance(raw, list):
        cleaned = [str(x).strip() for x in raw if str(x).strip()]
        return cleaned if cleaned else _absent_sentinel_for(name)
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped if stripped else _absent_sentinel_for(name)
    return _absent_sentinel_for(name)


def _absent_sentinel_for(name: str):
    return [] if name in ("mood", "color_palette") else ""


def _resolve_field(profile_value, house_value, name: str):
    """Per-field fallback. profile -> house style -> empty.

    Lists (mood / color_palette) merge into a unique-preserving union
    capped at the field cap; strings fall back by exact non-empty match."""
    cap = _MOOD_CAP if name == "mood" else _COLOR_PALETTE_CAP if name == "color_palette" else 1
    if name in ("mood", "color_palette"):
        out: list[str] = []
        for v in (profile_value, house_value):
            if not isinstance(v, list):
                continue
            for x in v:
                s = str(x).strip()
                if s and s not in out:
                    out.append(s)
                if len(out) >= cap:
                    break
            if len(out) >= cap:
                break
        return out
    # String-valued field.
    if isinstance(profile_value, str) and profile_value.strip():
        return profile_value.strip()
    if isinstance(house_value, str) and house_value.strip():
        return house_value.strip()
    return ""


def resolve_span_vibe(profile: SpanProfile, reference_house: dict | None) -> dict:
    """Per-field fallback: profile value -> house style -> empty.

    Returns a dict with the seven vibe keys, each in their absence-sentinel
    form when empty (lists for ``mood`` / ``color_palette``, empty string
    for the enums). Never raises."""
    out: dict = {}
    for name in _VIBE_FIELD_NAMES:
        profile_v = _profile_vibe_value(profile, name)
        house_v = None
        if isinstance(reference_house, dict):
            house_v = reference_house.get(name)
        out[name] = _resolve_field(profile_v, house_v, name)
    return out


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

_LIST_VIBE_FIELDS = ("mood", "color_palette")
_ENUM_VIBE_FIELDS = ("energy", "lighting", "shot_type", "camera_motion", "depth_of_field")


def _mode_or_first(values: list[str]) -> str:
    """Pick the mode of a list of non-empty strings (ties broken by first
    occurrence). Empty input -> ``""``."""
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for i, v in enumerate(values):
        if not isinstance(v, str) or not v.strip():
            continue
        counts[v] = counts.get(v, 0) + 1
        first_seen.setdefault(v, i)
    if not counts:
        return ""
    best = max(counts.items(), key=lambda kv: (kv[1], -first_seen[kv[0]]))
    return best[0]


def build_reference_house_style(analysis) -> dict:
    """Aggregate span-level tags into a single "house style" vector.

    Computed ONCE per job (in ``fetch_broll_cut_variations`` / friends) and
    passed down to every per-span scoring call. Used as a fallback when an
    individual span's vibe fields are empty (e.g. a close-up on a face
    whose ``mood`` field didn't take).

    Input tolerance: every entry of ``analysis.broll_span_tags`` may be a
    ``dict``, a mapping-like object, or anything else — non-dicts skip
    silently. ``mood`` / ``color_palette`` are unioned (capped); the enum
    fields take the mode across non-empty values.

    Returns a dict shaped like one span's vibe fields. Empty fields stay
    empty (the absence sentinel)."""
    house: dict = {
        "mood": [],
        "energy": "",
        "lighting": "",
        "shot_type": "",
        "camera_motion": "",
        "depth_of_field": "",
        "color_palette": [],
    }
    tags_seq = getattr(analysis, "broll_span_tags", None) or []
    if not isinstance(tags_seq, list):
        return house

    mood_union: list[str] = []
    palette_union: list[str] = []
    enum_buckets: dict[str, list[str]] = {k: [] for k in _ENUM_VIBE_FIELDS}

    for entry in tags_seq:
        if not isinstance(entry, dict):
            continue
        ms = entry.get("mood")
        if isinstance(ms, list):
            for m in ms:
                s = str(m).strip()
                if s and s not in mood_union:
                    mood_union.append(s)
                if len(mood_union) >= _MOOD_CAP:
                    break
        pal = entry.get("color_palette")
        if isinstance(pal, list):
            for c in pal:
                s = str(c).strip()
                if s and s not in palette_union:
                    palette_union.append(s)
                if len(palette_union) >= _COLOR_PALETTE_CAP:
                    break
        for name in _ENUM_VIBE_FIELDS:
            v = entry.get(name)
            if isinstance(v, str) and v.strip():
                enum_buckets[name].append(v.strip())

    house["mood"] = mood_union[:_MOOD_CAP]
    house["color_palette"] = palette_union[:_COLOR_PALETTE_CAP]
    for name in _ENUM_VIBE_FIELDS:
        house[name] = _mode_or_first(enum_buckets[name])
    return house


# ---------------------------------------------------------------------------
# Vibe score (resolved-span vs clip)
# ---------------------------------------------------------------------------

#: Vibe score weights. Same shape as the legacy ``_VIBE_FIELDS`` table —
#: the camera_motion component is intentionally absent here (it lives in
#: the cinema_match path so a static-vs-handheld clip can still get a
#: partial vibe match without being skipped entirely).
_VIBE_FIELDS_WEIGHTED: tuple[tuple[str, float], ...] = (
    ("mood",           0.40),
    ("lighting",       0.20),
    ("energy",         0.15),
    ("shot_type",      0.15),
    ("depth_of_field", 0.10),
)


def _vibe_subscore(a, b) -> float:
    """Compare two vibe values. Lists -> Jaccard. Strings -> exact match
    or 0.5 on a small adjacent bucket. Mixed types -> 0.0. Mirrors the
    legacy ``_vibe_subscore`` in app/broll.py but tolerant of the absence
    sentinels (``""`` / ``[]``)."""
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
        # Same small adjacency buckets as the legacy scorer (kept verbatim
        # so a span asking for "low-key" doesn't completely zero a clip
        # tagged "mixed").
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


def _vibe_score_for_resolved(resolved_span_vibe: dict, clip: LibraryClip) -> float:
    """Compare a RESOLVED span vibe (already back-filled from house style)
    to a clip. Returns 0.0 when neither side carries vibe fields."""
    if not isinstance(resolved_span_vibe, dict) or not isinstance(clip, LibraryClip):
        return 0.0
    total_weight = 0.0
    weighted = 0.0
    for name, weight in _VIBE_FIELDS_WEIGHTED:
        sv = resolved_span_vibe.get(name)
        cv = getattr(clip, name, None)
        if _is_absent(sv) or _is_absent(cv):
            continue
        weighted += weight * _vibe_subscore(sv, cv)
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return max(0.0, min(1.0, weighted / total_weight))


def _is_absent(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, list):
        return not any(str(x).strip() for x in v)
    return False


# ---------------------------------------------------------------------------
# Cinema match + sub-scores (SPEC §7)
# ---------------------------------------------------------------------------


def _shot_type_subscore(a: str, b: str):
    """Both empty / one empty -> ``None`` (drop out). ``a == b`` -> 1.0.
    Both in same broad class (WIDE / TIGHT) -> 0.6. Both ``medium`` -> 0.6.
    Cross-class -> 0.25."""
    a_s = (a or "").strip().lower()
    b_s = (b or "").strip().lower()
    if not a_s or not b_s:
        return None
    if a_s == b_s:
        return 1.0
    if a_s in _SHOT_TYPE_GROUP_WIDE and b_s in _SHOT_TYPE_GROUP_WIDE:
        return 0.6
    if a_s in _SHOT_TYPE_GROUP_TIGHT and b_s in _SHOT_TYPE_GROUP_TIGHT:
        return 0.6
    if a_s == "medium" and b_s == "medium":
        return 0.6
    return 0.25


def _camera_motion_subscore(a: str, b: str):
    """Both empty / one empty -> ``None``. ``a == b`` -> 0.4 (deliberately
    low). HIGH intent pair -> 1.0. LOW intent pair -> 0.4. Otherwise 0.0."""
    a_s = (a or "").strip().lower()
    b_s = (b or "").strip().lower()
    if not a_s or not b_s:
        return None
    if a_s == b_s:
        return 0.4
    pair1 = (a_s, b_s)
    pair2 = (b_s, a_s)
    if pair1 in _CAMERA_MOTION_INTENT_HIGH or pair2 in _CAMERA_MOTION_INTENT_HIGH:
        return 1.0
    if pair1 in _CAMERA_MOTION_INTENT_LOW or pair2 in _CAMERA_MOTION_INTENT_LOW:
        return 0.4
    return 0.0


def _depth_of_field_subscore(a: str, b: str):
    """Both empty / one empty -> ``None``. Same -> 1.0. Mismatch -> 0.2."""
    a_s = (a or "").strip().lower()
    b_s = (b or "").strip().lower()
    if not a_s or not b_s:
        return None
    if a_s == b_s:
        return 1.0
    if {a_s, b_s} <= {"deep", "shallow"}:
        return 0.2
    return 0.0


def cinema_match(profile: SpanProfile, clip: LibraryClip) -> float:
    """0..1 cinematography match score — combines shot_type, camera_motion,
    depth_of_field. Empty / missing fields on either side drop out of the
    calculation (don't pull the score toward 0 for an absent field — that's
    a data gap, not a mismatch). Floored at ``_CINEMA_FLOOR`` (0.18) so a
    true cinematographic mismatch cannot rise out of the bottom."""
    subs = [
        _shot_type_subscore(getattr(profile, "shot_type", ""), getattr(clip, "shot_type", "")),
        _camera_motion_subscore(getattr(profile, "camera_motion", ""), getattr(clip, "camera_motion", "")),
        _depth_of_field_subscore(getattr(profile, "depth_of_field", ""), getattr(clip, "depth_of_field", "")),
    ]
    present = [v for v in subs if v is not None]
    if not present:
        return 0.6  # neutral default — no cinema data on either side
    raw = sum(present) / len(present)
    # Apply the floor so a true cinematic mismatch cannot escape the bottom.
    return max(_CINEMA_FLOOR, raw)


# ---------------------------------------------------------------------------
# Continuity ledger (SPEC §9)
# ---------------------------------------------------------------------------

#: 6-dim FeatureVector bucket sizes. The encoding is:
#:   dims 0..0   shot_type    (7 buckets)
#:   dim  1      camera_motion (7 buckets)
#:   dim  2      depth_of_field (2 buckets)
#:   dims 3..5   mood histogram over 3 canonical groups (mystery/intensity/
#:               stillness; padded to 3 with 0.0)
_SHOT_TYPE_BUCKETS: tuple[str, ...] = (
    "wide", "medium", "close-up", "extreme-close-up", "aerial", "overhead", "two-shot",
)
_CAMERA_MOTION_BUCKETS: tuple[str, ...] = (
    "static", "pan", "tilt", "dolly", "handheld", "tracking", "zoom",
)
_DOF_BUCKETS: tuple[str, ...] = ("deep", "shallow")
# 3 canonical mood groups — chosen to spread the 16-item vocab across 3 axes
# that humans can read: mystery/intensity/stillness. Empty buckets get
# weighted 0.0 and don't disturb the cosine.
_MOOD_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"mysterious", "tense", "ominous", "sinister", "dramatic"}),
    frozenset({"epic", "energetic", "aggressive", "uplifting", "joyful", "playful"}),
    frozenset({"calm", "melancholic", "nostalgic", "romantic", "neutral"}),
)


def feature_vector_for_clip(clip: LibraryClip) -> dict:
    """Six-dim vibe vector for cosine similarity.

    Drops the long-tail fields (energy, lighting, depth_of_field) which
    are noisy on single-frame tags and replaces them with the three that
    matter most visually: shot_type, camera_motion, plus a 3-element mood
    histogram.

    Returned shape (for the cosine similarity inputs):

        {
            "shot_type_onehot": [0/1, ...]   length 7
            "camera_motion_onehot": [0/1, ...] length 7
            "dof_onehot": [0/1, 0/1]         length 2
            "mood_hist": [0..1, 0..1, 0..1]  length 3, normalized to sum
        }
    """
    shot = _onehot((getattr(clip, "shot_type", "") or "").strip().lower(), _SHOT_TYPE_BUCKETS)
    cam = _onehot((getattr(clip, "camera_motion", "") or "").strip().lower(), _CAMERA_MOTION_BUCKETS)
    dof = _onehot((getattr(clip, "depth_of_field", "") or "").strip().lower(), _DOF_BUCKETS)
    mood_hist = _mood_histogram(getattr(clip, "mood", []) or [])
    return {
        "shot_type_onehot": shot,
        "camera_motion_onehot": cam,
        "dof_onehot": dof,
        "mood_hist": mood_hist,
    }


def _onehot(value: str, buckets: tuple[str, ...]) -> list[float]:
    out = [0.0] * len(buckets)
    if not value:
        return out
    for i, b in enumerate(buckets):
        if b == value:
            out[i] = 1.0
            break
    return out


def _mood_histogram(moods) -> list[float]:
    if not isinstance(moods, list) or not moods:
        return [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    for raw in moods:
        s = str(raw).strip().lower()
        for i, group in enumerate(_MOOD_GROUPS):
            if s in group:
                counts[i] += 1
                break
    total = sum(counts)
    if total <= 0:
        return [0.0, 0.0, 0.0]
    return [c / total for c in counts]


def cosine_similarity_6d(a: dict, b: dict) -> float:
    """Cosine similarity over the six-dim FeatureVector. Returns 0.0 when
    either side is empty / wrong shape / degenerate. Range [0, 1] (the
    encoding is non-negative so true cosine never goes negative; we
    clip to [0, 1] defensively)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0

    def _flat(v: dict) -> list[float]:
        return (
            list(v.get("shot_type_onehot") or [])
            + list(v.get("camera_motion_onehot") or [])
            + list(v.get("dof_onehot") or [])
            + list(v.get("mood_hist") or [])
        )

    va = _flat(a)
    vb = _flat(b)
    n = min(len(va), len(vb))
    if n == 0:
        return 0.0
    va = va[:n]
    vb = vb[:n]
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(y * y for y in vb) ** 0.5
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


class ContinuityLedger:
    """Job-scoped ledger that holds the last few picks' FeatureVectors.

    Used by ``match_local`` to apply a small negative penalty to
    consecutive B-roll slots that look visually identical. The ledger is
    mutable, monotonic, and intentionally simple — agents A and C
    own the threading shape around it; this class is the storage +
    scoring primitive.

    Usage:
        ledger = ContinuityLedger(max_history=2)
        ledger.note(feature_vec_for_clip_a)
        penalty = ledger.penalty_for(feature_vec_for_clip_b,
                                     threshold=0.92, max_penalty=-0.08)

    Single-threaded. The caller (typically the per-span serial loop in
    ``_gather_span_pool``) owns the lock.
    """

    def __init__(self, max_history: int = 2):
        self._max_history = max(2, max_history)
        self._history: list[dict] = []

    def note_picked(self, clip_or_vec) -> None:
        """Record one pick. Accepts either a ``LibraryClip`` or a raw
        6-dim dict (for tests). Idempotent: silently skips duplicates of
        the last entry (no-op when the same vec is appended twice)."""
        if isinstance(clip_or_vec, LibraryClip):
            vec = feature_vector_for_clip(clip_or_vec)
        else:
            vec = clip_or_vec
        if not isinstance(vec, dict):
            return
        if self._history and self._history[-1] == vec:
            return
        self._history.append(vec)
        # Keep only the last `max_history` entries — the penalty uses the
        # moving average of the last 2 picks.
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def penalty_for(self, candidate, *, threshold: float = 0.92,
                    max_penalty: float = -0.08) -> float:
        """Compute the continuity penalty for ``candidate``.

        ``candidate`` may be a ``LibraryClip`` or a raw 6-dim dict.
        Returns a non-positive float in [``max_penalty``, 0.0]. Zero
        when the ledger is empty or every entry scores below the
        threshold.

        Per SPEC §9, penalty fires when the cosine similarity of the
        candidate against the moving average of the last 2 picks meets
        ``threshold`` (default 0.92).
        """
        if isinstance(candidate, LibraryClip):
            cand_vec = feature_vector_for_clip(candidate)
        else:
            cand_vec = candidate
        if not isinstance(cand_vec, dict):
            return 0.0
        if not self._history:
            return 0.0
        avg = _moving_average(self._history)
        if avg is None:
            return 0.0
        cos = cosine_similarity_6d(cand_vec, avg)
        if cos >= threshold:
            return max_penalty
        return 0.0


def _moving_average(history: list[dict]) -> dict | None:
    """Average the last up-to-2 entries component-wise. Returns None when
    nothing to average."""
    if not history:
        return None
    window = history[-2:]
    keys = ("shot_type_onehot", "camera_motion_onehot", "dof_onehot", "mood_hist")
    out: dict = {}
    for key in keys:
        rows: list[list[float]] = []
        for entry in window:
            row = entry.get(key) or []
            if isinstance(row, list) and row:
                rows.append([float(x) for x in row])
        if not rows:
            out[key] = []
            continue
        length = min(len(r) for r in rows)
        if length == 0:
            out[key] = []
            continue
        out[key] = [sum(r[i] for r in rows) / len(rows) for i in range(length)]
    return out


# ---------------------------------------------------------------------------
# Re-export convenience
# ---------------------------------------------------------------------------

__all__ = [
    "ContinuityLedger",
    "build_reference_house_style",
    "cinema_match",
    "cosine_similarity_6d",
    "feature_vector_for_clip",
    "resolve_span_vibe",
    "_CINEMA_FLOOR",
    "_CINEMA_LIFT_TERM",
    "_vibe_score_for_resolved",
]
