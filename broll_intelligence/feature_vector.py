"""FeatureVector dataclass + helpers.

The contract for downstream tasks. Schema is defined verbatim in
CONTRACT.md — keep these two in lock-step when adding fields. Validation
rejects free-form strings for the closed enums (category, energy, lighting,
shot_type, camera_motion, depth_of_field) so a bad LLM response can't poison
the index.

The dataclass is the in-memory representation; library_indexer serialises it
with feature_vector_to_dict / feature_vector_from_dict (no numpy/PIL bytes;
everything is JSON-safe).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Vocabularies (closed enums — rejected on parse, not just warned)
# ---------------------------------------------------------------------------

VALID_CATEGORIES: frozenset[str] = frozenset(
    {"movie", "sports", "tech", "lifestyle", "money", "other"}
)

VALID_MOODS: frozenset[str] = frozenset(
    {
        "tense", "uplifting", "mysterious", "epic", "melancholic",
        "energetic", "calm", "aggressive", "romantic", "nostalgic",
        "ominous", "joyful", "neutral", "dramatic", "playful", "sinister",
    }
)

VALID_ENERGY: frozenset[str] = frozenset({"low", "medium", "high"})

VALID_LIGHTING: frozenset[str] = frozenset(
    {"low-key", "high-key", "natural", "neon", "golden-hour", "mixed"}
)

VALID_SHOT_TYPES: frozenset[str] = frozenset(
    {"wide", "medium", "close-up", "extreme-close-up", "aerial", "overhead", "two-shot"}
)

VALID_CAMERA_MOTION: frozenset[str] = frozenset(
    {"static", "pan", "tilt", "dolly", "handheld", "tracking", "zoom"}
)

VALID_DEPTH_OF_FIELD: frozenset[str] = frozenset({"deep", "shallow"})


class FeatureVectorError(ValueError):
    """Raised when a FeatureVector is constructed from invalid input. Used by
    the test suite to confirm the validator rejects unknown enums."""


@dataclass
class FeatureVector:
    """Multi-dimensional description of a single video clip / span.

    See CONTRACT.md — schema here must match the file 1:1.
    """

    # Subject matter (compatible with existing campeditor broll tags)
    subjects: list[str] = field(default_factory=list)
    setting: list[str] = field(default_factory=list)
    action: list[str] = field(default_factory=list)
    category: str = "other"
    query: str = ""

    # Vibe / aesthetic (NEW)
    mood: list[str] = field(default_factory=list)
    energy: str = "medium"
    lighting: str = "natural"
    color_palette: list[str] = field(default_factory=list)

    # Cinematography (NEW)
    shot_type: str = "wide"
    camera_motion: str = "static"
    depth_of_field: str = "deep"

    # Quantitative features (OpenCV-derived)
    palette_warmth: float = 0.0
    palette_saturation: float = 0.0
    palette_brightness: float = 0.0
    motion_intensity: float = 0.0
    contrast: float = 0.0
    edge_density: float = 0.0

    # Provenance
    confidence: float = 0.0
    source: str = "library"
    media_path: str = ""

    # ---- normalisation helpers ----
    def normalised(self) -> "FeatureVector":
        """Return a copy with all string-list fields lowercased / trimmed and
        nested lists capped to their documented sizes. The on-disk cache uses
        normalised vectors so case differences never desync the index."""
        cap_subjects = self.subjects[:3]
        cap_setting = self.setting[:2]
        cap_action = self.action[:2]
        cap_mood = self.mood[:3]
        cap_palette = self.color_palette[:3]
        return FeatureVector(
            subjects=[s.strip().lower() for s in cap_subjects if s and s.strip()],
            setting=[s.strip().lower() for s in cap_setting if s and s.strip()],
            action=[s.strip().lower() for s in cap_action if s and s.strip()],
            category=(self.category or "other").strip().lower() or "other",
            query=" ".join((self.query or "").split()),
            mood=[m.strip().lower() for m in cap_mood if m and m.strip()],
            energy=(self.energy or "medium").strip().lower() or "medium",
            lighting=(self.lighting or "natural").strip().lower() or "natural",
            color_palette=[
                c.strip().lower() for c in cap_palette if c and c.strip()
            ],
            shot_type=(self.shot_type or "wide").strip().lower() or "wide",
            camera_motion=(self.camera_motion or "static").strip().lower() or "static",
            depth_of_field=(self.depth_of_field or "deep").strip().lower() or "deep",
            palette_warmth=_clip01(self.palette_warmth),
            palette_saturation=_clip01(self.palette_saturation),
            palette_brightness=_clip01(self.palette_brightness),
            motion_intensity=_clip01(self.motion_intensity),
            contrast=_clip01(self.contrast),
            edge_density=_clip01(self.edge_density),
            confidence=_clip01(self.confidence),
            source=(self.source or "library").strip().lower() or "library",
            media_path=str(self.media_path or ""),
        )

    def validate(self, *, strict: bool = False) -> list[str]:
        """Return a list of human-readable validation errors. Empty = OK.

        When `strict=True`, raises FeatureVectorError on the first error
        instead of accumulating. Callers should default to non-strict so the
        indexer can log-and-skip a bad row without aborting the whole run.
        """
        errors: list[str] = []
        if self.category not in VALID_CATEGORIES:
            errors.append(
                f"category {self.category!r} not in {sorted(VALID_CATEGORIES)}"
            )
        if self.energy not in VALID_ENERGY:
            errors.append(f"energy {self.energy!r} not in {sorted(VALID_ENERGY)}")
        if self.lighting not in VALID_LIGHTING:
            errors.append(
                f"lighting {self.lighting!r} not in {sorted(VALID_LIGHTING)}"
            )
        if self.shot_type not in VALID_SHOT_TYPES:
            errors.append(
                f"shot_type {self.shot_type!r} not in {sorted(VALID_SHOT_TYPES)}"
            )
        if self.camera_motion not in VALID_CAMERA_MOTION:
            errors.append(
                f"camera_motion {self.camera_motion!r} not in {sorted(VALID_CAMERA_MOTION)}"
            )
        if self.depth_of_field not in VALID_DEPTH_OF_FIELD:
            errors.append(
                f"depth_of_field {self.depth_of_field!r} not in {sorted(VALID_DEPTH_OF_FIELD)}"
            )
        for m in self.mood:
            if m not in VALID_MOODS:
                errors.append(f"mood {m!r} not in {sorted(VALID_MOODS)}")
        for field_name, cap in (
            ("subjects", 3),
            ("setting", 2),
            ("action", 2),
            ("mood", 3),
            ("color_palette", 3),
        ):
            value = getattr(self, field_name)
            if len(value) > cap:
                errors.append(f"{field_name} has {len(value)} items (cap {cap})")
        for float_field in (
            "palette_warmth", "palette_saturation", "palette_brightness",
            "motion_intensity", "contrast", "edge_density", "confidence",
        ):
            v = getattr(self, float_field)
            if not _is_finite_unit(v):
                errors.append(f"{float_field}={v!r} is not a finite number in [0,1]")
        if self.source not in {"library", "reference", "youtube"}:
            errors.append(f"source {self.source!r} not in library|reference|youtube")
        if errors and strict:
            raise FeatureVectorError("; ".join(errors))
        return errors


# ---------------------------------------------------------------------------
# Factories / serialisation
# ---------------------------------------------------------------------------


def empty_feature_vector(media_path: str = "", *, source: str = "library") -> FeatureVector:
    """The default-empty FeatureVector returned when vision fails entirely.

    Centralised so callers (vibe_extractor, tests) all agree on the
    sentinel. `confidence=0.0` means 'no signal' — downstream matchers can
    treat this as a last-resort clip."""
    return FeatureVector(
        category="other",
        energy="medium",
        lighting="natural",
        shot_type="wide",
        camera_motion="static",
        depth_of_field="deep",
        confidence=0.0,
        source=source,
        media_path=str(media_path or ""),
    )


def feature_vector_to_dict(fv: FeatureVector) -> dict[str, Any]:
    """Plain-dict serialisation for the on-disk JSON cache. Round-trips with
    `feature_vector_from_dict`."""
    return asdict(fv.normalised())


def feature_vector_from_dict(d: dict[str, Any]) -> FeatureVector:
    """Inverse of `feature_vector_to_dict`. Tolerates extra keys (forward
    compat) and missing keys (backward compat with v0 indexes). Unknown enum
    values fall back to the empty-vector default rather than raising —
    calibration / schema drift between LLM training cuts and our vocab must
    NOT brick a library build."""
    if not isinstance(d, dict):
        raise FeatureVectorError(f"expected dict, got {type(d).__name__}")
    defaults = empty_feature_vector()
    return FeatureVector(
        subjects=_list_or(d.get("subjects"), defaults.subjects, cap=3),
        setting=_list_or(d.get("setting"), defaults.setting, cap=2),
        action=_list_or(d.get("action"), defaults.action, cap=2),
        category=_enum_or(d.get("category"), VALID_CATEGORIES, defaults.category),
        query=str(d.get("query") or ""),
        mood=_intersect_or(d.get("mood"), VALID_MOODS, defaults.mood, cap=3),
        energy=_enum_or(d.get("energy"), VALID_ENERGY, defaults.energy),
        lighting=_enum_or(d.get("lighting"), VALID_LIGHTING, defaults.lighting),
        color_palette=_list_or(d.get("color_palette"), defaults.color_palette, cap=3),
        shot_type=_enum_or(d.get("shot_type"), VALID_SHOT_TYPES, defaults.shot_type),
        camera_motion=_enum_or(
            d.get("camera_motion"), VALID_CAMERA_MOTION, defaults.camera_motion
        ),
        depth_of_field=_enum_or(
            d.get("depth_of_field"), VALID_DEPTH_OF_FIELD, defaults.depth_of_field
        ),
        palette_warmth=_finite_or(d.get("palette_warmth"), defaults.palette_warmth),
        palette_saturation=_finite_or(
            d.get("palette_saturation"), defaults.palette_saturation
        ),
        palette_brightness=_finite_or(
            d.get("palette_brightness"), defaults.palette_brightness
        ),
        motion_intensity=_finite_or(
            d.get("motion_intensity"), defaults.motion_intensity
        ),
        contrast=_finite_or(d.get("contrast"), defaults.contrast),
        edge_density=_finite_or(d.get("edge_density"), defaults.edge_density),
        confidence=_finite_or(d.get("confidence"), defaults.confidence),
        source=_enum_or(
            d.get("source"), {"library", "reference", "youtube"}, defaults.source
        ),
        media_path=str(d.get("media_path") or ""),
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _clip01(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _is_finite_unit(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0


def _list_or(value: Any, fallback: list[str], *, cap: int) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)
    out: list[str] = []
    for v in value:
        if not isinstance(v, str):
            continue
        s = v.strip().lower()
        if s:
            out.append(s)
        if len(out) >= cap:
            break
    return out


def _intersect_or(value: Any, vocab: frozenset[str], fallback: list[str], *, cap: int) -> list[str]:
    """Like _list_or, but drops any item not in the closed vocabulary. Used
    for `mood` so a hallucinated LLM word doesn't poison downstream
    similarity math."""
    if not isinstance(value, list):
        return list(fallback)
    out: list[str] = []
    seen: set[str] = set()
    for v in value:
        if not isinstance(v, str):
            continue
        s = v.strip().lower()
        if s in vocab and s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= cap:
            break
    return out


def _enum_or(value: Any, vocab: frozenset[str] | set[str], fallback: str) -> str:
    if isinstance(value, str):
        s = value.strip().lower()
        if s in vocab:
            return s
    return fallback


def _finite_or(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return _clip01(float(value))
    return fallback
