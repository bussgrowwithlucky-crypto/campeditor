"""Tests for vibe_extractor.extract_from_video.

All offline — the vision ladder is monkeypatched via the
`vibe_extractor.vision_call` injection point. ffmpeg is invoked for the
real flow (because extract_from_video runs ffmpeg on the source to grab
sample frames) but only against the tiny fake mp4s we generate in
conftest.make_fake_video; ffmpeg's per-call cost is well under a second.

We cover:
  * happy path: vision returns the full canned JSON, FeatureVector is
    populated, confidence > 0.5.
  * empty vision: ladder returns "", we get empty_feature_vector().
  * broken JSON: ladder returns "{ this is not JSON", we get
    empty_feature_vector() (no crash).
  * unknown enum values from the model: get coerced to defaults.
  * robustness: ladder raising an exception → empty vector.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from broll_intelligence import (
    FeatureVector,
    empty_feature_vector,
    extract_from_video,
)
from broll_intelligence.config import Settings
from broll_intelligence.feature_vector import (
    VALID_CATEGORIES,
    VALID_LIGHTING,
    FeatureVectorError,
    feature_vector_from_dict,
)
from broll_intelligence.vibe_extractor import (
    VISION_PROMPT,
    _parse_vision_json,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_populates_feature_vector(
    video_files: list[Path],
    index_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    """When vision returns the full canned JSON, every field lands on the
    FeatureVector with the right normalisation, and confidence reflects
    the completeness blend."""
    from broll_intelligence import vibe_extractor

    sample = {
        "subjects": ["Astronaut", "LUNAR ROVER"],   # uppercase should be normalised
        "setting": ["Lunar Surface"],
        "action": ["Walking"],
        "category": "movie",
        "query": "astronaut walking on lunar surface",
        "mood": ["mysterious", "epic", "uplifting"],
        "energy": "low",
        "lighting": "low-key",
        "color_palette": ["deep blue", "silver"],
        "shot_type": "wide",
        "camera_motion": "tracking",
        "depth_of_field": "deep",
    }
    monkeypatch.setattr(
        vibe_extractor, "vision_call", _caller(sample), raising=False
    )

    fv = extract_from_video(video_files[0], index_settings)

    # Subject matter normalisation
    assert fv.subjects == ["astronaut", "lunar rover"]
    assert fv.setting == ["lunar surface"]
    assert fv.action == ["walking"]
    assert fv.category == "movie"
    assert fv.query == "astronaut walking on lunar surface"

    # Vibe / aesthetic
    assert set(fv.mood) == {"mysterious", "epic", "uplifting"}
    assert fv.energy == "low"
    assert fv.lighting == "low-key"
    assert fv.color_palette == ["deep blue", "silver"]

    # Cinematography
    assert fv.shot_type == "wide"
    assert fv.camera_motion == "tracking"
    assert fv.depth_of_field == "deep"

    # Quantitative fields are present (we can't assert exact values because
    # the fake mp4 decodes differently each run, but they must be valid floats
    # in [0, 1]).
    for f in (
        "palette_warmth",
        "palette_saturation",
        "palette_brightness",
        "motion_intensity",
        "contrast",
        "edge_density",
    ):
        assert 0.0 <= getattr(fv, f) <= 1.0, f"{f} out of range: {getattr(fv, f)}"

    # Provenance
    assert fv.source == "library"
    assert fv.media_path == str(video_files[0].resolve())
    # Confidence: 12/12 vision fields + all numerics → 1.0
    assert fv.confidence == pytest.approx(1.0, abs=1e-6)


def test_extract_uses_middle_frame_for_vision(
    video_files: list[Path],
    index_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    """Vision must be called on the middle frame (50%), not the first/last.
    We assert this by recording the image_path the mock receives and
    verifying it sorts to the middle of the sampled frames."""
    from broll_intelligence import vibe_extractor

    captured: list[Path] = []
    sample = {"subjects": ["x"], "category": "movie", "mood": ["calm"],
              "energy": "low", "lighting": "natural", "shot_type": "wide",
              "camera_motion": "static", "depth_of_field": "deep",
              "color_palette": [], "setting": [], "action": [],
              "query": "test query"}

    def _capture(image_path, prompt, settings):
        captured.append(Path(image_path))
        return json.dumps(sample)

    monkeypatch.setattr(vibe_extractor, "vision_call", _capture, raising=False)

    extract_from_video(video_files[0], index_settings)

    assert len(captured) == 1, "vision should be called exactly once per extract"
    used_frame = captured[0]
    # The captured frame path encodes "frame-NN.jpg" — middle should be -01.
    assert used_frame.name == "frame-01.jpg", (
        f"expected middle frame, got {used_frame.name}"
    )


# ---------------------------------------------------------------------------
# Robustness — vision returns nothing or junk
# ---------------------------------------------------------------------------


def test_empty_vision_returns_empty_feature_vector(
    video_files: list[Path],
    index_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    from broll_intelligence import vibe_extractor

    monkeypatch.setattr(
        vibe_extractor, "vision_call", _caller(""), raising=False
    )
    fv = extract_from_video(video_files[0], index_settings)
    # Vision produced nothing parseable → vision half of confidence is 0.
    # CV still runs on the extracted frames so the numeric fields are
    # populated; categorical fields fall back to their defaults. Net
    # confidence = 0.0 * 0.5 + 1.0 * 0.5 = 0.5 (CV-only signal).
    assert fv.confidence == pytest.approx(0.5, abs=1e-6)
    assert fv.source == "library"
    assert fv.media_path == str(video_files[0].resolve())
    # Categorical defaults when vision gives nothing.
    assert fv.category == "other"
    assert fv.energy == "medium"
    assert fv.lighting == "natural"
    assert fv.shot_type == "wide"
    assert fv.camera_motion == "static"
    assert fv.depth_of_field == "deep"
    assert fv.subjects == []
    assert fv.mood == []
    assert fv.color_palette == []
    # CV is independent of vision: at least one numeric must be non-zero on
    # a real extracted frame.
    cv_values = [
        fv.palette_warmth, fv.palette_saturation, fv.palette_brightness,
        fv.contrast, fv.edge_density, fv.motion_intensity,
    ]
    assert any(0.0 < v <= 1.0 for v in cv_values), (
        f"expected at least one CV feature populated, got {cv_values}"
    )


def test_broken_json_returns_empty_feature_vector(
    video_files: list[Path],
    index_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    from broll_intelligence import vibe_extractor

    monkeypatch.setattr(
        vibe_extractor,
        "vision_call",
        _caller("{ this is not JSON at all"),
        raising=False,
    )
    fv = extract_from_video(video_files[0], index_settings)
    # Same shape as the empty-vision case: unparseable JSON is treated as
    # "no vision signal", confidence drops to 0.5, CV populates independently.
    assert fv.confidence == pytest.approx(0.5, abs=1e-6)
    assert fv.category == "other"
    assert fv.mood == []
    assert fv.media_path == str(video_files[0].resolve())


def test_ladder_exception_returns_empty_feature_vector(
    video_files: list[Path],
    index_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    from broll_intelligence import vibe_extractor

    def _boom(image_path, prompt, settings):
        raise RuntimeError("simulated ladder outage")

    monkeypatch.setattr(vibe_extractor, "vision_call", _boom, raising=False)
    fv = extract_from_video(video_files[0], index_settings)
    # Ladder raised → no vision signal → confidence 0.5; CV still runs.
    assert fv.confidence == pytest.approx(0.5, abs=1e-6)
    assert fv.source == "library"
    assert fv.media_path == str(video_files[0].resolve())
    assert fv.subjects == []
    assert fv.mood == []
    cv_values = [
        fv.palette_warmth, fv.palette_saturation, fv.palette_brightness,
        fv.contrast, fv.edge_density, fv.motion_intensity,
    ]
    assert any(0.0 < v <= 1.0 for v in cv_values)


def test_markdown_fenced_json_is_parsed(
    video_files: list[Path],
    index_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    """Real LLMs often wrap JSON in ```json ... ``` fences. Confirm we strip."""
    from broll_intelligence import vibe_extractor

    payload = {
        "subjects": ["car"],
        "category": "movie",
        "mood": ["energetic"],
        "energy": "high",
        "lighting": "natural",
        "shot_type": "wide",
        "camera_motion": "tracking",
        "depth_of_field": "deep",
        "color_palette": [],
        "setting": [],
        "action": [],
        "query": "car driving highway at sunset",
    }
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    monkeypatch.setattr(
        vibe_extractor, "vision_call", _caller(fenced), raising=False
    )
    fv = extract_from_video(video_files[0], index_settings)
    assert fv.subjects == ["car"]
    assert fv.category == "movie"
    assert "energetic" in fv.mood


def test_unknown_enum_values_fall_back(
    video_files: list[Path],
    index_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    """LLMs occasionally invent enums. feature_vector_from_dict must coerce
    them to defaults; FeatureVector.validate(strict=True) must reject."""
    from broll_intelligence import vibe_extractor

    payload = {
        "subjects": ["x"],
        "category": "not-a-real-category",
        "lighting": "bioluminescent",        # not in vocab
        "shot_type": "drone-follow",         # not in vocab
        "mood": ["hallucinated-mood"],       # not in vocab
        "energy": "medium",
        "camera_motion": "static",
        "depth_of_field": "deep",
        "color_palette": [],
        "setting": [],
        "action": [],
        "query": "x",
    }
    monkeypatch.setattr(
        vibe_extractor, "vision_call", _caller(payload), raising=False
    )
    fv = extract_from_video(video_files[0], index_settings)
    # Unknown enums coerced to defaults (see feature_vector_from_dict).
    assert fv.category in VALID_CATEGORIES
    assert fv.lighting in VALID_LIGHTING
    assert fv.shot_type in {
        "wide", "medium", "close-up", "extreme-close-up", "aerial",
        "overhead", "two-shot",
    }
    assert fv.mood == [], "unknown mood terms must be dropped, not preserved"
    # strict=True validation catches the dropped/unknown values only if the
    # caller passed them in via direct construction — from_dict already
    # coerced them, so a built-from-dict vector validates clean.
    errors = fv.validate()
    assert errors == []


def test_validate_strict_rejects_unknown_category():
    """FeatureVector.validate(strict=True) raises on unknown enum values
    even after default construction, because the caller can mutate fields
    in place."""
    fv = FeatureVector()
    fv.category = "not-a-real-category"
    with pytest.raises(FeatureVectorError) as excinfo:
        fv.validate(strict=True)
    assert "category" in str(excinfo.value)


def test_validate_strict_rejects_unknown_lighting():
    fv = FeatureVector()
    fv.lighting = "bioluminescent"
    with pytest.raises(FeatureVectorError):
        fv.validate(strict=True)


def test_validate_strict_rejects_unknown_shot_type():
    fv = FeatureVector()
    fv.shot_type = "drone-follow"
    with pytest.raises(FeatureVectorError):
        fv.validate(strict=True)


# ---------------------------------------------------------------------------
# JSON parsing edge cases (unit-level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_subjects",
    [
        ('{"subjects":["a","b"]}', ["a", "b"]),
        ('noise before {"subjects":["a"]} noise after', ["a"]),
        ('```json\n{"subjects":["a"]}\n```', ["a"]),
        ('```\n{"subjects":["a"]}\n```', ["a"]),
        ('{"subjects":["a"]} trailing prose', ["a"]),
        ('{"a":1}', []),  # missing key → empty default
    ],
)
def test_parse_vision_json_robust(raw: str, expected_subjects: list[str]):
    parsed = _parse_vision_json(raw)
    assert parsed.get("subjects", []) == expected_subjects


def test_parse_vision_json_returns_empty_on_garbage():
    for bad in ("", "not json at all", "}{", "{", "}"):
        assert _parse_vision_json(bad) == {}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_vision_prompt_contains_required_keys():
    """The exported VISION_PROMPT must mention every field the spec
    requires downstream tasks to consume."""
    required = [
        "subjects", "setting", "action", "category", "query", "mood",
        "energy", "lighting", "color_palette", "shot_type",
        "camera_motion", "depth_of_field",
    ]
    for key in required:
        assert key in VISION_PROMPT, f"VISION_PROMPT missing key: {key}"


def test_extract_rejects_zero_duration(
    tmp_path: Path, index_settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    """A non-video file (zero duration) returns the empty vector without
    crashing — the indexer depends on this to skip bad files cleanly."""
    from broll_intelligence import vibe_extractor

    # Make a "video" with no decodable frames — an empty file is fine for
    # this test because probe_duration returns 0 before any frame extraction.
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"")

    called = {"count": 0}

    def _track(image_path, prompt, settings):
        called["count"] += 1
        return ""

    monkeypatch.setattr(vibe_extractor, "vision_call", _track, raising=False)
    fv = extract_from_video(bogus, index_settings)
    assert fv.confidence == 0.0
    assert fv.source == "library"
    assert fv.media_path == str(bogus.resolve())
    assert called["count"] == 0, "vision must not be called when probe_duration is 0"


def test_feature_vector_from_dict_tolerates_missing_keys():
    """Forward/backward compat: a v0 dict (only subject matter fields) still
    parses — missing keys fall back to defaults."""
    fv = feature_vector_from_dict({
        "subjects": ["car"],
        "category": "movie",
        "query": "car at night",
    })
    assert fv.subjects == ["car"]
    assert fv.category == "movie"
    assert fv.query == "car at night"
    assert fv.energy == "medium"
    assert fv.mood == []
    assert fv.lighting == "natural"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _caller(payload):
    """Build a vision_ladder-compatible callable that returns `payload`
    (serialised as JSON if it's a dict)."""

    def _call(image_path, prompt, settings):
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    return _call