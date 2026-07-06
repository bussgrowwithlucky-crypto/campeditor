"""Tests for matcher.rank_candidates.

All offline — no API calls, no ffmpeg. The matcher ships its own
self-contained scoring formula (it doesn't depend on the YouTube
fallback's formula), so we test the matcher's contract here:

  * identical vectors score high
  * orthogonal vectors score low
  * used_clips set is respected
  * top_k gates results
  * missing fields degrade gracefully
  * top-K stability across larger candidate sets
  * compatibility tables influence ranking (lighting + shot_type + camera_motion)
  * empty-sentinel vectors (confidence==0) sort to the back
"""

from __future__ import annotations

from pathlib import Path

import pytest

from broll_intelligence.feature_vector import (
    FeatureVector,
    empty_feature_vector,
)
from broll_intelligence.matcher import (
    LIGHTING_COMPATIBLE,
    SHOT_TYPE_COMPATIBLE,
    rank_candidates,
    score_clip,
)


def _fv(**kw) -> FeatureVector:
    """Shorthand FeatureVector builder with sensible defaults."""
    return FeatureVector(
        subjects=kw.get("subjects", []),
        setting=kw.get("setting", []),
        action=kw.get("action", []),
        category=kw.get("category", "movie"),
        query=kw.get("query", ""),
        mood=kw.get("mood", []),
        energy=kw.get("energy", "medium"),
        lighting=kw.get("lighting", "natural"),
        color_palette=kw.get("color_palette", []),
        shot_type=kw.get("shot_type", "wide"),
        camera_motion=kw.get("camera_motion", "static"),
        depth_of_field=kw.get("depth_of_field", "deep"),
        palette_warmth=kw.get("palette_warmth", 0.5),
        palette_saturation=kw.get("palette_saturation", 0.5),
        palette_brightness=kw.get("palette_brightness", 0.5),
        motion_intensity=kw.get("motion_intensity", 0.5),
        contrast=kw.get("contrast", 0.5),
        edge_density=kw.get("edge_density", 0.5),
        confidence=kw.get("confidence", 1.0),
        source=kw.get("source", "library"),
        media_path=kw.get("media_path", ""),
    )


def _entry(name: str, fv: FeatureVector, tmp_path: Path) -> tuple[Path, FeatureVector]:
    p = tmp_path / name
    p.write_bytes(b"x")
    fv.media_path = str(p)
    return (p, fv)


# ---------------------------------------------------------------------------
# Identical / orthogonal / matching
# ---------------------------------------------------------------------------


def test_identical_vectors_score_high(tmp_path: Path):
    ref = _fv(subjects=["car"], category="movie", mood=["tense"])
    cand = _entry(
        "clip.mp4", _fv(subjects=["car"], category="movie", mood=["tense"]), tmp_path
    )
    out = rank_candidates(ref, [cand], top_k=1)
    assert len(out) == 1
    _, _, score = out[0]
    assert score > 0.7, f"identical vectors should score high, got {score}"


def test_orthogonal_vectors_score_low(tmp_path: Path):
    ref = _fv(
        subjects=["astronaut"], category="movie", mood=["mysterious"],
        lighting="low-key", color_palette=["deep blue"],
    )
    cand = _entry(
        "bright_comedy.mp4",
        _fv(
            subjects=["dog"], category="lifestyle", mood=["joyful"],
            lighting="high-key", color_palette=["yellow"],
            palette_warmth=0.9, palette_brightness=0.9, motion_intensity=0.9,
        ),
        tmp_path,
    )
    out = rank_candidates(ref, [cand], top_k=1)
    assert out[0][2] < 0.5


def test_matching_subjects_and_mood_score_high(tmp_path: Path):
    ref = _fv(
        subjects=["astronaut", "lunar rover"], category="movie",
        mood=["mysterious", "epic"], lighting="low-key",
        color_palette=["deep blue", "silver"],
    )
    good = _entry(
        "good.mp4",
        _fv(
            subjects=["astronaut"], category="movie",
            mood=["mysterious"], lighting="low-key",
            color_palette=["deep blue"],
            palette_warmth=0.2, palette_saturation=0.4, palette_brightness=0.3,
            motion_intensity=0.2, contrast=0.6,
        ),
        tmp_path,
    )
    out = rank_candidates(ref, [good], top_k=1)
    assert out[0][2] > 0.6


# ---------------------------------------------------------------------------
# used_clips / top_k
# ---------------------------------------------------------------------------


def test_used_clips_are_excluded(tmp_path: Path):
    ref = _fv(subjects=["car"], category="movie")
    a = _entry("a.mp4", _fv(subjects=["car"], category="movie"), tmp_path)
    b = _entry("b.mp4", _fv(subjects=["car"], category="movie"), tmp_path)
    used = {a[0]}
    out = rank_candidates(ref, [a, b], top_k=5, used_clips=used)
    paths = {p for p, _, _ in out}
    assert a[0] not in paths
    assert b[0] in paths


def test_top_k_caps_results(tmp_path: Path):
    ref = _fv(subjects=["x"], category="movie")
    entries = [
        _entry(f"c{i}.mp4", _fv(subjects=["x"], category="movie"), tmp_path)
        for i in range(10)
    ]
    out = rank_candidates(ref, entries, top_k=3)
    assert len(out) == 3
    scores = [s for _, _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_top_k_zero_returns_empty(tmp_path: Path):
    ref = _fv(subjects=["x"], category="movie")
    cand = _entry("c.mp4", _fv(subjects=["x"], category="movie"), tmp_path)
    out = rank_candidates(ref, [cand], top_k=0)
    assert out == []


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_missing_fields_dont_crash(tmp_path: Path):
    """An empty-vibe FeatureVector on the candidate should still rank
    sensibly (subject-only match) without crashing."""
    ref = _fv(subjects=["car"], category="movie", mood=["tense"])
    sparse = empty_feature_vector(media_path="sparse", source="library")
    sparse.subjects = ["car"]
    sparse.category = "movie"
    sparse.confidence = 0.5
    cand = _entry("sparse.mp4", sparse, tmp_path)
    out = rank_candidates(ref, [cand], top_k=1)
    assert len(out) == 1
    _, _, score = out[0]
    assert 0.0 <= score <= 1.0


def test_all_zero_vectors_dont_crash():
    a = _fv()
    b = _fv()
    s = score_clip(a, b)
    assert 0.0 <= s <= 1.0


def test_empty_index_returns_empty():
    ref = _fv(subjects=["car"])
    assert rank_candidates(ref, [], top_k=5) == []


def test_empty_sentinel_pushed_to_back(tmp_path: Path):
    """A confidence==0 (empty-sentinel) clip should appear AFTER any
    signal-bearing clip, even if its raw score happens to be higher."""
    ref = _fv(subjects=["car"], category="movie")
    real = _entry(
        "real.mp4",
        _fv(subjects=["car"], category="movie", confidence=0.9),
        tmp_path,
    )
    empty = _entry(
        "empty.mp4", empty_feature_vector(media_path="empty", source="library"), tmp_path
    )
    out = rank_candidates(ref, [empty, real], top_k=2)
    paths = [p.name for p, _, _ in out]
    assert paths[0] == "real.mp4"
    assert paths[1] == "empty.mp4"


# ---------------------------------------------------------------------------
# Top-K stability
# ---------------------------------------------------------------------------


def test_top_k_stable_across_pool_sizes(tmp_path: Path):
    ref = _fv(
        subjects=["astronaut"], category="movie", mood=["mysterious"], lighting="low-key",
    )
    good = _entry(
        "good.mp4",
        _fv(subjects=["astronaut"], category="movie", mood=["mysterious"], lighting="low-key"),
        tmp_path,
    )
    medium = _entry(
        "medium.mp4",
        _fv(subjects=["astronaut"], category="movie"),
        tmp_path,
    )
    third = _entry(
        "third.mp4",
        _fv(subjects=["astronaut"], category="movie", mood=["mysterious"]),
        tmp_path,
    )
    filler = [
        _entry(
            f"f{i}.mp4",
            _fv(subjects=["dog"], category="lifestyle", mood=["joyful"]),
            tmp_path,
        )
        for i in range(20)
    ]
    small = rank_candidates(ref, [good, medium, third], top_k=3)
    large = rank_candidates(ref, [good, medium, third] + filler, top_k=3)
    small_names = [p.name for p, _, _ in small]
    large_names = [p.name for p, _, _ in large]
    assert small_names == large_names, (
        f"top-3 mismatch: small={small_names} large={large_names}"
    )


# ---------------------------------------------------------------------------
# Compatibility tables influence ranking
# ---------------------------------------------------------------------------


def test_lighting_compatible_ranks_above_incompatible(tmp_path: Path):
    """Reference is `low-key`. Per the compatibility table, `natural` is
    a compatible pair while `high-key` is not."""
    ref = _fv(
        lighting="low-key", category="other", shot_type="wide",
        camera_motion="static", depth_of_field="deep", energy="medium",
    )
    natural = _entry("natural.mp4", _fv(lighting="natural"), tmp_path)
    highkey = _entry("highkey.mp4", _fv(lighting="high-key"), tmp_path)
    out = rank_candidates(ref, [highkey, natural], top_k=2)
    paths = [p.name for p, _, _ in out]
    assert paths.index("natural.mp4") < paths.index("highkey.mp4"), (
        f"expected natural (compatible) before highkey (incompatible), got {paths}"
    )


def test_shot_type_adjacent_ranks_above_far(tmp_path: Path):
    """Reference is `close-up`. Per the shot-type adjacency table,
    `extreme-close-up` is a compatible pair while `aerial` is not."""
    ref = _fv(shot_type="close-up")
    ecu = _entry("extreme-close-up.mp4", _fv(shot_type="extreme-close-up"), tmp_path)
    aerial = _entry("aerial.mp4", _fv(shot_type="aerial"), tmp_path)
    out = rank_candidates(ref, [aerial, ecu], top_k=2)
    paths = [p.name for p, _, _ in out]
    assert paths.index("extreme-close-up.mp4") < paths.index("aerial.mp4")


def test_camera_motion_match_ranks_above_mismatch(tmp_path: Path):
    ref = _fv(camera_motion="static")
    match = _entry("static.mp4", _fv(camera_motion="static"), tmp_path)
    mismatch = _entry("dolly.mp4", _fv(camera_motion="dolly"), tmp_path)
    out = rank_candidates(ref, [mismatch, match], top_k=2)
    paths = [p.name for p, _, _ in out]
    assert paths.index("static.mp4") < paths.index("dolly.mp4")


# ---------------------------------------------------------------------------
# Compatibility tables are well-formed
# ---------------------------------------------------------------------------


def test_lighting_compatible_table_is_non_empty():
    assert len(LIGHTING_COMPATIBLE) > 0
    assert ("low-key", "natural") in LIGHTING_COMPATIBLE


def test_shot_type_compatible_table_is_non_empty():
    assert len(SHOT_TYPE_COMPATIBLE) > 0
    assert ("close-up", "extreme-close-up") in SHOT_TYPE_COMPATIBLE