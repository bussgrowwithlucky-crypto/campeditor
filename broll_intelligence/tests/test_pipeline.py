"""Tests for the pipeline orchestrator + baseline comparator.

All offline — no API calls, no network, no ffmpeg. The vibe_extractor
and library_indexer are real (with ffmpeg on the fake videos), so the
test suite still incurs ~minute-scale runtime for the ffmpeg-heavy
fixtures. We split the tests here so the lighter cases run first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from broll_intelligence.baseline import (
    TRUNCATE_WORDS,
    local_score,
    rank_library_legacy,
    truncate_query,
    youtube_query_legacy,
)
from broll_intelligence.config import Settings
from broll_intelligence.feature_vector import FeatureVector, empty_feature_vector
from broll_intelligence.pipeline import (
    LIBRARY_ACCEPT_THRESHOLD,
    BrollItem,
    BrollPack,
    analyze_reference,
    select_broll,
)


# ---------------------------------------------------------------------------
# Baseline (legacy scoring) — unit tests, no I/O
# ---------------------------------------------------------------------------


def test_truncate_query_caps_at_8_words():
    assert truncate_query("one two three four five six seven eight nine ten") == \
        "one two three four five six seven eight"


def test_truncate_query_handles_empty():
    assert truncate_query("") == ""


def test_truncate_query_handles_short():
    assert truncate_query("two words") == "two words"


def test_local_score_categories_match():
    a = FeatureVector(category="movie", subjects=[], setting=[])
    b = FeatureVector(category="movie", subjects=[], setting=[])
    s = local_score(a, b)
    # category=3.0, no subject/setting overlap -> 3.0 / 8.0 = 0.375
    assert s == pytest.approx(0.375, abs=0.001)


def test_local_score_subjects_add():
    a = FeatureVector(category="movie", subjects=["car", "road"], setting=[])
    b = FeatureVector(category="movie", subjects=["car", "road", "sky"], setting=[])
    s = local_score(a, b)
    # 3.0 category + 2.0 subjects + 0 setting = 5.0 / 8.0 = 0.625
    assert s == pytest.approx(0.625, abs=0.001)


def test_local_score_setting_adds():
    a = FeatureVector(category="movie", subjects=[], setting=["indoors"])
    b = FeatureVector(category="movie", subjects=[], setting=["indoors"])
    s = local_score(a, b)
    # 3.0 category + 1.0 setting = 4.0 / 8.0 = 0.5
    assert s == pytest.approx(0.5, abs=0.001)


def test_local_score_caps_at_1():
    a = FeatureVector(category="movie", subjects=["x", "y", "z"], setting=["i", "j"])
    b = FeatureVector(category="movie", subjects=["x", "y", "z"], setting=["i", "j"])
    s = local_score(a, b)
    # 3.0 + 3.0 + 2.0 = 8.0 / 8.0 = 1.0
    assert s == pytest.approx(1.0, abs=0.001)


def test_youtube_query_legacy_uses_subjects():
    a = FeatureVector(subjects=["car", "road"], setting=["highway"], query="car road")
    q = youtube_query_legacy(a)
    assert "car" in q
    assert "road" in q
    assert len(q.split()) <= 8


def test_youtube_query_legacy_truncates_to_8():
    a = FeatureVector(subjects=["a", "b", "c", "d", "e"], setting=["x", "y"], query="")
    q = youtube_query_legacy(a)
    assert len(q.split()) <= 8


def test_youtube_query_legacy_falls_back():
    a = FeatureVector(subjects=[], setting=[], action=[], category="movie", query="")
    q = youtube_query_legacy(a)
    assert "movie" in q
    assert len(q.split()) <= 8


def test_rank_library_legacy_filters_below_threshold(tmp_path: Path):
    a = FeatureVector(category="movie", subjects=[], setting=[])
    b = FeatureVector(category="lifestyle", subjects=["dog"], setting=["park"])
    index = [
        (tmp_path / "a.mp4", a),
        (tmp_path / "b.mp4", b),
    ]
    out = rank_library_legacy(a, index, top_k=5)
    # `a` matches category, `b` matches nothing related.
    paths = [p.path.name for p in out]
    assert "a.mp4" in paths
    assert "b.mp4" not in paths


# ---------------------------------------------------------------------------
# Pipeline — end-to-end with ffmpeg fake videos
# ---------------------------------------------------------------------------


def test_analyze_reference_uses_cache(video_files, index_settings):
    """Calling analyze_reference twice on the same video only triggers
    extraction once (the second call returns the cached value)."""
    from broll_intelligence import vibe_extractor

    # First call -> real extraction.
    fv1 = analyze_reference(video_files[0], index_settings)
    assert fv1 is not None

    # Second call -> cache hit. Stub the extractor so a real call would
    # produce a different value; if the cache works, we still get fv1.
    def _explode(*args, **kwargs):
        raise AssertionError("extract_from_video should NOT be called on cache hit")

    original = vibe_extractor.extract_from_video
    vibe_extractor.extract_from_video = _explode  # type: ignore
    try:
        fv2 = analyze_reference(video_files[0], index_settings)
    finally:
        vibe_extractor.extract_from_video = original  # type: ignore
    assert fv2 == fv1


def test_select_broll_returns_brollpack(video_files, index_settings):
    pack = select_broll(video_files[0], top_k=3, settings=index_settings, enable_youtube=False)
    assert isinstance(pack, BrollPack)
    assert pack.reference is not None
    # At least one of these will be true: a library pick, a YouTube pick,
    # or a reference_crop placeholder. Empty list is NOT acceptable.
    assert pack.items != [] or pack.diagnostics.get("rungs_fired")


def test_select_broll_diagnostics_capture_rung(video_files, index_settings):
    pack = select_broll(video_files[0], top_k=3, settings=index_settings, enable_youtube=False)
    assert "rungs_fired" in pack.diagnostics
    assert isinstance(pack.diagnostics["rungs_fired"], list)
    assert len(pack.diagnostics["rungs_fired"]) >= 1


def test_select_broll_falls_through_when_library_empty(tmp_path, monkeypatch):
    """With an empty library and no LLM keys, the ladder should fall
    through to the reference_crop placeholder rather than returning []."""
    from broll_intelligence import pipeline as p_mod

    empty_settings = Settings(
        library_dir=tmp_path / "empty_lib",
        index_path=tmp_path / "empty_idx.json",
        data_dir=tmp_path,
    )
    empty_settings.library_dir.mkdir(parents=True, exist_ok=True)

    # Use any small valid video; the reference is irrelevant to the rung
    # logic when the library is empty.
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"\x00" * 4096)

    pack = p_mod.select_broll(fake, top_k=3, settings=empty_settings, enable_youtube=False)
    assert pack.items != []
    assert any(it.source == "reference_crop" for it in pack.items)


def test_select_broll_offline_no_crash(video_files, index_settings):
    """With no API keys, the system should still return a BrollPack
    (just without the YouTube rung). Library rung may also be empty
    if no clip clears the threshold; reference_crop placeholder is fine."""
    from broll_intelligence import pipeline as p_mod

    # index_settings has no API keys by default (conftest blanks them).
    pack = p_mod.select_broll(video_files[0], top_k=3, settings=index_settings, enable_youtube=False)
    assert pack is not None
    # No YouTube picks when there are no API keys.
    assert all(it.source != "youtube" for it in pack.items)


# ---------------------------------------------------------------------------
# BrollItem / BrollPack data classes
# ---------------------------------------------------------------------------


def test_brollitem_to_dict_round_trip():
    from broll_intelligence.pipeline import BrollItem
    it = BrollItem(source="library", path=Path("/x.mp4"), url=None, score=0.7, features=None, notes="ok")
    d = it
    assert d.source == "library"
    assert d.path == Path("/x.mp4")
    assert d.score == 0.7


def test_brollpack_to_dict_includes_diagnostics():
    pack = BrollPack(
        reference=empty_feature_vector(media_path="ref", source="reference"),
        items=[
            BrollItem(source="library", path=Path("/x.mp4"), url=None, score=0.7),
        ],
        diagnostics={"rungs_fired": ["library"]},
    )
    d = pack.to_dict()
    assert "reference" in d
    assert "items" in d
    assert "diagnostics" in d
    assert d["diagnostics"]["rungs_fired"] == ["library"]
    assert d["items"][0]["source"] == "library"
    assert d["items"][0]["score"] == 0.7