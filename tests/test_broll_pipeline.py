"""Unit tests for the rewritten B-roll pipeline (app/broll.py). No network:
every test either exercises pure functions or monkeypatches the network/LLM
edges so a developer's real .env can never make a test issue a live call.
"""
import urllib.error

from app import broll
from app.config import Settings
from app.models import ReferenceAnalysis


def _settings(tmp_path) -> Settings:
    settings = Settings(
        groq_api_key="",
        llm_api_key="",
        nvidia_api_key="",
        nvidia_fallback_api_key="",
        nvidia_fallback_api_key_2="",
        nvidia_fallback_api_key_3="",
        gemini_api_key="",
        youtube_data_api_key="",
        youtube_data_api_key_2="",
        ollama_vision_model="",
        ollama_text_model="",
    )
    settings.data_dir = tmp_path
    settings.broll_library_dir = tmp_path / "no_library"
    return settings


def _clip(tmp_path, name, category, subjects, setting, folder="misc") -> broll.LibraryClip:
    path = tmp_path / name
    path.write_bytes(b"x")
    return broll.LibraryClip(
        path=path, mtime=0.0, size=1, subjects=subjects, setting=setting, category=category, folder=folder,
    )


# ---------------------------------------------------------------------------
# Span detection
# ---------------------------------------------------------------------------


def test_spans_from_shot_flags_splits_on_hard_cuts():
    primary_flags = (
        [False] * 18  # B-roll shot: 0.0-1.8
        + [False] * 12  # B-roll shot: 1.8-3.0
        + [True] * 21  # talking head: 3.0-5.1
        + [False] * 9  # B-roll: 5.1-6.0
        + [False] * 10  # B-roll: 6.0-7.0
        + [True] * 20
    )
    visual_flags = [True] * len(primary_flags)
    spans = broll._spans_from_shot_flags(primary_flags, visual_flags, cut_indices=[18, 30, 51, 60, 70])
    assert spans == [(0.0, 1.8), (1.8, 3.0), (5.1, 6.0), (6.0, 7.0)]


def test_spans_from_flags_fills_single_frame_gap():
    # A single missed frame surrounded by True on both sides is bridged, so
    # a real cutaway isn't split into two spurious tiny spans.
    assert broll._spans_from_flags([True, False, True, True]) == [(0.0, 0.4)]


def test_spans_from_flags_all_false_yields_no_spans():
    assert broll._spans_from_flags([False, False, False, False]) == []


# ---------------------------------------------------------------------------
# _align_span_to_clip
# ---------------------------------------------------------------------------


def test_align_span_unchanged_when_within_bounds():
    assert broll._align_span_to_clip(1.2, 1.3, 15.0, 15.0) == (1.2, 1.3)


def test_align_span_scales_proportionally_when_overflowing():
    s, e = broll._align_span_to_clip(8.0, 12.0, 12.0, 10.0)
    assert abs(s - 6.6667) < 0.01
    assert e == 10.0


def test_align_span_clips_tail_to_clip_duration():
    s, e = broll._align_span_to_clip(11.0, 12.0, 12.0, 10.0)
    assert e == 10.0
    assert s < e


# ---------------------------------------------------------------------------
# Local matching
# ---------------------------------------------------------------------------


def test_match_local_prefers_category_match(tmp_path):
    settings = _settings(tmp_path)
    profile = broll.SpanProfile(
        start=0, end=1, subjects=["ball"], setting=["stadium"], category="sports", query="basketball",
    )
    clips = [
        _clip(tmp_path, "a.mp4", category="movie", subjects=["car"], setting=["street"], folder="Good Stuff"),
        _clip(tmp_path, "b.mp4", category="sports", subjects=["hoop"], setting=["court"], folder="NBA_Clips"),
    ]
    picked = broll.match_local(profile, clips, set(), settings)
    assert picked is not None
    clip, _vibe, _cont, _score = picked
    assert clip.category == "sports"


def test_match_local_matches_subject_overlap_across_settings(tmp_path):
    """'person on laptop in a house' should match 'person on PC in a house'
    (semantic sense, not pixel-exact) — same category + subject overlap wins
    over an unrelated clip even though neither library entry is an exact
    wording match."""
    settings = _settings(tmp_path)
    profile = broll.SpanProfile(
        start=0, end=1, subjects=["person", "laptop"], setting=["home"],
        category="tech", query="person on laptop at home",
    )
    clips = [
        _clip(tmp_path, "a.mp4", category="tech", subjects=["person", "pc"], setting=["home"], folder="Miscellaneous"),
        _clip(tmp_path, "b.mp4", category="sports", subjects=["ball"], setting=["stadium"], folder="NBA_Clips"),
    ]
    picked = broll.match_local(profile, clips, set(), settings)
    assert picked is not None
    clip, _vibe, _cont, _score = picked
    assert clip.path.name == "a.mp4"


def test_match_local_returns_none_when_nothing_close(tmp_path):
    settings = _settings(tmp_path)
    profile = broll.SpanProfile(start=0, end=1, subjects=["yacht"], setting=["ocean"], category="lifestyle")
    clips = [_clip(tmp_path, "a.mp4", category="sports", subjects=["ball"], setting=["stadium"], folder="NBA_Clips")]
    assert broll.match_local(profile, clips, set(), settings) is None


def test_match_local_skips_already_used_clips(tmp_path):
    settings = _settings(tmp_path)
    profile = broll.SpanProfile(start=0, end=1, subjects=["ball"], setting=["stadium"], category="sports")
    clip = _clip(tmp_path, "a.mp4", category="sports", subjects=["ball"], setting=["stadium"], folder="NBA_Clips")
    used = {clip.path.resolve()}
    assert broll.match_local(profile, [clip], used, settings) is None


# ---------------------------------------------------------------------------
# YouTube: ISO-8601 duration, junk-title filter, Data-API -> yt-dlp fallback
# ---------------------------------------------------------------------------


def test_iso8601_duration_parses_typical_formats():
    assert broll._iso8601_duration_to_seconds("PT1M30S") == 90.0
    assert broll._iso8601_duration_to_seconds("PT45S") == 45.0
    assert broll._iso8601_duration_to_seconds("PT2H") == 7200.0
    assert broll._iso8601_duration_to_seconds("") == 0.0
    assert broll._iso8601_duration_to_seconds("garbage") == 0.0


def test_junk_youtube_entry_filters_shorts_and_commentary():
    assert broll._is_junk_youtube_entry("https://www.youtube.com/shorts/abc", "cool clip") is True
    assert broll._is_junk_youtube_entry("https://youtu.be/abc", "My Podcast Episode 12") is True
    assert broll._is_junk_youtube_entry("https://youtu.be/abc", "I REACT to this") is True
    assert broll._is_junk_youtube_entry("https://youtu.be/abc", "city street stock footage") is False


def test_youtube_data_api_search_returns_none_when_all_keys_403(monkeypatch, tmp_path):
    """All configured keys quota-exceeded/blocked -> the caller must fall back
    to yt-dlp ytsearch (returning None, not [], is the fallback signal)."""
    settings = _settings(tmp_path)
    settings.youtube_data_api_key = "key1"
    settings.youtube_data_api_key_2 = "key2"

    def fake_urlopen(url, timeout=None):
        raise urllib.error.HTTPError(url, 403, "quota exceeded", None, None)

    monkeypatch.setattr(broll.urllib.request, "urlopen", fake_urlopen)
    assert broll._youtube_data_api_search("city street", 10, settings) is None


def test_youtube_search_entries_falls_back_to_yt_dlp_when_api_unavailable(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    settings.youtube_data_api_key = "key1"

    monkeypatch.setattr(broll, "_youtube_data_api_search", lambda *a, **k: None)

    def fake_yt_dlp_search(query, per_query, settings):
        return [{"id": "vid1", "url": "https://youtu.be/vid1", "duration": 20, "title": "city street stock footage"}]

    monkeypatch.setattr(broll, "_yt_dlp_search", fake_yt_dlp_search)
    entries = broll._youtube_search_entries(["city street"], settings)
    assert [e["id"] for e in entries] == ["vid1"]


def test_youtube_search_entries_dedupes_and_filters_junk(monkeypatch, tmp_path):
    settings = _settings(tmp_path)

    def fake_yt_dlp_search(query, per_query, settings):
        return [
            {"id": "good", "url": "https://youtu.be/good", "duration": 20, "title": "city street stock footage"},
            {"id": "pod", "url": "https://youtu.be/pod", "duration": 30, "title": "My Podcast Episode 12"},
            {"id": "good", "url": "https://youtu.be/good", "duration": 20, "title": "city street stock footage"},
        ]

    monkeypatch.setattr(broll, "_youtube_data_api_search", lambda *a, **k: None)
    monkeypatch.setattr(broll, "_yt_dlp_search", fake_yt_dlp_search)
    entries = broll._youtube_search_entries(["city street"], settings)
    assert [e["id"] for e in entries] == ["good"]


# ---------------------------------------------------------------------------
# Count parity: every span always yields a cut
# ---------------------------------------------------------------------------


def test_every_span_gets_a_cut_via_reference_crop_last_resort(monkeypatch, tmp_path):
    """With local + YouTube mocked to fail, the reference-crop rung must still
    produce a cut for every span — count parity is guaranteed, no span is
    ever left empty."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(broll, "build_library_index", lambda settings: [])
    monkeypatch.setattr(broll, "search_youtube", lambda *a, **k: None)

    def fake_crop(reference_path, span, output_path, settings):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x" * 1000)
        return output_path

    def fake_extract_segment(source, duration, output_path, settings):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x" * 1000)
        return output_path

    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)
    monkeypatch.setattr(broll, "_extract_segment", fake_extract_segment)

    analysis = ReferenceAnalysis(
        duration=20.0,
        broll_spans=[(1.0, 2.0, "a"), (5.0, 6.0, "b"), (10.0, 11.0, "c")],
    )
    cuts = broll.fetch_broll_cuts(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "cache", settings,
    )
    assert len(cuts) == 3
    assert cuts[0].start == 1.0 and cuts[0].end == 2.0


def test_variations_all_get_broll_when_only_reference_crop_available(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(broll, "build_library_index", lambda settings: [])
    monkeypatch.setattr(broll, "search_youtube", lambda *a, **k: None)

    def fake_crop(reference_path, span, output_path, settings):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x" * 1000)
        return output_path

    def fake_extract_segment(source, duration, output_path, settings):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x" * 1000)
        return output_path

    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)
    monkeypatch.setattr(broll, "_extract_segment", fake_extract_segment)

    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[(1.0, 2.0, "a")])
    cut_lists = broll.fetch_broll_cut_variations(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "cache", settings, variations=3,
    )
    assert len(cut_lists) == 3
    for cuts in cut_lists:
        assert len(cuts) == 1, "every variation must still have B-roll (reference-crop fallback)"
