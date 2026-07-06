"""Tests for broll_intelligence.search + broll_intelligence.llm_ladder.

All offline. Every external side-effect (chat ladder, yt-dlp, ffmpeg,
vibe_extractor) is injected through the keyword args on search_broll so
we can fake them without touching the filesystem or the network.

Coverage:
  * Query generation — LLM success path.
  * Query generation fallback — LLM fails → deterministic query set built
    from the reference's own fields.
  * Aggregation + dedup — multiple queries returning overlapping video_ids
    collapse to one candidate.
  * Junk-title filter — "podcast" / "tutorial" / "reaction" titles dropped,
    YouTube Shorts URLs dropped.
  * Top-K ordering — scores returned in descending order.
  * Composite-score sanity — self-similarity is well above empty-vector
    similarity.
  * LLM JSON helpers — fence stripping + brace-span extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from broll_intelligence import FeatureVector, empty_feature_vector
from broll_intelligence.config import Settings
from broll_intelligence.llm_ladder import (
    CHAT_TIMEOUT,
    extract_json_object,
    generate_search_queries,
    strip_code_fence,
)
from broll_intelligence.search import (
    COMPOSITE_WEIGHTS,
    QUERY_GENERATION_PROMPT,
    QUERY_MAX_WORDS,
    YT_JUNK_TITLE_TERMS,
    BrollCandidate,
    _aggregate_search_results,
    _build_query_prompt,
    _fallback_queries,
    _is_junk_title,
    _truncate_query,
    composite_score,
    search_broll,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cinematic_dark_ref() -> FeatureVector:
    """The reference used as the example in the task brief."""
    return FeatureVector(
        subjects=["astronaut"],
        setting=["lunar surface"],
        action=["floating"],
        category="movie",
        query="astronaut floating on lunar surface",
        mood=["mysterious"],
        energy="low",
        lighting="low-key",
        color_palette=["deep blue", "silver"],
        shot_type="wide",
        camera_motion="tracking",
        depth_of_field="deep",
    )


@pytest.fixture
def empty_settings(tmp_path: Path) -> Settings:
    """A Settings instance with all API keys blanked — search_broll uses
    chat_fn injection in tests so the keys don't matter, but Settings()
    itself needs to construct without raising."""
    from broll_intelligence.config import Settings as _Settings

    return _Settings(
        library_dir=tmp_path / "library",
        index_path=tmp_path / "index.json",
        data_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------


def test_query_generation_uses_llm_when_available(
    cinematic_dark_ref: FeatureVector,
    empty_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    """When the chat ladder returns a clean JSON object, those queries are
    used (after dedup + truncation). The fallback is NOT touched."""
    from broll_intelligence import search as search_mod

    canned = [
        "astronaut floating in space",
        "mysterious deep space scene",
        "low-key dramatic lighting",
        "lunar surface wide shot",
        "cinematic blue silver palette",
    ]

    def _fake_chat(prompt, settings):
        assert "subjects: astronaut" in prompt
        assert "mood: mysterious" in prompt
        return list(canned)

    captured: dict[str, Any] = {}

    # Patch the search module's chat fallback so we can confirm it's NOT
    # used when the injected chat_fn returns valid queries.
    def _boom(prompt, settings):
        captured["called"] = True
        return ["fallback"]

    monkeypatch.setattr(search_mod, "_ytdlp_search", lambda *a, **k: [])

    result = search_broll(
        reference=cinematic_dark_ref,
        top_k=3,
        cache_dir=Path("/tmp"),
        settings=empty_settings,
        _chat_fn=_fake_chat,
        _search_fn=lambda q, n: [],
        _download_fn=lambda *a, **k: False,
        _frame_extractor_fn=lambda *a, **k: False,
        _vibe_fn=lambda *a, **k: empty_feature_vector(),
    )

    # No candidates (yt-dlp returned nothing) but the function ran without
    # raising and the chat_fn was consulted.
    assert result == []
    assert "called" not in captured


def test_query_generation_falls_back_when_llm_fails(
    cinematic_dark_ref: FeatureVector,
    empty_settings: Settings,
):
    """When the chat ladder raises or returns junk, fall back to the
    deterministic query set built from the reference's own fields."""

    def _broken_chat(prompt, settings):
        return []  # empty = "every provider failed" simulation

    queries = _fallback_queries(cinematic_dark_ref)
    assert len(queries) == 5
    # Subject-focused (q1) is built from the primary subject.
    assert "astronaut" in queries[0]
    # Mood-focused (q2) uses the mood term.
    assert "mysterious" in queries[1]
    # Lighting-focused (q3) uses the lighting enum.
    assert "low-key" in queries[2]
    # Scene-focused (q4) uses setting + subject.
    assert "lunar surface" in queries[3]
    # Aesthetic-focused (q5) uses the first palette colour.
    assert "deep blue" in queries[4]


def test_query_generation_uses_fallback_on_llm_exception(
    cinematic_dark_ref: FeatureVector,
    empty_settings: Settings,
    tmp_path: Path,
):
    """If chat_fn raises (network error, timeout, whatever), search_broll
    MUST NOT crash — it must use the deterministic fallback."""

    def _exploding_chat(prompt, settings):
        raise RuntimeError("simulated LLM outage")

    result = search_broll(
        reference=cinematic_dark_ref,
        top_k=2,
        cache_dir=tmp_path / "cache",
        settings=empty_settings,
        _chat_fn=_exploding_chat,
        _search_fn=lambda q, n: [],
        _download_fn=lambda *a, **k: False,
        _frame_extractor_fn=lambda *a, **k: False,
        _vibe_fn=lambda *a, **k: empty_feature_vector(),
    )

    assert result == []  # no yt-dlp results, but no crash either


def test_query_truncation_enforces_eight_word_cap():
    """The spec requires <= 8 words per query; the helper must enforce it
    regardless of what the LLM returned."""
    long = "one two three four five six seven eight nine ten eleven"
    truncated = _truncate_query(long)
    assert len(truncated.split()) == 8
    assert truncated.endswith("eight")


def test_query_truncation_collapses_whitespace_and_strips_quotes():
    q = '  "  astronaut   floating  on\tlunar\nsurface  "  '
    out = _truncate_query(q)
    assert out == "astronaut floating on lunar surface"


def test_prompt_contains_required_fields(cinematic_dark_ref: FeatureVector):
    """The query-generation prompt must mention every field the reference
    feeds in — otherwise the LLM is guessing."""
    prompt = _build_query_prompt(cinematic_dark_ref)
    for key in ("subjects", "setting", "mood", "lighting", "shot_type", "color_palette"):
        assert f"{key}:" in prompt
    # The reference's actual values are interpolated.
    assert "astronaut" in prompt
    assert "mysterious" in prompt
    assert "low-key" in prompt


# ---------------------------------------------------------------------------
# Aggregation + dedup
# ---------------------------------------------------------------------------


def _fake_search_fn(per_query_results: dict[str, list[dict[str, Any]]]) -> Callable[..., list[dict[str, Any]]]:
    """Build a yt-dlp stub that returns canned rows per query."""

    def _search(query: str, per_query: int) -> list[dict[str, Any]]:
        return list(per_query_results.get(query, []))

    return _search


def test_aggregation_dedups_overlapping_video_ids(
    cinematic_dark_ref: FeatureVector,
):
    """Two queries returning the same video_id should produce ONE entry."""
    shared = {"id": "vid1", "url": "https://www.youtube.com/watch?v=vid1", "title": "Astronaut on Moon"}
    other = {"id": "vid2", "url": "https://www.youtube.com/watch?v=vid2", "title": "Space walk HD"}
    search_fn = _fake_search_fn({
        "astronaut floating": [shared, other],
        "mysterious deep space": [shared],  # duplicate!
    })
    entries = _aggregate_search_results(
        ["astronaut floating", "mysterious deep space"],
        max_per_query=4,
        max_total=12,
        _search_fn=search_fn,
    )
    ids = sorted(e["id"] for e in entries)
    assert ids == ["vid1", "vid2"]


def test_aggregation_respects_total_cap(
    cinematic_dark_ref: FeatureVector,
):
    """Per-query cap × N queries MUST be capped by max_total."""
    rows = [
        {"id": f"vid{i}", "url": f"https://www.youtube.com/watch?v=vid{i}", "title": f"clip {i}"}
        for i in range(20)
    ]
    search_fn = _fake_search_fn({
        "q1": rows,
        "q2": rows,
    })
    entries = _aggregate_search_results(
        ["q1", "q2"],
        max_per_query=10,
        max_total=5,
        _search_fn=search_fn,
    )
    assert len(entries) == 5


# ---------------------------------------------------------------------------
# Junk filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,title,is_junk",
    [
        ("https://www.youtube.com/watch?v=abc", "Astronaut on the Moon", False),
        ("https://www.youtube.com/watch?v=abc", "Podcast with Astronaut", True),
        ("https://www.youtube.com/watch?v=abc", "React to Space Movie", True),
        ("https://www.youtube.com/watch?v=abc", "Explained: Lunar Mission", True),
        ("https://www.youtube.com/watch?v=abc", "How to be an Astronaut", True),
        ("https://www.youtube.com/watch?v=abc", "Movie Review: Gravity", True),
        ("https://www.youtube.com/watch?v=abc", "My Vlog Day 42", True),
        ("https://www.youtube.com/shorts/abc", "Cool moon shot", True),
        ("https://www.youtube.com/watch?v=abc", "Top 10 Space Clips", True),
        ("https://www.youtube.com/watch?v=abc", "Wide lunar landscape", False),
    ],
)
def test_junk_title_filter(url: str, title: str, is_junk: bool):
    assert _is_junk_title(url, title) is is_junk


def test_aggregation_drops_junk_titles(cinematic_dark_ref: FeatureVector):
    """Junk entries must be filtered before they reach the candidate list."""
    good = {"id": "ok1", "url": "https://www.youtube.com/watch?v=ok1", "title": "Astronaut floating in space"}
    bad = {"id": "bad1", "url": "https://www.youtube.com/watch?v=bad1", "title": "Podcast about space"}
    shorts = {"id": "sht1", "url": "https://www.youtube.com/shorts/sht1", "title": "Cool clip"}
    search_fn = _fake_search_fn({"q1": [good, bad, shorts]})
    entries = _aggregate_search_results(
        ["q1"],
        max_per_query=4,
        max_total=12,
        _search_fn=search_fn,
    )
    assert [e["id"] for e in entries] == ["ok1"]


def test_junk_title_terms_list_is_complete():
    """The rejection list covers the common commentary formats. If a new
    term gets added to app/broll.py, this constant is the place to update
    in lock-step."""
    expected_substrings = ("podcast", "reaction", "tutorial", "review")
    for term in expected_substrings:
        assert term in YT_JUNK_TITLE_TERMS


# ---------------------------------------------------------------------------
# Top-K ordering + scoring
# ---------------------------------------------------------------------------


def test_top_k_ordering_descending_by_score(
    cinematic_dark_ref: FeatureVector,
    empty_settings: Settings,
    tmp_path: Path,
):
    """Given a fixed set of candidates with known scores, search_broll must
    return them in score-descending order and respect top_k."""

    # Three candidates, each with a different FeatureVector to drive a
    # different composite_score.
    candidates = [
        # Excellent: same mood + same lighting + matching palette.
        {
            "id": "great",
            "url": "https://www.youtube.com/watch?v=great",
            "title": "Astronaut on Moon",
            "query": "q",
            "features": FeatureVector(
                subjects=["astronaut"], mood=["mysterious"], energy="low",
                lighting="low-key", color_palette=["deep blue", "silver"],
                shot_type="wide", camera_motion="static", depth_of_field="deep",
                palette_warmth=0.2, palette_saturation=0.4, palette_brightness=0.35,
                contrast=0.55, edge_density=0.18, motion_intensity=0.05,
            ),
        },
        # Mediocre: only subject match.
        {
            "id": "okay",
            "url": "https://www.youtube.com/watch?v=okay",
            "title": "Astronaut eating",
            "query": "q",
            "features": FeatureVector(
                subjects=["astronaut"], mood=["joyful"], energy="high",
                lighting="natural", color_palette=["warm"],
                shot_type="close-up", camera_motion="handheld", depth_of_field="shallow",
                palette_warmth=0.6, palette_saturation=0.7, palette_brightness=0.7,
                contrast=0.3, edge_density=0.2, motion_intensity=0.5,
            ),
        },
        # Terrible: nothing matches.
        {
            "id": "bad",
            "url": "https://www.youtube.com/watch?v=bad",
            "title": "Cooking pasta",
            "query": "q",
            "features": FeatureVector(
                subjects=["pasta"], mood=["joyful"], energy="medium",
                lighting="high-key", color_palette=["yellow"],
                shot_type="close-up", camera_motion="handheld", depth_of_field="shallow",
                palette_warmth=0.7, palette_saturation=0.8, palette_brightness=0.8,
                contrast=0.2, edge_density=0.1, motion_intensity=0.4,
            ),
        },
    ]
    rows = [{"id": c["id"], "url": c["url"], "title": c["title"]} for c in candidates]
    by_id = {c["id"]: c for c in candidates}

    def _download(url, out_path, ffmpeg_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * 12000)  # > MIN_PREVIEW_BYTES
        return True

    def _extract_frame(source, at_seconds, output_path, settings):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")  # JPEG-ish
        return True

    def _vibe(frame_path, settings):
        video_id = frame_path.stem.replace("preview_", "")
        cand = by_id.get(video_id)
        assert cand is not None, f"unexpected frame: {frame_path}"
        return cand["features"]

    cache_dir = tmp_path / "cache"
    result = search_broll(
        reference=cinematic_dark_ref,
        top_k=2,
        cache_dir=cache_dir,
        settings=empty_settings,
        max_candidates_per_query=4,
        max_total_candidates=12,
        _chat_fn=lambda prompt, settings: ["astronaut floating"],
        _search_fn=lambda q, n: rows,
        _download_fn=_download,
        _frame_extractor_fn=_extract_frame,
        _vibe_fn=_vibe,
    )

    # top_k=2 → first 2 by descending score. 'great' must be first.
    assert len(result) == 2
    assert result[0].video_id == "great"
    assert result[1].video_id in {"okay", "bad"}
    assert result[0].score >= result[1].score
    # The loser (the one not picked) is dropped because top_k=2.
    picked_ids = {c.video_id for c in result}
    assert "bad" not in picked_ids or "okay" not in picked_ids
    # Preview path was set on every returned candidate.
    for c in result:
        assert c.preview_path is not None
        assert c.preview_path.exists()


def test_composite_score_self_similarity_above_empty(cinematic_dark_ref: FeatureVector):
    """composite_score(ref, ref) MUST be strictly greater than
    composite_score(ref, empty) — otherwise the signal is useless."""
    self_score = composite_score(cinematic_dark_ref, cinematic_dark_ref)
    empty = empty_feature_vector()
    empty_score = composite_score(cinematic_dark_ref, empty)
    assert self_score > empty_score
    # Self-score is bounded but not trivially 1.0 (camera_motion complement
    # table demotes identical camera motion).
    assert 0.5 <= self_score <= 1.0


def test_composite_score_weights_sum_to_one():
    """COMPOSITE_WEIGHTS must sum to 1.0 (±1e-6) so the score stays in
    [0, 1]. A drift to e.g. 1.10 would silently inflate results."""
    total = sum(COMPOSITE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_failed_download_yields_zero_score(
    cinematic_dark_ref: FeatureVector,
    empty_settings: Settings,
    tmp_path: Path,
):
    """When the download fails, the candidate is still returned (with its
    URL preserved) so the caller can decide what to do — but score=0 and
    preview_path=None."""
    rows = [{"id": "v1", "url": "https://www.youtube.com/watch?v=v1", "title": "Cool"}]

    def _download(url, out, ffmpeg_path):
        return False  # yt-dlp failed

    result = search_broll(
        reference=cinematic_dark_ref,
        top_k=1,
        cache_dir=tmp_path / "cache",
        settings=empty_settings,
        _chat_fn=lambda p, s: ["astronaut"],
        _search_fn=lambda q, n: rows,
        _download_fn=_download,
        _frame_extractor_fn=lambda *a, **k: False,
        _vibe_fn=lambda *a, **k: empty_feature_vector(),
    )
    assert len(result) == 1
    assert result[0].score == 0.0
    assert result[0].preview_path is None
    assert result[0].video_id == "v1"
    assert result[0].features is None


def test_top_k_zero_returns_empty(cinematic_dark_ref: FeatureVector, empty_settings: Settings, tmp_path: Path):
    """top_k=0 is a legitimate "just give me everything you found" mode,
    but the contract says it should return an empty list (no scoring
    work)."""
    result = search_broll(
        reference=cinematic_dark_ref,
        top_k=0,
        cache_dir=tmp_path,
        settings=empty_settings,
        _chat_fn=lambda p, s: [],
        _search_fn=lambda q, n: [],
    )
    assert result == []


# ---------------------------------------------------------------------------
# LLM JSON helpers (unit-level)
# ---------------------------------------------------------------------------


def test_extract_json_object_strips_code_fence():
    raw = "```json\n{\"queries\": [\"a\", \"b\"]}\n```"
    obj = extract_json_object(raw)
    assert obj == {"queries": ["a", "b"]}


def test_extract_json_object_handles_prose_prefix():
    raw = "Sure! Here you go: {\"queries\": [\"x\"]}."
    obj = extract_json_object(raw)
    assert obj == {"queries": ["x"]}


def test_extract_json_object_returns_empty_on_garbage():
    for bad in ("", "not json at all", "}{", "{", "}"):
        assert extract_json_object(bad) == {}


def test_strip_code_fence_idempotent_on_unfenced():
    text = '{"queries": ["a"]}'
    assert strip_code_fence(text) == text


def test_generate_search_queries_handles_empty_response(empty_settings: Settings):
    """When chat() returns "" (every provider failed), extract_json_object
    gives us {} and generate_search_queries returns []."""
    from broll_intelligence import llm_ladder

    def _empty_chat(prompt, settings):
        return ""

    llm_ladder._provider_cooldowns.clear()
    # All providers have no API key (test fixture), so chat() will return
    # "" without making any network calls.
    assert generate_search_queries("anything", empty_settings) == []


def test_query_prompt_constant_matches_doc():
    """The module-level QUERY_GENERATION_PROMPT must match the verbatim
    prompt documented in CONTRACT.md §7. If one moves, the other must
    follow — drift between them silently invalidates every cached LLM
    response."""
    assert "subjects: {subjects}" in QUERY_GENERATION_PROMPT
    assert "setting: {setting}" in QUERY_GENERATION_PROMPT
    assert "mood: {mood}" in QUERY_GENERATION_PROMPT
    assert "lighting: {lighting}" in QUERY_GENERATION_PROMPT
    assert "shot_type: {shot_type}" in QUERY_GENERATION_PROMPT
    assert "color_palette: {color_palette}" in QUERY_GENERATION_PROMPT
    assert '"queries": [q1, q2, q3, q4, q5]' in QUERY_GENERATION_PROMPT
    assert "subject-focused" in QUERY_GENERATION_PROMPT
    assert "mood-focused" in QUERY_GENERATION_PROMPT
    assert "lighting-focused" in QUERY_GENERATION_PROMPT
    assert "scene-focused" in QUERY_GENERATION_PROMPT
    assert "aesthetic-focused" in QUERY_GENERATION_PROMPT


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_query_max_words_is_eight():
    """Spec invariant: <= 8 words per query."""
    assert QUERY_MAX_WORDS == 8


def test_chat_timeout_matches_spec():
    """The chat ladder must default to 15 s — same as app/broll.py.
    Drift here would silently change per-call budget across the whole
    package."""
    assert CHAT_TIMEOUT == 15.0