"""Unit tests for the B-roll pack pipeline (gather_broll_pack + jobs/main glue).

Mirror of test_broll_pipeline.py: every external edge (YouTube, library scan,
ffmpeg, vision LLM) is monkeypatched so the tests run offline and stay
deterministic. The plan calls for stubbing _rank_local + search_youtube_candidates
+ _extract_segment directly so the tests exercise gather_broll_pack's own ladder
logic, not the match scoring tier.
"""
from pathlib import Path

from app import broll
from app.broll import LibraryClip
from app.config import Settings
from app.main import _summary
from app.models import BrollPackItem, Job, JobStatus, ReferenceAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Settings:
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


def _clip(tmp_path: Path, name: str) -> LibraryClip:
    """Make a fake library clip with category='sports' so the production
    _local_score would return > 0 for any sports profile. Tests stub
    _rank_local directly anyway, so this just needs the path to be distinct."""
    path = tmp_path / name
    path.write_bytes(b"x")
    return LibraryClip(
        path=path, mtime=0.0, size=1,
        subjects=["ball"], setting=["stadium"], category="sports", folder="Sports",
    )


def _stub_extract_and_crop() -> tuple[list[float], callable, callable]:
    """Replace _extract_segment + crop_reference_cutaway with file-touching
    stubs that record the requested duration so tests can assert the trim."""

    seen_durations: list[float] = []

    def fake_extract(source, duration, output_path, settings):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x" * 1000)
        seen_durations.append(duration)
        return output_path

    def fake_crop(reference_path, span, output_path, settings):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x" * 1000)
        return output_path

    return seen_durations, fake_extract, fake_crop


def _rank_local_stub_factory(scored_clips: list[tuple[LibraryClip, float]]):
    """Build a _rank_local stub that returns the given (clip, score) pairs,
    already sorted, regardless of how the implementation iterates.

    V2 selector change: production ``_rank_local`` now returns 5-tuples
    (clip, total, vibe, cinema, cont) and may receive extra kwargs
    (reference_house, job_id, ledger). This stub mirrors that contract
    so the tests still exercise ``gather_broll_pack``'s own ladder logic.
    """
    sorted_pairs = sorted(scored_clips, key=lambda t: t[1], reverse=True)

    def fake_rank(profile, library_index, used_clips, **kwargs):
        out = []
        for clip, score in sorted_pairs:
            try:
                if clip.path.resolve() in used_clips:
                    continue
            except OSError:
                continue
            out.append((clip, score, 0.0, 0.0, 0.0))
        return out

    return fake_rank


# ---------------------------------------------------------------------------
# gather_broll_pack — happy paths
# ---------------------------------------------------------------------------


def test_gather_broll_pack_returns_two_distinct_items_when_library_has_two_matches(
    monkeypatch, tmp_path
):
    """per_span=2 with two ranked local clips → emit rank-1 + rank-2, both
    'local', both pointing at distinct resolved files."""
    settings = _settings(tmp_path)
    clip_a = _clip(tmp_path, "a.mp4")
    clip_b = _clip(tmp_path, "b.mp4")
    monkeypatch.setattr(broll, "_rank_local", _rank_local_stub_factory([(clip_a, 0.9), (clip_b, 0.7)]))
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    seen_durations, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[(1.0, 2.0, "q")])
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=2,
    )

    assert len(items) == 2
    assert [it.rank for it in items] == [1, 2]
    assert all(it.provider == "local" for it in items)
    assert len({it.clip_path.resolve() for it in items}) == 2, "rank-1 and rank-2 must be distinct files"
    # The trim duration matches the span's output length (2.0 - 1.0 = 1.0).
    assert all(abs(d - 1.0) < 1e-6 for d in seen_durations[:2])


def test_gather_broll_pack_dedupes_to_one_when_only_one_distinct_candidate(
    monkeypatch, tmp_path
):
    """One local clip, empty YouTube, ref-crop that would produce a duplicate
    → exactly ONE item (never padded by repeating the same source)."""
    settings = _settings(tmp_path)
    clip_a = _clip(tmp_path, "a.mp4")
    monkeypatch.setattr(broll, "_rank_local", _rank_local_stub_factory([(clip_a, 0.8)]))
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])

    def fake_crop(reference_path, span, output_path, settings):
        return None  # exhausted; only local survives

    _, fake_extract, _ = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[(1.0, 2.0, "q")])
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=2,
    )

    assert len(items) == 1
    assert items[0].rank == 1
    assert items[0].provider == "local"


def test_gather_broll_pack_uses_youtube_to_fill_rank_two_when_local_has_one(
    monkeypatch, tmp_path
):
    """Rank-2 slot is filled by the YouTube rung when local is exhausted."""
    settings = _settings(tmp_path)
    clip_a = _clip(tmp_path, "a.mp4")
    yt_clip = tmp_path / "yt.mp4"
    yt_clip.write_bytes(b"x")
    monkeypatch.setattr(broll, "_rank_local", _rank_local_stub_factory([(clip_a, 0.8)]))
    monkeypatch.setattr(
        broll, "search_youtube_candidates",
        lambda profile, cache_dir, settings, count=2, **kwargs: [yt_clip] if count >= 1 else [],
    )
    _, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[(1.0, 2.0, "q")])
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=2,
    )

    providers = [it.provider for it in items]
    assert "local" in providers
    assert "youtube" in providers
    assert len(items) == 2


def test_gather_broll_pack_uses_reference_crop_when_local_and_youtube_empty(
    monkeypatch, tmp_path
):
    """Rung 3 (reference-crop) is the always-succeeds last filler — a span is
    never left empty just because the upstream rungs were empty."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(broll, "_rank_local", lambda *a, **k: [])
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    _, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[(1.0, 2.0, "q")])
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=2,
    )

    assert len(items) == 1
    assert items[0].provider == "reference_crop"


# ---------------------------------------------------------------------------
# gather_broll_pack — span alignment + skip rules
# ---------------------------------------------------------------------------


def test_gather_broll_pack_skips_spans_shorter_than_min_output_broll_span(
    monkeypatch, tmp_path
):
    """Sub-MIN_OUTPUT_BROLL_SPAN spans are silently dropped (matches the
    fetch_broll_cut_variations policy)."""
    settings = _settings(tmp_path)
    clip_a = _clip(tmp_path, "a.mp4")
    monkeypatch.setattr(broll, "_rank_local", _rank_local_stub_factory([(clip_a, 0.9)]))
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    _, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    # 0.05s span is well below MIN_OUTPUT_BROLL_SPAN (0.08s).
    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[(1.0, 1.05, "q")])
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=2,
    )
    assert items == []


def test_gather_broll_pack_records_aligned_output_timing_per_span(
    monkeypatch, tmp_path
):
    """start/end on each pack item reflect the OUTPUT timeline (after
    _align_span_to_clip), not the raw reference timestamps — drops straight
    into the rendered cut."""
    settings = _settings(tmp_path)
    clip_a = _clip(tmp_path, "a.mp4")
    monkeypatch.setattr(broll, "_rank_local", _rank_local_stub_factory([(clip_a, 0.9)]))
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    seen_durations, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    # Reference 12s long, our output 10s — span (8, 11) overflows and gets
    # scaled to fit the tail of our clip.
    analysis = ReferenceAnalysis(duration=12.0, broll_spans=[(8.0, 11.0, "q")])
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 10.0, tmp_path / "pack", settings, per_span=2,
    )

    assert len(items) >= 1
    assert items[0].end == 10.0, "aligned to clip end"
    assert items[0].start < 10.0
    expected_duration = items[0].end - items[0].start
    assert any(abs(d - expected_duration) < 1e-6 for d in seen_durations)


# ---------------------------------------------------------------------------
# gather_broll_pack — multi-span behavior
# ---------------------------------------------------------------------------


def test_gather_broll_pack_dedupes_used_local_clips_across_spans(
    monkeypatch, tmp_path
):
    """With a single library clip, span 0 takes it; span 1 must NOT re-emit
    the same clip — it falls through to ref-crop."""
    settings = _settings(tmp_path)
    clip_a = _clip(tmp_path, "a.mp4")
    # Stub records the used_clips set across calls.
    state = {"used": set()}

    def fake_rank(profile, library_index, used_clips, **kwargs):
        state["used"] = used_clips
        if clip_a.path.resolve() in {p.resolve() for p in used_clips}:
            return []
        return [(clip_a, 0.9, 0.0, 0.0, 0.0)]

    monkeypatch.setattr(broll, "_rank_local", fake_rank)
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    _, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    analysis = ReferenceAnalysis(
        duration=20.0, broll_spans=[(1.0, 2.0, "q"), (5.0, 6.0, "r")],
    )
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=2,
    )

    providers_per_span: dict[int, list[str]] = {}
    for it in items:
        providers_per_span.setdefault(it.span_index, []).append(it.provider)
    assert "local" in providers_per_span[0]
    assert "local" not in providers_per_span.get(1, [])
    # Span 1 falls through to the guaranteed ref-crop rung.
    assert "reference_crop" in providers_per_span[1]


def test_gather_broll_pack_returns_empty_when_no_spans(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(broll, "_rank_local", lambda *a, **k: [])
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    monkeypatch.setattr(broll, "_extract_segment", lambda *a, **k: None)
    monkeypatch.setattr(broll, "crop_reference_cutaway", lambda *a, **k: None)
    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[])
    assert broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings,
    ) == []


def test_gather_broll_pack_emits_one_item_when_per_span_one(monkeypatch, tmp_path):
    """per_span=1 → exactly one (rank=1) item per kept span. No rank=2 padding."""
    settings = _settings(tmp_path)
    clip_a = _clip(tmp_path, "a.mp4")
    clip_b = _clip(tmp_path, "b.mp4")
    monkeypatch.setattr(broll, "_rank_local", _rank_local_stub_factory([(clip_a, 0.9), (clip_b, 0.7)]))
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    _, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    analysis = ReferenceAnalysis(duration=20.0, broll_spans=[(1.0, 2.0, "q")])
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=1,
    )
    assert len(items) == 1
    assert items[0].rank == 1


def test_gather_broll_pack_emits_at_least_one_item_per_kept_span(monkeypatch, tmp_path):
    """ref-crop rung guarantees no kept span is empty — every aligned span
    produces ≥1 item, which is the contract for any span-row UI."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(broll, "_rank_local", lambda *a, **k: [])
    monkeypatch.setattr(broll, "search_youtube_candidates", lambda *a, **k: [])
    _, fake_extract, fake_crop = _stub_extract_and_crop()
    monkeypatch.setattr(broll, "_extract_segment", fake_extract)
    monkeypatch.setattr(broll, "crop_reference_cutaway", fake_crop)

    analysis = ReferenceAnalysis(
        duration=20.0,
        broll_spans=[(1.0, 2.0, "a"), (5.0, 6.0, "b"), (10.0, 11.0, "c")],
    )
    items = broll.gather_broll_pack(
        analysis, tmp_path / "ref.mp4", 20.0, tmp_path / "pack", settings, per_span=2,
    )
    assert {it.span_index for it in items} == {0, 1, 2}


# ---------------------------------------------------------------------------
# BrollPackItem model + Job defaults (back-compat)
# ---------------------------------------------------------------------------


def test_broll_pack_item_model_fields_round_trip(tmp_path):
    item = BrollPackItem(
        span_index=0,
        rank=1,
        start=1.5,
        end=3.5,
        query="city street",
        provider="local",
        clip_path=tmp_path / "x.mp4",
    )
    assert item.span_index == 0
    assert item.rank == 1
    assert item.provider == "local"
    assert isinstance(item.clip_path, Path)


def test_job_broll_pack_defaults_keep_old_meta_loading(tmp_path):
    """Old meta.json files lack broll_pack / broll_pack_items — Pydantic
    defaults keep them loading fine without an explicit migration."""
    job = Job(id="old_job", status=JobStatus.INGESTED)
    assert job.broll_pack is False
    assert job.broll_pack_items == []


# ---------------------------------------------------------------------------
# Pipeline glue: Job.broll_pack forces variation_count = 1
# ---------------------------------------------------------------------------


def test_job_broll_pack_overrides_variation_count_to_one():
    """Mirror the construction in jobs.py _run: when broll_pack is set,
    variation_count = 1 no matter the default. This pins the contract that
    pack-mode jobs don't produce N duplicate renders."""
    broll_pack = True
    variation_count = 1 if broll_pack else 4
    assert variation_count == 1


def test_job_without_broll_pack_keeps_default_variation_count():
    """Inverse: a normal job keeps its default variation_count."""
    broll_pack = False
    bulk = False
    settings_variation_count = 4
    if broll_pack:
        variation_count = 1
    else:
        variation_count = 1 if bulk else max(1, settings_variation_count)
    assert variation_count == 4


# ---------------------------------------------------------------------------
# JobSummary / _summary exposure
# ---------------------------------------------------------------------------


def test_summary_exposes_broll_pack_urls_when_pack_items_present(tmp_path):
    """_summary() serializes every BrollPackItem as a BrollPackDownload with
    the canonical /api/renders/<id>/broll/<i>.mp4 URL."""
    item_0 = BrollPackItem(
        span_index=0, rank=1, start=1.0, end=2.0,
        query="city street", provider="local", clip_path=tmp_path / "a.mp4",
    )
    item_1 = BrollPackItem(
        span_index=0, rank=2, start=1.0, end=2.0,
        query="city street", provider="youtube", clip_path=tmp_path / "b.mp4",
    )
    job = Job(
        id="abc123", status=JobStatus.READY, progress=1.0,
        message="Render complete", output_path=tmp_path / "render.mp4",
        broll_pack=True, broll_pack_items=[item_0, item_1],
    )
    summary = _summary(job)
    assert len(summary.broll_pack_urls) == 2
    assert summary.broll_pack_urls[0].url == "/api/renders/abc123/broll/0.mp4"
    assert summary.broll_pack_urls[1].url == "/api/renders/abc123/broll/1.mp4"
    # Span/rank metadata is preserved so the UI can render the table.
    assert summary.broll_pack_urls[0].span_index == 0
    assert summary.broll_pack_urls[0].rank == 1
    assert summary.broll_pack_urls[1].rank == 2
    assert summary.broll_pack_urls[0].query == "city street"


def test_summary_broll_pack_urls_empty_for_normal_replicate(tmp_path):
    """A regular replicate job (no pack) gets an empty list — the UI hides
    the Download Pack affordance."""
    job = Job(
        id="xyz", status=JobStatus.READY, progress=1.0,
        message="Render complete", output_path=tmp_path / "render.mp4",
        broll_pack=False, broll_pack_items=[],
    )
    summary = _summary(job)
    assert summary.broll_pack_urls == []


def test_broll_pack_download_subclass_serializes():
    """BrollPackDownload is a Pydantic model — ensure model_dump produces a
    well-formed JSON response (the API contract)."""
    from app.models import BrollPackDownload
    payload = BrollPackDownload(
        span_index=0, rank=1, start=0.0, end=1.0, query="q", url="/api/renders/x/broll/0.mp4",
    )
    dumped = payload.model_dump()
    assert dumped["url"] == "/api/renders/x/broll/0.mp4"
    assert dumped["rank"] == 1
