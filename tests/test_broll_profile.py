from app.broll_profile import (
    BrollProfile,
    load_broll_profile,
    save_broll_profile,
    synthesize_broll_spans,
    update_broll_profile,
)
from app.config import Settings
from app.models import ReferenceAnalysis


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.data_dir = tmp_path
    return settings


def test_empty_profile_loads_cleanly(tmp_path):
    profile = load_broll_profile(_settings(tmp_path))
    assert profile.sample_count == 0
    assert profile.common_queries == []
    assert profile.common_placement_fractions == []


def test_update_profile_with_one_reference_records_averages(tmp_path):
    analysis = ReferenceAnalysis(
        duration=30.0,
        broll_spans=[(3.0, 5.0, "city street"), (12.0, 14.0, "coffee cup")],
    )
    profile = update_broll_profile(BrollProfile(), analysis)
    assert profile.sample_count == 1
    assert profile.avg_span_count_per_30s == 2.0
    assert profile.avg_span_duration_s == 2.0
    assert profile.avg_first_span_start_s == 3.0
    assert dict(profile.common_queries).get("city street") == 1

    settings = _settings(tmp_path)
    save_broll_profile(profile, settings)
    reloaded = load_broll_profile(settings)
    assert reloaded.sample_count == 1
    assert dict(reloaded.common_queries).get("coffee cup") == 1


def test_update_profile_with_multiple_references_converges():
    a1 = ReferenceAnalysis(duration=30.0, broll_spans=[(0.0, 3.0, "q1")])
    a2 = ReferenceAnalysis(duration=30.0, broll_spans=[(0.0, 3.0, "q1"), (10.0, 13.0, "q2")])
    profile = update_broll_profile(BrollProfile(), a1)
    profile = update_broll_profile(profile, a2)
    assert profile.sample_count == 2
    assert abs(profile.avg_span_count_per_30s - 1.5) < 1e-6
    assert dict(profile.common_queries)["q1"] == 2


def test_generated_spans_match_clip_duration():
    analysis = ReferenceAnalysis(
        duration=30.0,
        broll_spans=[(3.0, 5.0, "a"), (12.0, 14.0, "b"), (20.0, 22.0, "c")],
    )
    profile = update_broll_profile(BrollProfile(), analysis)
    spans = synthesize_broll_spans(profile, clip_duration=15.0)
    assert spans
    for start, end, _query in spans:
        assert 0.0 <= start < end <= 15.0
