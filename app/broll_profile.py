"""Global B-roll learning profile.

Every analyzed reference short feeds a running-average profile of where B-roll
tends to sit (placement fractions), how long/frequent cutaways are, and the
most common scene queries. Raw uploads with no reference reuse this profile to
synthesize B-roll spans, so the app develops "a good sense of B-roll" over time.
"""

import logging
import threading
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.models import ReferenceAnalysis

logger = logging.getLogger(__name__)

MIN_SYNTH_SPAN = 0.4
_MAX_QUERIES = 50
_MAX_PLACEMENTS = 40

# Process-wide lock around the load -> modify -> save sequence. The Pipeline
# uses worker_count=2 by default, so two replicate-mode jobs really do run
# concurrently and their `analyze_reference` calls can land within milliseconds
# of each other. Without this lock both threads read the same on-disk profile,
# compute updates independently, and the second save silently overwrites the
# first's contribution — one job's B-roll learning is lost with no error.
_PROFILE_LOCK = threading.Lock()


class BrollProfile(BaseModel):
    sample_count: int = 0
    avg_span_count_per_30s: float = 0.0
    avg_span_duration_s: float = 0.0
    avg_first_span_start_s: float = 0.0
    avg_gap_between_spans_s: float = 0.0
    query_counts: dict[str, int] = Field(default_factory=dict)
    placement_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def common_queries(self) -> list[tuple[str, int]]:
        return sorted(self.query_counts.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_QUERIES]

    @property
    def common_placement_fractions(self) -> list[tuple[float, float]]:
        top = sorted(self.placement_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
        fractions: list[tuple[float, float]] = []
        for key, _count in top:
            start_str, _, dur_str = key.partition(",")
            try:
                fractions.append((float(start_str), float(dur_str)))
            except ValueError:
                continue
        return fractions


def _profile_path(settings: Settings) -> Path:
    return settings.data_dir / "broll_profile.json"


def load_broll_profile(settings: Settings) -> BrollProfile:
    path = _profile_path(settings)
    if not path.exists():
        return BrollProfile()
    try:
        return BrollProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return BrollProfile()


def save_broll_profile(profile: BrollProfile, settings: Settings) -> None:
    path = _profile_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def update_broll_profile_atomic(
    settings: Settings, analysis: ReferenceAnalysis
) -> BrollProfile | None:
    """Atomic load -> modify -> save under the process-wide profile lock.

    Returns the new profile on success, or None when the analysis contributed
    nothing (no spans / zero duration — `update_broll_profile` returns the
    input unchanged in that case so there's nothing to persist).

    Callers from concurrent workers MUST go through this rather than the raw
    `load_broll_profile` + `update_broll_profile` + `save_broll_profile`
    sequence, or two simultaneous updates will race and one will be lost.
    """
    with _PROFILE_LOCK:
        try:
            profile = load_broll_profile(settings)
            updated = update_broll_profile(profile, analysis)
            if updated is profile:
                return None  # nothing to persist (empty analysis)
            save_broll_profile(updated, settings)
            return updated
        except Exception:
            logger.exception("Atomic broll profile update failed")
            return None


def update_broll_profile(profile: BrollProfile, analysis: ReferenceAnalysis) -> BrollProfile:
    spans = list(analysis.broll_spans or [])
    duration = analysis.duration or 0.0
    if not spans or duration <= 0:
        return profile

    n = profile.sample_count
    span_count_per_30s = len(spans) / duration * 30.0
    span_durations = [max(0.0, end - start) for start, end, _q in spans]
    span_duration_s = sum(span_durations) / len(span_durations)
    first_span_start_s = spans[0][0]
    gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
    gap_between_spans_s = sum(gaps) / len(gaps) if gaps else 0.0

    def running(old: float, new: float) -> float:
        return (old * n + new) / (n + 1)

    query_counts = dict(profile.query_counts)
    for _start, _end, query in spans:
        cleaned = (query or "").strip()
        if cleaned:
            query_counts[cleaned] = query_counts.get(cleaned, 0) + 1
    if len(query_counts) > _MAX_QUERIES:
        query_counts = dict(
            sorted(query_counts.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_QUERIES]
        )

    placement_counts = dict(profile.placement_counts)
    for start, end, _q in spans:
        start_frac = round(min(1.0, max(0.0, start / duration)), 2)
        dur_frac = round(min(1.0, max(0.0, (end - start) / duration)), 2)
        key = f"{start_frac},{dur_frac}"
        placement_counts[key] = placement_counts.get(key, 0) + 1
    if len(placement_counts) > _MAX_PLACEMENTS:
        placement_counts = dict(
            sorted(placement_counts.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_PLACEMENTS]
        )

    return profile.model_copy(
        update={
            "sample_count": n + 1,
            "avg_span_count_per_30s": running(profile.avg_span_count_per_30s, span_count_per_30s),
            "avg_span_duration_s": running(profile.avg_span_duration_s, span_duration_s),
            "avg_first_span_start_s": running(profile.avg_first_span_start_s, first_span_start_s),
            "avg_gap_between_spans_s": running(profile.avg_gap_between_spans_s, gap_between_spans_s),
            "query_counts": query_counts,
            "placement_counts": placement_counts,
        }
    )


def synthesize_broll_spans(
    profile: BrollProfile, clip_duration: float
) -> list[tuple[float, float, str]]:
    """Build B-roll spans for a raw clip from the learned profile.

    Span count scales with the learned per-30s frequency; placements rotate
    through the most common learned fractions; queries rotate by frequency.
    All spans are clamped to 0..clip_duration.
    """
    if profile.sample_count <= 0 or clip_duration <= 0:
        return []
    span_count = round(profile.avg_span_count_per_30s * clip_duration / 30.0)
    if span_count <= 0:
        return []

    placements = profile.common_placement_fractions
    if not placements:
        fallback_dur = profile.avg_span_duration_s / clip_duration if clip_duration > 0 else 0.1
        placements = [(0.1, max(0.05, min(0.3, fallback_dur)))]
    queries = [query for query, _count in profile.common_queries] or [""]

    spans: list[tuple[float, float, str]] = []
    for i in range(span_count):
        start_frac, dur_frac = placements[i % len(placements)]
        start = max(0.0, min(clip_duration, start_frac * clip_duration))
        duration = dur_frac * clip_duration if dur_frac > 0 else profile.avg_span_duration_s
        if duration <= 0:
            duration = MIN_SYNTH_SPAN
        end = min(clip_duration, start + duration)
        if end - start < MIN_SYNTH_SPAN:
            continue
        spans.append((start, end, queries[i % len(queries)]))
    spans.sort(key=lambda span: span[0])
    return spans
