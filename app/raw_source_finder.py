"""Find the raw source video and exact timestamp range from which a reference
short was cut, given a Frame.io folder containing long-form source footage.

Workflow:
  1. Sync every video from the Frame.io share into a local directory.
  2. Extract keyframes from the reference short.
  3. Two-pass scan:
     - Pass 1 (fast): one thumbnail per candidate → quick score.
     - Pass 2 (slow): top-3 candidates → full window search with multiple
       keyframes to pinpoint the matching timestamp range.
  4. Return (source_path, start, end, confidence_score) for the best match.
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.frameio_source import sync_frameio_share
from app.rendering import probe_duration

logger = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
# Cap on how many window centers a single candidate's Pass-2 scan will probe.
_MAX_WINDOW_PROBES = 12


@dataclass
class RecoveryCandidate:
    provider: str
    source: str
    clip_path: Path
    duration: float
    source_url: str = ""
    sample_time: float | None = None


def _library_index(directory: Path, settings: Settings) -> list[tuple[Path, float]]:
    """Every video file under `directory` with its probed duration."""
    if not directory.exists():
        return []
    results: list[tuple[Path, float]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _VIDEO_EXTS:
            continue
        try:
            duration = probe_duration(path, settings)
        except Exception:
            continue
        if duration > 0:
            results.append((path, duration))
    return results


def _library_sample_times(duration: float, count: int = 3) -> list[float]:
    """A few evenly-spread sample timestamps across a clip, for a quick
    thumbnail-based pre-score before the full window scan."""
    if duration <= 0:
        return []
    fractions = (0.2, 0.5, 0.8)[:max(1, count)]
    return [max(0.0, min(duration - 0.1, duration * frac)) for frac in fractions]


def _extract_frame_at(source: Path, at_seconds: float, output_path: Path, settings: Settings) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                settings.ffmpeg_path, "-y",
                "-ss", f"{max(0.0, at_seconds):.3f}",
                "-i", str(source),
                "-frames:v", "1", "-vf", "scale=320:-1",
                str(output_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 500


def _extract_reference_keyframes(
    source: Path,
    start: float,
    end: float,
    frames_dir: Path,
    settings: Settings,
    deadline: float | None = None,
    count: int = 5,
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    span = max(0.1, end - start)
    count = max(1, count)
    frames: list[Path] = []
    for i in range(count):
        if deadline is not None and time.monotonic() >= deadline:
            break
        at = start + span * (i + 0.5) / count
        frame_path = frames_dir / f"kf_{i:02d}.jpg"
        if _extract_frame_at(source, at, frame_path, settings):
            frames.append(frame_path)
    return frames


def _extract_reference_thumbnail(
    source: Path,
    at_seconds: float,
    output_path: Path,
    settings: Settings,
    deadline: float | None = None,
) -> Path | None:
    if output_path.exists() and output_path.stat().st_size > 1000:
        return output_path
    if deadline is not None and time.monotonic() >= deadline:
        return None
    return output_path if _extract_frame_at(source, at_seconds, output_path, settings) else None


def _phash(image_path: Path) -> int | None:
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    bit_index = 0
    for row in range(8):
        for col in range(8):
            if small[row, col] > small[row, col + 1]:
                bits |= 1 << bit_index
            bit_index += 1
    return bits


def _histogram_similarity(image_a: Path, image_b: Path) -> float:
    try:
        import cv2
    except ImportError:
        return 0.0
    a = cv2.imread(str(image_a))
    b = cv2.imread(str(image_b))
    if a is None or b is None:
        return 0.0
    hist_a = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0, 1], None, [8, 4], [0, 180, 0, 256])
    hist_b = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2HSV)], [0, 1], None, [8, 4], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return max(0.0, float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)))


def _frame_match_score(reference_frame: Path, candidate_frame: Path) -> tuple[float, bool]:
    """0..1 similarity plus an 'exact same source frame' flag (pHash)."""
    ref_hash = _phash(reference_frame)
    cand_hash = _phash(candidate_frame)
    phash_score = 0.0
    exact = False
    if ref_hash is not None and cand_hash is not None:
        distance = bin(ref_hash ^ cand_hash).count("1")
        exact = distance <= 8
        phash_score = max(0.0, 1.0 - distance / 32.0)
    hist_score = _histogram_similarity(reference_frame, candidate_frame)
    score = max(phash_score, 0.6 * phash_score + 0.4 * hist_score)
    return min(1.0, score), exact


def _score_candidate_frames(ref_frames: list[Path], candidate_frames: list[Path]) -> tuple[float, str]:
    if not ref_frames or not candidate_frames:
        return 0.0, "rejected"
    pair_scores: list[float] = []
    exact_hits = 0
    for ref in ref_frames:
        best = 0.0
        for candidate in candidate_frames:
            score, exact = _frame_match_score(ref, candidate)
            best = max(best, score)
            if exact:
                exact_hits += 1
        pair_scores.append(best)
    score = sum(pair_scores) / len(pair_scores)
    if exact_hits >= max(1, len(ref_frames) // 2):
        return max(score, 0.92), "exact_frame"
    if score >= 0.72:
        return score, "same_setup"
    return score, "rejected"


def _candidate_window_centers(source_duration: float, target_duration: float) -> list[float]:
    """Timestamps to probe across a candidate video, spaced ~target_duration/2
    apart so no plausible window is skipped, capped at _MAX_WINDOW_PROBES."""
    half = max(0.5, target_duration / 2)
    if source_duration <= target_duration:
        return [source_duration / 2]
    step = max(2.0, target_duration / 2)
    centers: list[float] = []
    t = half
    while t <= source_duration - half and len(centers) < _MAX_WINDOW_PROBES:
        centers.append(t)
        t += step
    return centers or [source_duration / 2]


def _find_best_candidate_window(
    ref_frames: list[Path],
    candidate: RecoveryCandidate,
    target_duration: float,
    output_dir: Path,
    settings: Settings,
    deadline: float | None = None,
) -> tuple[float, str, float, list[Path]] | None:
    """Slide a probe across the candidate video and return the best-matching
    (score, match_type, center_seconds, frames) window, or None if nothing
    could be sampled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    centers = _candidate_window_centers(candidate.duration, target_duration)
    best: tuple[float, str, float, list[Path]] | None = None
    for index, center in enumerate(centers):
        if deadline is not None and time.monotonic() >= deadline:
            break
        thumb = output_dir / f"probe_{index:02d}.jpg"
        extracted = _extract_reference_thumbnail(candidate.clip_path, center, thumb, settings, deadline=deadline)
        if extracted is None:
            continue
        score, match_type = _score_candidate_frames(ref_frames, [extracted])
        if best is None or score > best[0]:
            best = (score, match_type, center, [extracted])
    return best

# Number of reference keyframes to extract for matching.
SOURCE_FIND_KEYFRAMES = 5
# Minimum confidence to accept a match (same scale as _score_candidate_frames).
SOURCE_FIND_MIN_SCORE = 0.65
# How many top candidates from Pass 1 get a full Pass 2 deep scan.
SOURCE_FIND_DEEP_CANDIDATES = 3
# Wall-clock budget for the entire source-finding step (seconds).
SOURCE_FIND_BUDGET = 150.0


def find_raw_source(
    reference_path: Path,
    frameio_url: str,
    work_dir: Path,
    settings: Settings,
    deadline: float | None = None,
) -> tuple[Path, float, float, float] | None:
    """Find the raw source video in a Frame.io folder that contains the
    reference's footage, and return the exact timestamp range.

    Returns ``(source_video_path, start_seconds, end_seconds, confidence)``
    or ``None`` if no match above the confidence threshold is found.
    """
    if deadline is None:
        deadline = time.monotonic() + SOURCE_FIND_BUDGET

    raw_dir = work_dir / "frameio_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Sync Frame.io folder ──────────────────────────────────
    logger.info("Syncing Frame.io folder: %s", frameio_url)
    sync_result = sync_frameio_share(frameio_url, raw_dir)
    total_videos = len(sync_result["downloaded"]) + len(sync_result["skipped"])
    logger.info(
        "Frame.io sync complete: %d new, %d skipped, %d failed",
        len(sync_result["downloaded"]),
        len(sync_result["skipped"]),
        len(sync_result["failed"]),
    )
    if total_videos == 0:
        logger.warning("No videos found in the Frame.io folder")
        return None

    # ── Step 2: Extract reference keyframes ───────────────────────────
    ref_duration = probe_duration(reference_path, settings)
    if ref_duration <= 0:
        logger.warning("Could not probe reference duration")
        return None

    ref_frames_dir = work_dir / "source_find" / "ref_frames"
    ref_frames = _extract_reference_keyframes(
        reference_path,
        0.0,
        ref_duration,
        ref_frames_dir,
        settings,
        deadline=deadline,
        count=SOURCE_FIND_KEYFRAMES,
    )
    if not ref_frames:
        logger.warning("Could not extract keyframes from reference")
        return None
    logger.info("Extracted %d reference keyframes", len(ref_frames))

    # ── Step 3: Build candidate index from synced videos ─────────────
    indexed = _library_index(raw_dir, settings)
    # Filter: skip candidates shorter than the reference (can't contain it)
    candidates = [(p, d) for p, d in indexed if d >= ref_duration * 0.8]
    if not candidates:
        logger.warning(
            "No candidate videos long enough (ref=%.1fs, need >=%.1fs)",
            ref_duration,
            ref_duration * 0.8,
        )
        return None
    logger.info("Scanning %d candidate videos (of %d total)", len(candidates), len(indexed))

    # ── Pass 1: Fast single-frame scan ────────────────────────────────
    pass1_dir = work_dir / "source_find" / "pass1"
    pass1_dir.mkdir(parents=True, exist_ok=True)
    scored_candidates: list[tuple[float, Path, float]] = []

    for clip_path, duration in candidates:
        if time.monotonic() >= deadline:
            logger.warning("Pass 1 deadline reached after %d candidates", len(scored_candidates))
            break
        # Extract one thumbnail at the first sample time for a quick score
        sample_times = _library_sample_times(duration)
        if not sample_times:
            continue
        thumb = pass1_dir / f"{clip_path.stem}_{clip_path.suffix[1:]}_probe.jpg"
        extracted = _extract_reference_thumbnail(
            clip_path, sample_times[0], thumb, settings, deadline=deadline
        )
        if extracted is None:
            continue
        score, _ = _score_candidate_frames(ref_frames, [extracted])
        scored_candidates.append((score, clip_path, duration))

    if not scored_candidates:
        logger.warning("Pass 1 found no candidates with extractable frames")
        return None

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    logger.info(
        "Pass 1 top scores: %s",
        [(f"{s:.3f}", p.name) for s, p, d in scored_candidates[:5]],
    )

    # ── Pass 2: Deep window search on top candidates ─────────────────
    top_n = min(SOURCE_FIND_DEEP_CANDIDATES, len(scored_candidates))
    best_match: tuple[float, float, float, Path] | None = None  # (score, start, end, clip)

    for rank in range(top_n):
        _, clip_path, duration = scored_candidates[rank]
        if time.monotonic() >= deadline:
            break
        candidate = RecoveryCandidate(
            provider="frameio_raw",
            source=str(clip_path),
            clip_path=clip_path,
            duration=duration,
            sample_time=duration / 2,
        )
        window_dir = work_dir / "source_find" / f"pass2_cand{rank}"
        window_result = _find_best_candidate_window(
            ref_frames=ref_frames,
            candidate=candidate,
            target_duration=ref_duration,
            output_dir=window_dir,
            settings=settings,
            deadline=deadline,
        )
        if window_result is None:
            continue
        score, match_type, center, frames = window_result
        start = max(0.0, center - ref_duration / 2)
        end = min(duration, start + ref_duration)

        logger.info(
            "Pass 2 candidate %d (%s): score=%.3f type=%s range=[%.1f, %.1f]",
            rank,
            clip_path.name,
            score,
            match_type,
            start,
            end,
        )

        if best_match is None or score > best_match[0]:
            best_match = (score, start, end, clip_path)

    if best_match is None:
        logger.warning("Pass 2 found no matching window in any candidate")
        return None

    score, start, end, clip_path = best_match
    if score < SOURCE_FIND_MIN_SCORE:
        logger.warning(
            "Best match score %.3f below threshold %.3f — no source found",
            score,
            SOURCE_FIND_MIN_SCORE,
        )
        return None

    logger.info(
        "Source found: %s @ [%.1f, %.1f]s (confidence=%.3f)",
        clip_path.name,
        start,
        end,
        score,
    )
    return clip_path, start, end, score
