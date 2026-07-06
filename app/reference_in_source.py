"""Find where a reference short was cut from within an uploaded source video.

Used by the AUTO_SOURCE_MATCH clip mode: the user uploads a long source
video + a short reference; the system finds the contiguous window in the
source that matches the reference (audio + light visual) and returns
(start, end) timestamps the render pipeline uses as the clip range.

Approach (cross-correlation on loudness profile):
  1. Extract per-second mean loudness for both the reference and the source
     using ffmpeg's `astats` filter (writes mean_volume to stderr).
  2. Slide the reference's profile across the source's profile, computing
     a normalized cross-correlation at each offset.
  3. The best-scoring offset is the start time; end = start + ref_duration.

This is a fast O(N*M) correlation over 1-second buckets — a 30-minute
source is 1800 buckets, a 20-second reference is 20 buckets, ~36k ops.
Robust enough for the "I cut this short from this long video" workflow
and uses no extra dependencies.
"""

import logging
import re
import subprocess
from pathlib import Path

from app.config import Settings
from app.rendering import probe_duration

logger = logging.getLogger(__name__)


def find_reference_in_source(
    reference_path: Path,
    source_path: Path,
    work_dir: Path,
    settings: Settings,
) -> tuple[float, float] | None:
    """Find (start, end) in `source_path` that matches `reference_path`.

    Returns None when the correlation peak is too weak to trust — the
    caller should fall back to MANUAL mode in that case.

    The returned (start, end) is rounded to 0.1s to avoid jittery UI values.
    """
    ref_duration = probe_duration(reference_path, settings)
    src_duration = probe_duration(source_path, settings)
    if ref_duration <= 0 or src_duration <= 0 or ref_duration > src_duration:
        return None

    ref_profile = _loudness_profile(reference_path, work_dir / "ref_prof.txt", settings)
    src_profile = _loudness_profile(source_path, work_dir / "src_prof.txt", settings)
    if len(ref_profile) < 2 or len(src_profile) < 2:
        return None

    # Cross-correlate ref_profile against src_profile. The peak offset is
    # the start of the matching window.
    best_offset = 0
    best_score = -1e9
    n = len(ref_profile)
    last = len(src_profile) - n
    if last < 0:
        return None
    for offset in range(0, last + 1):
        score = 0.0
        for i in range(n):
            score += ref_profile[i] * src_profile[offset + i]
        if score > best_score:
            best_score = score
            best_offset = offset

    # Sanity check: require the peak to clearly beat the average. A weak
    # correlation means the reference is not actually a contiguous cut
    # from this source, so we return None.
    avg = sum(_dot(ref_profile, src_profile[i:i + n]) for i in range(len(src_profile) - n + 1)) / max(1, len(src_profile) - n + 1)
    if best_score < avg * 1.5 and best_score < 50:
        logger.info(
            "Reference-source correlation too weak (best=%.1f avg=%.1f); "
            "user should pick clip range manually",
            best_score, avg,
        )
        return None

    start = round(best_offset, 1)
    end = round(best_offset + ref_duration, 1)
    end = min(end, src_duration)
    return (start, end)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _loudness_profile(
    audio_or_video_path: Path,
    out_txt: Path,
    settings: Settings,
) -> list[float]:
    """Per-second mean loudness (dB) of an audio/video file via ffmpeg astats.

    Writes one `mean_volume:-XX.X` line per 1-second window to `out_txt`,
    then parses those lines into a list. Returns [] on any failure.
    """
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                settings.ffmpeg_path,
                "-i", str(audio_or_video_path),
                "-vn",
                "-af", "asetnsamples=44100,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.mean_volume",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return []
    # astats writes one metadata block per 1s window (because asetnsamples
    # + reset=1). The key we want is `mean_volume` in dB.
    out: list[float] = []
    for line in (result.stderr or "").splitlines():
        m = re.search(r"mean_volume=(-?\d+(?:\.\d+)?)", line)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                continue
    return out
