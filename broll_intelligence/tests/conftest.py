"""Shared fixtures for the broll_intelligence test suite.

* `sample_vision_response()` — a fully-populated, valid vision JSON dict
  matching the spec exactly. Tests use this for "happy path" scenarios.
* `empty_vision_response()` — an empty dict, mimicking "every provider
  failed / model hallucinated non-JSON".
* `mock_vision_response()` — wraps a canned dict in a callable that mimics
  the signature of `vision_ladder.call` so tests can monkeypatch
  `vibe_extractor.vision_call` with it.
* `make_fake_video(tmp_path)` — generates a small mp4 with ffmpeg if
  available, otherwise writes a tiny placeholder file. Either way the
  returned Path has a real mtime + size that the indexer can stat.
* `video_files` fixture — yields 3 such files in a tmp library dir.
* `index_settings` fixture — a `Settings` instance pointed at a tmp
  library dir + tmp index cache path. Also patches the env so
  `pydantic-settings` doesn't pick up real API keys from the workspace
  .env during tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest


# ---------------------------------------------------------------------------
# Vision response fixtures
# ---------------------------------------------------------------------------


SAMPLE_VISION_RESPONSE: dict[str, Any] = {
    "subjects": ["astronaut", "lunar rover"],
    "setting": ["lunar surface"],
    "action": ["walking"],
    "category": "movie",
    "query": "astronaut walking on lunar surface",
    "mood": ["mysterious", "epic"],
    "energy": "low",
    "lighting": "low-key",
    "color_palette": ["deep blue", "silver"],
    "shot_type": "wide",
    "camera_motion": "tracking",
    "depth_of_field": "deep",
}


EMPTY_VISION_RESPONSE: dict[str, Any] = {}


def sample_vision_response() -> dict[str, Any]:
    """Fresh deep copy so a test can mutate without bleeding into siblings."""
    return json.loads(json.dumps(SAMPLE_VISION_RESPONSE))


def empty_vision_response() -> dict[str, Any]:
    return dict(EMPTY_VISION_RESPONSE)


def mock_vision_response(
    response: dict[str, Any] | str,
    *,
    raise_exc: Exception | None = None,
) -> Callable[..., str]:
    """Return a vision_ladder-compatible callable that yields `response` (as
    JSON if it's a dict). If `raise_exc` is given, the callable raises it
    instead of returning — simulating a totally broken ladder."""

    def _call(image_path, prompt, settings):  # signature matches vision_ladder.call
        if raise_exc is not None:
            raise raise_exc
        if isinstance(response, dict):
            return json.dumps(response)
        return str(response)

    return _call


# ---------------------------------------------------------------------------
# Fake video files
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def make_fake_video(
    target: Path,
    *,
    duration_seconds: float = 0.5,
    color: str = "0x202030",
    width: int = 64,
    height: int = 48,
) -> Path:
    """Create a tiny mp4 at `target`. Tries real ffmpeg first; falls back to a
    tiny placeholder file (still has valid mtime + size so the indexer can
    stat it, but won't be decodable)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if _ffmpeg_available():
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f", "lavfi",
                    "-i", f"color=c={color}:s={width}x{height}:d={duration_seconds}",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
                return target
        except (subprocess.TimeoutExpired, OSError):
            pass
    # Fallback: a small but non-zero file. The indexer's mtime/size check
    # works fine on this; the extractor is monkeypatched in tests so the
    # content never has to be a real video.
    target.write_bytes(b"\x00" * 4096)
    return target


# ---------------------------------------------------------------------------
# Settings + tmpdir fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_library_dir(tmp_path: Path) -> Path:
    """A fresh empty library dir per test."""
    d = tmp_path / "library"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def video_files(fake_library_dir: Path) -> list[Path]:
    """Three tiny mp4 files in the fake library."""
    paths = [
        make_fake_video(fake_library_dir / "clip_a.mp4", color="0x101040"),
        make_fake_video(fake_library_dir / "clip_b.mp4", color="0x402010"),
        make_fake_video(fake_library_dir / "clip_c.mp4", color="0x104020"),
    ]
    # Ensure stable sort order — give each a distinct mtime.
    for i, p in enumerate(paths):
        stat = p.stat()
        os.utime(p, (stat.st_atime, stat.st_mtime - (len(paths) - i)))
    return paths


@pytest.fixture
def index_settings(fake_library_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Settings instance pointed at the fake library + a tmp index path.
    Also blanks the API keys in os.environ so pydantic-settings doesn't pick
    up a real key from the workspace .env (which would change behavior if a
    key is missing here but present there — we want hermetic tests)."""
    from broll_intelligence.config import Settings

    # Blank API keys for hermetic tests. The vision ladder is monkeypatched
    # in every test, so these values don't matter beyond "Settings() loads".
    for k in (
        "GROQ_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_FALLBACK_API_KEY",
        "NVIDIA_FALLBACK_API_KEY_2",
        "NVIDIA_FALLBACK_API_KEY_3",
        "GEMINI_API_KEY",
        "OLLAMA_VISION_MODEL",
    ):
        monkeypatch.setenv(k, "")

    settings = Settings(
        library_dir=fake_library_dir,
        index_path=tmp_path / "broll_intelligence_index.json",
        data_dir=tmp_path,
    )
    return settings


@pytest.fixture
def reset_vision_cooldowns():
    """Clear vision_ladder's cooldown dict between tests so a 429 in one
    test doesn't poison the next."""
    from broll_intelligence import vision_ladder

    vision_ladder.reset_cooldowns()
    yield
    vision_ladder.reset_cooldowns()


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Force Settings.data_dir to a tmp path so any cache writes land
    somewhere disposable rather than the real workspace."""
    monkeypatch.setenv("CAMPEDITOR_DATA_DIR", str(tmp_path))
    yield tmp_path