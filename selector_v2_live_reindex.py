"""Live re-index sanity check (SPEC section 13, verification step 2).

Temporarily bumps `_INDEX_CACHE_VERSION` in app/broll.py to 3 (so the
existing v1 cache is treated as empty), monkeypatches `app.broll._vision`
AND `app.broll.probe_duration` to keep the run fast and deterministic,
then runs `build_library_index` against the real B-Roll library at
C:/campeditor/data/broll_library/B-Roll and prints one rebuilt clip
entry (all 12 fields).

The version bump is REVERTED at the end so the runtime cache version
remains 2.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import app.broll as broll_mod
import app.rendering as rendering_mod
from app.config import Settings

# Plain logging — no print bypass of sys.excepthook.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selector_v2_live_reindex")


# ---------------------------------------------------------------------------
# Canned per-hash vision responses — mirror the shape from
# broll_intelligence/tests/conftest.py::SAMPLE_VISION_RESPONSE plus a
# couple of variants so different frames don't all collapse to one clip
# tag.
# ---------------------------------------------------------------------------
CANNED_VISION_RESPONSE_A: dict = {
    "subjects": ["player", "court"],
    "setting": ["stadium", "indoors"],
    "action": ["dribbling"],
    "category": "sports",
    "query": "basketball player dribbling on court",
    "mood": ["energetic", "epic"],
    "energy": "high",
    "lighting": "mixed",
    "shot_type": "medium",
    "camera_motion": "tracking",
    "depth_of_field": "shallow",
    "color_palette": ["amber", "deep blue"],
}

CANNED_VISION_RESPONSE_B: dict = {
    "subjects": ["player", "crowd"],
    "setting": ["stadium", "indoors"],
    "action": ["celebrating"],
    "category": "sports",
    "query": "celebration in stadium with crowd",
    "mood": ["uplifting"],
    "energy": "high",
    "lighting": "high-key",
    "shot_type": "wide",
    "camera_motion": "pan",
    "depth_of_field": "deep",
    "color_palette": ["white", "orange"],
}


def _hash_to_response(frame_path: Path) -> dict:
    name = frame_path.name
    if "_33." in name:
        return CANNED_VISION_RESPONSE_A
    return CANNED_VISION_RESPONSE_B


# ---------------------------------------------------------------------------
# Step 1: bump _INDEX_CACHE_VERSION to 3 in-memory only.
# ---------------------------------------------------------------------------
ORIGINAL_VERSION = broll_mod._INDEX_CACHE_VERSION
logger.info("Original _INDEX_CACHE_VERSION = %s", ORIGINAL_VERSION)
broll_mod._INDEX_CACHE_VERSION = 3
logger.info("Bumped _INDEX_CACHE_VERSION to 3 (in-memory only)")

# Save originals so we can revert the monkeypatches.
ORIGINAL_VISION = broll_mod._vision
ORIGINAL_PROBE = rendering_mod.probe_duration


def _canned_vision(image_path, prompt, settings_arg, budget=None):  # noqa: ARG001
    payload = _hash_to_response(Path(image_path))
    return json.dumps(payload)


def _fake_probe_duration(source_path, settings_arg):  # noqa: ARG001
    """Return a deterministic duration so we don't shell out to ffprobe for
    every one of 290 clips. 2.0s is enough for `_extract_single_frame` to
    pick a midpoint; we also monkeypatch that to a no-op below."""
    return 2.0


def _noop_extract_single_frame(*_a, **_kw):  # noqa: ANN001
    """Create an empty placeholder file at the requested path. _frame_tags
    only checks `.exists()` so the file just needs to be on disk; the
    monkeypatched _vision ignores the bytes anyway."""
    target = _a[2] if len(_a) >= 3 else _kw.get("target")
    if target is None:
        return False
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)  # tiny jpeg-ish stub
    return True


try:
    # ------------------------------------------------------------------
    # Step 2: build a temp Settings pointed at a temp data_dir so the
    # real data/cache is not clobbered.
    # ------------------------------------------------------------------
    tmp_data = Path(tempfile.mkdtemp(prefix="campeditor_v2_live_"))
    logger.info("Temp data_dir = %s", tmp_data)

    settings = Settings(
        data_dir=tmp_data,
        broll_library_dir=Path("C:/campeditor/data/broll_library/B-Roll"),
    )
    settings.groq_api_key = ""
    settings.nvidia_api_key = ""
    settings.nvidia_fallback_api_key = ""
    settings.nvidia_fallback_api_key_2 = ""
    settings.nvidia_fallback_api_key_3 = ""
    settings.gemini_api_key = ""

    # ------------------------------------------------------------------
    # Step 3: monkeypatch app.broll._vision AND probe_duration AND
    # _extract_single_frame — keep the run fast and deterministic.
    # ------------------------------------------------------------------
    broll_mod._vision = _canned_vision
    rendering_mod.probe_duration = _fake_probe_duration
    broll_mod.probe_duration = _fake_probe_duration
    broll_mod._extract_single_frame = _noop_extract_single_frame
    logger.info("Monkeypatched _vision, probe_duration, _extract_single_frame")

    # ------------------------------------------------------------------
    # Step 4: force a rebuild — build_library_index.
    # ------------------------------------------------------------------
    logger.info("Calling build_library_index...")
    clips = broll_mod.build_library_index(settings)
    logger.info("build_library_index returned %d LibraryClip objects", len(clips))

    # ------------------------------------------------------------------
    # Step 5: verify the rebuilt index.json on disk has clip entries
    # with all 12 fields populated, print one.
    # ------------------------------------------------------------------
    cache_path = settings.data_dir / "cache" / "broll_index.json"
    if not cache_path.exists():
        raise SystemExit(f"FATAL: expected index cache at {cache_path} but it doesn't exist")
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert raw.get("version") == 3, f"cache version should be 3, got {raw.get('version')}"
    clip_entries = raw.get("clips", {})
    print()
    print(f"Rebuilt index.json: version={raw['version']}  clip_count={len(clip_entries)}")
    if clip_entries:
        sample_key, sample_entry = next(iter(clip_entries.items()))
        print(f"Sample clip entry (key={sample_key!r}):")
        print(json.dumps(sample_entry, indent=2, ensure_ascii=False))
        required_fields = [
            "mtime", "size", "subjects", "setting", "category", "folder", "query",
            "mood", "energy", "lighting", "shot_type", "camera_motion",
            "depth_of_field", "color_palette",
        ]
        missing = [f for f in required_fields if f not in sample_entry]
        if missing:
            print(f"MISSING FIELDS on the sample entry: {missing}")
            raise SystemExit(1)
        print(f"All 14 required fields present on the sample entry.")
    else:
        raise SystemExit("FATAL: rebuilt index had no clip entries.")

    # And confirm the same LibraryClip list mirrors the on-disk cache.
    populated_in_memory = [c for c in clips if c.mood or c.energy or c.lighting
                           or c.shot_type or c.camera_motion or c.depth_of_field]
    print(f"LibraryClip objects with non-empty cinema/vibe fields: {len(populated_in_memory)} / {len(clips)}")

finally:
    # ------------------------------------------------------------------
    # Step 6: revert the in-memory version bump AND the monkeypatches.
    # The source file was never modified.
    # ------------------------------------------------------------------
    broll_mod._INDEX_CACHE_VERSION = ORIGINAL_VERSION
    broll_mod._vision = ORIGINAL_VISION
    rendering_mod.probe_duration = ORIGINAL_PROBE
    broll_mod.probe_duration = ORIGINAL_PROBE
    logger.info("Reverted _INDEX_CACHE_VERSION to %s and restored _vision / probe_duration",
                ORIGINAL_VERSION)
    try:
        shutil.rmtree(tmp_data, ignore_errors=True)
        logger.info("Removed temp data_dir %s", tmp_data)
    except Exception:
        pass

print()
print("LIVE RE-INDEX SANITY CHECK: PASS")