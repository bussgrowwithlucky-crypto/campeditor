"""CLI demo for the broll_intelligence system.

Usage::

    python -m broll_intelligence.demo --reference <video> --top-k 5
    python -m broll_intelligence.demo --reference <video> --library-dir <path>
    python -m broll_intelligence.demo --reference <video> --json-out pack.json

Pretty-prints the BrollPack to the terminal and (optionally) writes the
full JSON to ``--json-out``. Runs in OFFLINE mode if no API keys are
present — the system warns clearly and runs with whatever quantitative
fields it can extract (no vision calls).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .pipeline import BrollPack, select_broll


def _format_pack(pack: BrollPack) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("B-roll Intelligence — selection")
    lines.append("=" * 72)

    ref = pack.reference
    lines.append(f"Reference  : subjects={ref.subjects!r}  setting={ref.setting!r}")
    lines.append(f"            category={ref.category!r}  mood={ref.mood!r}  energy={ref.energy!r}")
    lines.append(f"            lighting={ref.lighting!r}  shot={ref.shot_type!r}  motion={ref.camera_motion!r}")
    lines.append(
        f"            palette warmth/sat/bright = "
        f"{ref.palette_warmth:.2f}/{ref.palette_saturation:.2f}/{ref.palette_brightness:.2f}"
    )
    lines.append("")

    lines.append(f"Picks (top {len(pack.items)}, rungs fired: {pack.diagnostics.get('rungs_fired')}):")
    for i, it in enumerate(pack.items, start=1):
        fv = it.features
        vibe = ""
        if fv is not None:
            vibe = (
                f"mood={fv.mood!r}  light={fv.lighting!r}  "
                f"palette={fv.color_palette!r}  shot={fv.shot_type!r}"
            )
        loc = str(it.path) if it.path else (it.url or "<n/a>")
        lines.append(f"  {i}. [{it.source:>15}]  score={it.score:.3f}  -> {loc}")
        if vibe:
            lines.append(f"      {vibe}")
        if it.notes:
            lines.append(f"      notes: {it.notes}")

    diag = pack.diagnostics
    lines.append("")
    lines.append(
        f"Diagnostics: elapsed={diag.get('elapsed_seconds')}s  "
        f"library_picks={diag.get('library_picks', 0)}  "
        f"youtube_picks={diag.get('youtube_picks', 0)}  "
        f"ref_crop={diag.get('reference_crop_picks', 0)}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m broll_intelligence.demo",
        description=(
            "Run the broll_intelligence selection pipeline against a reference video. "
            "Prints a ranked B-roll pack to stdout; optionally writes the full pack to JSON."
        ),
    )
    parser.add_argument(
        "--reference", required=True, type=Path,
        help="Path to the reference video whose B-rolls you want to replace.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of picks to return (default 5).",
    )
    parser.add_argument(
        "--library-dir", type=Path, default=None,
        help="Override the local B-roll library root (defaults to settings.library_dir).",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="If set, write the full BrollPack JSON to this path.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override the YouTube preview cache directory.",
    )
    args = parser.parse_args(argv)

    if not args.reference.exists():
        print(f"error: reference video not found: {args.reference}", file=sys.stderr)
        return 2

    settings = get_settings()
    if args.library_dir is not None:
        settings.library_dir = args.library_dir

    # Offline-mode detection: if no LLM keys, warn clearly.
    has_llm = bool(settings.groq_api_key or settings.nvidia_keys() or settings.gemini_api_key)
    if not has_llm:
        print(
            "warning: no LLM API keys configured (GROQ_API_KEY / NVIDIA_API_KEY / "
            "GEMINI_API_KEY). Running in OFFLINE mode — library rung only, no YouTube rung.",
            file=sys.stderr,
        )

    pack = select_broll(
        ref_video=args.reference,
        top_k=args.top_k,
        cache_dir=args.cache_dir,
        settings=settings,
    )
    print(_format_pack(pack))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(pack.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())