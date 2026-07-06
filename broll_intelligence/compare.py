"""Side-by-side comparison: new pipeline vs legacy campeditor heuristic.

Usage::

    python -m broll_intelligence.compare --reference <video> --top-k 5
    python -m broll_intelligence.compare --reference <video> --output report.md

Produces a markdown report with three sections:

  1. Reference — path + FeatureVector summary
  2. Picks — side-by-side table of new vs old rankings
  3. Vibe similarity (new only) — per-pick, one-line vibe note
  4. Notes — what the new system did differently + caveats

The "old" path runs :mod:`broll_intelligence.baseline` which is a faithful
inline re-implementation of the legacy ``app/broll.py::_local_score`
weights and `_truncate_query` cap. It MUST NOT import from app.*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .baseline import BaselinePick, local_score, rank_library_legacy, youtube_query_legacy
from .config import get_settings
from .feature_vector import FeatureVector
from .vibe_extractor import extract_from_video
from .library_indexer import build_library_index, load_index_as_clips
from .matcher import score_clip
from .pipeline import LIBRARY_ACCEPT_THRESHOLD, BrollItem, BrollPack, select_broll


def _vibe_summary(fv: FeatureVector) -> str:
    if fv is None:
        return "—"
    parts: list[str] = []
    if fv.mood:
        parts.append(f"mood={','.join(fv.mood[:2])}")
    if fv.lighting:
        parts.append(f"light={fv.lighting}")
    if fv.color_palette:
        parts.append(f"palette={','.join(fv.color_palette[:2])}")
    if fv.shot_type:
        parts.append(f"shot={fv.shot_type}")
    if fv.camera_motion:
        parts.append(f"motion={fv.camera_motion}")
    return " / ".join(parts) if parts else "—"


def _legacy_query(reference: FeatureVector) -> str:
    return youtube_query_legacy(reference)


def _build_legacy_picks(
    reference: FeatureVector,
    settings,
    top_k: int,
) -> list[BaselinePick]:
    if not settings.library_dir.exists():
        return []
    try:
        build_library_index(settings)
    except Exception:
        pass
    clips = load_index_as_clips(settings)
    return rank_library_legacy(reference, [(c.path, c.features) for c in clips], top_k=top_k)


def _vibe_note(reference: FeatureVector, candidate_features: FeatureVector) -> str:
    """A short human-readable vibe match note for the new pipeline's pick."""
    if candidate_features is None:
        return "no preview features available"
    if candidate_features.confidence <= 0.0:
        return "empty-sentinel fallback"
    s = score_clip(reference, candidate_features)
    if s >= 0.8:
        return f"strong match ({s:.2f})"
    if s >= 0.6:
        return f"good match ({s:.2f})"
    if s >= 0.4:
        return f"weak match ({s:.2f})"
    return f"poor match ({s:.2f})"


def _render_markdown(
    reference_path: Path,
    reference: FeatureVector,
    new_pack: BrollPack,
    legacy_picks: list[BaselinePick],
) -> str:
    out: list[str] = []
    out.append("# B-roll Selection: New vs Old")
    out.append("")
    out.append("## Reference")
    out.append("")
    out.append(f"- path: `{reference_path}`")
    fv = reference
    out.append(
        f"- subjects: `{fv.subjects}`  setting: `{fv.setting}`  "
        f"category: `{fv.category}`"
    )
    out.append(f"- mood: `{fv.mood}`  energy: `{fv.energy}`  lighting: `{fv.lighting}`")
    out.append(
        f"- shot_type: `{fv.shot_type}`  camera_motion: `{fv.camera_motion}`  "
        f"dof: `{fv.depth_of_field}`"
    )
    out.append(
        f"- palette: `{fv.color_palette}`  "
        f"warmth/sat/bright: `{fv.palette_warmth:.2f}/{fv.palette_saturation:.2f}/{fv.palette_brightness:.2f}`"
    )
    out.append("")

    # Side-by-side table
    out.append("## Picks (side-by-side)")
    out.append("")
    out.append("| # | New (source, score, vibe) | Old (subject score, top subject) |")
    out.append("|---|---|---|")
    rows = max(len(new_pack.items), len(legacy_picks))
    for i in range(rows):
        new_cell = "—"
        if i < len(new_pack.items):
            it = new_pack.items[i]
            vibe = _vibe_summary(it.features) if it.features else "—"
            new_cell = f"`{it.source}` / {it.score:.3f} / {vibe}"
        old_cell = "—"
        if i < len(legacy_picks):
            p = legacy_picks[i]
            top_subj = p.subjects[0] if p.subjects else "—"
            old_cell = f"{p.subject_score:.3f} / `{top_subj}` ({p.category})"
        out.append(f"| {i+1} | {new_cell} | {old_cell} |")
    out.append("")

    # Vibe similarity (new only)
    out.append("## Vibe similarity (new system only)")
    out.append("")
    for i, it in enumerate(new_pack.items, start=1):
        note = _vibe_note(reference, it.features) if it.features else "no features"
        out.append(f"- Pick {i} (`{it.source}`, score={it.score:.3f}): {note}")
    out.append("")

    # Notes
    out.append("## Notes")
    out.append("")
    out.append(
        f"- New ladder rungs fired: `{new_pack.diagnostics.get('rungs_fired')}` "
        f"(library threshold {LIBRARY_ACCEPT_THRESHOLD})."
    )
    out.append(
        f"- Old heuristic uses subject/category overlap only "
        f"(`local_score` weights 3.0/1.0/1.0, max 8.0). Library threshold default 0.35."
    )
    out.append(
        f"- Old YouTube query (single string, 8 words): "
        f"\"{_legacy_query(reference)}\""
    )
    if len(new_pack.items) == 1 and new_pack.items[0].source == "reference_crop":
        out.append(
            "- The new system fell through to a reference-crop placeholder. "
            "Either the library is empty / untagged, or no B-roll cleared the "
            "vibe threshold. The placeholder carries the reference FeatureVector "
            "so a follow-up crop implementation can consume it directly."
        )
    out.append("")
    out.append(
        "- Caveat: the old system was applied only to the local library. The new "
        "system, when LLM keys are present, additionally tries a YouTube rung with "
        "vibe-aware multi-angle queries (subject / mood / lighting / scene / aesthetic)."
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m broll_intelligence.compare",
        description=(
            "Run the broll_intelligence new pipeline AND a faithful baseline of the "
            "legacy campeditor B-roll scoring, then write a side-by-side markdown "
            "report so a human can compare picks."
        ),
    )
    parser.add_argument(
        "--reference", required=True, type=Path,
        help="Path to the reference video.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of picks per system (default 5).",
    )
    parser.add_argument(
        "--library-dir", type=Path, default=None,
        help="Override the local B-roll library root.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output markdown path (default data/broll_intelligence_comparison.md).",
    )
    args = parser.parse_args(argv)

    if not args.reference.exists():
        print(f"error: reference video not found: {args.reference}", file=sys.stderr)
        return 2

    settings = get_settings()
    if args.library_dir is not None:
        settings.library_dir = args.library_dir

    # New system
    new_pack = select_broll(
        ref_video=args.reference, top_k=args.top_k, settings=settings,
    )

    # Old / baseline system — build the reference FeatureVector with the same
    # pipeline as the new system so both see the same input.
    reference = extract_from_video(args.reference, settings)
    legacy_picks = _build_legacy_picks(reference, settings, args.top_k)

    md = _render_markdown(args.reference, reference, new_pack, legacy_picks)
    out_path = args.output or (settings.data_dir / "broll_intelligence_comparison.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())