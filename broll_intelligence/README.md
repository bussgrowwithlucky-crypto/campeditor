# broll_intelligence

Intelligent B-roll selection for campeditor. Captures deeper aesthetics
— mood, lighting, cinematography, motion, color palette — that the
existing `app/broll.py` matching ignores, so a cinematic dark moody
reference cutaway can find a library clip that *looks* like it, not
just one with overlapping category tags.

**This package is not yet integrated into the running campeditor
pipeline.** It lives as a sibling under `C:\campeditor\broll_intelligence\`
and can be exercised standalone against the same `.env`, library
folder, and vision providers that `app/broll.py` uses.

---

## What's here

| module | purpose |
| --- | --- |
| `feature_vector.py` | The public dataclass schema (`FeatureVector`) — subject, vibe, cinematography, CV features, provenance. |
| `vision_ladder.py` | Groq → NVIDIA (up to 4 keys) → Gemini → Ollama ladder. Same cooldown semantics as `app/broll.py`. |
| `vibe_extractor.py` | `extract_from_video(path, settings)` — ffmpeg frame grab + vision pass + OpenCV quantitative pass → `FeatureVector`. |
| `library_indexer.py` | `build_library_index(settings)` — incremental, mtime-aware tagging of every clip in the local library. |
| `matcher.py` | `rank_candidates(reference, index, top_k)` — multi-dimensional composite score (subject + vibe + cinema + energy). |
| `search.py` | `search_broll(reference, top_k, cache_dir, settings)` — vibe-aware YouTube fallback with multi-angle queries. |
| `baseline.py` | Faithful re-implementation of the legacy `app/broll.py::_local_score` weights, used by the comparison report. |
| `pipeline.py` | `select_broll(ref_video, top_k)` — the full ladder: LIBRARY → YOUTUBE → REFERENCE_CROP. |
| `demo.py` | `python -m broll_intelligence.demo --reference <video> --top-k 5` — CLI demo. |
| `compare.py` | `python -m broll_intelligence.compare --reference <video> --top-k 5` — side-by-side new vs old markdown report. |
| `CONTRACT.md` | The single source of truth for the schema, vision prompt, and cache format. Read this before changing anything. |

---

## Quick start (CLI)

```bash
# PowerShell, from C:\campeditor, .venv active
python -m broll_intelligence.demo    --reference <video.mp4> --top-k 5
python -m broll_intelligence.compare --reference <video.mp4> --top-k 5 --output report.md
```

Both commands work **offline** (no API keys) — the system warns clearly
and uses the library rung only. When keys are present, the YouTube rung
activates and the compare report shows multi-angle vibe queries.

Optional flags:

```
--library-dir <path>     # override the B-roll library root
--json-out <path>        # demo: write the full BrollPack JSON
--cache-dir <path>       # demo: override the YouTube preview cache
```

---

## Quick start (Python)

```python
from pathlib import Path
from broll_intelligence import get_settings
from broll_intelligence.pipeline import select_broll

settings = get_settings()
pack = select_broll(Path("my_reference.mp4"), top_k=5, settings=settings)
for i, item in enumerate(pack.items, 1):
    print(f"{i}. {item.source:>15}  score={item.score:.3f}  {item.path or item.url}")
print("rungs fired:", pack.diagnostics["rungs_fired"])
```

The ladder tries:

1. **LIBRARY** — composite score against every cached `FeatureVector`,
   accept any clip ≥ 0.55.
2. **YOUTUBE** — vibe-aware multi-angle queries → re-rank with the same
   composite formula, accept any candidate ≥ 0.50.
3. **REFERENCE_CROP** — placeholder. (Real crop implementation lives in
   `app/broll.py`; the new system is intentionally agnostic here.)

---

## How to run the tests

```bash
# PowerShell, from C:\campeditor, .venv active
python -m pytest broll_intelligence/tests -q
```

The vibe_extractor and library_indexer tests use tiny generated mp4s
and ffmpeg (~20 s each). All other tests are pure-Python and finish
in < 2 s. To run only the fast tests:

```bash
python -m pytest broll_intelligence/tests -q -k "not (vibe or library_indexer or pipeline)"
```

To exercise the full pipeline end-to-end (with ffmpeg + index build):

```bash
python -m pytest broll_intelligence/tests/test_pipeline.py -q --timeout=120
```

---

## Configuration

The package reads from the same `.env` file as campeditor
(`C:\campeditor\.env`) via `pydantic-settings`. Keys it cares about:

| env var | purpose |
| --- | --- |
| `GROQ_API_KEY` | primary cloud vision provider |
| `NVIDIA_API_KEY` (+ 3 fallbacks) | secondary cloud vision |
| `GEMINI_API_KEY` | tertiary cloud vision |
| `OLLAMA_VISION_MODEL` (+ URL) | offline last-resort |
| `BROLL_INTELLIGENCE_LIBRARY_DIR` | default `data/broll_library` |
| `BROLL_INTELLIGENCE_INDEX_PATH` | default `data/cache/broll_intelligence_index.json` |
| `BROLL_INTELLIGENCE_FFMPEG_PATH` / `FFMPEG_PATH` | ffmpeg binary |
| `BROLL_INTELLIGENCE_FFPROBE_PATH` / `FFPROBE_PATH` | ffprobe binary |
| `CAMPEDITOR_DATA_DIR` | root for `data/` defaults |

Vision timeouts: 25 s cloud, 120 s Ollama. Provider cooldown on 429: 90 s.

---

## Design rules

* **No `app.*` imports.** The package stays a standalone sibling so a
  regression here can never break the production campeditor pipeline.
* **Tolerant parsing, strict validation.** `feature_vector_from_dict`
  accepts hallucinated enums and falls back to defaults;
  `FeatureVector.validate(strict=True)` is for callers that want a
  hard rejection.
* **Atomic persistence.** Index files are written via `<path>.tmp` +
  `os.replace` — a crash mid-write leaves the previous good copy in place.
* **Incremental by default.** Re-running the indexer is cheap; only
  files whose mtime or size changed pay the vision cost.
* **Composite score is shared.** Matcher and YouTube fallback both use
  the same formula in `broll_intelligence.search.composite_score`, so
  library picks and YouTube picks live on the same [0, 1] scale.
* **Provenance tracked.** Every `FeatureVector` carries `source`
  (`library` / `reference` / `youtube`) and `media_path` so downstream
  matchers can weight, filter, and audit.

See `CONTRACT.md` for the full schema, prompt, compatibility matrices,
and cache format.