# broll_intelligence — Public Contract

This document is the **single source of truth** for the schema, vision prompt,
and on-disk cache shape used by the `broll_intelligence` package. Downstream
tasks (matcher, YouTube comparator, etc.) MUST treat this file as the spec;
the Python code is the implementation.

The package is **not yet integrated into** the running campeditor app
(`app/broll.py`). It is a sibling for offline testing and A/B comparison.

---

## 1. FeatureVector schema

```python
@dataclass
class FeatureVector:
    # Subject matter (compatible with existing campeditor broll tags)
    subjects: list[str]              # 0-3 concrete nouns (lowercase)
    setting: list[str]               # 0-2 location descriptors
    action: list[str]                # 0-2 verbs
    category: str                    # movie|sports|tech|lifestyle|money|other
    query: str                       # 3-8 word stock search phrase
    # Vibe / aesthetic (NEW)
    mood: list[str]                  # 0-3 words from fixed mood vocab
    energy: str                      # low|medium|high
    lighting: str                    # low-key|high-key|natural|neon|golden-hour|mixed
    color_palette: list[str]         # 0-3 dominant colors
    # Cinematography (NEW)
    shot_type: str                   # wide|medium|close-up|extreme-close-up|aerial|overhead|two-shot
    camera_motion: str               # static|pan|tilt|dolly|handheld|tracking|zoom
    depth_of_field: str              # deep|shallow
    # Quantitative features (OpenCV-derived)
    palette_warmth: float            # 0..1 (cool→warm)
    palette_saturation: float        # 0..1 (desaturated→vivid)
    palette_brightness: float        # 0..1 (dark→bright)
    motion_intensity: float          # 0..1 (static→chaotic)
    contrast: float                  # 0..1 (flat→high-contrast)
    edge_density: float              # 0..1 (smooth→busy)
    # Provenance
    confidence: float                # 0..1 (how complete extraction was)
    source: str                      # "library" | "reference" | "youtube"
    media_path: str                  # absolute path to source video
```

### 1.1 Closed vocabularies

| field          | allowed values |
|----------------|----------------|
| `category`     | `movie`, `sports`, `tech`, `lifestyle`, `money`, `other` |
| `mood`         | `tense`, `uplifting`, `mysterious`, `epic`, `melancholic`, `energetic`, `calm`, `aggressive`, `romantic`, `nostalgic`, `ominous`, `joyful`, `neutral`, `dramatic`, `playful`, `sinister` |
| `energy`       | `low`, `medium`, `high` |
| `lighting`     | `low-key`, `high-key`, `natural`, `neon`, `golden-hour`, `mixed` |
| `shot_type`    | `wide`, `medium`, `close-up`, `extreme-close-up`, `aerial`, `overhead`, `two-shot` |
| `camera_motion`| `static`, `pan`, `tilt`, `dolly`, `handheld`, `tracking`, `zoom` |
| `depth_of_field`| `deep`, `shallow` |
| `source`       | `library`, `reference`, `youtube` |

`FeatureVector.validate(strict=False)` returns a list of human-readable errors.
Empty list = OK. `FeatureVector.validate(strict=True)` raises
`FeatureVectorError` on the first problem.

`feature_vector_from_dict` is **forgiving** — unknown enum values fall back to
the default rather than raising (so a drift in LLM training data doesn't
brick a library build). Production callers that want strict acceptance
should call `.validate(strict=True)` after parsing.

### 1.2 Field caps

| field           | max items |
|-----------------|-----------|
| `subjects`      | 3 |
| `setting`       | 2 |
| `action`        | 2 |
| `mood`          | 3 |
| `color_palette` | 3 |

`FeatureVector.normalised()` enforces these caps and lowercases strings.

### 1.3 Confidence formula

```
vision_completeness = non_empty_vision_fields / 12          # 0..1
cv_completeness    = 1.0 if all six CV fields computed else 0.5
confidence         = vision_completeness * 0.5 + cv_completeness * 0.5
```

`confidence == 0.0` ⇒ downstream matchers should treat this as a last-resort
clip (returned by `empty_feature_vector()`).

---

## 2. Vision prompt

Exactly this string is sent to the vision ladder for each clip:

```
You are analyzing a video frame for an intelligent B-roll matching system.
Reply with ONLY a JSON object (no markdown, no prose) with these exact keys:

{
  "subjects": list of 1-3 concrete nouns (objects/people, lowercase),
  "setting": list of 1-2 location descriptors,
  "action": list of 0-2 verbs,
  "category": one of [movie, sports, tech, lifestyle, money, other],
  "query": a short stock-footage search phrase of 3-8 words,
  "mood": list of 1-3 words from [tense, uplifting, mysterious, epic,
          melancholic, energetic, calm, aggressive, romantic, nostalgic,
          ominous, joyful, neutral, dramatic, playful, sinister],
  "energy": one of [low, medium, high],
  "lighting": one of [low-key, high-key, natural, neon, golden-hour, mixed],
  "color_palette": list of 2-3 dominant colors (e.g. ["deep blue", "amber"]),
  "shot_type": one of [wide, medium, close-up, extreme-close-up, aerial, overhead, two-shot],
  "camera_motion": one of [static, pan, tilt, dolly, handheld, tracking, zoom],
  "depth_of_field": one of [deep, shallow]
}

Ignore any burned-in text/captions. Use empty lists/empty strings when nothing fits.
```

Robust parsing strips any ``` ```json ``` fences, finds the outermost `{...}`
span, and tolerates trailing prose. Failures degrade to an empty FeatureVector.

---

## 3. Quantitative (CV) feature formulas

All numerics are clipped to `[0.0, 1.0]`.

| field                 | formula |
|-----------------------|---------|
| `palette_warmth`      | `mean( (R - B) / 255 )` across the central 64% crop of the frame (cv2 loads BGR, so we use `r - b`). |
| `palette_saturation`  | `mean(HSV.saturation) / 255` over the central 64% crop. |
| `palette_brightness`  | `mean(HSV.value) / 255` over the central 64% crop. |
| `contrast`            | `std(grayscale) / 128` over the central 64% crop. |
| `edge_density`        | `mean(Canny(80, 180)) / 255` over the central 64% crop. |
| `motion_intensity`    | mean of consecutive grayscale absdiff (divided by 255) across the 3 sampled frames. |

Frames are sampled at 25 / 50 / 75 % of duration (or 0 / 50 / 100 % when the
clip is ≤ 1.5 s — otherwise the three samples collapse to the same instant).
Single-frame numerics (`palette_*`, `contrast`, `edge_density`) are averaged
across the 3 frames.

If a feature cannot be computed (cv2 missing, frame unreadable, etc.), it
defaults to `0.0` and `confidence` reflects the partial extraction (CV
completeness drops from 1.0 → 0.5).

---

## 4. Library index cache schema

Path: `BROLL_INTELLIGENCE_INDEX_PATH` (default: `data/cache/broll_intelligence_index.json`).

```json
{
  "version": 1,
  "clips": {
    "C:\\campeditor\\data\\broll_library\\movie\\clip.mp4": {
      "mtime": 1717530000.123,
      "size": 4820304,
      "features": {
        "subjects": ["astronaut", "moon"],
        "setting": ["lunar surface"],
        "action": [],
        "category": "movie",
        "query": "astronaut walking on lunar surface",
        "mood": ["mysterious", "epic"],
        "energy": "low",
        "lighting": "low-key",
        "color_palette": ["deep blue", "white", "silver"],
        "shot_type": "wide",
        "camera_motion": "tracking",
        "depth_of_field": "deep",
        "palette_warmth": 0.21,
        "palette_saturation": 0.42,
        "palette_brightness": 0.38,
        "motion_intensity": 0.07,
        "contrast": 0.55,
        "edge_density": 0.18,
        "confidence": 0.92,
        "source": "library",
        "media_path": "C:\\campeditor\\data\\broll_library\\movie\\clip.mp4"
      }
    }
  }
}
```

* Incremental: a clip is re-tagged only when `mtime` OR `size` changes.
* Library root: `BROLL_INTELLIGENCE_LIBRARY_DIR` (default `data/broll_library`).
* `VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}`.
* Unreadable / zero-duration clips are skipped (logged at debug, never added).
* Persistence is atomic: write `<index>.tmp`, then `os.replace` over `<index>`.
* Corrupt or wrong-version cache files are treated as empty (never raised).

---

## 5. Package public API

```python
from broll_intelligence import (
    FeatureVector,                       # dataclass
    empty_feature_vector,                # default-empty factory
    feature_vector_to_dict,              # serialiser
    feature_vector_from_dict,            # parser (enum-tolerant)
    extract_from_video,                  # vision + CV pass on one clip
    build_library_index,                 # incremental index build
    load_index,                          # {abs_path: cached_entry}
    load_index_as_clips,                 # list[IndexedClip]
    invalidate_clip,                     # drop one clip from the cache
    Settings, get_settings,              # pydantic-settings config
)
```

Helper modules (advanced use):

```python
from broll_intelligence.vision_ladder import call, reset_cooldowns
from broll_intelligence.library_indexer import IndexedClip, IndexReport
from broll_intelligence.vibe_extractor import VISION_PROMPT
```

All symbols are also re-exported from the top-level `broll_intelligence`
package.

---

## 6. Configuration

The package reads from the same `.env` file as campeditor (`C:\campeditor\.env`)
via `pydantic-settings`. Keys it cares about:

| env var                          | purpose |
|----------------------------------|---------|
| `GROQ_API_KEY`                   | primary cloud vision provider |
| `NVIDIA_API_KEY` (+ 3 fallbacks) | secondary cloud vision |
| `GEMINI_API_KEY`                 | tertiary cloud vision |
| `OLLAMA_VISION_MODEL` (+ URL)    | offline last-resort |
| `BROLL_INTELLIGENCE_LIBRARY_DIR` | default `data/broll_library` |
| `BROLL_INTELLIGENCE_INDEX_PATH`  | default `data/cache/broll_intelligence_index.json` |
| `BROLL_INTELLIGENCE_FFMPEG_PATH` / `FFMPEG_PATH` | ffmpeg binary |
| `BROLL_INTELLIGENCE_FFPROBE_PATH` / `FFPROBE_PATH` | ffprobe binary |
| `CAMPEDITOR_DATA_DIR`            | root for `data/` defaults |

Vision timeouts (matches `app/broll.py`): 25 s cloud, 120 s Ollama.
Provider cooldown on 429: 90 s.

---

## 7. Query generation prompt

This is the verbatim text `broll_intelligence.search` sends to the chat
ladder when generating YouTube search queries from a reference
FeatureVector. The placeholder slots are filled by the reference's own
fields. The model is asked for exactly five queries, one per aesthetic
angle, each at most 8 words.

```
You generate stock-footage search queries that capture the VIBE of a B-roll scene.
Given this scene description:
  subjects: {subjects}
  setting: {setting}
  mood: {mood}
  lighting: {lighting}
  shot_type: {shot_type}
  color_palette: {color_palette}
Reply with ONLY a JSON object: {"queries": [q1, q2, q3, q4, q5]}
Each query is <= 8 words. Each captures a DIFFERENT angle:
  q1 = subject-focused (main object/action)
  q2 = mood-focused (emotional tone)
  q3 = lighting-focused (visual lighting)
  q4 = scene-focused (location + action)
  q5 = aesthetic-focused (overall visual feel)
```

Behaviour:

* The chat ladder is `broll_intelligence.llm_ladder.chat`, with the
  Groq -> NVIDIA -> Gemini -> Ollama chain, 15 s timeout, temperature
  0.4, max_tokens 200, 90 s cooldown on 429 / RateLimitError.
* The response is parsed by `broll_intelligence.llm_ladder.extract_json_object`
  (forgiving — strips ```json fences, finds the outermost `{...}` span).
* Each query is truncated to <= 8 words before being passed to yt-dlp.
* When the LLM returns an empty list, malformed JSON, or every provider
  is rate-limited, `search_broll` falls back to a deterministic query
  set built from the reference's own fields (subject + action /
  mood / lighting / setting / palette).

---

## 8. Compatibility matrices (matcher)

Used by `broll_intelligence.matcher.score_clip` / `rank_candidates` for the
"compatible but not exact" partial credit (0.4 for lighting and camera
motion; 0.5 for shot type). All pairs are **symmetric** — if `(a, b)` is
listed, `(b, a)` is also compatible. Any pair not listed below scores 0.0.

### 8.1 Lighting compatibility

| lighting A    | lighting B    | credit |
|---------------|---------------|--------|
| `low-key`     | `natural`     | 0.4    |
| `high-key`    | `natural`     | 0.4    |
| `neon`        | `mixed`       | 0.4    |
| `golden-hour` | `natural`     | 0.4    |
| anything else | anything else | 0.0    |

Exact match (`a == b`) always scores 1.0 regardless of the table.

### 8.2 Shot-type compatibility

| shot_type A          | shot_type B          | credit |
|----------------------|----------------------|--------|
| `close-up`           | `extreme-close-up`   | 0.5    |
| `wide`               | `medium`             | 0.5    |
| `aerial`             | `overhead`           | 0.5    |
| anything else        | anything else        | 0.0    |

Exact match scores 1.0.

### 8.3 Camera-motion compatibility

| camera_motion A | camera_motion B | credit |
|-----------------|-----------------|--------|
| `pan`           | `tilt`          | 0.4    |
| `dolly`         | `tracking`      | 0.4    |
| anything else   | anything else   | 0.0    |

`static`, `handheld`, and `zoom` have no compatibility neighbours — only an
exact match scores credit.

### 8.4 Category families

`score_clip` gives 0.5 credit (rather than 1.0 for exact, 0.0 for unrelated)
when the two categories share a family. Pairs (symmetric):

* `sports` <-> `lifestyle`
* `money` <-> `lifestyle`

`movie`, `tech`, `other` have no family — they only score 1.0 on exact match.

### 8.5 Mood synonym groups

For mood Jaccard, each mood is expanded to its synonym group before
intersection. Two moods in the same group match at full strength (1.0).

| group          | members                       |
|----------------|-------------------------------|
| mystery        | `mysterious`, `ominous`       |
| intensity      | `energetic`, `aggressive`     |
| stillness      | `calm`, `melancholic`         |
| uplift         | `joyful`, `uplifting`         |
| grandeur       | `epic`, `dramatic`            |

Standalone moods (no synonym neighbour): `tense`, `romantic`, `nostalgic`,
`neutral`, `playful`, `sinister`.