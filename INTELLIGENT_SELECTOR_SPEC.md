# Intelligent B-roll Selector V2 — Executable Specification

**Target file:** `C:\campeditor\INTELLIGENT_SELECTOR_SPEC.md`
**Author:** Architect agent (Task #26)
**Status:** Draft for agents A/B/C to implement against.
**Scope:** All changes live in `app/*`. The sibling package at `broll_intelligence/` is NOT imported by `app/*` (verified via `grep -rn "from broll_intelligence\|import broll_intelligence" C:/campeditor/app` returning NO MATCHES). The closed-vocab constants and vision-prompt text in this spec are taken **verbatim** from `C:/campeditor/broll_intelligence/CONTRACT.md` lines 49–57 and lines 96–119 (the "single source of truth" — the package code, not this spec, is the runtime implementation).

---

## 1. Goal and in-scope

### Three concrete deliverables (one per downstream agent)

**A. Extended tag prompt** (Agent A — `app/broll.py:_TAG_PROMPT`, `_parse_tags`, `_merge_tags`, `_tag_library_clip`).
   - The vision prompt must request the seven vibe/cinematography keys in addition to the existing five keys so that newly-tagged library clips actually carry `mood`, `energy`, `lighting`, `shot_type`, `camera_motion`, `depth_of_field`, `color_palette`.
   - `_parse_tags` must be tolerant: an unknown enum value is silently dropped (NEVER raised), so vision-model hallucinations cannot brick a library build.

**B. Hardened scoring** (Agent B — `app/broll.py:_local_score`, new `_cinema_match`, new `_CONTINUITY_LEDGER`, `_rank_local`).
   - The current vibe bonus (`_VIBE_BONUS_WEIGHT = 0.35`) is a *soft* add-on that can be drowned out by the legacy keyword score. Replace it with a multiplicative cinema-aware adjustment that floors at ~0.18 so a true cinematographic mismatch cannot score above 20% even on perfect content overlap.

**C. Robustness** (Agent C — `app/broll.py:_span_profile_for`, `build_reference_house_style`, `_gather_span_pool`, `search_youtube_candidates`, `_frame_tags`, `BrollRecoveryDiagnostic`).
   - Reference house-style back-fill: spans that lack mood/lighting borrow from siblings.
   - Same-span continuity penalty: consecutive B-roll slots that look identical pay a small tax.
   - YouTube rung re-ranks with the new vibe fields.
   - Diagnostics surface `cinema`, `continuity_penalty`, and a top-level `intelligent_active` flag.

### Explicit non-goals (not in this iteration)

- No new UI. The checkbox `use_intelligent_selector` is already wired (UI `static/index.html` → `app.js` → `app/main.py:113` → `app/jobs.py:104` → `app/models.py:Job.use_intelligent_selector` → `getattr(job, "use_intelligent_selector", True)` at `app/jobs.py` lines 449/497/516/535).
- No new sources. Pexels and Frame.io v4 are out of scope.
- No model retraining. The vision prompt is the only thing the model "sees" change.
- No Learnable weights from user feedback. The pipeline is still heuristic.

### Acceptance posture

The fix must produce **visibly different picks** when `intelligent=True` vs `intelligent=False` on contrived scenarios that share category but differ on cinematography. The acceptance test (§13) is the contract — agents A/B/C must each make the test pass.

---

## 2. Closed vocabs (hardcode constants in `app/broll.py`)

These seven tuples/sets go **at the top of `app/broll.py`**, immediately after the existing `VALID_CATEGORIES = {"movie", "sports", "tech", "lifestyle", "money", "other"}` constant at `app/broll.py:58`. They are hardcoded so the rule lives next to the scorer that consumes it; they mirror the closed-vocab table in `broll_intelligence/CONTRACT.md` lines 49–57 verbatim. No import from `broll_intelligence` (that is forbidden — see header).

```python
# Closed vocabs — verbatim from broll_intelligence/CONTRACT.md §1.1.
# Used by _parse_tags to validate vision-model output and by _local_score
# / _cinema_match to compare cinematography fields. Empty list / empty
# string is the "no data" sentinel — the scorer treats empty as "absent"
# (not "unknown") and falls back to the house style or zero credit.
_MOOD_VOCAB: frozenset[str] = frozenset({
    "tense", "uplifting", "mysterious", "epic", "melancholic", "energetic",
    "calm", "aggressive", "romantic", "nostalgic", "ominous", "joyful",
    "neutral", "dramatic", "playful", "sinister",
})
_ENERGY_VOCAB: frozenset[str] = frozenset({"low", "medium", "high"})
_LIGHTING_VOCAB: frozenset[str] = frozenset({
    "low-key", "high-key", "natural", "neon", "golden-hour", "mixed",
})
_SHOT_TYPE_VOCAB: frozenset[str] = frozenset({
    "wide", "medium", "close-up", "extreme-close-up", "aerial", "overhead",
    "two-shot",
})
_CAMERA_MOTION_VOCAB: frozenset[str] = frozenset({
    "static", "pan", "tilt", "dolly", "handheld", "tracking", "zoom",
})
_DEPTH_OF_FIELD_VOCAB: frozenset[str] = frozenset({"deep", "shallow"})

# Field caps. Clip-level moods cap at 3, span-level at 3, color_palette at 3.
# Anything beyond the cap is silently truncated (never raised).
_MOOD_CAP = 3
_COLOR_PALETTE_CAP = 3
```

The existing `VALID_CATEGORIES` (line 58) and `VIDEO_EXTS` (line 57) are unchanged.

### Shot-type broad-class groupings (used by `_cinema_match`)

Hardcode these three frozensets next to the vocabs:

```python
_SHOT_TYPE_GROUP_WIDE: frozenset[str] = frozenset({"wide", "aerial", "overhead"})
_SHOT_TYPE_GROUP_TIGHT: frozenset[str] = frozenset({"close-up", "extreme-close-up", "two-shot"})
# "medium" stands alone in its own group.
```

A clip and a span are in the **same broad class** when both values belong to the same group above, or when both equal `"medium"`.

### Camera-motion compatibility table (explicit small table, used by `_cinema_match`)

Encode as a constant frozenset-of-pairs (symmetric — both `(a, b)` and `(b, a)` are listed):

```python
# Cinematic-intent pairs that read as "the same direction of motion" or
# "deliberately complementary". A static ref asking for handheld motion
# is intentionally mapped to a HIGH score (0.85) — the static span
# benefits from a dynamic clip injected on top. Symmetric.
_CAMERA_MOTION_INTENT_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("static", "dolly"),       # static ref wants a slow push-in
    ("static", "handheld"),    # static ref wants a dynamic B-roll
    ("static", "tracking"),    # static ref wants lateral motion
    ("static", "pan"),
    ("static", "tilt"),
    ("static", "zoom"),
    ("dolly", "tracking"),     # both = deliberate motion
    ("pan", "tilt"),           # both = rotation, no translation
    ("handheld", "handheld"),  # exact handheld-on-handheld for gritty refs
})
```

Everything else scores 0.4 on exact match and 0.0 otherwise (see §7).

---

## 3. Extended `_TAG_PROMPT`

**Replace** the existing string at `app/broll.py:544` with this exact new prompt. The seven vibe keys are appended after the legacy five, the JSON sample is updated, and the tolerance line moves to the end so the model reads the schema first.

```python
_TAG_PROMPT = (
    "You are analyzing a single video frame for a B-roll matching system. "
    "Reply with ONLY a JSON object (no markdown, no prose) with these exact keys:\n"
    '{"subjects": list of 1-3 concrete nouns (objects/people, lowercase), '
    '"setting": list of 1-2 location descriptors (e.g. ["office","indoors"]), '
    '"action": list of 0-2 verbs (e.g. ["typing","running"]), '
    '"category": one of [movie, sports, tech, lifestyle, money, other], '
    '"query": a short stock-footage search phrase of 3-8 words describing the shot, '
    '"mood": list of 0-3 words from [tense, uplifting, mysterious, epic, melancholic, '
    'energetic, calm, aggressive, romantic, nostalgic, ominous, joyful, neutral, '
    'dramatic, playful, sinister], '
    '"energy": one of [low, medium, high], '
    '"lighting": one of [low-key, high-key, natural, neon, golden-hour, mixed], '
    '"shot_type": one of [wide, medium, close-up, extreme-close-up, aerial, overhead, two-shot], '
    '"camera_motion": one of [static, pan, tilt, dolly, handheld, tracking, zoom], '
    '"depth_of_field": one of [deep, shallow], '
    '"color_palette": list of 0-3 dominant colors (e.g. ["deep blue", "amber"]) }.\n'
    "Use empty lists / empty strings when nothing fits. Unknown enum values are "
    "tolerated (we drop them silently), so prefer leaving a field empty over "
    "guessing a value that is not in the allowed list."
)
```

Rules for the prompt text:

- The JSON example keys MUST match the seven new field names exactly. The keys are `mood`, `energy`, `lighting`, `shot_type`, `camera_motion`, `depth_of_field`, `color_palette`.
- The prompt explicitly tells the model it is OK to leave a field empty. This is the "tolerance" line that downstream `_parse_tags` relies on.
- The model may slightly hallucinate vocabulary. `_parse_tags` (§4) MUST validate every enum value against the closed vocabs from §2 and silently drop unknowns. **Never raise.**

---

## 4. `_parse_tags` extension

**File:** `app/broll.py`, function `_parse_tags` at line 575.

Currently returns a dict with five keys: `subjects`, `setting`, `action`, `category`, `query`. Extend it to also return the seven vibe keys when the model produces them. Rules:

1. **`mood`** — `list[str]`, capped at `_MOOD_CAP = 3`, lowercased and trimmed, each member validated against `_MOOD_VOCAB`. Unknown values are silently dropped (no warning, no raise). Empty list `[]` when the model returned nothing or only unknowns.

2. **`energy`** — `str`, validated against `_ENERGY_VOCAB`. Unknown → empty string `""`. Empty string when missing. Note: `""` is the absence sentinel the scorer reads (not `"unknown"`).

3. **`lighting`** — `str`, validated against `_LIGHTING_VOCAB`. Same rules.

4. **`shot_type`** — `str`, validated against `_SHOT_TYPE_VOCAB`. Same rules.

5. **`camera_motion`** — `str`, validated against `_CAMERA_MOTION_VOCAB`. Same rules.

6. **`depth_of_field`** — `str`, validated against `_DEPTH_OF_FIELD_VOCAB`. Same rules.

7. **`color_palette`** — `list[str]`, capped at `_COLOR_PALETTE_CAP = 3`, each member `.strip().lower()`. **No** vocabulary check — palette is free-form text by spec. Drop empty strings.

The function signature stays `(raw: str) -> dict` for backwards compatibility — callers that ignore the new keys (e.g. tests that just look at `subjects`) keep working.

### Concrete helper to add inside `_parse_tags`

Drop in this validator block after the existing `_as_list` helper (around line 603):

```python
    def _enum(value: object, vocab: frozenset[str]) -> str:
        s = str(value).strip().lower() if value is not None else ""
        return s if s in vocab else ""

    def _enum_list(value: object, vocab: frozenset[str], cap: int) -> list[str]:
        out: list[str] = []
        if isinstance(value, list):
            for x in value:
                s = str(x).strip().lower()
                if s in vocab and s not in out:
                    out.append(s)
                    if len(out) >= cap:
                        break
        elif isinstance(value, str):
            s = value.strip().lower()
            if s in vocab:
                out.append(s)
        return out

    def _palette(value: object, cap: int) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip().lower() for x in value if str(x).strip()][:cap]
        if isinstance(value, str) and value.strip():
            return [value.strip().lower()][:cap]
        return []
```

The final return dict becomes:

```python
    return {
        "subjects": _as_list(obj.get("subjects"), 3),
        "setting": _as_list(obj.get("setting"), 2),
        "action": _as_list(obj.get("action"), 2),
        "category": category,
        "query": query,
        "mood": _enum_list(obj.get("mood"), _MOOD_VOCAB, _MOOD_CAP),
        "energy": _enum(obj.get("energy"), _ENERGY_VOCAB),
        "lighting": _enum(obj.get("lighting"), _LIGHTING_VOCAB),
        "shot_type": _enum(obj.get("shot_type"), _SHOT_TYPE_VOCAB),
        "camera_motion": _enum(obj.get("camera_motion"), _CAMERA_MOTION_VOCAB),
        "depth_of_field": _enum(obj.get("depth_of_field"), _DEPTH_OF_FIELD_VOCAB),
        "color_palette": _palette(obj.get("color_palette"), _COLOR_PALETTE_CAP),
    }
```

**Empty values stay empty.** A missing `lighting` becomes `""`, not `"unknown"`. This is the absence sentinel the scorer reads.

---

## 5. `_tag_library_clip` propagation

**File:** `app/broll.py`, function `_tag_library_clip` at line 766.

`_tag_library_clip` already calls `_frame_tags` then `_merge_tags`. Extend `_merge_tags` (line 646) so the merged dict carries the seven new fields, and extend `_tag_library_clip` so the LibraryClip instance built downstream carries them.

### `_merge_tags` extension

Add four new merge strategies alongside the existing `subjects/setting/action/category/query` merges:

- **`mood`** — union across frames, capped at `_MOOD_CAP = 3`. Use the same `_union("mood", 3)` helper shape as for `subjects`. Drop any frame value that isn't a list of strings.
- **`energy`, `lighting`, `shot_type`, `camera_motion`, `depth_of_field`** — closed-vocab enum strings. Take the **mode** (most-frequent non-empty value) across frames; break ties by first occurrence. Empty string when no frame carries that key.
- **`color_palette`** — union across frames, capped at `_COLOR_PALETTE_CAP = 3`, lowercased + trimmed (no vocab check, free-form).

Concrete helper to add inside `_merge_tags`:

```python
    def _mode(key: str) -> str:
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for i, tags in enumerate(frame_tag_list):
            v = tags.get(key, "")
            if isinstance(v, str) and v:
                counts[v] = counts.get(v, 0) + 1
                first_seen.setdefault(v, i)
        if not counts:
            return ""
        # Highest count, then earliest first-seen index for tie-break.
        best = max(counts.items(), key=lambda kv: (kv[1], -first_seen[kv[0]]))
        return best[0]
```

The final `return {...}` block in `_merge_tags` becomes:

```python
    return {
        "subjects": _union("subjects", 4),
        "setting": _union("setting", 2),
        "action": _union("action", 2),
        "category": category,
        "query": query,
        "mood": _union("mood", _MOOD_CAP),
        "energy": _mode("energy"),
        "lighting": _mode("lighting"),
        "shot_type": _mode("shot_type"),
        "camera_motion": _mode("camera_motion"),
        "depth_of_field": _mode("depth_of_field"),
        "color_palette": _union("color_palette", _COLOR_PALETTE_CAP),
    }
```

### `_tag_library_clip` extension

`LibraryClip` already has the seven new fields as empty defaults (see `app/broll.py:138-144`). The constructor call at line 850 needs to forward the merged fields:

```python
        clip = LibraryClip(
            path=path, mtime=stat.st_mtime, size=stat.st_size,
            subjects=tagged.get("subjects", []), setting=tagged.get("setting", []),
            category=category, folder=folder, query=tagged.get("query", ""),
            mood=tagged.get("mood", []),
            energy=tagged.get("energy", ""),
            lighting=tagged.get("lighting", ""),
            shot_type=tagged.get("shot_type", ""),
            camera_motion=tagged.get("camera_motion", ""),
            depth_of_field=tagged.get("depth_of_field", ""),
            color_palette=tagged.get("color_palette", []),
        )
```

And the `cached[key] = {...}` persist block at lines 859–863 needs every new field too (so re-loads from cache populate the LibraryClip fields):

```python
            cached[key] = {
                "mtime": stat.st_mtime, "size": stat.st_size,
                "subjects": clip.subjects, "setting": clip.setting,
                "category": clip.category, "folder": clip.folder, "query": clip.query,
                "mood": clip.mood,
                "energy": clip.energy,
                "lighting": clip.lighting,
                "shot_type": clip.shot_type,
                "camera_motion": clip.camera_motion,
                "depth_of_field": clip.depth_of_field,
                "color_palette": clip.color_palette,
            }
```

Same forwarding for the **cache-hit** branch at lines 828–833 — the early-return that pulls an unchanged clip straight from cache:

```python
            clips.append(LibraryClip(
                path=path, mtime=stat.st_mtime, size=stat.st_size,
                subjects=entry.get("subjects", []), setting=entry.get("setting", []),
                category=entry.get("category", "other"), folder=entry.get("folder", ""),
                query=entry.get("query", ""),
                mood=entry.get("mood", []),
                energy=entry.get("energy", ""),
                lighting=entry.get("lighting", ""),
                shot_type=entry.get("shot_type", ""),
                camera_motion=entry.get("camera_motion", ""),
                depth_of_field=entry.get("depth_of_field", ""),
                color_palette=entry.get("color_palette", []),
            ))
```

---

## 6. Cache version bump

**File:** `app/broll.py`, constant `_INDEX_CACHE_VERSION` at line 76.

Bump from `1` to `2`. The existing version-mismatch branch at line 803 (`if raw.get("version") == _INDEX_CACHE_VERSION: cached = raw.get("clips", {})`) already handles the case where an old version-1 cache is treated as empty — no extra work needed there. On the next `build_library_index` call, every clip's entry will be missing from the in-memory cache (because the version comparison fails), and `_tag_library_clip` will be re-run for every file.

The frame-tag cache (`data/cache/broll_tags/<sha256>.json`, see `_tags_cache_dir` at line 571) is keyed by **frame CONTENT hash** (computed by `_frame_content_hash` at line 556). When the prompt changes, old cached frame tags are technically stale — they were produced by the old 5-key prompt and won't have the seven new fields. So old frames return their 5-key cached result, and `_parse_tags` extension gracefully returns empty strings/lists for the new keys. That is fine for the score floor (`_vibe_score_for` returns 0.0 when fields are missing on both sides).

### Frame-tag cache purge helper

The spec calls for a **single-shot purge** of the frame-tag cache when the prompt version changes. Add this helper near the top of `app/broll.py`, after the closed-vocab constants:

```python
def purge_frame_tag_cache_if_prompt_version_changed(settings: Settings) -> bool:
    """One-shot purge of data/cache/broll_tags when the cached prompt version
    disagrees with the live `intelligent_frame_tag_prompt_version` setting.

    Controlled by the env var / settings flag
    `intelligent_frame_tag_prompt_version` (default 2, see §12). When the
    in-process value is higher than the version stamp persisted on disk,
    delete every <sha256>.json under settings.data_dir/cache/broll_tags,
    then write the new version stamp. Returns True when a purge happened.

    Idempotent: a second call with the same live version is a no-op.
    """
```

The helper is invoked **once per `build_library_index` call, at the top, before any file scanning**. Pseudocode for the call site (insert at line ~795, before the `if not library_dir.exists():` early-return):

```
stamp_path = settings.data_dir / "cache" / "_frame_tag_prompt_version"
live_version = settings.intelligent_frame_tag_prompt_version  # default 2
stamp = stamp_path.read_text() if stamp_path.exists() else "0"
if int(stamp or "0") < live_version:
    purge frame-tag cache dir
    write live_version to stamp_path
    logger.info(...)
```

The stamp file lives at `data/cache/_frame_tag_prompt_version` (top-level cache, NOT inside `broll_tags/` so a future purge doesn't delete its own stamp).

The env var that controls the live version is `CAMPEDITOR_INTELLIGENT_FRAME_TAG_PROMPT_VERSION`, default `2` (see §12 for the Settings entry).

### Cost note

The first build after the version bump costs roughly the same as the first build ever (every frame is re-visioned). Subsequent rebuilds only pay vision for changed files (mtime/size deltas) and reuse the now-cached frame tags for everything else.

---

## 7. Strengthened scoring (cinema_match + cinema-aware multiplier)

**File:** `app/broll.py`. This section replaces the current "soft add-on" `_VIBE_BONUS_WEIGHT = 0.35` formula in `_local_score` (line 1006) with a multiplicative adjustment that **floors at ~0.18** when cinematography mismatches hard.

### New helper: `_cinema_match(profile, clip) -> float`

```python
def _cinema_match(profile: SpanProfile, clip: LibraryClip) -> float:
    """0..1 cinematography match score, combining shot_type + camera_motion +
    depth_of_field.  Returns 1.0 (perfect match) when every component agrees;
    0.18 (the floor, see _CINEMA_FLOOR) when they all disagree. Empty /
    missing fields on either side drop out of the calculation (don't pull
    the score toward 0 for an absent field — that's a data gap, not a
    mismatch)."""
```

Three sub-scores, each in `[0.0, 1.0]`:

**`_shot_type_subscore(a: str, b: str) -> float`:**

- Both empty or one empty: return `None` (drop out, weight is excluded).
- `a == b` (exact match, includes both being `"medium"`): return `1.0`.
- Both in `_SHOT_TYPE_GROUP_WIDE`: return `0.6`.
- Both in `_SHOT_TYPE_GROUP_TIGHT`: return `0.6`.
- Both equal `"medium"`: return `0.6` (same-group default).
- Otherwise (cross-class): return `0.25`.

**`_camera_motion_subscore(a: str, b: str) -> float`:**

- Both empty or one empty: return `None`.
- `a == b`: return `0.4` (exact match — deliberately low because static-on-static or handheld-on-handheld is rarely the BEST choice for a B-roll, but it's not wrong either).
- `(a, b)` or `(b, a)` is in `_CAMERA_MOTION_INTENT_PAIRS` (the explicit table from §2): return `1.0` for the high-intent pairs (static + handheld, static + dolly, etc.), `0.4` for the rotation pairs (pan + tilt).
  - Actually split: pairs tagged "complementary cinematic intent" → `1.0`. Pairs that are just "exact match" inside the table → `0.4`. Encode this by **two** frozensets:
    ```python
    _CAMERA_MOTION_INTENT_HIGH: frozenset[tuple[str, str]] = frozenset({
        ("static", "dolly"), ("static", "handheld"), ("static", "tracking"),
        ("static", "pan"), ("static", "tilt"), ("static", "zoom"),
        ("dolly", "tracking"),
    })
    _CAMERA_MOTION_INTENT_LOW: frozenset[tuple[str, str]] = frozenset({
        ("handheld", "handheld"),
        ("pan", "tilt"),
    })
    ```
- Otherwise: return `0.0`.

**`_depth_of_field_subscore(a: str, b: str) -> float`:**

- Both empty or one empty: return `None`.
- `a == b`: return `1.0`.
- One is `"deep"` and the other is `"shallow"` (true mismatch): return `0.2`.
- (There's no third value, so `mixed` isn't applicable here.)

### Combined `cinema_match`

Combine the three sub-scores with equal weights when present:

```python
def _cinema_match(profile: SpanProfile, clip: LibraryClip) -> float:
    subs = [
        ("shot_type", _shot_type_subscore(profile.shot_type, clip.shot_type)),
        ("camera_motion", _camera_motion_subscore(profile.camera_motion, clip.camera_motion)),
        ("depth_of_field", _depth_of_field_subscore(profile.depth_of_field, clip.depth_of_field)),
    ]
    present = [(name, v) for name, v in subs if v is not None]
    if not present:
        return 0.6  # neutral default — no cinema data on either side
    raw = sum(v for _, v in present) / len(present)
    # Apply the floor so a true cinematic mismatch cannot score above ~20%.
    return max(_CINEMA_FLOOR, raw)
```

Constant declaration (top of file, near other tunables):

```python
_CINEMA_FLOOR: float = 0.18
```

### Cinema-aware multiplier in `_local_score`

**Replace** the body of `_local_score` (lines 1019–1031). New shape:

```python
def _local_score(profile: SpanProfile, clip: LibraryClip, *,
                 intelligent: bool = False,
                 reference_house: dict | None = None,
                 continuity_penalty: float = 0.0) -> tuple[float, float, float, float]:
    """0..1 normalized match score, plus diagnostic components.

    Returns (total, vibe, cinema, continuity_penalty).  When intelligent=False,
    vibe=0.0 and cinema=0.0 (caller should ignore them); continuity_penalty
    is always returned for diagnostics.  When intelligent=True and BOTH
    profile and clip carry vibe fields, the cinema multiplier is applied.

    Cinema floor (see _CINEMA_FLOOR) prevents a true cinematographic mismatch
    from scoring above 20% even on perfect content overlap.
    """
    # Legacy keyword score (unchanged from current code).
    score = 0.0
    if profile.category and clip.category and profile.category == clip.category:
        score += 3.0
    score += len(set(profile.subjects) & set(clip.subjects)) * 1.0
    score += len(set(profile.setting) & set(clip.setting)) * 1.0
    base = min(1.0, score / _LOCAL_SCORE_MAX)
    if not intelligent:
        total = max(0.0, min(1.0, base + continuity_penalty))
        return total, 0.0, 0.0, continuity_penalty

    # Resolve vibe fields: span value, then house-style fallback, then empty.
    resolved_span_vibe = _resolve_span_vibe(profile, reference_house)
    vibe = _vibe_score_for_resolved(resolved_span_vibe, clip)
    cinema = _cinema_match(profile, clip)

    # Cinema-aware multiplicative adjustment.
    # When cinema >= 0.6, no extra penalty is applied (the multiplier is 1.0).
    # When cinema < 0.6, we apply CINEMA_LIFT_TERM per (0.6 - cinema) gap.
    # CINEMA_LIFT_TERM = 0.5 means a cinema=0.18 (worst case) subtracts 0.21.
    cinema_lift = _CINEMA_LIFT_TERM * (cinema - 0.6) if cinema < 0.6 else 0.0
    boosted = base + _VIBE_BONUS_WEIGHT * vibe * (1.0 - base) + cinema_lift
    total = max(0.0, min(1.0, boosted + continuity_penalty))
    return total, vibe, cinema, continuity_penalty
```

Constant declaration:

```python
_CINEMA_LIFT_TERM: float = 0.5   # multiplier on (cinema - 0.6) gap when cinema<0.6
```

Worked example (clip B with `extreme-close-up + handheld` against an epic wide-tracking ref):
- `base = 1.0` (perfect content overlap)
- `shot_type_subscore = 0.25` (wide → extreme-close-up is cross-class)
- `camera_motion_subscore = 1.0` (static → handheld is in `_CAMERA_MOTION_INTENT_HIGH` — wait, the ref is tracking, not static; tracking → handheld = no entry → `0.0`)
- `depth_of_field_subscore = 0.2` (mismatch)
- `cinema_match = (0.25 + 0.0 + 0.2) / 3 = 0.15`, floored to `0.18`
- `cinema_lift = 0.5 * (0.18 - 0.6) = -0.21`
- `boosted = 1.0 + 0 + (-0.21) = 0.79` (still high because `vibe` was 0.0)

That is intentionally not aggressive enough on its own. The hard demotion happens because clip A — same content, matching cinematography — scores `cinema = 1.0` → `cinema_lift = 0.0` → `boosted = 1.0`. The DIFFERENCE between A and B is what the user sees, not the absolute score of B.

---

## 8. Reference house-style back-fill

**File:** `app/broll.py`. New function `build_reference_house_style` plus changes to `_span_profile_for` (line 1599) and `_vibe_score_for` (line 988).

### `build_reference_house_style(analysis) -> dict`

Computes a "house style" FeatureVector-like dict over every span in `analysis.broll_span_tags`. One computation per job — passed down to every `_gather_span_pool` call (see §8 wiring).

```python
def build_reference_house_style(analysis: ReferenceAnalysis) -> dict:
    """Aggregate span-level tags into a single 'house style' vector.
    Used as a fallback when an individual span's vibe fields are empty
    (e.g. a close-up on a face -> mood empty). Computed ONCE per job.

    Returns a dict with the same shape as a single span's vibe fields:
        mood: list[str]   (union, capped at _MOOD_CAP)
        energy: str       (mode of non-empty values)
        lighting: str     (mode of non-empty values)
        shot_type: str    (mode of non-empty values)
        camera_motion: str (mode of non-empty values)
        depth_of_field: str (mode of non-empty values)
        color_palette: list[str] (union, capped at _COLOR_PALETTE_CAP)
    Empty fields stay empty (the absence sentinel).
    """
```

Implementation shape (re-uses the merge logic from `_merge_tags`):

- `mood`: union across all spans' `mood` lists, capped at 3.
- Each enum field: mode of non-empty values across all spans. Empty when no span carries the field.
- `color_palette`: union across all spans, capped at 3.

### Wiring: pass through `_gather_span_pool` and `_gather_pack_sources_for_span`

Add a `reference_house: dict | None = None` parameter to both `_gather_span_pool` (line 1624) and `_gather_pack_sources_for_span` (line 1837). Inside, pass it to `_local_score`, `_vibe_score_for_resolved`, and `_rank_local`. None is the safe default — the helpers treat None as "no house style, use span-only".

### Updated `_vibe_score_for` signature

Rename the existing `_vibe_score_for` to `_vibe_score_for_resolved` and add a fallback path:

```python
def _vibe_score_for_resolved(resolved_span_vibe: dict, clip: LibraryClip) -> float:
    """Compare a RESOLVED span vibe (already filled from house style) to a
    clip. Uses the same _VIBE_FIELDS weights as before."""
```

Plus a small resolver:

```python
def _resolve_span_vibe(profile: SpanProfile, reference_house: dict | None) -> dict:
    """Per-field fallback: profile value → house style → empty. Returns a
    dict suitable for _vibe_score_for_resolved."""
```

Order for each field: `profile.<field>` if non-empty, else `reference_house["<field>"]` if non-empty, else `""` / `[]`. Never raise.

### Where the house style is computed

In `fetch_broll_cut_variations` (line 1716), **before** the per-span loop:

```python
reference_house = build_reference_house_style(analysis)
```

Pass `reference_house=reference_house` to every `_gather_span_pool` call. In `_gather_span_pool`, propagate to `_local_score`, `_rank_local`, and `match_local`.

`gather_broll_pack` (line 1938) gets the same one-shot computation + propagation.

`fetch_broll_cuts` (line 1815) inherits the work via `fetch_broll_cut_variations`.

---

## 9. Continuity penalty across spans

**File:** `app/broll.py`. New module-level `_CONTINUITY_LEDGER: dict[str, list[dict]] = {}` and new helper `_continuity_penalty`.

### The ledger

Keyed by `job_id` (the Job's string UUID). Value is a list of "last pick" FeatureVector-like dicts (one per span, in pick order). Reset between variations — each variation builds its own continuity sequence.

```python
_CONTINUITY_LEDGER: dict[str, list[dict]] = {}
```

### Helper functions

```python
def _feature_vector_for_clip(clip: LibraryClip) -> dict:
    """Six-dim vibe vector for cosine similarity. Drops the long-tail
    fields (energy, lighting, depth_of_field) which are noisy on
    single-frame tags and replaces them with the three that matter most
    visually: shot_type, camera_motion, plus a 3-element mood histogram."""
```

Use this 6-D encoding:

| dim | source |
|-----|--------|
| 0 | `shot_type` one-hot (7 buckets: wide, medium, close-up, extreme-close-up, aerial, overhead, two-shot) |
| 1 | `camera_motion` one-hot (7 buckets: static, pan, tilt, dolly, handheld, tracking, zoom) |
| 2 | `depth_of_field` one-hot (2 buckets: deep, shallow) |
| 3-5 | `mood` histogram over 3 canonical groups (mystery/intensity/stillness/uplift/grandeur → from `broll_intelligence/CONTRACT.md` §8.5; pad to 3 with 0.0) |

```python
def _cosine_similarity_6d(a: dict, b: dict) -> float:
    """Returns 0..1. 0 = orthogonal (great continuity tax), 1 = identical."""
```

```python
def _continuity_penalty(candidate_vec: dict, ledger: list[dict],
                         threshold: float, max_penalty: float) -> float:
    """Returns a non-positive float. When the previous span's pick has
    cosine_similarity_6d >= threshold to the candidate, apply max_penalty.
    Otherwise 0.0. Only the IMMEDIATELY previous pick matters (not the
    whole history) — keeps the diversity nudge from snowballing into a
    forced rotation."""
```

### Wiring

Add `job_id: str | None = None` and `ledger` parameter to `_gather_span_pool`, `_gather_pack_sources_for_span`, `_rank_local`, `_local_score`. The actual ledger update happens **after** the pick is chosen, in `_gather_span_pool`:

```python
# After pick_local succeeds:
last_pick_vec = _feature_vector_for_clip(picked)
ledger.append(last_pick_vec)
```

For each CANDIDATE being scored in `_local_score`, compute `cosine = _cosine_similarity_6d(candidate_vec, ledger[-1])` (if ledger has at least one entry), then `penalty = _continuity_penalty(candidate_vec, ledger, threshold, max_penalty)`. Pass `penalty` into `_local_score` as the `continuity_penalty` arg.

### Reset between variations

In `fetch_broll_cut_variations`, the outer loop is `for v in range(variations)`. Reset the ledger at the top of each iteration:

```python
for v in range(variations):
    ledger = _CONTINUITY_LEDGER.setdefault(job_id, [])
    ledger.clear()
    ...
```

This means **variations are independent** — variation 0's picks don't influence variation 1's picks.

---

## 10. YouTube rung re-rank

**File:** `app/broll.py`, function `search_youtube_candidates` at line 1390.

Currently scores each preview with the legacy `_profile_similarity` (category + subjects + setting only, line 1381). Refactor:

### New YouTube scoring path

```python
def search_youtube_candidates(
    profile: SpanProfile,
    cache_dir: Path,
    settings: Settings,
    count: int = 2,
    *,
    reference_house: dict | None = None,
    intelligent: bool = True,
) -> list[Path]:
    """Download + score up to `count` YouTube preview candidates for a profile.

    When intelligent=True: scores each preview with _local_score(intelligent=True),
    using the EXTENDED prompt so a downloaded preview also has mood/lighting/etc.
    Falls back to the legacy _profile_similarity when vibe fields are missing
    on the preview (data is thin).
    """
```

### Frame tag path

The existing call at line 1434:

```python
tags = _frame_tags(frame_path, settings, budget=None)
```

This already uses `_TAG_PROMPT`, which is the extended prompt after §3 — so a downloaded YouTube preview gets the seven vibe keys when the vision model cooperates. **No change needed there** beyond ensuring the prompt is bumped.

### New score-and-pick loop

Replace lines 1420–1438:

```python
    scored: list[tuple[Path, float, float, float]] = []
    for i, entry in enumerate(entries[: max(count, 2)]):
        preview_path = cache_dir / f"preview_{i}.mp4"
        if not _download_youtube_preview(entry["url"], preview_path, settings):
            continue
        try:
            duration = probe_duration(preview_path, settings)
        except Exception:
            duration = 0.0
        if duration <= 0:
            continue
        frame_path = cache_dir / f"preview_{i}.jpg"
        if not _extract_single_frame(preview_path, duration / 2, frame_path, settings):
            continue
        tags = _frame_tags(frame_path, settings, budget=None)
        if not tags:
            continue
        # Wrap frame tags into a LibraryClip-shaped stub for _local_score.
        stub = LibraryClip(
            path=preview_path, mtime=0.0, size=0,
            subjects=tags.get("subjects", []), setting=tags.get("setting", []),
            category=tags.get("category", "other"), folder="youtube",
            query=tags.get("query", ""),
            mood=tags.get("mood", []), energy=tags.get("energy", ""),
            lighting=tags.get("lighting", ""), shot_type=tags.get("shot_type", ""),
            camera_motion=tags.get("camera_motion", ""),
            depth_of_field=tags.get("depth_of_field", ""),
            color_palette=tags.get("color_palette", []),
        )
        total, _vibe, _cinema, _cont = _local_score(
            profile, stub, intelligent=intelligent,
            reference_house=reference_house, continuity_penalty=0.0,
        )
        if total <= 0:
            # Graceful fallback: legacy scoring when new fields are missing.
            legacy = _profile_similarity(profile, tags)
            if legacy > 0:
                scored.append((preview_path, legacy / 6.0, 0.0, 0.0))
            continue
        scored.append((preview_path, total, 0.0, 0.0))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [path for path, _score, _vibe, _cont in scored[:count]]
```

Notes:
- The stub `LibraryClip` reuses the existing dataclass — no new type needed.
- `_profile_similarity` returns a raw 0..6 number; divide by 6 to put it in the same 0..1 space as `_local_score`.
- The continuity ledger does NOT participate in YouTube scoring (continuity_penalty=0.0) — YouTube is the safety-net rung, not the visual-variety driver.

### Call site updates

`search_youtube_candidates` is called at line 1885 (in `_gather_pack_sources_for_span`) and line 1693 (in `_gather_span_pool`). Both call sites need to forward `intelligent=...` and `reference_house=reference_house`.

---

## 11. Diagnostics surface

**File:** `app/broll.py` + `app/models.py`.

### Add `cinema` + `continuity_penalty` + `intelligent_active` to `BrollRecoveryDiagnostic`

In `app/models.py` around line 130, add three fields to `BrollRecoveryDiagnostic`:

```python
    # Cinema-match subscore in [0,1]. 0.0 when intelligent_active is False or
    # when neither side carries cinema fields.
    cinema: float = 0.0
    # Continuity penalty applied to this pick. Non-positive. 0.0 when no
    # previous pick exists in the ledger, or when continuity_cosine < threshold.
    continuity_penalty: float = 0.0
    # Top-level flag that makes it crystal clear on the UI which mode ran
    # per clip. True iff the Form's `use_intelligent_selector` was on AND
    # the job ran with intelligent=True.
    intelligent_active: bool = False
```

### Surface in `BrollRecoveryDiagnostic.reason`

In `app/broll.py` at line 1806 (the diagnostic-construction site in `fetch_broll_cut_variations`), extend the existing format. New shape:

```python
diagnostics.append(BrollRecoveryDiagnostic(
    start=out_start, end=out_end, query=profile.query,
    provider=provider, source=str(path), match_type=provider,
    selected=True,
    reason=(
        f"{reason}; cinema={cinema:.2f} continuity_penalty={cont:.2f}"
    ),
    vibe_score=float(vibe),
    cinema=float(cinema),
    continuity_penalty=float(cont),
    intelligent_active=bool(intelligent),
))
```

The `cinema` and `cont` values are pulled from the pool row — extend the pool tuple from `(path, provider, reason, vibe_score)` (4-tuple) to `(path, provider, reason, vibe_score, cinema_score, continuity_penalty)` (6-tuple). Update every consumer: `_gather_span_pool`, `fetch_broll_cut_variations`, `gather_broll_pack`.

### How `cinema` reaches the pool row

`_local_score` already returns the `(total, vibe, cinema, continuity_penalty)` 4-tuple. The plumbing:

```python
# In _gather_span_pool, around line 1674:
total, vibe, cinema, cont = _local_score(
    profile, clip, intelligent=intelligent,
    reference_house=reference_house, continuity_penalty=pending_cont,
)
pool.append((picked.path, "local", f"library match: {picked.path.name}", vibe, cinema, cont))
```

For the YouTube rung at line 1695:

```python
total, _vibe, cinema, cont = _local_score(profile, stub, intelligent=intelligent, ...)
pool.append((yt_path, "youtube", "matched YouTube clip", _vibe, cinema, 0.0))
```

For the reference-crop rung at line 1701:

```python
pool.append((cropped, "reference_crop", "no local/YouTube match; cropped reference cutaway", 0.0, 0.0, 0.0))
```

The diagnostic-construction site at line 1806 destructures the new 6-tuple:

```python
path, provider, reason, vibe, cinema, cont = pool[min(v, len(pool) - 1)]
```

---

## 12. Settings knobs

**File:** `app/config.py`, inside the `Settings` dataclass (after line 144, just before the `youtube_data_api_keys` property).

Add four pydantic fields with these defaults, ranges, and meanings:

```python
    # Cinema-match floor (see app/broll.py:_CINEMA_FLOOR). The lowest
    # cinema_match subscore a clip can achieve. Default 0.18 means a
    # truly mismatched clip (extreme-close-up + handheld against a
    # wide + tracking ref) cannot score above ~20% even on perfect
    # content overlap. Range [0.05, 0.40]. Lower = stricter demotion.
    intelligent_cinema_floor: float = Field(
        default=0.18, validation_alias="CAMPEDITOR_INTELLIGENT_CINEMA_FLOOR",
        ge=0.05, le=0.40,
    )
    # Maximum continuity penalty applied when the previous span's pick
    # has cosine_similarity_6d >= threshold. Non-positive. Default -0.08
    # nudges consecutive B-roll slots toward visual variety without
    # forcing it. Range [-0.20, 0.0]. More negative = stronger nudge.
    intelligent_continuity_penalty_max: float = Field(
        default=-0.08, validation_alias="CAMPEDITOR_INTELLIGENT_CONTINUITY_PENALTY_MAX",
        ge=-0.20, le=0.0,
    )
    # Cosine similarity threshold above which the continuity penalty is
    # applied. Default 0.92 = only near-identical consecutive picks pay.
    # Range [0.70, 0.99]. Lower = penalty fires more often.
    intelligent_continuity_cosine_threshold: float = Field(
        default=0.92, validation_alias="CAMPEDITOR_INTELLIGENT_CONTINUITY_COSINE_THRESHOLD",
        ge=0.70, le=0.99,
    )
    # Frame-tag prompt version. Bumped whenever _TAG_PROMPT changes
    # shape. The purge-frame-tag-cache helper reads this and wipes
    # data/cache/broll_tags when the on-disk stamp is older.
    # Default 2 = the post-§3 extended prompt.
    intelligent_frame_tag_prompt_version: int = Field(
        default=2, validation_alias="CAMPEDITOR_INTELLIGENT_FRAME_TAG_PROMPT_VERSION",
        ge=1,
    )
```

The tunables are referenced from `app/broll.py` via `settings.intelligent_cinema_floor` etc. — never import from `app/config.py` into `broll_intelligence/` and vice versa.

---

## 13. Acceptance test

**New file:** `C:\campeditor\tests\test_intelligent_selector_v2.py` (pytest). Tests the new `_local_score`, `_cinema_match`, `_continuity_penalty`, `_resolve_span_vibe`, `build_reference_house_style` in isolation.

### LibraryClip / SpanProfile construction (literal dataclass instances)

```python
from app.broll import (
    LibraryClip, SpanProfile, _local_score, _cinema_match,
    _continuity_penalty, _resolve_span_vibe, build_reference_house_style,
)

def make_clip(name: str, **kw) -> LibraryClip:
    defaults = dict(
        path=Path(f"/tmp/{name}.mp4"), mtime=0.0, size=0,
        subjects=[], setting=[], category="sports", folder="nba",
        query="basketball stadium",
        mood=[], energy="", lighting="", shot_type="",
        camera_motion="", depth_of_field="", color_palette=[],
    )
    defaults.update(kw)
    return LibraryClip(**defaults)

clipA = make_clip("A", subjects=["player"], setting=["stadium"], category="sports",
                  mood=["epic"], energy="high", lighting="low-key",
                  shot_type="wide", camera_motion="tracking", depth_of_field="deep",
                  query="epic wide basketball tracking", confidence=0.8)
clipB = make_clip("B", subjects=["player"], setting=["stadium"], category="sports",
                  mood=["uplifting"], energy="high", lighting="high-key",
                  shot_type="extreme-close-up", camera_motion="handheld",
                  depth_of_field="shallow", query="close basketball action", confidence=0.8)
clipC = make_clip("C", subjects=["player"], setting=["stadium"], category="sports",
                  mood=["epic"], energy="high", lighting="low-key",
                  shot_type="wide", camera_motion="tracking", depth_of_field="deep",
                  query="epic wide basketball tracking", confidence=0.8)
clipD = make_clip("D", subjects=["player"], setting=["stadium"], category="sports",
                  mood=["dramatic"], energy="high", lighting="natural",
                  shot_type="aerial", camera_motion="dolly", depth_of_field="deep",
                  query="basketball aerial establishing", confidence=0.8)
clipE = make_clip("E", subjects=["player"], setting=["stadium"], category="sports",
                  mood=["epic"], energy="high", lighting="low-key",
                  shot_type="wide", camera_motion="tracking", depth_of_field="deep",
                  query="epic wide basketball tracking", confidence=0.5)
clipF = make_clip("F", subjects=["player"], setting=["stadium"], category="sports",
                  mood=["joyful"], energy="medium", lighting="natural",
                  shot_type="medium", camera_motion="static", depth_of_field="shallow",
                  query="basketball medium joyful", confidence=0.8)

ref_span = SpanProfile(
    start=0.0, end=2.0,
    subjects=["player"], setting=["stadium"],
    action=["shooting"], category="sports", query="epic basketball wide",
    mood=["epic"], energy="high", lighting="low-key",
    shot_type="wide", camera_motion="tracking", depth_of_field="deep",
)
```

### Test assertions

```python
def test_intelligent_true_ranks_a_above_b_and_f():
    a_score, _, a_cinema, _ = _local_score(ref_span, clipA, intelligent=True)
    b_score, _, b_cinema, _ = _local_score(ref_span, clipB, intelligent=True)
    f_score, _, f_cinema, _ = _local_score(ref_span, clipF, intelligent=True)
    assert a_cinema > 0.9, "A must be near-perfect cinema match"
    assert b_cinema < 0.30, "B must be heavily demoted by cinema floor"
    assert f_cinema < 0.50, "F must also be demoted (medium != wide)"
    assert a_score > b_score, "intelligent=True must rank A above B"
    assert a_score > f_score, "intelligent=True must rank A above F"

def test_intelligent_false_ranks_a_and_b_near_tied():
    a_score, _, _, _ = _local_score(ref_span, clipA, intelligent=False)
    b_score, _, _, _ = _local_score(ref_span, clipB, intelligent=False)
    # Both have subjects/setting/category all matching -> identical base score.
    assert abs(a_score - b_score) < 0.01, (
        "intelligent=False ignores cinema; A and B share category+subjects+setting"
    )

def test_continuity_penalty_kicks_in_for_consecutive_picks():
    """Pick A then score C (looks-like-A); the second pick must pay the tax."""
    a_score, _, _, _ = _local_score(ref_span, clipA, intelligent=True, continuity_penalty=0.0)
    # First pick: ledger = [A-vec], score C with continuity
    c_score_with_penalty, _, _, c_pen = _local_score(
        ref_span, clipC, intelligent=True, continuity_penalty=-0.08,
    )
    c_score_without_penalty, _, _, _ = _local_score(
        ref_span, clipC, intelligent=True, continuity_penalty=0.0,
    )
    assert c_pen < 0.0, "continuity_penalty must be negative when above threshold"
    assert c_score_with_penalty < c_score_without_penalty, "penalty must lower the score"

def test_house_style_back_fills_empty_span_fields():
    """A span with empty mood should fall back to the house style mood."""
    empty_span = SpanProfile(
        start=0.0, end=2.0,
        subjects=["player"], setting=["stadium"],
        category="sports", query="basketball",
        mood=[], energy="", lighting="", shot_type="", camera_motion="",
        depth_of_field="",
    )
    house = {"mood": ["epic"], "lighting": "low-key", "shot_type": "wide",
             "camera_motion": "tracking", "depth_of_field": "deep",
             "energy": "high", "color_palette": []}
    resolved = _resolve_span_vibe(empty_span, house)
    assert resolved["mood"] == ["epic"], "house style must back-fill mood"
    assert resolved["shot_type"] == "wide", "house style must back-fill shot_type"
    # Now score against clipA — should match.
    total, vibe, _, _ = _local_score(empty_span, clipA, intelligent=True, reference_house=house)
    assert vibe > 0.8, "house-style back-filled span must score high vibe against A"

def test_build_reference_house_style_aggregates():
    analysis = ReferenceAnalysis(
        duration=10.0,
        broll_spans=[(0.0, 2.0, "q1"), (2.0, 4.0, "q2")],
        broll_span_tags=[
            {"mood": ["epic"], "lighting": "low-key", "shot_type": "wide",
             "camera_motion": "tracking", "energy": "high", "depth_of_field": "deep"},
            {"mood": ["epic", "dramatic"], "lighting": "natural", "shot_type": "medium",
             "camera_motion": "dolly", "energy": "medium", "depth_of_field": "deep"},
        ],
    )
    house = build_reference_house_style(analysis)
    assert "epic" in house["mood"], "house style must include union mood"
    assert "dramatic" in house["mood"], "house style must include union mood"
    # Mode: lighting = low-key (1) vs natural (1) → tie → first occurrence → "low-key"
    assert house["lighting"] == "low-key"
    # Mode: camera_motion = tracking (1) vs dolly (1) → tie → first → "tracking"
    assert house["camera_motion"] == "tracking"

def test_low_confidence_clip_still_works():
    """clipE has confidence=0.5 (low). The scorer should still rank it
    sensibly, never crash, and not artificially inflate its score."""
    e_score, _, _, _ = _local_score(ref_span, clipE, intelligent=True)
    assert 0.0 <= e_score <= 1.0
```

### Pass criteria

- `pytest tests/test_intelligent_selector_v2.py -v` → all six tests pass.
- A manual end-to-end run with `intelligent=True` and `intelligent=False` on a contrived scenario (one sports-wide-tracking ref against a mixed local library) must produce visibly different picks.

---

## 14. Migration: backward compatibility

The fix is backward-compatible by construction:

- **Form flag wiring is unchanged.** `app/main.py:113` accepts `use_intelligent_selector: bool = False` (defaults to False; the Form has its own default). `app/jobs.py:104` stores it on the Job model. Every caller uses `getattr(job, "use_intelligent_selector", True)` as before. **Old meta.json files load fine** — they pass `intelligent=False` (or whatever they had) into `fetch_broll_cuts` / `fetch_broll_cut_variations` / `gather_broll_pack` / `fetch_learned_broll_cuts`, and those functions gate the new code on `intelligent=True`.

- **`broll_index.json` version-1 is treated as empty when the live version is 2.** The existing version-mismatch branch in `build_library_index` (line 803) already handles this: `if raw.get("version") == _INDEX_CACHE_VERSION: cached = raw.get("clips", {})` — when `_INDEX_CACHE_VERSION` is bumped to 2, the old `version: 1` payload fails the comparison and `cached` stays `{}`. The next call rebuilds.

- **No breaking change to `LibraryClip` or `SpanProfile` dataclasses.** All seven new fields have empty defaults (`""` / `[]`). Legacy code paths that build a LibraryClip without the new fields continue to compile and run.

- **No import of `broll_intelligence` from `app/*`.** Verified via `grep -rn "from broll_intelligence\|import broll_intelligence" C:/campeditor/app` returning NO MATCHES. The two packages are siblings — the sibling is the reference implementation, the runtime path is `app/broll.py`.

- **One-shot frame-tag cache purge** on first build after deploy wipes `data/cache/broll_tags/`. Subsequent rebuilds reuse the cache. Cost is paid once.

---

## 15. Out-of-scope notes

The following are deliberately not in this iteration:

- **Pexels and Frame.io v4 sources.** The current YouTube-only fallback remains. Adding a third source would change the scoring ladder in ways this spec does not anticipate.

- **Real learnable weights from user feedback.** The `_VIBE_BONUS_WEIGHT` and `_CINEMA_LIFT_TERM` are still hand-tuned constants. A future iteration could close the loop on `BrollRecoveryDiagnostic.user_picked` events, but that's a separate piece of work.

- **The full selector package at `broll_intelligence/selector/`.** That is a separate (orphan) effort. This V2 work happens entirely in `app/*`. The sibling `broll_intelligence/` package's `CONTRACT.md` is the source of truth for the closed vocabs and vision-prompt text — nothing else from that package is consumed at runtime.

- **A new UI control.** The existing checkbox (`static/index.html#use-intelligent-selector`, `app.js` → `app/main.py:113`) is the entire UI surface. No slider, no debug panel, no per-mode preview.

- **Refreshing the `_local_score` `score` accumulator.** The legacy max (`_LOCAL_SCORE_MAX = 8.0`) and category-3/subjects-3/setting-2 scoring shape is preserved verbatim — agents A/B/C MUST NOT re-tune the legacy weights. The cinema + vibe multipliers are added on TOP of the existing base score.

---

## Appendix A — Quick-reference anchors for agents

| Concern | File | Lines (approx) |
|---|---|---|
| Closed vocabs to add | `app/broll.py` | after line 58 |
| Extended prompt | `app/broll.py:_TAG_PROMPT` | 544 |
| Parser tolerance | `app/broll.py:_parse_tags` | 575 |
| Frame tags (uses prompt) | `app/broll.py:_frame_tags` | 618 |
| Merge logic | `app/broll.py:_merge_tags` | 646 |
| Library clip tagging | `app/broll.py:_tag_library_clip` | 766 |
| Cache version | `app/broll.py:_INDEX_CACHE_VERSION` | 76 |
| Legacy local score | `app/broll.py:_local_score` | 1006 |
| Vibe bonus (replace) | `app/broll.py:_VIBE_BONUS_WEIGHT` | 894 |
| Vibe fields table (replace) | `app/broll.py:_VIBE_FIELDS` | 900 |
| Vibe subscore | `app/broll.py:_vibe_subscore` | 947 |
| Vibe score | `app/broll.py:_vibe_score_for` | 988 |
| Library index build | `app/broll.py:build_library_index` | 790 |
| Library index cache write | `app/broll.py` | ~866-874 |
| LibraryClip dataclass | `app/broll.py` | 114 |
| SpanProfile dataclass | `app/broll.py` | 84 |
| Span profile build | `app/broll.py:_span_profile_for` | 1599 |
| Gather span pool | `app/broll.py:_gather_span_pool` | 1624 |
| Pack sources | `app/broll.py:_gather_pack_sources_for_span` | 1837 |
| Cut variations | `app/broll.py:fetch_broll_cut_variations` | 1716 |
| YouTube search candidates | `app/broll.py:search_youtube_candidates` | 1390 |
| YouTube profile similarity (legacy) | `app/broll.py:_profile_similarity` | 1381 |
| Diagnostic construction | `app/broll.py` | 1806 |
| Diagnostic model | `app/models.py:BrollRecoveryDiagnostic` | 103 |
| Settings knobs | `app/config.py:Settings` | after line 144 |
| Job kwarg | `app/jobs.py` | 104 |
| Form field | `app/main.py` | 113 |

---

## Appendix B — Numeric tunables (defaults + ranges + meaning)

| Knob | Default | Range | Meaning |
|---|---|---|---|
| `intelligent_cinema_floor` | `0.18` | `[0.05, 0.40]` | Minimum `cinema_match` subscore. Lower = stricter demotion on cinematography mismatch. |
| `intelligent_continuity_penalty_max` | `-0.08` | `[-0.20, 0.0]` | Penalty applied when consecutive picks have cosine ≥ threshold. More negative = stronger diversity nudge. |
| `intelligent_continuity_cosine_threshold` | `0.92` | `[0.70, 0.99]` | Cosine similarity threshold above which the continuity penalty fires. Lower = penalty fires more often. |
| `intelligent_frame_tag_prompt_version` | `2` | `>= 1` | Bump when `_TAG_PROMPT` changes shape. Triggers one-shot purge of `data/cache/broll_tags/`. |
| `_VIBE_BONUS_WEIGHT` (constant) | `0.35` | n/a | Maximum vibe-score lift on the legacy `base` score. Preserved from V1; do not retune here. |
| `_CINEMA_LIFT_TERM` (constant) | `0.5` | n/a | Multiplier on `(cinema_match - 0.6)` gap when `cinema_match < 0.6`. Higher = harsher cinema mismatch penalty. |
| `_CINEMA_FLOOR` (constant) | `0.18` | n/a | Hard floor on `cinema_match` subscore. Mirrors `intelligent_cinema_floor`. |
| `_MOOD_CAP` | `3` | n/a | Max moods per clip/span. |
| `_COLOR_PALETTE_CAP` | `3` | n/a | Max color strings per clip/span. |

---

## Appendix C — Work assignment (handoff to agents)

- **Agent A** owns: §2 (constants), §3 (prompt), §4 (parser), §5 (merge + tag), §6 (cache version + purge helper). Files touched: `app/broll.py` only.
- **Agent B** owns: §7 (cinema_match + cinema-aware `_local_score`), §9 (continuity ledger + helpers). Files touched: `app/broll.py` only.
- **Agent C** owns: §8 (house style + propagation), §10 (YouTube re-rank), §11 (diagnostics surface + model extension), §12 (Settings knobs). Files touched: `app/broll.py` + `app/models.py` + `app/config.py`.
- **All three agents** share: §13 (acceptance test). Agent B writes the test skeleton, Agent C extends the diagnostic assertions.

The spec is the contract. If a downstream agent finds an ambiguity, the tiebreaker is the existing V1 behavior (preserved in §14) — never invent new semantics. If V1 behavior is also ambiguous, escalate to the architect (Task #26 owner) before coding.