from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    INGESTED = "ingested"
    ANALYZING = "analyzing"
    TRANSCRIBING = "transcribing"
    SELECTING = "selecting"
    TITLING = "titling"
    BROLL_RECOVERY = "broll_recovery"
    RENDERING = "rendering"
    READY = "ready"
    FAILED = "failed"


class ColorGrade(str, Enum):
    NONE = "none"
    CINEMATIC = "cinematic"
    WARM = "warm"
    COOL = "cool"
    PUNCHY = "punchy"


class TitleMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class ClipMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class TranscriptWord(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    words: list[TranscriptWord] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        if self.segments:
            return " ".join(segment.text for segment in self.segments)
        return " ".join(word.word for word in self.words)


class Title(BaseModel):
    line1: str
    line2: str
    highlight_words: list[str] = Field(default_factory=list)
    # The single word/short phrase the LLM identified as the scroll-stopping
    # hook of this title. Purely diagnostic — the renderer reads
    # `highlight_words`; refinement logic uses this to ensure the right word
    # ends up red. MUST always be a substring of `line1 + " " + line2` when set.
    hook_word: str | None = None


class BrollCut(BaseModel):
    """A B-roll cutaway in the OUTPUT timeline, backed by a local clip file."""

    start: float
    end: float
    clip_path: Path
    query: str = ""


class BrollPackItem(BaseModel):
    """One entry in a downloadable B-roll pack for a single detected span.

    The pack is a flat list of 1–2 trimmed clips per span (rank 1 = best match,
    rank 2 = first viable alternate). `start`/`end` are coordinates on the
    OUTPUT clip's timeline (what fetch_broll_cut_variations would insert), not
    the source clip's intrinsic timeline — this is what lets the user drop a
    pack item straight into Premiere without re-mapping.

    `clip_path` is an absolute path under `data/jobs/<job_id>/broll/` so the
    download endpoint can serve it via FileResponse with the same path
    containment check used for variations.
    """

    span_index: int
    rank: int  # 1 (best match) or 2 (first viable alternate)
    start: float  # output-timeline start
    end: float  # output-timeline end
    query: str
    provider: str  # local / youtube / reference_crop
    clip_path: Path


class BrollRecoveryDiagnostic(BaseModel):
    """Per-span source-recovery details for high-accuracy B-roll matching."""

    start: float
    end: float
    query: str = ""
    provider: str = ""
    source: str = ""
    score: float = 0.0
    match_type: str = "rejected"
    selected: bool = False
    reason: str = ""
    thumbnails: list[Path] = Field(default_factory=list)
    required_concepts: list[str] = Field(default_factory=list)
    candidate_description: str = ""
    # New in Step 6 — vibe/rotation telemetry for the replicate pool path:
    # vibe_score  | 0.0-1.0 fused vibe of the picked candidate vs reference
    #              | span tags (visual*0.55 + tag_overlap*0.35 + provider*0.10).
    #              | 0.0 when no ref tags were available (pre-Step-3 spans).
    # pool_size   | how many concept-passing candidates _pick_recovered_candidate
    #             | collected for this span (rotation fallback picks from this).
    # variation_index | which of N variations this diagnostic row belongs to
    #                  | (0..variation_count-1). 0 = primary pipeline run.
    # fallback_query_used | True when the span used the scene-tag-derived
    #                      | vibe query (no ref_tags signal / canonical query
    #                      | came back empty), False when the canonical query
    #                      | drove candidate search, "" pre-Step-6 rows.
    vibe_score: float = 0.0
    pool_size: int = 0
    variation_index: int = 0
    fallback_query_used: str = ""
    # Spec §11 — V2 intelligent-selector diagnostics. Added here with safe
    # defaults so existing rows (without these fields) continue to load.
    # Cinema-match subscore in [0,1]. 0.0 when intelligent_active is False or
    # when neither side carries cinema fields.
    cinema: float = 0.0
    # Continuity penalty applied to this pick. Non-positive. 0.0 when no
    # previous pick exists in the ledger, or when continuity_cosine <
    # threshold.
    continuity_penalty: float = 0.0
    # Top-level flag that makes it crystal clear on the UI which mode ran
    # per clip. True iff the Form's `use_intelligent_selector` was on AND
    # the job ran with intelligent=True.
    intelligent_active: bool = False


class ReferenceAnalysis(BaseModel):
    """What we reverse-engineered from an uploaded reference short."""

    duration: float = 0.0
    title_text: str = ""
    transcript_text: str = ""
    # B-roll spans in the reference timeline (seconds) with a stock-search query each.
    broll_spans: list[tuple[float, float, str]] = Field(default_factory=list)
    # Structured scene tags per span, indexed in parallel with `broll_spans`.
    # Each entry is a dict of: subject (list[str]), setting (list[str]),
    # action (list[str]), mood (list[str]), shot_type (str), lighting (str),
    # vibe_query (str). Populated lazily by replicate._describe_broll_span with
    # disk-cache by frame hash. Empty dict when not yet scored (non-replicate
    # modes never populate this; replicate mode populates it for vibe matching).
    broll_span_tags: list[dict] = Field(default_factory=list)
    # Per-span flag (parallel to broll_spans) describing where each span's
    # search query came from: "canonical" (mid-frame vision description),
    # "tag_fallback" (canonical query empty → rebuilt from span tags via
    # _span_tags_to_vibe_query), "" for pre-Step-3 rows. Surfaced via
    # BrollRecoveryDiagnostic.fallback_query_used for operator visibility.
    broll_query_source: list[str] = Field(default_factory=list)
    music_path: Path | None = None
    # Mean volume (dBFS) of the reference's separated music / background-audio
    # track, measured once when music is extracted. The renderer applies it to
    # our final output ("R7") so the music loudness matches the reference.
    music_volume_db: float | None = None
    # Hook window at the start of the reference (start, end) in seconds — the
    # lead-in B-roll cutaway before the speaker starts talking. A common
    # pattern in viral shorts: 0.5-3s of music-over-broll to grab attention,
    # then the A-roll. None when the speaker starts at frame 0 (no hook).
    hook_span: tuple[float, float] | None = None
    # Visual tags for the hook window: subjects / setting / action / category
    # (from the existing frame-tagger) + `personality` (name of any famous
    # business figure / celebrity the vision model identified, or ""). Used
    # by fetch_hook_broll to find a matching library clip to prepend to
    # the output so the rendered short opens with the same kind of hook.
    hook_tags: dict | None = None


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.INGESTED
    progress: float = 0.0
    message: str = ""
    # Estimated total editing time in seconds (set when the job starts).
    # Updated as stages progress so the UI can show remaining ETA.
    eta_seconds: float | None = None
    # Stage timing telemetry — filled in as stages complete. Keys are
    # JobStatus values (e.g. "analyzing", "rendering"). Values are elapsed
    # seconds for that stage. Used to refine ETA estimates for future jobs
    # and to surface per-stage timing on the dashboard.
    stage_timings: dict[str, float] = Field(default_factory=dict)
    # Wall-clock timestamp (seconds since epoch) the job entered its current
    # stage. Lets us compute elapsed-so-far for the running stage.
    stage_started_at: float | None = None
    source_path: Path | None = None
    clip_mode: ClipMode = ClipMode.MANUAL
    clip_reason: str = ""
    start: float = 0.0
    end: float = 0.0
    title_mode: TitleMode = TitleMode.AUTO
    manual_title: str = ""
    color_grade: ColorGrade = ColorGrade.NONE
    replicate: bool = False
    reference_url: str = ""
    reference_path: Path | None = None
    music_path: Path | None = None
    # User-uploaded logo image for Replicate mode. The system searches the
    # reference for this image (or detects any static overlay if absent), and
    # the renderer applies the user's logo at the matched position+opacity.
    logo_path: Path | None = None
    reference: ReferenceAnalysis | None = None
    broll_cuts: list[BrollCut] = Field(default_factory=list)
    broll_recovery: list[BrollRecoveryDiagnostic] = Field(default_factory=list)
    # ── B-roll pack (downloadable stock alternatives) ─────────────────────
    # When True (only meaningful in replicate mode), the pipeline runs
    # gather_broll_pack instead of fetch_broll_cuts and stores the 1–2 trimmed
    # clips per span here for download. The main rendered video gets NO cuts
    # in this mode — the pack is the deliverable. Defaults keep pre-pack
    # meta.json files loading without modification.
    broll_pack: bool = False
    broll_pack_items: list[BrollPackItem] = Field(default_factory=list)
    # ── Learned B-roll (non-replicate jobs) ──────────────────────────────
    # When True (the historical default), a non-replicate job will reuse the
    # running B-roll profile to synthesize placement spans and insert local
    # library matches at those points. Set to False when the user wants a
    # plain caption + title render with NO B-roll injected — the profile is
    # still updated from replicate jobs regardless. Defaults keep old
    # meta.json files loading fine.
    enable_learned_broll: bool = True
    # ── Intelligent B-roll selector (replicate jobs) ─────────────────────
    # When True (default), the local-library B-roll ranking layer adds a
    # vibe / lighting / shot-type bonus on top of the historical
    # category/subjects/setting score. Helps the picker choose a clip that
    # *looks* like the reference cutaway, not just one that shares keywords.
    # Off = historical keyword-only matching. Persisted on the Job so the
    # audit row can surface which path ran.
    use_intelligent_selector: bool = True
    transcript: Transcript = Field(default_factory=Transcript)
    title: Title | None = None
    output_path: Path | None = None
    error: str | None = None
    # How many rendered variations to produce. Single-upload mode sets this to
    # settings.variation_count (default 4). Bulk mode forces 1 per pair.
    variation_count: int = 1
    # Per-variation render outputs (paths to rendered mp4 files). Index 0 is
    # the "primary" render and equals `output_path` for backward compat.
    variations: list["Variation"] = Field(default_factory=list)
    # Bulk-mode marker (set when this job is one entry of a bulk upload batch).
    bulk: bool = False
    # Frame.io folder URL provided for auto source-finding. When set, the
    # pipeline syncs the folder, visually matches the reference against every
    # video, and auto-sets source_path/start/end to the matched segment.
    frameio_source_url: str = ""
    # Path to the matched raw source video (set by source-finding step).
    raw_source_path: Path | None = None
    # Confidence score (0..1) of the visual match that identified the source.
    raw_source_match_score: float = 0.0


class Variation(BaseModel):
    """One rendered variation of a job. All variations share the same music,
    caption (ASS) file, and B-roll placement timeline (start/end seconds); they
    differ in title text and B-roll source clips. The user picks the best one.
    """

    index: int
    title: Title
    broll_cuts: list[BrollCut] = Field(default_factory=list)
    output_path: Path | None = None
    # Optional short label for the UI ("Variation 2 of 4", etc.).
    label: str = ""


class BrollPackDownload(BaseModel):
    """One row in JobSummary.broll_pack_urls — UI/discovery metadata for a
    single trimmed pack clip. `url` is the download endpoint, the rest mirrors
    the on-disk BrollPackItem so the client can label/sort the rows without a
    second fetch."""

    span_index: int
    rank: int
    start: float
    end: float
    query: str
    url: str


class JobSummary(BaseModel):
    id: str
    status: JobStatus
    progress: float
    message: str
    output_url: str | None = None
    # Per-variation download URLs (when variation_count > 1). Index 0 mirrors
    # output_url for backward compatibility.
    variation_urls: list[str] = Field(default_factory=list)
    # Per-pack-clip download URLs (only populated when the job runs in
    # broll_pack mode). Each entry points at /api/renders/{id}/broll/{i}.mp4
    # and carries the span + rank metadata so the UI can group them.
    broll_pack_urls: list[BrollPackDownload] = Field(default_factory=list)
    error: str | None = None
    # Estimated total editing time in seconds. UI uses this to show "ETA: ~2m
    # remaining" while the job is in progress.
    eta_seconds: float | None = None
    # Per-stage elapsed seconds so far (only filled after each stage completes).
    stage_timings: dict[str, float] = Field(default_factory=dict)
