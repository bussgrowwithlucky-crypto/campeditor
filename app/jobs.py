import logging
import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from fastapi import UploadFile

from app.clip_selection import select_viral_clip, slice_transcript
from app.config import Settings
from app.models import ClipMode, ColorGrade, Job, JobStatus, Title, TitleMode
from app.rendering import probe_duration, render
from app.broll_profile import (
    update_broll_profile_atomic,
)
from app.broll import (
    ContinuityLedger,
    build_reference_house_style,
    fetch_broll_cuts,
    fetch_broll_cut_variations,
    fetch_hook_broll,
    gather_broll_pack,
)
from app.frameio_source import ensure_frameio_library
from app.replicate import analyze_reference, download_reference
from app.raw_source_finder import find_raw_source
from app.store import JobStore
from app.title_generation import fallback_title, generate_title, manual_title, replicate_title
from app.transcription import transcribe

logger = logging.getLogger(__name__)


def _job_broll_span_count(job: Job) -> int:
    """Number of B-roll spans detected in the job's reference analysis.

    Used to scale the broll_recovery stage prior in ETA forecasting. Returns 0
    when the reference has not been analyzed yet (early stages) â€” the ETA
    scaler then falls back to a single-span prior, which is a safe
    underestimate."""
    reference = getattr(job, "reference", None)
    if reference is None:
        return 0
    spans = getattr(reference, "broll_spans", None)
    return len(spans) if spans else 0


class JobCancelled(Exception):
    pass


class Pipeline:
    # How long a non-terminal job may go without meta.json being updated
    # before the watchdog kills it. INGESTED jobs that don't advance within
    # this window are typically a sign that the executor silently dropped
    # the future â€” better to surface a clear failure than spin forever.
    # Raised from 120s â†’ 900s because the per-job heartbeat thread is a
    # Python thread blocked on the GIL while the worker runs CPU-bound code
    # inside fetch_broll_cuts / fetch_broll_cut_variations (5-10 min on a
    # full reference), so mtime can legitimately go stale for minutes even
    # with a healthy worker. Per-stage exemptions below override this for
    # known-fast stages.
    STUCK_JOB_TIMEOUT_SECONDS: float = 900.0
    WATCHDOG_POLL_INTERVAL_SECONDS: float = 15.0
    # How often the per-job heartbeat thread re-saves meta.json during long
    # stages (broll_recovery can run 5-10 min per job). Keeps mtime fresh so
    # the watchdog can distinguish "worker actively running" from "worker
    # silently dropped the future".
    WORKER_HEARTBEAT_SECONDS: float = 30.0
    # Per-stage stuck-job timeout. broll_recovery is the long pole â€” its
    # ETA forecast alone is up to ~600s and on slow cloud vision calls it
    # can run significantly longer. Keep the early stages on the global
    # default (900s) so a stuck INGESTED/ANALYZING job is still caught.
    _STUCK_TIMEOUT_BY_STATUS: dict[str, float] = {
        "ingested": 180.0,
        "analyzing": 600.0,
        "transcribing": 300.0,
        "selecting": 120.0,
        "titling": 120.0,
        "broll_recovery": 1200.0,
        "rendering": 600.0,
    }

    def __init__(self, settings: Settings, store: JobStore):
        self.settings = settings
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=settings.worker_count)
        self.job_events: dict[str, Event] = {}
        # Track the Future for every submitted job so the watchdog can tell
        # the difference between a queued-but-alive worker (PENDING future)
        # and a dropped worker. Without this, jobs queued behind a long stage
        # get killed by the watchdog even though their worker is alive and
        # waiting in the executor's task queue.
        self.job_futures: dict[str, concurrent.futures.Future] = {}
        # Job dirs the watchdog couldn't process (corrupt/unloadable
        # meta.json, or a job_id missing from the store). Logged once here,
        # never retried - without this, a single unrecoverable directory
        # logs a full ERROR traceback every WATCHDOG_POLL_INTERVAL_SECONDS
        # forever.
        self._watchdog_unrecoverable: set[str] = set()
        # True while shutdown() runs: lets the JobCancelled handler tell a
        # server-restart apart from a real user cancel, so interrupted jobs
        # stay non-terminal and get resumed on the next startup instead of
        # being mislabelled "Cancelled by user".
        self._shutting_down = False
        self._watchdog_stop = Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="pipeline-watchdog",
        )
        self._watchdog_thread.start()
        self._resume_interrupted_jobs()

    def create_job(
        self,
        upload: UploadFile,
        clip_mode: ClipMode,
        start: float,
        end: float,
        title_mode: TitleMode,
        manual_title_text: str,
        color_grade: ColorGrade,
        replicate: bool = False,
        reference_upload: UploadFile | None = None,
        reference_url: str = "",
        music_upload: UploadFile | None = None,
        logo_upload: UploadFile | None = None,
        broll_pack: bool = False,
        enable_learned_broll: bool = True,
        use_intelligent_selector: bool = True,
        add_caption: bool = True,
        broll_source: str = "both",
        use_broll_frameio_2: bool = False,
    ) -> Job:
        broll_source = (broll_source or "both").strip().lower()
        if broll_source not in ("frameio", "youtube", "both"):
            raise ValueError("broll_source must be one of: frameio, youtube, both")
        if clip_mode == ClipMode.MANUAL:
            if end <= start:
                raise ValueError("End time must be after start time")
            if end - start > 90:
                raise ValueError("Clip length is capped at 90 seconds")
        if title_mode == TitleMode.MANUAL and not manual_title_text.strip():
            raise ValueError("Manual title mode requires a title")
        has_reference_file = reference_upload is not None and bool(reference_upload.filename)
        if replicate and not has_reference_file and not reference_url.strip():
            raise ValueError("Replicate mode requires a reference video file or link")
        if replicate and reference_url.strip() and not reference_url.strip().lower().startswith(
            ("http://", "https://")
        ):
            raise ValueError("Reference link must be an http(s) URL")
        job = self.store.create()
        try:
            job.source_path = self.store.save_upload(upload, job.id, self.settings.max_upload_mb)
            if replicate and has_reference_file and reference_upload is not None:
                job.reference_path = self.store.save_upload(
                    reference_upload, job.id, self.settings.max_upload_mb, name="reference"
                )
            if music_upload is not None and music_upload.filename:
                job.music_path = self.store.save_upload(
                    music_upload, job.id, self.settings.max_upload_mb, name="music", allow_audio=True
                )
            if logo_upload is not None and logo_upload.filename:
                job.logo_path = self.store.save_upload(
                    logo_upload,
                    job.id,
                    self.settings.max_upload_mb,
                    name="logo",
                    allow_image=True,
                )
        except Exception:
            self.store.save(job)
            raise
        job.clip_mode = clip_mode
        job.replicate = replicate
        job.reference_url = reference_url.strip()
        job.start = start
        job.end = end
        job.title_mode = title_mode
        job.manual_title = manual_title_text.strip()
        job.color_grade = color_grade
        # broll_pack is only meaningful with replicate=True (we need the
        # reference's detected spans to source alternatives). Store the flag
        # even when False so meta.json persistence round-trips cleanly.
        job.broll_pack = bool(broll_pack)
        # enable_learned_broll gates the auto-learned-B-roll insertion for
        # non-replicate jobs. Defaults to True to preserve historical
        # behavior; users who want a caption+title-only render set False.
        job.enable_learned_broll = bool(enable_learned_broll)
        # use_intelligent_selector adds a vibe/lighting/shot-type bonus on
        # top of the keyword-only local-library matcher. Defaults to True so
        # B-roll picks drift toward the reference's look; off = the legacy
        # score. Only the B-roll ladder reads this; non-replicate jobs
        # ignore it.
        job.use_intelligent_selector = bool(use_intelligent_selector)
        # add_caption gates the ASS caption burn-in for replicate jobs.
        # Defaults to True to preserve historical behavior; unchecked users
        # get a title-only render. Transcription still runs (the title +
        # B-roll queries depend on it) â€” only the burn-in step is skipped.
        job.add_caption = bool(add_caption)
        # Where span B-roll may come from (replicate jobs): the synced
        # Frame.io share, YouTube search, or both. Reference-crop stays the
        # guaranteed fallback in every mode.
        job.broll_source = broll_source
        # Per-job opt-in to merge a secondary Frame.io share (configured at
        # the server via BROLL_FRAMEIO_SHARE_URL_2) into this job's B-roll
        # library. Persisted on the Job so audit rows can surface which
        # jobs pulled from the secondary share, and so a re-run picks up the
        # same flag from meta.json.
        job.use_broll_frameio_2 = bool(use_broll_frameio_2)
        job.status = JobStatus.INGESTED
        job.progress = 0.1
        job.message = "Upload received"
        self.store.save(job)
        self.job_events[job.id] = Event()
        future = self.executor.submit(self._run, job.id)
        self.job_futures[job.id] = future
        return job

    def create_bulk_job(
        self,
        upload: UploadFile,
        reference_upload: UploadFile,
        clip_mode: ClipMode = ClipMode.AUTO,
        title_mode: TitleMode = TitleMode.AUTO,
        manual_title_text: str = "",
        color_grade: ColorGrade = ColorGrade.NONE,
        music_upload: UploadFile | None = None,
        logo_upload: UploadFile | None = None,
        enable_learned_broll: bool = True,
    ) -> Job:
        if reference_upload is None or not reference_upload.filename:
            raise ValueError("Bulk mode requires a reference video for every pair")
        if title_mode == TitleMode.MANUAL and not manual_title_text.strip():
            raise ValueError("Manual title mode requires a title")
        job = self.store.create()
        try:
            job.source_path = self.store.save_upload(upload, job.id, self.settings.max_upload_mb)
            job.reference_path = self.store.save_upload(
                reference_upload, job.id, self.settings.max_upload_mb, name="reference"
            )
            if music_upload is not None and music_upload.filename:
                job.music_path = self.store.save_upload(
                    music_upload, job.id, self.settings.max_upload_mb, name="music", allow_audio=True
                )
            if logo_upload is not None and logo_upload.filename:
                job.logo_path = self.store.save_upload(
                    logo_upload, job.id, self.settings.max_upload_mb, name="logo", allow_image=True
                )
        except Exception:
            self.store.save(job)
            raise
        job.clip_mode = clip_mode
        job.replicate = True
        job.bulk = True
        job.variation_count = 1
        job.title_mode = title_mode
        job.manual_title = manual_title_text.strip()
        job.color_grade = color_grade
        # Bulk mode is always replicate, so the learned-B-roll path is moot
        # for these jobs. Store the flag anyway for meta.json round-trip
        # parity with create_job. Intelligent selector is also on by default
        # since every bulk row runs the B-roll ladder.
        job.enable_learned_broll = bool(enable_learned_broll)
        job.use_intelligent_selector = True
        job.status = JobStatus.INGESTED
        job.progress = 0.1
        job.message = "Bulk upload received"
        self.store.save(job)
        self.job_events[job.id] = Event()
        future = self.executor.submit(self._run, job.id)
        self.job_futures[job.id] = future
        return job

    def create_folder_job(
        self,
        frameio_url: str,
        clip_mode: ClipMode = ClipMode.AUTO,
        title_mode: TitleMode = TitleMode.AUTO,
        manual_title_text: str = "",
        color_grade: ColorGrade = ColorGrade.NONE,
        reference_upload: UploadFile | None = None,
        reference_url: str = "",
        music_upload: UploadFile | None = None,
        logo_upload: UploadFile | None = None,
    ) -> Job:
        """Create a job that auto-finds its raw source from a Frame.io folder.

        Instead of uploading a raw video, the user provides a Frame.io folder
        URL containing long-form source footage and a reference short. The
        pipeline syncs the folder, visually matches the reference to find which
        source video it was cut from, and auto-sets source_path/start/end.
        """
        if not frameio_url.strip():
            raise ValueError("Frame.io folder URL is required for folder mode")
        if not frameio_url.strip().lower().startswith(("http://", "https://")):
            raise ValueError("Frame.io URL must be an http(s) URL")
        has_reference_file = reference_upload is not None and bool(reference_upload.filename)
        if not has_reference_file and not reference_url.strip():
            raise ValueError("Folder mode requires a reference video file or link")
        if reference_url.strip() and not reference_url.strip().lower().startswith(
            ("http://", "https://")
        ):
            raise ValueError("Reference link must be an http(s) URL")

        job = self.store.create()
        try:
            if has_reference_file and reference_upload is not None:
                job.reference_path = self.store.save_upload(
                    reference_upload, job.id, self.settings.max_upload_mb, name="reference"
                )
            if music_upload is not None and music_upload.filename:
                job.music_path = self.store.save_upload(
                    music_upload, job.id, self.settings.max_upload_mb, name="music", allow_audio=True
                )
            if logo_upload is not None and logo_upload.filename:
                job.logo_path = self.store.save_upload(
                    logo_upload,
                    job.id,
                    self.settings.max_upload_mb,
                    name="logo",
                    allow_image=True,
                )
        except Exception:
            self.store.save(job)
            raise

        job.clip_mode = clip_mode
        job.replicate = True
        job.frameio_source_url = frameio_url.strip()
        job.reference_url = reference_url.strip()
        job.title_mode = title_mode
        job.manual_title = manual_title_text.strip()
        job.color_grade = color_grade
        job.status = JobStatus.INGESTED
        job.progress = 0.1
        job.message = "Folder source-finding job created"
        self.store.save(job)
        self.job_events[job.id] = Event()
        future = self.executor.submit(self._run, job.id)
        self.job_futures[job.id] = future
        return job

    def cancel(self, job_id: str) -> None:
        event = self.job_events.get(job_id)
        if event:
            event.set()

    def shutdown(self) -> None:
        # Mark BEFORE tripping the events so workers that bail out with
        # JobCancelled know this is a restart, not a user cancel.
        self._shutting_down = True
        # Stop the watchdog first so it doesn't race against the executor
        # teardown on the same job metadata.
        self._watchdog_stop.set()
        if self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)
        for event in self.job_events.values():
            event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)

    # Interrupted jobs younger than this are resubmitted at startup;
    # anything older is left alone (stale experiments, not active work).
    RESUME_MAX_AGE_SECONDS: float = 24 * 3600

    def _resume_interrupted_jobs(self) -> None:
        """Resubmit every recent non-terminal job found on disk at startup.

        A server restart (deploy, --reload, Ctrl+C) used to strand running
        jobs: shutdown() cancelled them and they surfaced as FAILED
        "Cancelled by user" even though nobody cancelled anything. Now the
        JobCancelled handler leaves restart-interrupted jobs non-terminal,
        and this sweep re-runs them from the top. The re-run is fast where
        it matters: reference analysis, Frame.io sync, the library index,
        and vision tags are all disk-cached, so a job that died at the
        B-roll stage gets back there in seconds."""
        terminal = {JobStatus.READY, JobStatus.FAILED}
        now = time.time()
        try:
            job_dirs = list(self.store.jobs_dir.iterdir())
        except OSError:
            return
        for job_dir in job_dirs:
            meta_path = job_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                if now - meta_path.stat().st_mtime > self.RESUME_MAX_AGE_SECONDS:
                    continue
                job = self.store.get(job_dir.name)
            except Exception:
                continue
            if job.status in terminal or job.id in self.job_futures:
                continue
            job.message = "Resumed after server restart"
            self.store.save(job)
            self.job_events[job.id] = Event()
            self.job_futures[job.id] = self.executor.submit(self._run, job.id)
            logger.info("Resumed interrupted job %s (was %s)", job.id, job.status.value)

    def _run(self, job_id: str) -> None:
        # Load the job + cancellation event INSIDE the try block. Any
        # exception here (missing job, evicted event, deserialization error)
        # used to be swallowed by the ThreadPoolExecutor and the job sat in
        # INGESTED forever with status='ingested' and no error to debug.
        # Now: we catch it, log it, and mark the job FAILED so the UI
        # surfaces a real reason instead of an infinite spinner.
        try:
            job = self.store.get(job_id)
            cancellation = self.job_events[job_id]
        except Exception as exc:
            logger.exception("Failed to load job %s for execution", job_id)
            try:
                job = self.store.get(job_id)
                job.status = JobStatus.FAILED
                job.progress = min(job.progress, 0.99)
                job.message = "Job failed"
                job.error = f"Could not start: {exc!r}"[-1000:]
                self.store.save(job)
            except Exception:
                logger.exception("Also failed to mark job %s FAILED", job_id)
            return
        # Bump meta.json mtime NOW so the watchdog can see the worker has
        # picked up the job, even if it's about to spend 10 minutes in
        # broll_recovery (which doesn't call _advance() until it returns).
        # Without this, a queued-but-just-picked-up job looks identical to a
        # dropped future until the first stage transition, and the watchdog
        # fires before the worker gets a chance to advance.
        try:
            self.store.save(job)
        except Exception:
            logger.exception("Could not bump mtime at start of job %s", job_id)
        # Heartbeat thread: keep meta.json's mtime fresh during long stages
        # by re-saving every WORKER_HEARTBEAT_SECONDS. The watchdog uses
        # this signal to know the worker is still alive â€” without it, jobs
        # in broll_recovery (5-10 min) get killed mid-stage because their
        # meta.json looks stale. Cheap: just a disk stat + write of ~1KB.
        stop_heartbeat = Event()
        def _heartbeat() -> None:
            while not stop_heartbeat.wait(self.WORKER_HEARTBEAT_SECONDS):
                try:
                    current = self.store.get(job_id)
                    self.store.save(current)
                except Exception:
                    logger.exception("Heartbeat failed for job %s", job_id)
        heartbeat_thread = threading.Thread(
            target=_heartbeat, daemon=True, name=f"job-heartbeat-{job_id[:8]}",
        )
        heartbeat_thread.start()
        try:
            # Folder jobs auto-discover source_path later; all other jobs
            # require a pre-uploaded raw video. Only fail when BOTH are
            # missing â€” a normal upload job legitimately has an empty
            # frameio_source_url but a set source_path.
            if not job.frameio_source_url and job.source_path is None:
                raise ValueError(
                    f"Job {job.id} has no source video uploaded and is not a "
                    "Frame.io folder job â€” cannot run."
                )
            work_dir = self.store.job_dir(job.id)
            job.stage_started_at = time.time()

            if job.replicate and not job.reference_path and job.reference_url:
                self._advance(job, cancellation, JobStatus.INGESTED, 0.12, "Downloading reference short")
                job.reference_path = download_reference(job.reference_url, work_dir, self.settings)
                self.store.save(job)

            # B-roll source gating (job.broll_source): only the cheap mode
            # booleans are resolved here. The actual Frame.io sync (folder 1
            # + optional folder 2) is DEFERRED to the B-roll stage further
            # down so the visible pipeline keeps its historical order:
            # upload -> analyzing -> transcribing -> titling -> b-roll ->
            # rendering. (Syncing at INGESTED made "Syncing Frame.io
            # B-roll library" appear as step 2, before analysis.)
            # - "frameio": library rung reads the synced Frame.io share
            #   mirror(s) ONLY; YouTube rung disabled.
            # - "youtube": library rung disabled; YouTube rung on.
            # - "both": Frame.io mirror(s) + YouTube rung on. Never fall
            #   back to the machine-local B-roll folder.
            broll_source_mode = (getattr(job, "broll_source", "both") or "both").lower()
            broll_use_library = broll_source_mode in ("frameio", "both")
            broll_use_youtube = broll_source_mode in ("youtube", "both")
            # None = no library rung; a list = synced Frame.io mirror dirs
            # (folder 1 [+ folder 2]) merged into one index.
            broll_library_dir: Path | list[Path] | None = None
            youtube_errors: list[str] = []
            # Fail fast on impossible configuration (before any heavy work).
            if (
                job.replicate
                and broll_source_mode == "frameio"
                and not self.settings.broll_frameio_share_url.strip()
            ):
                raise ValueError(
                    "Frame.io B-roll source selected but no share URL is configured "
                    "(set BROLL_FRAMEIO_SHARE_URL)"
                )

            if job.replicate and job.reference_path:
                self._advance(job, cancellation, JobStatus.ANALYZING, 0.15, "Analyzing reference short")
                job.reference = analyze_reference(job.reference_path, work_dir, self.settings)
                self.store.save(job)
                if self.settings.broll_learning_enabled and job.reference:
                    # update_broll_profile_atomic holds the process-wide
                    # profile lock across load -> modify -> save, so two
                    # concurrent replicate-mode jobs don't race-pick the
                    # same on-disk state and silently overwrite each other.
                    update_broll_profile_atomic(self.settings, job.reference)

            # â”€â”€ Auto source-finding from Frame.io folder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if job.frameio_source_url and job.reference_path:
                self._advance(
                    job, cancellation, JobStatus.ANALYZING, 0.18,
                    "Finding source video in Frame.io folder",
                )
                find_deadline = time.monotonic() + 150.0
                result = find_raw_source(
                    job.reference_path,
                    job.frameio_source_url,
                    work_dir,
                    self.settings,
                    deadline=find_deadline,
                )
                if result is None:
                    raise ValueError(
                        "Could not find the source video in the Frame.io folder. "
                        "Make sure the folder contains the raw/long-form video that "
                        "the reference short was edited from."
                    )
                source_path, raw_start, raw_end, match_score = result
                job.raw_source_path = source_path
                job.raw_source_match_score = match_score
                job.source_path = source_path
                job.start = raw_start
                job.end = raw_end
                job.clip_mode = ClipMode.MANUAL  # timestamps already set
                job.message = (
                    f"Found source: {source_path.name} @ [{raw_start:.1f}sâ€“{raw_end:.1f}s] "
                    f"(score={match_score:.2f})"
                )
                self.store.save(job)

            if job.clip_mode == ClipMode.AUTO:
                self._select_clip(job, cancellation, work_dir)
            else:
                self._advance(job, cancellation, JobStatus.TRANSCRIBING, 0.35, "Transcribing audio")
                job.transcript = transcribe(
                    job.source_path, job.start, job.end, work_dir, self.settings
                )
                if not job.transcript.words:
                    job.message = "No speech detected - rendering without captions"
                    self.store.save(job)

            # Learned local B-roll is disabled. B-roll sourcing is limited to
            # Frame.io shares and YouTube.

            self._advance(job, cancellation, JobStatus.TITLING, 0.6, "Writing title")
            job.title = self._make_title(job)

            # Determine variation count: bulk mode = 1 video, single mode = N
            # (default 4). All variations share the same music, transcript,
            # color_grade, and B-roll placement timeline. They differ in title
            # text and B-roll source clips.
            #
            # broll_pack forces variation_count = 1: in replicate mode every
            # variation shares one title, so the only thing that would differ
            # across renders is the B-roll source clips â€” and when broll_pack
            # is on those clips are exported separately rather than inserted,
            # so multiple renders would be byte-identical files. One render is
            # the correct output.
            if job.broll_pack:
                variation_count = 1
            else:
                variation_count = 1 if job.bulk else max(1, self.settings.variation_count)

            # â”€â”€ B-roll library preparation (deferred Frame.io sync) â”€â”€â”€â”€â”€â”€â”€
            # Runs AFTER titling so the stage order the user sees stays
            # upload -> analyzing -> transcribing -> titling -> b-roll.
            # Under BROLL_RECOVERY so the watchdog grants the long-stage
            # budget (a first-time share download can take many minutes).
            if job.replicate and job.reference and broll_use_library:
                frameio_dirs: list[Path] = []
                share_url = self.settings.broll_frameio_share_url.strip()
                if share_url:
                    try:
                        self._advance(
                            job, cancellation, JobStatus.BROLL_RECOVERY, 0.65,
                            "Preparing B-roll library (Frame.io folder 1)",
                        )
                        frameio_dirs.append(ensure_frameio_library(share_url, self.settings))
                    except JobCancelled:
                        raise
                    except Exception as exc:
                        if broll_source_mode == "frameio":
                            raise ValueError(
                                f"Frame.io B-roll source is not working: {exc}"
                            ) from exc
                        # both-mode: keep YouTube as the only remaining
                        # external source. Never fall back to local files.
                        broll_use_library = False
                        job.warning = f"Frame.io B-roll library unavailable ({exc}); used YouTube only."
                        logger.warning("Frame.io B-roll sync failed; continuing with YouTube only", exc_info=True)
                # Optional secondary share (BROLL_FRAMEIO_SHARE_URL_2),
                # toggled per-job via the "Also search folder 2" checkbox.
                # Its mirror is MERGED into the same searchable index as
                # folder 1 â€” the matcher sees one combined library.
                use_secondary = bool(getattr(job, "use_broll_frameio_2", False))
                secondary_url = self.settings.broll_frameio_share_url_2.strip()
                if use_secondary and secondary_url:
                    try:
                        self._advance(
                            job, cancellation, JobStatus.BROLL_RECOVERY, 0.66,
                            "Preparing B-roll library (Frame.io folder 2)",
                        )
                        frameio_dirs.append(ensure_frameio_library(secondary_url, self.settings))
                    except JobCancelled:
                        raise
                    except Exception as exc:
                        if broll_source_mode == "frameio" and not frameio_dirs:
                            raise ValueError(
                                f"Frame.io B-roll source 2 is not working: {exc}"
                            ) from exc
                        job.warning = (
                            (job.warning + " " if job.warning else "")
                            + f"Frame.io folder 2 unavailable ({exc}); searched folder 1 only."
                        )
                        logger.warning(
                            "Frame.io secondary sync failed; continuing without folder 2",
                            exc_info=True,
                        )
                if frameio_dirs:
                    broll_library_dir = frameio_dirs
                else:
                    broll_use_library = False

                # Hook replication: the reference's first 0.5-3.5s of
                # music-over-broll is captured in job.reference.hook_span +
                # hook_tags. Prepend a matching library clip so the rendered
                # short opens with the same kind of hook. Failure here is
                # non-fatal â€” the regular broll_recovery ladder still
                # produces the rest of the cuts. Skipped in youtube-only
                # mode (the hook is a library feature).
                if broll_use_library and job.reference.hook_span and job.reference.hook_tags:
                    hook_start, hook_end = job.reference.hook_span
                    hook_dur = max(0.0, hook_end - hook_start)
                    if hook_dur > 0:
                        try:
                            hook_clip = fetch_hook_broll(
                                job.reference.hook_tags,
                                hook_dur,
                                work_dir / "hook",
                                self.settings,
                                library_dir=broll_library_dir,
                            )
                            if hook_clip is not None:
                                job.hook_clip_path = hook_clip
                                job.hook_span = (0.0, hook_dur)
                        except Exception:
                            logger.exception("Hook B-roll fetch failed")

            broll_cut_lists: list[list] = [job.broll_cuts]
            if job.replicate and job.reference and job.reference.broll_spans:
                span_count = len(job.reference.broll_spans)
                assert job.reference_path is not None
                # Compute reference_house_style ONCE per job (before the
                # first B-roll rung) so per-span scoring can back-fill
                # empty span vibe fields from a stable per-job aggregate
                # (SPEC Â§8). This is the SAME aggregate for the pack path
                # AND the variations path, so they're consistent.
                reference_house = build_reference_house_style(job.reference)
                # ContinuityLedger per job (SPEC Â§9): every fetch_* call
                # below shares this instance so consecutive span picks pay
                # the diversity tax. The ledger's internal history is
                # reset at the start of each variation inside
                # fetch_broll_cut_variations.
                continuity_ledger = ContinuityLedger(max_history=2)
                if job.broll_pack:
                    # Pack mode: gather 1-2 trimmed clips per span into
                    # job.broll_pack_items for download; do NOT insert any
                    # cuts into the main render. Stage message mirrors the
                    # span count so the UI shows progress against the
                    # ladder the user is actually waiting on.
                    self._advance(
                        job,
                        cancellation,
                        JobStatus.BROLL_RECOVERY,
                        0.72,
                        f"Finding B-roll pack for {span_count} span"
                        f"{'s' if span_count != 1 else ''}",
                    )
                    job.broll_pack_items = gather_broll_pack(
                        job.reference,
                        job.reference_path,
                        job.end - job.start,
                        work_dir / "broll",
                        self.settings,
                        per_span=2,
                        intelligent=getattr(job, "use_intelligent_selector", True),
                        reference_house=reference_house,
                        ledger=continuity_ledger,
                        library_dir=broll_library_dir,
                        use_library=broll_use_library,
                        use_youtube=broll_use_youtube,
                        youtube_errors=youtube_errors,
                    )
                    self.store.save(job)
                elif variation_count > 1:
                    self._advance(
                        job,
                        cancellation,
                        JobStatus.BROLL_RECOVERY,
                        0.72,
                        f"Recovering B-roll for {span_count} span{'s' if span_count != 1 else ''}",
                    )
                    broll_cut_lists = fetch_broll_cut_variations(
                        job.reference,
                        job.reference_path,
                        job.end - job.start,
                        work_dir / "broll",
                        self.settings,
                        variations=variation_count,
                        diagnostics=job.broll_recovery,
                        intelligent=getattr(job, "use_intelligent_selector", True),
                        reference_house=reference_house,
                        ledger=continuity_ledger,
                        library_dir=broll_library_dir,
                        use_library=broll_use_library,
                        use_youtube=broll_use_youtube,
                        youtube_errors=youtube_errors,
                    )
                    job.broll_cuts = broll_cut_lists[0]
                    self.store.save(job)
                else:
                    self._advance(
                        job,
                        cancellation,
                        JobStatus.BROLL_RECOVERY,
                        0.72,
                        f"Recovering B-roll for {span_count} span{'s' if span_count != 1 else ''}",
                    )
                    job.broll_cuts = fetch_broll_cuts(
                        job.reference,
                        job.reference_path,
                        job.end - job.start,
                        work_dir / "broll",
                        self.settings,
                        diagnostics=job.broll_recovery,
                        intelligent=getattr(job, "use_intelligent_selector", True),
                        reference_house=reference_house,
                        ledger=continuity_ledger,
                        library_dir=broll_library_dir,
                        use_library=broll_use_library,
                        use_youtube=broll_use_youtube,
                        youtube_errors=youtube_errors,
                    )
                    broll_cut_lists[0] = job.broll_cuts
                    self.store.save(job)

            # Surface hard YouTube failures collected during the B-roll
            # ladder (bot-check, cookie problems, rate limits, timeouts).
            # In youtube-only mode the user picked YouTube as THE source, so
            # a broken YouTube must fail the job loudly rather than shipping
            # a render quietly made of reference crops. In both-mode the
            # Frame.io/library rung carried the job, so a warning suffices.
            if youtube_errors:
                unique_errors = list(dict.fromkeys(youtube_errors))
                if broll_source_mode == "youtube":
                    raise ValueError(
                        "YouTube B-roll source is not working: "
                        f"{unique_errors[0]}. Upload fresh cookies via the "
                        "cookie box or switch the B-roll source."
                    )
                job.warning = (
                    f"YouTube B-roll source had problems ({unique_errors[0]}) â€” "
                    "B-roll came from the library/Frame.io rung."
                )
                self.store.save(job)

            # Prepend the hook B-roll (if any) as the first cut in every
            # variation list. The reference's first 0.5-3.5s of music-over-
            # broll is the "hook" that grabs attention; the output should
            # open with a matching clip from the local library so the
            # rendered short has the same kind of lead-in.
            if job.hook_clip_path is not None and job.hook_span is not None:
                from app.models import BrollCut as _HookCut
                _hook = _HookCut(
                    start=0.0,
                    end=max(0.0, job.hook_span[1] - job.hook_span[0]),
                    clip_path=job.hook_clip_path,
                    query="hook",
                )
                broll_cut_lists = [
                    [_hook, *cuts] for cuts in broll_cut_lists
                ]

            # Pad broll_cut_lists to variation_count if broll didn't supply
            # enough (e.g. no reference or spans = 0). Each slot gets a
            # possibly-empty list so the render loop below still produces
            # variation_count output files.
            while len(broll_cut_lists) < variation_count:
                broll_cut_lists.append(broll_cut_lists[0])

            music_path, music_volume_db, music_loop = _music_render_options(job)
            # Replicate mode: all variations share ONE title so the user
            # sees a consistent A/B/C/D comparison of the same edit. Bulk mode
            # already forces variation_count=1 above so the multiply is a no-op.
            base_title = job.title or fallback_title(job.transcript.text)
            titles: list[Title] = [base_title] * variation_count

            for v_idx in range(variation_count):
                if v_idx > 0:
                    self._advance(
                        job, cancellation, JobStatus.RENDERING,
                        0.85 + 0.15 * v_idx / max(1, variation_count - 1),
                        f"Rendering variation {v_idx + 1} of {variation_count}",
                    )
                elif variation_count > 1:
                    self._advance(job, cancellation, JobStatus.RENDERING, 0.85, "Rendering variations")
                else:
                    self._advance(job, cancellation, JobStatus.RENDERING, 0.85, "Rendering 9:16 video")

                output_path = work_dir / ("render.mp4" if v_idx == 0 else f"render_v{v_idx}.mp4")
                render(
                    source_path=job.source_path,
                    output_path=output_path,
                    start=job.start,
                    end=job.end,
                    transcript=job.transcript,
                    title=titles[v_idx],
                    color_grade=job.color_grade,
                    settings=self.settings,
                    broll_cuts=broll_cut_lists[v_idx],
                    music_path=music_path,
                    music_volume_db=music_volume_db,
                    music_loop=music_loop,
                    add_caption=getattr(job, "add_caption", True),
                )

                from app.models import Variation
                job.variations.append(Variation(
                    index=v_idx,
                    title=titles[v_idx],
                    broll_cuts=broll_cut_lists[v_idx],
                    output_path=output_path,
                    label=f"Variation {v_idx + 1} of {variation_count}" if variation_count > 1 else "",
                ))

            # Backward compat: primary render is always index 0.
            job.output_path = job.variations[0].output_path
            job.status = JobStatus.READY
            job.progress = 1.0
            # Pack mode succeeds when both the render AND the pack are
            # produced; surface the pack count so the UI's "Download" panel
            # has a meaningful "X B-roll clips" tally without an extra fetch.
            if job.broll_pack:
                clip_count = len(job.broll_pack_items)
                job.message = (
                    f"Render complete (+ {clip_count} B-roll clip"
                    f"{'s' if clip_count != 1 else ''})"
                )
            else:
                job.message = (
                    f"Render complete ({variation_count} variation"
                    f"{'s' if variation_count != 1 else ''})"
                )
            self.store.save(job)
        except JobCancelled:
            if self._shutting_down:
                # Server restart, not a user action: keep the job
                # non-terminal so _resume_interrupted_jobs re-runs it on
                # the next startup. Cached stages make the re-run cheap.
                job.message = "Interrupted by server restart â€” resumes automatically"
                self.store.save(job)
            else:
                job.status = JobStatus.FAILED
                job.message = "Cancelled"
                job.error = "Cancelled by user"
                self.store.save(job)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            job.status = JobStatus.FAILED
            job.progress = min(job.progress, 0.99)
            job.message = "Job failed"
            job.error = str(exc)[-1000:]
            self.store.save(job)
        finally:
            # Stop the heartbeat thread first so it doesn't fire after the
            # worker has already saved its terminal state and torn down
            # job_events (it would log spurious "Heartbeat failed" exceptions
            # when job_events.pop() races with store.get()).
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=self.WORKER_HEARTBEAT_SECONDS + 1)
            # Pop the cancellation event once the worker is done so the dict
            # doesn't grow unbounded over the server's lifetime. `cancel()`
            # uses `.get()` so it tolerates the event already being gone.
            # The event is set by the time we reach here (normal completion
            # has no need to check it, JobCancelled has already tripped, and
            # exceptions don't consume it).
            self.job_events.pop(job_id, None)
            # Drop the Future reference so the watchdog treats this slot as
            # "unknown" rather than "still alive". Once the worker is done
            # (success, failure, or cancel) the job is either in a terminal
            # status (READY/FAILED) or the watchdog's own FAILED write will
            # cover it.
            self.job_futures.pop(job_id, None)

    # â”€â”€ ETA estimation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Empirical stage durations (seconds) used as priors when no historical
    # stage_timings are available.  These are tuned to a typical 20s reference
    # with 8 B-roll spans on a mid-range GPU.  The _advance helper refines
    # them with the actual elapsed time of each completed stage so later jobs
    # get progressively more accurate ETA figures.
    _DEFAULT_STAGE_SECONDS: dict[str, float] = {
        "ingested": 1.0,
        "analyzing": 25.0,
        "transcribing": 8.0,
        "selecting": 2.0,
        "titling": 2.0,
        "broll_recovery": 180.0,  # scaled by span count below
        "rendering": 35.0,
    }
    # Ordered pipeline for ETA forecasting (last â†’ first = remaining stages).
    _STAGE_ORDER: list[str] = [
        "ingested", "analyzing", "transcribing", "selecting",
        "titling", "broll_recovery", "rendering",
    ]

    def _estimate_eta(self, job: Job) -> float:
        """Estimate total wall-clock seconds to finish this job.

        Uses a two-signal approach:
          1. If we've already completed at least one stage, use its *actual*
             elapsed time (stage_timings) as the anchor â€” better than the
             static prior.
          2. For stages not yet reached, fall back to the static prior,
             scaled by clip_duration (longer clips need proportionally more
             rendering time) and variation_count (more variations = more
             render passes).

        Returns estimated *remaining* seconds from NOW, NOT total seconds.
        """
        clip_factor = max(0.5, min(2.0, (job.end - job.start) / 20.0))
        var_factor = max(1.0, float(job.variation_count or 1))

        # Identify the current stage in the ordered pipeline.
        current = job.status.value
        try:
            current_idx = self._STAGE_ORDER.index(current)
        except ValueError:
            return 0.0

        # Per-job active stage list â€” manual-mode jobs skip the SELECTING
        # stage entirely (they go straight from TRANSCRIBING to TITLING), and
        # the "selecting" prior was inflating manual-mode ETA forecasts by ~2s
        # for no reason. Future per-job exclusions should land here.
        active_stages = list(self._STAGE_ORDER)
        if getattr(job, "clip_mode", None) and job.clip_mode.value == "manual":
            active_stages = [s for s in active_stages if s != "selecting"]

        remaining = 0.0
        # Sum remaining stages AFTER the current one.
        for stage_name in active_stages[active_stages.index(current) + 1:]:
            prior = self._DEFAULT_STAGE_SECONDS.get(stage_name, 5.0)
            # Apply scaling factors to the static prior.
            scaled = prior * clip_factor
            if stage_name == "rendering":
                scaled *= var_factor  # rendering scales with variation count
            elif stage_name == "broll_recovery":
                # B-roll recovery prior is per-span; scale by the number of
                # spans detected in the reference (each span needs ~180s of
                # ytsearch + ffmpeg scoring). Cap at the REAL worst-case ceiling
                # derived from the configured recovery budget (was a hardcoded
                # 3600.0 â€” stale as soon as BROLL_RECOVERY_BUDGET_SECONDS is
                # changed, e.g. to 180 for a fast render, which used to forecast
                # a wildly wrong ~17 minutes even though the pipeline itself was
                # bounded to a few minutes).
                span_count = _job_broll_span_count(job)
                scaled = prior * max(1, span_count) * clip_factor
                scaled = min(scaled, self._broll_recovery_ceiling_seconds())
            # Replace static prior with historical average when available.
            historical = job.stage_timings.get(stage_name)
            if historical and historical > 0:
                # Blend: 60% historical, 40% scaled prior (smooth convergence).
                scaled = historical * 0.6 + scaled * 0.4
            remaining += scaled

        # Remaining budget for the CURRENT stage = prior minus elapsed so far.
        if job.stage_started_at is not None:
            elapsed = time.time() - job.stage_started_at
            stage_prior = job.stage_timings.get(current) or self._DEFAULT_STAGE_SECONDS.get(current, 10.0)
            stage_prior *= clip_factor
            if current == "rendering":
                stage_prior *= var_factor
            elif current == "broll_recovery":
                span_count = _job_broll_span_count(job)
                stage_prior = stage_prior * max(1, span_count)
                stage_prior = min(stage_prior, self._broll_recovery_ceiling_seconds())
            # Use the historical data as a better prior if available.
            historical = job.stage_timings.get(current)
            if historical and historical > 0:
                stage_prior = historical * 0.6 + stage_prior * 0.4
            remaining += max(0.0, stage_prior - elapsed)
        else:
            stage_prior = self._DEFAULT_STAGE_SECONDS.get(current, 10.0) * clip_factor
            if current == "broll_recovery":
                span_count = _job_broll_span_count(job)
                stage_prior = stage_prior * max(1, span_count)
                stage_prior = min(stage_prior, self._broll_recovery_ceiling_seconds())
            remaining += stage_prior

        return max(1.0, remaining)

    def _broll_recovery_ceiling_seconds(self) -> float:
        """Flat wall-clock ceiling for the broll_recovery stage ETA forecast.

        app/broll.py has no configurable global budget: each span resolves via
        a guaranteed Local -> YouTube -> reference-crop ladder bounded by flat
        per-call timeouts, so there's no per-job budget setting left to mirror
        here. This is purely a forecasting cap on the ETA display, not an
        enforced deadline.

        With the 4-way parallel span sourcing in app.broll, wall-clock is
        closer to (longest span Ã— ceil(spans / 4)) than to (per-span Ã— spans);
        10 min is a still-conservative upper bound that doesn't overshoot 8-span
        jobs AND doesn't pretend they'll hit the old 15 min sequential ceiling.
        """
        return 600.0

    def _watchdog_loop(self) -> None:
        """Background sweeper that fails jobs whose meta.json hasn't been
        updated in STUCK_JOB_TIMEOUT_SECONDS.

        Without this, a job whose worker future silently dropped (e.g. the
        executor is in a bad state, the worker thread crashed) sits in
        INGESTED forever and the user gets an infinite spinner with no
        error to debug. The watchdog converts that into a visible FAILED
        status with a clear message.

        Runs every WATCHDOG_POLL_INTERVAL_SECONDS on its own daemon thread;
        joins cleanly on shutdown via _watchdog_stop.
        """
        while not self._watchdog_stop.wait(self.WATCHDOG_POLL_INTERVAL_SECONDS):
            try:
                self._fail_stuck_jobs()
            except Exception:
                logger.exception("Watchdog sweep failed")

    def _fail_stuck_jobs(self) -> None:
        """One watchdog sweep. Iterate every known job, find any non-terminal
        one whose meta.json hasn't been touched in STUCK_JOB_TIMEOUT_SECONDS,
        and mark it FAILED with a clear cause.

        Important: a stale meta.json does NOT always mean a dropped worker.
        Two legitimate scenarios produce stale mtimes with an alive worker:
          1. The job is queued behind another job in the executor's task
             queue (worker_count < concurrent submissions). The future is
             PENDING â€” `future.running()` is False and `future.done()` is
             False.
          2. The job is mid-stage in a long-running operation (broll_recovery
             can run 5-10 min). The heartbeat thread in _run keeps mtime
             fresh, but only every WORKER_HEARTBEAT_SECONDS.
        In both cases we skip â€” killing them would just generate false
        positives for jobs that would have completed successfully.
        """
        now = time.time()
        cutoff = now - self.STUCK_JOB_TIMEOUT_SECONDS
        terminal = {JobStatus.READY, JobStatus.FAILED}
        for job_dir in self.store.jobs_dir.iterdir():
            meta_path = job_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                mtime = meta_path.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                continue  # recently touched â€” probably still progressing
            if job_dir.name in self._watchdog_unrecoverable:
                continue  # already logged once; won't fix itself, don't retry forever
            try:
                import json as _json
                data = _json.loads(meta_path.read_text(encoding="utf-8"))
                status_str = data.get("status")
                if status_str in {s.value for s in terminal}:
                    continue
                job_id = data.get("id") or job_dir.name
                # Cross-check the future state. If the worker is still
                # alive (running or queued in the executor), the stale mtime
                # is either queueing latency or a missing heartbeat tick â€”
                # NOT a dropped future. Bail out and let the next sweep
                # decide once the heartbeat has caught up.
                future = self.job_futures.get(job_id)
                if future is not None:
                    if future.running() or not future.done():
                        # Worker is alive (executing or queued). Skip.
                        continue
                    # Future is done with no exception â†’ worker completed
                    # cleanly but somehow the job isn't in a terminal state.
                    # That's a real bug (missing READY/FAILED save) and
                    # deserves the watchdog kill.
                # Per-stage timeout override: broll_recovery in particular
                # can legitimately run longer than the global cap when the
                # worker is stuck inside fetch_broll_cuts behind the GIL,
                # because the heartbeat thread can't fire to refresh mtime.
                stage_timeout = self._STUCK_TIMEOUT_BY_STATUS.get(
                    status_str or "", self.STUCK_JOB_TIMEOUT_SECONDS,
                )
                if mtime >= now - stage_timeout:
                    # Still within the per-stage window â€” give it more time.
                    continue
                # Trip the cancellation event so a still-running worker
                # (race between the future.running() check above and now)
                # bails out at its next _advance() instead of overwriting
                # our FAILED save with a stale non-terminal status on its
                # way out of fetch_broll_cuts.
                event = self.job_events.get(job_id)
                if event is not None:
                    event.set()
                # Re-load via the store (so the type-discriminated enum + Pydantic
                # validation run again), then mark FAILED and persist.
                job = self.store.get(job_id)
                job.status = JobStatus.FAILED
                job.progress = min(job.progress, 0.99)
                job.message = "Job failed"
                job.error = (
                    f"Job did not advance within "
                    f"{stage_timeout:.0f}s in stage '{status_str}' â€” the "
                    "worker thread likely dropped the task. Restart the "
                    "server and resubmit."
                )[-1000:]
                self.store.save(job)
                logger.error(
                    "Watchdog failed stuck job %s (status was %s, stage timeout %.0fs)",
                    job.id, status_str, stage_timeout,
                )
            except Exception:
                # Log once and never retry this directory again - a corrupt
                # meta.json or a job_id missing from the store won't fix
                # itself, and without this guard it would log a full
                # traceback every WATCHDOG_POLL_INTERVAL_SECONDS forever.
                self._watchdog_unrecoverable.add(job_dir.name)
                logger.exception(
                    "Watchdog could not process %s - giving up on it (won't retry)",
                    job_dir.name,
                )

    def _advance(
        self, job: Job, cancellation: Event, status: JobStatus, progress: float, message: str
    ) -> None:
        if cancellation.is_set():
            raise JobCancelled()
        # Record the wall-clock time the previous stage started so we can
        # measure its actual elapsed duration for future ETA refinement.
        if job.stage_started_at is not None:
            elapsed = time.time() - job.stage_started_at
            prev_stage = job.status.value
            # Running average: blend old timing (if any) with this run's elapsed.
            old = job.stage_timings.get(prev_stage)
            if old and old > 0:
                job.stage_timings[prev_stage] = old * 0.7 + elapsed * 0.3
            else:
                job.stage_timings[prev_stage] = elapsed
        job.status = status
        job.progress = progress
        job.message = message
        job.stage_started_at = time.time()
        job.eta_seconds = self._estimate_eta(job)
        self.store.save(job)

    def _make_title(self, job: Job) -> Title:
        transcript_text = job.transcript.text
        if job.title_mode == TitleMode.MANUAL:
            return manual_title(job.manual_title)
        if job.replicate and job.reference and job.reference.title_text:
            return replicate_title(job.reference.title_text, transcript_text, self.settings)
        if transcript_text.strip():
            return generate_title(transcript_text, self.settings)
        return fallback_title(transcript_text)

    def _make_variation_title(self, job: Job, variation_index: int) -> Title:
        if job.title_mode == TitleMode.MANUAL:
            return job.title or manual_title(job.manual_title)
        transcript_text = job.transcript.text
        if not transcript_text.strip():
            return job.title or fallback_title(transcript_text)
        primary = job.title
        candidate = primary or fallback_title(transcript_text)
        for _ in range(2):
            if job.replicate and job.reference and job.reference.title_text:
                candidate = replicate_title(job.reference.title_text, transcript_text, self.settings)
            else:
                candidate = generate_title(transcript_text, self.settings)
            if primary is None or (candidate.line1, candidate.line2) != (primary.line1, primary.line2):
                return candidate
        return candidate

    def _select_clip(self, job: Job, cancellation: Event, work_dir) -> None:
        assert job.source_path is not None
        duration = probe_duration(job.source_path, self.settings)
        analysis_end = min(duration, self.settings.auto_clip_scan_seconds)

        self._advance(
            job,
            cancellation,
            JobStatus.TRANSCRIBING,
            0.25,
            "Transcribing video for AI clip selection",
        )
        full_transcript = transcribe(job.source_path, 0.0, analysis_end, work_dir, self.settings)

        self._advance(
            job,
            cancellation,
            JobStatus.SELECTING,
            0.5,
            "Selecting viral clip",
        )
        selection = select_viral_clip(full_transcript, analysis_end, self.settings)
        job.start = selection.start
        job.end = selection.end
        job.clip_reason = selection.reason
        job.transcript = slice_transcript(full_transcript, job.start, job.end)
        job.message = f"Selected {job.start:.1f}s-{job.end:.1f}s"
        if selection.reason:
            job.message += f": {selection.reason}"
        self.store.save(job)


def _music_render_options(job: Job):
    """Resolve (music_path, target_volume_db, loop) for rendering.

    The reference's measured mean music loudness (music_volume_db) is the
    target loudness the rendered track should land at â€” this is independent
    of WHICH music file is used. When the user supplies their own music_path
    we still want the rendered music to sit at the reference's loudness, so
    we pass the reference dB through as the target and let
    _music_volume_coefficient() measure the new file's source dB and compute
    the gain that lands at the target. Returns loop=False for an arbitrary
    user track (we don't know if it loops cleanly).

    Disabled for now: music production is out of scope while the B-roll
    pipeline is debugged, so renders always get no music track regardless
    of source. Remove this early return to re-enable.
    """
    return None, None, True
    reference_music = job.reference.music_path if job.reference else None
    reference_db = job.reference.music_volume_db if job.reference else None
    if job.music_path:
        return job.music_path, reference_db, False
    if reference_music:
        return reference_music, reference_db, False
    return None, None, True
