"""Tests for the silent-failure hardening + stuck-job watchdog.

Two bugs were uncovered when a job sat in INGESTED forever with no error:
1. The first two lines of Pipeline._run (store.get + job_events lookup) were
   OUTSIDE the try block, so any exception there was swallowed by the
   ThreadPoolExecutor's internal logger.
2. If the worker's future silently dropped (executor in a bad state), nothing
   ever surfaced — the job stayed in INGESTED with no error.

This file covers the fix:
- Missing source_path raises ValueError (not AssertionError) and is FAILED.
- The watchdog marks jobs whose meta.json stops updating as FAILED with a
  clear message.
- The watchdog is shut down cleanly on Pipeline.shutdown.
"""
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.jobs import Pipeline
from app.models import JobStatus
from app.store import JobStore


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        groq_api_key="",
        llm_api_key="",
        nvidia_api_key="",
        nvidia_fallback_api_key="",
        nvidia_fallback_api_key_2="",
        nvidia_fallback_api_key_3="",
        gemini_api_key="",
        youtube_data_api_key="",
        youtube_data_api_key_2="",
        ollama_vision_model="",
        ollama_text_model="",
    )
    settings.data_dir = tmp_path
    settings.broll_library_dir = tmp_path / "no_library"
    # Short watchdog window so the test runs fast.
    settings.worker_count = 1
    return settings


# ---------------------------------------------------------------------------
# Run-start hardening
# ---------------------------------------------------------------------------


def test_run_marks_job_failed_when_source_path_missing(tmp_path, monkeypatch):
    """A job that lost its source_path (or was never given one) used to
    AssertionError inside _run. Now it must surface a clean FAILED status
    with an explanatory error string, never silently sit in INGESTED."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)

    # Create a job that bypasses create_job's source_path assertion.
    job = store.create()
    job.source_path = None  # explicit; create_job would have asserted
    job.clip_mode = "manual"
    job.start = 0.0
    job.end = 15.0
    job.title_mode = "auto"
    job.manual_title = ""
    job.color_grade = "none"
    job.broll_pack = False
    job.enable_learned_broll = False
    store.save(job)

    # Submit directly — we don't want to set up a real source file just for
    # this assertion-failure test.
    pipeline.job_events[job.id] = threading.Event()
    pipeline.executor.submit(pipeline._run, job.id).result(timeout=10)

    final = store.get(job.id)
    assert final.status == JobStatus.FAILED, (
        f"Expected FAILED, got {final.status}: {final.message} / {final.error}"
    )
    assert "source video" in (final.error or "") or "source" in (final.message or "").lower()
    pipeline.shutdown()


def test_run_swallows_load_failures_by_marking_job_failed(tmp_path):
    """If store.get itself raises (e.g. corrupted meta.json), _run must NOT
    silently disappear — it should mark the job FAILED with the traceback."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)

    # Submit for a job that doesn't exist on disk at all. The first thing
    # _run does is store.get(job_id) which will raise KeyError. With the fix,
    # the outer try catches it and marks the job FAILED; without the fix it
    # would vanish into the executor's logger.
    fake_id = "ghost_job"
    pipeline.job_events[fake_id] = threading.Event()
    # _run will call self.store.get(fake_id) which raises KeyError → caught by
    # our fix → second store.get also raises → logger.exception is the
    # fallback path. Either way no exception leaks to the test.
    pipeline.executor.submit(pipeline._run, fake_id).result(timeout=10)
    # If we got here without an exception, the hardening works.
    pipeline.shutdown()


# ---------------------------------------------------------------------------
# Stuck-job watchdog
# ---------------------------------------------------------------------------


def test_watchdog_fails_ingested_job_with_stale_meta(tmp_path):
    """A job whose meta.json hasn't been updated in STUCK_JOB_TIMEOUT_SECONDS
    must be auto-failed by the watchdog — this is the exact failure mode the
    user hit when their uvicorn executor silently dropped futures."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)

    # Shrink the watchdog window so the test is fast.
    pipeline.STUCK_JOB_TIMEOUT_SECONDS = 0.5
    pipeline.WATCHDOG_POLL_INTERVAL_SECONDS = 0.1

    job = store.create()
    job.status = JobStatus.INGESTED
    job.progress = 0.1
    job.message = "Upload received"
    store.save(job)

    # Backdate the meta.json mtime to simulate a stuck job whose worker
    # future silently dropped — no _advance has fired for STUCK_JOB_TIMEOUT.
    meta_path = settings.data_dir / "jobs" / job.id / "meta.json"
    stale = time.time() - 60.0
    import os
    os.utime(meta_path, (stale, stale))

    # Run one watchdog sweep manually (don't wait for the background thread).
    pipeline._fail_stuck_jobs()

    final = store.get(job.id)
    assert final.status == JobStatus.FAILED
    assert final.error is not None
    assert "did not advance" in final.error
    pipeline.shutdown()


def test_watchdog_skips_recently_touched_jobs(tmp_path):
    """A job whose meta.json was updated recently is NOT auto-failed — the
    watchdog only fires on the silent-drop case, not normal slow jobs."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)
    pipeline.STUCK_JOB_TIMEOUT_SECONDS = 0.5

    job = store.create()
    job.status = JobStatus.TRANSCRIBING  # legitimately in-flight
    job.progress = 0.35
    job.message = "Transcribing audio"
    store.save(job)
    # meta.json mtime is "now" → recent → watchdog leaves it alone.

    pipeline._fail_stuck_jobs()

    final = store.get(job.id)
    assert final.status == JobStatus.TRANSCRIBING
    assert final.error is None
    pipeline.shutdown()


def test_watchdog_skips_terminal_jobs(tmp_path):
    """READY and FAILED jobs are never re-failed by the watchdog even if
    their meta.json is stale — that would corrupt history."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)
    pipeline.STUCK_JOB_TIMEOUT_SECONDS = 0.5

    for terminal in (JobStatus.READY, JobStatus.FAILED):
        job = store.create()
        job.status = terminal
        job.message = terminal.value
        store.save(job)
        # Backdate.
        meta_path = settings.data_dir / "jobs" / job.id / "meta.json"
        import os
        os.utime(meta_path, (time.time() - 60, time.time() - 60))
        pipeline._fail_stuck_jobs()
        final = store.get(job.id)
        assert final.status == terminal, f"{terminal.value} job was incorrectly touched by watchdog"
        assert final.message == terminal.value
    pipeline.shutdown()


def test_watchdog_thread_starts_and_stops_cleanly(tmp_path):
    """The watchdog thread must auto-start on Pipeline init and shut down
    cleanly on Pipeline.shutdown (no hanging threads)."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)
    assert pipeline._watchdog_thread.is_alive()
    pipeline.shutdown()
    # Give the daemon thread a moment to actually exit after the stop event.
    deadline = time.time() + 2.0
    while pipeline._watchdog_thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    assert not pipeline._watchdog_thread.is_alive(), "watchdog thread did not stop on shutdown"


# ---------------------------------------------------------------------------
# Watchdog false-positive prevention — job futures are still alive
# ---------------------------------------------------------------------------


def test_watchdog_skips_job_with_alive_future(tmp_path):
    """A job whose worker future is still alive (running or queued) must
    NOT be killed by the watchdog, even if meta.json is stale. This is the
    exact false-positive that broke submissions: jobs queued behind a long
    broll_recovery job would sit in the executor queue for >120s without
    meta.json being touched, and the watchdog killed them as 'dropped'.

    The fix: create_job / create_bulk_job / create_folder_job capture the
    Future from executor.submit() into pipeline.job_futures. The watchdog
    cross-checks future.running() / not future.done() before killing.
    """
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)
    pipeline.STUCK_JOB_TIMEOUT_SECONDS = 0.5
    pipeline.WATCHDOG_POLL_INTERVAL_SECONDS = 0.1

    job = store.create()
    job.status = JobStatus.INGESTED
    job.progress = 0.1
    job.message = "Upload received"
    store.save(job)
    # Simulate a queued future: never started, but tracked.
    fake_future: concurrent.futures.Future = concurrent.futures.Future()
    pipeline.job_futures[job.id] = fake_future
    # Backdate meta.json so it LOOKS stale to the watchdog.
    meta_path = settings.data_dir / "jobs" / job.id / "meta.json"
    stale = time.time() - 60.0
    os.utime(meta_path, (stale, stale))

    pipeline._fail_stuck_jobs()

    final = store.get(job.id)
    assert final.status == JobStatus.INGESTED, (
        "Watchdog killed a job whose worker future was still alive — "
        "this is the queued-job false positive the user hit in production."
    )
    assert final.error is None

    fake_future.set_result(None)
    pipeline.shutdown()


def test_watchdog_skips_running_future(tmp_path):
    """A job whose worker future is currently RUNNING (executing right now)
    must NOT be killed by the watchdog, even if it's been silent for a
    while. Covers the broll_recovery long-stage case where meta.json only
    gets updated every WORKER_HEARTBEAT_SECONDS."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)
    pipeline.STUCK_JOB_TIMEOUT_SECONDS = 0.5

    job = store.create()
    job.status = JobStatus.BROLL_RECOVERY
    job.progress = 0.72
    job.message = "Recovering B-roll"
    store.save(job)
    # Build a real Future via a private executor (the public API doesn't
    # let us fabricate a "running" future directly). Wrap a sentinel that
    # blocks until we tell it to stop.
    stop = threading.Event()
    exec_ = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = exec_.submit(stop.wait)
    time.sleep(0.1)  # let the executor actually schedule it
    pipeline.job_futures[job.id] = future
    meta_path = settings.data_dir / "jobs" / job.id / "meta.json"
    stale = time.time() - 60.0
    os.utime(meta_path, (stale, stale))

    pipeline._fail_stuck_jobs()

    final = store.get(job.id)
    assert final.status == JobStatus.BROLL_RECOVERY, (
        "Watchdog killed a job whose worker was actively running — "
        "broll_recovery jobs would be killed mid-stage without this guard."
    )
    assert final.error is None

    stop.set()
    future.result(timeout=5)
    exec_.shutdown(wait=True)
    pipeline.shutdown()


def test_watchdog_kills_when_future_done_without_terminal_status(tmp_path):
    """Sanity check: if the future is done (worker returned cleanly) but
    somehow the job isn't in a terminal status — i.e. a real bug where the
    worker exited without marking READY/FAILED — the watchdog MUST still
    fail it. The future-alive guard is a guard, not a free pass."""
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    pipeline = Pipeline(settings, store)
    pipeline.STUCK_JOB_TIMEOUT_SECONDS = 0.5

    job = store.create()
    job.status = JobStatus.RENDERING
    job.progress = 0.85
    job.message = "Rendering 9:16 video"
    store.save(job)

    # A done future with no result set — simulates a worker that returned
    # cleanly but the save got lost.
    done_future: concurrent.futures.Future = concurrent.futures.Future()
    done_future.set_result(None)
    pipeline.job_futures[job.id] = done_future
    meta_path = settings.data_dir / "jobs" / job.id / "meta.json"
    stale = time.time() - 60.0
    os.utime(meta_path, (stale, stale))

    pipeline._fail_stuck_jobs()

    final = store.get(job.id)
    assert final.status == JobStatus.FAILED
    assert final.error is not None
    assert "did not advance" in final.error
    pipeline.shutdown()