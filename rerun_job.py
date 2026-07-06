"""Re-run the B-roll recovery for job 07f7cd2ebc22 with the updated pipeline.

Resets the job's broll_cuts / broll_recovery diagnostics, re-enters the
pipeline at the BROLL_RECOVERY stage, and waits for the render to finish.
"""
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.store import JobStore
from app.models import JobStatus
from app.jobs import Pipeline

JOB_ID = "07f7cd2ebc22"

def main() -> None:
    settings = get_settings()
    store = JobStore(settings.data_dir)

    # Load the existing job
    job = store.get(JOB_ID)

    # Reset the fields that the recovery / render stages will overwrite
    job.broll_cuts = []
    job.broll_recovery = []
    job.status = JobStatus.INGESTED
    job.progress = 0.0
    job.message = "Re-run scheduled"
    job.error = None
    job.output_path = None
    job.stage_started_at = None
    job.stage_timings = {}
    job.variations = []
    store.save(job)

    # Create a Pipeline and submit the job
    pipeline = Pipeline(settings, store)
    event = threading.Event()
    pipeline.job_events[JOB_ID] = event
    pipeline.executor.submit(pipeline._run, JOB_ID)

    print(f"Job {JOB_ID} submitted — watching progress…")
    last_status = ""
    while not event.is_set():
        job = store.get(JOB_ID)
        status_line = f"[{job.progress*100:5.1f}%] {job.status.value:20s}  {job.message}"
        if status_line != last_status:
            print(status_line)
            last_status = status_line
        if job.status in (JobStatus.READY, JobStatus.FAILED):
            break
        time.sleep(2)

    # Final status
    job = store.get(JOB_ID)
    print(f"\n=== DONE: {job.status.value} ===")
    print(f"message:  {job.message}")
    if job.error:
        print(f"error:    {job.error}")
    print(f"cuts:     {len(job.broll_cuts)}")
    print(f"diags:    {len(job.broll_recovery)}")
    if job.broll_recovery:
        providers = [d.provider for d in job.broll_recovery]
        selected = sum(1 for d in job.broll_recovery if d.selected)
        from collections import Counter
        print(f"providers: {dict(Counter(providers))}")
        print(f"selected:  {selected} / {len(job.broll_recovery)}")
        print("per-span:")
        for d in job.broll_recovery:
            flag = "✓" if d.selected else "✗"
            print(f"  {flag} [{d.start:.2f}-{d.end:.2f}] provider={d.provider or '(none)'} score={d.score:.3f} reason={d.reason[:80]}")

if __name__ == "__main__":
    main()
