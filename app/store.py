import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import UploadFile

from app.models import Job


class JobStore:
    """Disk layout: data/jobs/{job_id}/{source.mp4, render.mp4, meta.json}.

    Jobs live in an in-memory dict and are restored from meta.json on startup;
    meta.json is written on every save so a crashed server leaves inspectable
    state behind.
    """

    def __init__(self, data_dir: Path):
        self.jobs_dir = data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        for meta_path in self.jobs_dir.glob("*/meta.json"):
            try:
                job = Job.model_validate_json(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            self._jobs[job.id] = job

    def create(self) -> Job:
        job = Job(id=uuid4().hex[:12])
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        self.save(job)
        return job

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id].model_copy(deep=True)

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)
        meta_path = self.job_dir(job.id) / "meta.json"
        # Write via a temp file + atomic replace so a server crash/kill
        # mid-write can never leave a truncated/corrupt meta.json behind
        # (a plain write_text() could, and the watchdog sweep treats a
        # corrupt file as a permanent, unrecoverable job every 15s).
        tmp_path = meta_path.with_name(meta_path.name + ".tmp")
        tmp_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(meta_path)

    VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def save_upload(
        self,
        upload: UploadFile,
        job_id: str,
        max_mb: int,
        name: str = "source",
        allow_audio: bool = False,
        allow_image: bool = False,
    ) -> Path:
        suffix = Path(upload.filename or f"{name}.mp4").suffix.lower() or ".mp4"
        allowed = self.VIDEO_SUFFIXES | (
            self.AUDIO_SUFFIXES if allow_audio else set()
        ) | (self.IMAGE_SUFFIXES if allow_image else set())
        if suffix not in allowed:
            raise ValueError(f"Unsupported file type: {suffix}")
        target = self.job_dir(job_id) / f"{name}{suffix}"
        max_bytes = max_mb * 1024 * 1024
        written = 0
        try:
            with target.open("wb") as output:
                while chunk := upload.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"Upload exceeds {max_mb} MB limit")
                    output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        if written == 0:
            target.unlink(missing_ok=True)
            raise ValueError("Uploaded file is empty")
        return target
