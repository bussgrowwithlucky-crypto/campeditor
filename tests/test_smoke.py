"""End-to-end smoke test: upload a generated test clip, poll to READY, download render.

Requires FFmpeg on PATH. Makes real Groq/LLM calls if keys are set in .env;
without keys the pipeline still renders (no captions, fallback title).
"""

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SAMPLE = Path("data") / "sample.mp4"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg not on PATH"
)


@pytest.fixture(scope="module")
def sample_video() -> Path:
    if not SAMPLE.exists():
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=6:size=1280x720:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(SAMPLE),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    return SAMPLE


def test_upload_render_download(sample_video: Path) -> None:
    from app.main import app

    with TestClient(app) as client:
        with sample_video.open("rb") as f:
            response = client.post(
                "/api/jobs/upload",
                files={"file": ("sample.mp4", f, "video/mp4")},
                data={
                    "start": "1",
                    "end": "5",
                    "title_mode": "manual",
                    "manual_title": "He reveals why he works 12h+",
                    "color_grade": "cinematic",
                },
            )
        assert response.status_code == 200, response.text
        job_id = response.json()["id"]

        deadline = time.time() + 180
        status = ""
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            status = job["status"]
            if status in {"ready", "failed"}:
                break
            time.sleep(1)
        assert status == "ready", f"Job ended as {status}: {job.get('error')}"

        download = client.get(f"/api/renders/{job_id}.mp4")
        assert download.status_code == 200
        assert len(download.content) > 1024
