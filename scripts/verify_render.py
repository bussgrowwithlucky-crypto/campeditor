"""Verification driver: render the speech sample with manual mode + cinematic grade,
extract a mid-frame for visual comparison against the reference.

Run from C:/campeditor: python scripts/verify_render.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

SAMPLE = ROOT / "data" / "speech_sample.mp4"
OUT_DIR = ROOT / "data" / "verify"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_manual() -> Path:
    if not SAMPLE.exists():
        raise SystemExit(f"missing {SAMPLE}")
    with TestClient(app) as client:
        with SAMPLE.open("rb") as f:
            resp = client.post(
                "/api/jobs/upload",
                files={"file": ("speech_sample.mp4", f, "video/mp4")},
                data={
                    "start": "0.5",
                    "end": "10.0",
                    "title_mode": "manual",
                    "manual_title": "He explains how Jeff Bezos works",
                    "color_grade": "none",
                },
            )
        resp.raise_for_status()
        job_id = resp.json()["id"]
        print(f"job {job_id} submitted; polling…", flush=True)

        deadline = time.time() + 180
        final = None
        while time.time() < deadline:
            final = client.get(f"/api/jobs/{job_id}").json()
            if final["status"] in {"ready", "failed"}:
                break
            time.sleep(2)
        assert final is not None
        if final["status"] != "ready":
            raise SystemExit(f"job ended {final['status']}: {final.get('error')}")

        out_path = OUT_DIR / "manual.mp4"
        with (OUT_DIR / "manual.mp4").open("wb") as out:
            r = client.get(f"/api/renders/{job_id}.mp4")
            r.raise_for_status()
            out.write(r.content)
        return out_path


def extract_frame(video: Path, frame: Path, at: float = 5.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{at:.2f}",
            "-i", str(video),
            "-frames:v", "1",
            str(frame),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def main() -> None:
    mp4 = render_manual()
    print(f"rendered: {mp4} ({mp4.stat().st_size} bytes)")
    frame = OUT_DIR / "manual_frame.png"
    extract_frame(mp4, frame)
    print(f"frame: {frame}")


if __name__ == "__main__":
    main()