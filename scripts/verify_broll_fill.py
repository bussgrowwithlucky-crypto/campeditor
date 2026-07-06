import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.models import BrollCut, ColorGrade, Title, Transcript
from app.rendering import VIDEO_TOP, render

settings = get_settings()
work = Path("data/_verify")
work.mkdir(parents=True, exist_ok=True)
source = Path("data/sample.mp4")


def make_clip(path: Path, w: int, h: int, color: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:d=5:r=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def band_top_strip_mean(video: Path):
    frame_path = work / "probe.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "2.0", "-i", str(video), "-frames:v", "1", str(frame_path)],
        check=True, capture_output=True,
    )
    img = cv2.imread(str(frame_path))
    strip = img[VIDEO_TOP + 2 : VIDEO_TOP + 30, :, :]  # top of the video band
    b, g, r = strip[:, :, 0].mean(), strip[:, :, 1].mean(), strip[:, :, 2].mean()
    return r, g, b


title = Title(line1="Test Title", line2="Second Line", highlight_words=["Test"])
for name, (w, h) in {"landscape": (1920, 1080), "portrait": (1080, 1920)}.items():
    clip = make_clip(work / f"broll_{name}.mp4", w, h, "red")
    out = work / f"out_{name}.mp4"
    render(
        source_path=source, output_path=out, start=0.0, end=5.0,
        transcript=Transcript(), title=title, color_grade=ColorGrade.NONE, settings=settings,
        broll_cuts=[BrollCut(start=1.0, end=3.0, clip_path=clip, query="test")],
    )
    r, g, b = band_top_strip_mean(out)
    filled = r > 120 and b < 90  # red content, not black
    print(f"{name:9s} band-top mean R={r:.0f} G={g:.0f} B={b:.0f} -> {'FILLED (no gap)' if filled else 'BLACK GAP!'}")
