"""Bundled fonts for caption/title rendering.

assets/fonts ships TTF files (libass cannot load woff2):
  Inter-Regular.ttf, Inter-Bold.ttf -> family "Inter"
"""

from pathlib import Path

CAPTION_FONT = "Inter"
CAPTION_BOLD = -1  # ASS Bold on: Inter-Bold.ttf provides a real bold cut
TITLE_FONT = "Inter"
TITLE_BOLD = -1  # White title text uses the bundled Inter bold cut


def fonts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "fonts"
