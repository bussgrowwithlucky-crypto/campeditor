"""Single-pass FFmpeg render: trim + face-centered 1:1 crop + scale + pad + color grade + ASS burn-in.

Layout matches the reference frame:
- Source is cropped to a 1:1 square CENTERED ON THE SPEAKER'S FACE (OpenCV Haar
  detection over sampled frames; falls back to center crop if no face found),
  then scaled to 1080x1080.
- Padded into a 1080x1920 canvas with the video band starting at y=VIDEO_TOP
  (black above and below). Title sits in the upper black band; captions sit
  inside the video near the speaker's chest.
- One ASS file carries both the Caption and Title styles, so a single
  subtitles= filter handles captions and the title overlay together.
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from app.config import Settings
from app.fonts import (
    CAPTION_BOLD,
    CAPTION_FONT,
    TITLE_BOLD,
    TITLE_FONT,
    fonts_dir,
)
from app.models import BrollCut, ColorGrade, Title, Transcript, TranscriptWord

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 1080, 1920
VIDEO_TOP = 420  # top y of the 1080x1080 video band inside the 1080x1920 canvas

# User-specified highlight red #CD1111: R=CD G=11 B=11.
# ASS inline color override is {\c&HBBGGRR&} (blue-green-red order),
# so #CD1111 becomes &H1111CD.
# Title highlights keep the same font as the white title text; only color changes.
HIGHLIGHT_OPEN_TITLE = rf"{{\fn{TITLE_FONT}\b{TITLE_BOLD}\c&H1111CD&}}"
HIGHLIGHT_CLOSE_TITLE = rf"{{\fn{TITLE_FONT}\b{TITLE_BOLD}\c&HFFFFFF&}}"

# Caption cue sizing: single-line, at most 3 words per cue (reference style).
CAPTION_FONT_SIZE = 66
MAX_CUE_WORDS = 3
MAX_GAP_SECONDS = 1.0
MIN_BROLL_RENDER_SECONDS = 0.08

_GRADE_FILTERS = {
    ColorGrade.NONE: "",
    ColorGrade.CINEMATIC: "eq=contrast=1.05:saturation=0.9,curves=preset=darker",
    ColorGrade.WARM: "eq=saturation=1.1,colorbalance=rs=0.05:bs=-0.05",
    ColorGrade.COOL: "eq=saturation=0.95,colorbalance=bs=0.05:rs=-0.05",
    ColorGrade.PUNCHY: "eq=contrast=1.15:saturation=1.2",
}


def render(
    source_path: Path,
    output_path: Path,
    start: float,
    end: float,
    transcript: Transcript,
    title: Title,
    color_grade: ColorGrade,
    settings: Settings,
    broll_cuts: list[BrollCut] | None = None,
    music_path: Path | None = None,
    music_volume_db: float | None = None,
    music_loop: bool = True,
    add_caption: bool = True,
) -> Path:
    duration = probe_duration(source_path, settings)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid trim range {start}-{end}")
    if start >= duration:
        raise ValueError(f"Trim start {start:.1f}s is beyond source duration {duration:.1f}s")
    end = min(end, duration)
    clip_duration = end - start

    ass_path = output_path.parent / "captions.ass"
    ass_path.write_text(
        _build_ass(transcript, title, clip_duration, add_caption=add_caption),
        encoding="utf-8",
    )

    # Crop the landscape source to a 1:1 square centered on the speaker's face
    # (falls back to a plain center crop when no face is detected), scale to
    # 1080x1080, then pad vertically into 1080x1920 with the video band at
    # y=VIDEO_TOP (black bars above and below — title and caption text live
    # in those bars).
    focus_x = _detect_face_focus_x(source_path, start, end, settings)
    if focus_x is None:
        focus_x = 0.5
    crop_x = f"min(max(iw*{focus_x:.3f}-ow/2\\,0)\\,iw-ow)"
    main_chain = (
        f"crop=ih:ih:x='{crop_x}':y=0,scale=1080:1080,setsar=1,"
        f"pad=1080:1920:0:{VIDEO_TOP}:black"
    )
    grade = _GRADE_FILTERS[color_grade]
    subtitle_filter = (
        f"subtitles=filename='{_ffmpeg_path(ass_path)}':fontsdir='{_ffmpeg_path(fonts_dir())}'"
    )

    cuts = [
        cut
        for cut in (broll_cuts or [])
        if min(cut.end, clip_duration) - cut.start >= MIN_BROLL_RENDER_SECONDS
        and cut.start < clip_duration
        and cut.clip_path.exists()
    ]

    # Compute the music volume coefficient that produces the reference's mean
    # loudness in the final R7 output. We probe the rendered dry run later,
    # but here we use the (cached) reference volume if available — that's the
    # whole point: the user uploads a reference and our render's music lands
    # at the same dBFS.
    music_volume = _music_volume_coefficient(music_path, music_volume_db, settings)

    if not cuts and not music_path:
        filters = [main_chain] + ([grade] if grade else []) + [subtitle_filter]
        command = [
            settings.ffmpeg_path,
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_path),
            "-t", f"{clip_duration:.3f}",
            "-vf", ",".join(filters),
            "-map", "0:v:0",
            "-map", "0:a:0?",
        ]
    else:
        command = [
            settings.ffmpeg_path,
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_path),
        ]
        for cut in cuts:
            command.extend(["-i", str(cut.clip_path)])
        music_index = None
        if music_path:
            music_index = 1 + len(cuts)
            if music_loop:
                command.extend(["-stream_loop", "-1"])
            command.extend(["-i", str(music_path)])

        graph = [f"[0:v]{main_chain}[v0]"]
        label = "v0"
        for index, cut in enumerate(cuts):
            cut_end = min(cut.end, clip_duration)
            cut_duration = cut_end - cut.start
            # Always strip baked-in letterbox bars FIRST, regardless of
            # whether a face is detected. A 21:9 movie scene with a face
            # in the content area would otherwise be face-center-cropped
            # with the bars still attached, leaving a black gap above the
            # action and below the title. Doing the band-strip first means
            # the subsequent face detect + scale+crop operate on the real
            # content only.
            band = _content_band_crop(cut.clip_path, cut_duration / 2, settings)
            pre_crop = f"{band}," if band else ""
            # Frame the B-roll exactly like the A-roll: fill the full
            # 1080x1080 band (force_original_aspect_ratio=increase => no
            # letterbox / gap) then crop centered on the detected face so
            # a person's face lands in the middle. Falls back to a plain
            # center crop when no face.
            b_focus = _detect_face_focus(cut.clip_path, 0.0, cut_duration, settings)
            b_focus_x, b_focus_y = b_focus if b_focus else (0.5, 0.5)
            b_crop_x = f"min(max(iw*{b_focus_x:.3f}-ow/2\\,0)\\,iw-ow)"
            b_crop_y = f"min(max(ih*{b_focus_y:.3f}-oh/2\\,0)\\,ih-oh)"
            # B-roll frame 0 must appear at t=cut.start on the master timeline;
            # overlay repeats the last frame if the clip runs short of the window.
            graph.append(
                f"[{index + 1}:v]trim=duration={cut_duration:.3f},"
                f"setpts=PTS-STARTPTS+{cut.start:.3f}/TB,"
                f"{pre_crop}"
                f"scale=1080:1080:force_original_aspect_ratio=increase,"
                f"crop=1080:1080:x='{b_crop_x}':y='{b_crop_y}',setsar=1[b{index}]"
            )
            graph.append(
                f"[{label}][b{index}]overlay=0:{VIDEO_TOP}:"
                f"enable='between(t,{cut.start:.3f},{cut_end:.3f})'[v{index + 1}]"
            )
            label = f"v{index + 1}"
        tail = f"{grade},{subtitle_filter}" if grade else subtitle_filter
        graph.append(f"[{label}]{tail}[outv]")

        maps = ["-map", "[outv]"]
        if music_index is not None:
            if _probe_has_audio(source_path, settings):
                # Background music ducked under the speech track. When we
                # know the reference's mean music volume, scale to that;
                # otherwise the historical 0.18 (~-15dB) keeps music under
                # speech.
                graph.append(
                    f"[{music_index}:a]volume={music_volume:.3f}[ma];"
                    f"[0:a][ma]amix=inputs=2:duration=first:normalize=0[aout]"
                )
            else:
                graph.append(f"[{music_index}:a]volume={music_volume:.3f}[aout]")
            maps.extend(["-map", "[aout]"])
        else:
            maps.extend(["-map", "0:a:0?"])
        command.extend(["-t", f"{clip_duration:.3f}", "-filter_complex", ";".join(graph), *maps])

    command.extend(
        [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg render failed: {result.stderr[-1500:]}")
    return output_path


def _probe_has_audio(source_path: Path, settings: Settings) -> bool:
    result = subprocess.run(
        [
            settings.ffprobe_path,
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(source_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def probe_duration(source_path: Path, settings: Settings) -> float:
    command = [
        settings.ffprobe_path,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[-500:]}")
    return float(result.stdout.strip())


def _detect_face_focus(
    source_path: Path, start: float, end: float, settings: Settings
) -> tuple[float, float] | None:
    """Face center (x, y) each 0..1, averaged over sampled frames, or None.

    Samples up to 8 frames at 1fps from the window, runs OpenCV's Haar
    frontal-face cascade on each, and weights each frame's largest face by its
    area so brief misdetections don't drag the crop around.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not installed - falling back to center crop")
        return None

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None

    duration = max(0.1, min(8.0, end - start))
    try:
        with tempfile.TemporaryDirectory(prefix="campeditor-faces-") as tmp:
            frame_pattern = Path(tmp) / "frame-%03d.jpg"
            result = subprocess.run(
                [
                    settings.ffmpeg_path,
                    "-y",
                    "-ss", f"{start:.3f}",
                    "-i", str(source_path),
                    "-t", f"{duration:.3f}",
                    "-vf", "fps=1,scale=320:-1",
                    str(frame_pattern),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return None
            centers: list[tuple[float, float, float]] = []  # (center_x, center_y, weight)
            for frame_path in sorted(Path(tmp).glob("frame-*.jpg")):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32)
                )
                if len(faces) == 0:
                    continue
                x, y, face_width, face_height = max(faces, key=lambda f: f[2] * f[3])
                frame_height, frame_width = gray.shape[0], gray.shape[1]
                centers.append(
                    (
                        (x + face_width / 2) / frame_width,
                        (y + face_height / 2) / frame_height,
                        face_width * face_height,
                    )
                )
            if not centers:
                logger.info("No face detected - using center crop")
                return None
            total_weight = sum(weight for _, _, weight in centers)
            focus_x = sum(cx * weight for cx, _, weight in centers) / total_weight
            focus_y = sum(cy * weight for _, cy, weight in centers) / total_weight
            return max(0.0, min(1.0, focus_x)), max(0.0, min(1.0, focus_y))
    except Exception:
        logger.exception("Face detection failed - using center crop")
        return None


def _detect_face_focus_x(
    source_path: Path, start: float, end: float, settings: Settings
) -> float | None:
    """Horizontal face center (0..1) for the A-roll crop, or None."""
    focus = _detect_face_focus(source_path, start, end, settings)
    return focus[0] if focus else None


def _content_band_crop(clip_path: Path, at_seconds: float, settings: Settings) -> str | None:
    """FFmpeg `crop=...` isolating a B-roll clip's non-black content band, or None.

    Belt-and-suspenders for the "gap under the title" bug: a downloaded vertical
    with black bars baked into the pixels would otherwise be scaled + blind
    center-cropped with the bars intact. Detects the longest run of bright rows
    (same method as the reference-band detector in app.replicate) and returns a
    crop keeping only that band. Returns None for effectively full-frame clips.
    Implemented locally to avoid importing app.replicate (which imports this
    module) at load time.
    """
    try:
        import cv2
    except ImportError:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="campeditor-band-") as tmp:
            probe = Path(tmp) / "band.jpg"
            result = subprocess.run(
                [
                    settings.ffmpeg_path,
                    "-y",
                    "-ss", f"{max(0.0, at_seconds):.3f}",
                    "-i", str(clip_path),
                    "-frames:v", "1",
                    "-vf", "scale=270:-1",
                    str(probe),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0 or not probe.exists():
                return None
            frame = cv2.imread(str(probe))
            if frame is None:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row_means = gray.mean(axis=1)
            height = gray.shape[0]
            best_start, best_length = 0, 0
            run_start: int | None = None
            for index, value in enumerate([*row_means, 0.0]):  # sentinel closes a trailing run
                if value > 20 and run_start is None:
                    run_start = index
                elif value <= 20 and run_start is not None:
                    if index - run_start > best_length:
                        best_start, best_length = run_start, index - run_start
                    run_start = None
            band_height = best_length / height
            band_top = best_start / height
            # Ignore too-small bands (likely a dark scene, not letterbox) and
            # near-full-frame content (no meaningful bars).
            if band_height < 0.2 or (band_height >= 0.90 and band_top <= 0.05):
                return None
            return f"crop=iw:ih*{band_height:.4f}:0:ih*{band_top:.4f}"
    except Exception:
        logger.exception("B-roll content-band detection failed")
        return None


def _ffmpeg_path(path: Path) -> str:
    # Windows FFmpeg filter args need backslashes doubled and drive colons escaped.
    return str(path).replace("\\", "\\\\").replace(":", "\\:")


def _build_ass(
    transcript: Transcript,
    title: Title,
    clip_duration: float,
    add_caption: bool = True,
) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
            "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
            "Alignment,MarginL,MarginR,MarginV,Encoding"
        ),
        # Captions: Inter Bold, plain white, bottom-anchored centered. Matches the
        # user's editor preset: no outline, soft black drop shadow at 50% opacity
        # (ASS alpha = (1-0.50)*255 ~= 0x80 in BackColour). MarginV 790 puts the cue
        # ~63% down the video band (chin/upper-chest level in the reference edit).
        (
            f"Style: Caption,{CAPTION_FONT},{CAPTION_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
            f"{CAPTION_BOLD},0,0,0,100,100,0,0,1,0,2,2,60,60,790,1"
        ),
        # Title: bold Inter throughout; red words only change color.
        # bottom-left anchored (Alignment 1) so the text block's BOTTOM sits at a
        # fixed 24px gap above the video band top (y=VIDEO_TOP), regardless of
        # whether the title is 1 or 2 lines. MarginV counts from the frame bottom.
        (
            f"Style: Title,{TITLE_FONT},56,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
            f"{TITLE_BOLD},0,0,0,100,100,0,0,1,3,1,1,64,64,{HEIGHT - VIDEO_TOP + 24},1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    title_text = _title_ass_text(title)
    if title_text:
        lines.append(
            f"Dialogue: 1,0:00:00.00,{_ass_time(clip_duration)},Title,,0,0,0,,{title_text}"
        )

    # Caption burn-in is gated by the per-job add_caption flag. When False
    # we still write the ASS file (the title needs the filter) but emit no
    # caption Dialogue rows. This is the simplest correct way to skip just
    # the caption step without disturbing the title overlay or the ffmpeg
    # filter chain.
    if add_caption:
        for cue_start, cue_end, cue_text in _caption_cues(transcript.words, clip_duration):
            lines.append(
                f"Dialogue: 0,{_ass_time(cue_start)},{_ass_time(cue_end)},Caption,,0,0,0,,{cue_text}"
            )
    return "\n".join(lines) + "\n"


def _title_ass_text(title: Title) -> str:
    parts = [_sanitize(title.line1), _sanitize(title.line2)]
    highlights = [_sanitize(h) for h in title.highlight_words if _sanitize(h)]
    rendered_parts: list[str] = []
    remaining = highlights[:]
    for part in parts:
        if not part:
            continue
        rendered, used_index = _highlight_one_phrase(part, remaining)
        rendered_parts.append(rendered)
        if used_index is not None:
            del remaining[used_index]
    return r"\N".join(rendered_parts)


def _highlight_one_phrase(text: str, highlights: list[str]) -> tuple[str, int | None]:
    matches = [
        (index, highlight)
        for index, highlight in enumerate(highlights)
        if re.search(re.escape(highlight), text, flags=re.IGNORECASE)
    ]
    if not matches:
        return text, None
    used_index, highlight = max(matches, key=lambda item: len(item[1]))
    pattern = re.compile(re.escape(highlight), flags=re.IGNORECASE)
    return (
        pattern.sub(
            lambda m: f"{HIGHLIGHT_OPEN_TITLE}{m.group(0)}{HIGHLIGHT_CLOSE_TITLE}",
            text,
            count=1,
        ),
        used_index,
    )


def _caption_cues(
    words: list[TranscriptWord],
    clip_duration: float,
) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    group: list[TranscriptWord] = []

    def flush() -> None:
        nonlocal group
        if not group:
            return
        start = max(0.0, group[0].start)
        end = min(clip_duration, max(group[-1].end, start + 0.3))
        if end > start:
            cues.append((start, end, _cue_text(group)))
        group = []

    def visual_words(group_words: list[TranscriptWord]) -> int:
        # Groq sometimes returns multi-word tokens (e.g. "a $100"), so count the
        # words the viewer will actually see, not the token count.
        return sum(len(w.word.split()) for w in group_words)

    for word in words:
        if word.start >= clip_duration:
            break
        gap = word.start - group[-1].end if group else 0.0
        if group and (visual_words(group) + len(word.word.split()) > MAX_CUE_WORDS or gap > MAX_GAP_SECONDS):
            flush()
        group.append(word)
    flush()

    # Overlapping cues would make libass stack them vertically, pushing one cue up
    # over the speaker's face. Clamp each cue's end to the next cue's start so only
    # one caption ever occupies the fixed position at a time.
    clamped: list[tuple[float, float, str]] = []
    for index, (start, end, text) in enumerate(cues):
        if index + 1 < len(cues):
            end = min(end, cues[index + 1][0])
        if end > start:
            clamped.append((start, end, text))
    return clamped


def _cue_text(group: list[TranscriptWord]) -> str:
    return " ".join(_sanitize(word.word) for word in group)


def _sanitize(text: str) -> str:
    # ASS override blocks use {} and \ — strip them from user/LLM text.
    return text.replace("{", "(").replace("}", ")").replace("\\", "/").strip()


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


# ---------------------------------------------------------------------------
# Music loudness targeting
# ---------------------------------------------------------------------------


# When the reference's mean music loudness is unknown, fall back to a
# reasonable "background music under speech" coefficient (~-15 dB).
DEFAULT_MUSIC_VOLUME = 0.18
# Safety bounds for the loudness-targeted coefficient. Below 0.05 the music
# is inaudible; above 1.0 we'd clip. The renderer clamps.
MUSIC_VOLUME_FLOOR = 0.05
MUSIC_VOLUME_CEIL = 1.5


def _music_volume_coefficient(
    music_path: Path | None,
    target_db: float | None,
    settings: Settings,
) -> float:
    """Pick a volume multiplier for the music track.

    target_db is the reference's mean music loudness in dBFS (negative
    numbers). When present, we measure the music's native mean loudness and
    compute the multiplier that would land at target_db. When absent, fall
    back to a fixed ~0.18 (~-15 dB) so music sits under the speech track.
    """
    if music_path is None or not music_path.exists():
        return DEFAULT_MUSIC_VOLUME
    if target_db is None:
        return DEFAULT_MUSIC_VOLUME
    source_db = _volumedetect_db(music_path, settings)
    if source_db <= -90.0 or target_db <= -90.0:
        return DEFAULT_MUSIC_VOLUME
    # FFmpeg's volume filter takes a linear multiplier; 20*log10(g) = dB gain,
    # so g = 10^(dB/20). The user wants the RENDERED music to land at
    # target_db loudness — so g = 10^((target - source) / 20).
    delta_db = target_db - source_db
    raw = 10 ** (delta_db / 20.0)
    return max(MUSIC_VOLUME_FLOOR, min(MUSIC_VOLUME_CEIL, raw))


def _volumedetect_db(audio_path: Path, settings: Settings) -> float:
    """volumedetect the mean dBFS of an audio file. Returns -100 when the
    probe fails / the file is silent."""
    try:
        result = subprocess.run(
            [
                settings.ffmpeg_path,
                "-i", str(audio_path),
                "-af", "volumedetect",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return -100.0
    match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", result.stderr or "")
    return float(match.group(1)) if match else -100.0
