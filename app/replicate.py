"""Reference-short reverse engineering for Replicate mode.

Given a finished viral short (the "reference"), figure out how it was edited:
- read its on-screen title with vision (so the title LLM can clone the pattern),
- find its B-roll cutaways and describe/source matching footage (app/broll.py),
- pull background music from speech-free stretches of its audio,
- detect any logo/image overlay so the renderer can reproduce it.

B-roll detection, description, and sourcing live in app/broll.py; this module
only assembles the ReferenceAnalysis and owns everything else (title, music,
logo, reference download).
"""

import logging
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.broll import describe_spans, detect_broll_spans, extract_reference_frames
from app.broll import _vision as _broll_vision
from app.config import Settings
from app.models import ReferenceAnalysis, Transcript, TranscriptWord
from app.rendering import probe_duration
from app.transcription import transcribe

logger = logging.getLogger(__name__)

MAX_ANALYZE_SECONDS = 60.0
# Threshold for the speech-free "music window" the gap-fallback uses when
# MDX-Net can't produce a usable instrumental. 3s was too strict — a
# continuous-speech reference (e.g. a 21s reel with the speaker talking
# start-to-finish) has no 3s gap, so the fallback returned None and the
# rendered video played without music. 1.5s is permissive enough to find
# a usable section in the brief breaths of typical voiceovers.
MIN_MUSIC_GAP = 1.5

# Whisper model used for the REFERENCE clip's transcript. The reference
# transcript only feeds music gap-detection (looking for speech-free
# windows), so we don't need large-v3 word-level accuracy. `medium` runs
# ~3x realtime on CPU vs large-v3's ~1x, dropping reference transcription
# from ~60s to ~20s on a 21-second reel — a meaningful chunk of the
# `analyzing` stage. The source clip's transcript still uses large-v3
# (the default) so caption timing is unaffected.
REFERENCE_WHISPER_MODEL = "medium"

# Subprocess timeout for the MDX-Net music-separation worker. On CPU the
# HQ model can spend ~30s loading + 1-2x realtime per chunk. We cap at
# 120s so a stuck inference call can't burn the rest of the stage budget;
# the gap fallback then runs in <1s.
MDX_NET_SUBPROCESS_TIMEOUT = 120

# ONNX Runtime's CPU arena can fail to allocate ("bad allocation" /
# BFCArena::AllocateRawInternal) when two MDX-Net separation sessions run at
# the same time on the same process, which happens routinely with
# worker_count > 1 (two jobs analyzing their reference concurrently). That
# exception was previously swallowed and silently fell back to the
# speech-gap method, which finds nothing on continuous-speech references -
# music extraction failed with no visible error. Serializing just this
# inference call fixes it without losing any other stage's concurrency.
_SEPARATION_LOCK = threading.Lock()


def download_reference(url: str, work_dir: Path, settings: Settings) -> Path:
    """Download a reference short from a URL (YouTube/Shorts/etc.) via yt-dlp."""
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Reference link must be an http(s) URL")
    target = work_dir / "reference.mp4"
    target.unlink(missing_ok=True)
    command = _yt_dlp_command(url, target, settings)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0 and _is_browser_cookie_error(result.stderr):
        # A locked/unreadable/undecryptable browser cookie DB shouldn't kill
        # public downloads. yt-dlp #10927: Chrome 130+ DPAPI cookie encryption
        # breaks "--cookies-from-browser chrome" — the stderr says "decrypt with
        # DPAPI" (no "cookie" word), so a naive substring check lets the error
        # propagate and fail the whole job. Drop cookie args and retry.
        cookieless = [arg for index, arg in enumerate(command) if not (
            arg in ("--cookies", "--cookies-from-browser")
            or (index > 0 and command[index - 1] in ("--cookies", "--cookies-from-browser"))
        )]
        result = subprocess.run(cookieless, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not target.exists() or target.stat().st_size < 10_000:
        raise RuntimeError(f"Reference download failed: {result.stderr[-400:]}")
    return target


def _yt_dlp_command(url: str, target: Path, settings: Settings) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "bv*[height<=1920]+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(target),
    ]
    cookies_file = settings.ytdlp_cookies_file
    if cookies_file and cookies_file.exists():
        command.extend(["--cookies", str(cookies_file)])
    elif settings.ytdlp_cookies_from_browser.strip():
        command.extend(["--cookies-from-browser", settings.ytdlp_cookies_from_browser.strip()])
    command.append(url)
    return command


def _is_browser_cookie_error(stderr: str) -> bool:
    lowered = stderr.lower()
    if "could not copy" in lowered and "cookie" in lowered:
        return True
    if "failed to decrypt" in lowered and "dpapi" in lowered:
        return True
    if "extract cookies" in lowered and "browser" in lowered:
        return True
    return False


def analyze_reference(reference_path: Path, work_dir: Path, settings: Settings) -> ReferenceAnalysis:
    """Analyze the reference short. Heavy stages (Whisper, frame extraction,
    describe_spans vision, music separation, title read) take 1-4 min even on
    the optimized path. When the same reference is uploaded again — extremely
    common in the DPA workflow where a creator iterates on the same short
    with different cuts — we cache the full ReferenceAnalysis by file
    content hash and return it directly.

    The cache invalidates automatically: any change to the reference bytes
    produces a different hash, and the cache file is regenerated. music_path
    is dropped on cache hit because the file lives in the prior job's
    work_dir and isn't safe to point at across jobs — for refs ≤30s
    music_path is None anyway (MDX-Net skipped). For longer refs the music
    extraction is small enough that re-running it on cache hit is cheap.
    """
    cache_dir = settings.data_dir / "cache" / "reference_analysis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _reference_cache_key(reference_path)
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            import json as _json
            cached = _json.loads(cache_file.read_text(encoding="utf-8"))
            analysis = ReferenceAnalysis(
                duration=float(cached["duration"]),
                title_text=str(cached.get("title_text", "")),
                transcript_text=str(cached.get("transcript_text", "")),
                broll_spans=[tuple(span) for span in cached.get("broll_spans", [])],
                broll_span_tags=list(cached.get("broll_span_tags", [])),
                broll_query_source=list(cached.get("broll_query_source", [])),
                music_path=None,  # re-attempted below if cache says None
                music_volume_db=cached.get("music_volume_db"),
            )
            # Always re-attempt music extraction on cache hit. The cache was
            # written before music_path was persisted, and even when it
            # was, the prior run's MDX-Net attempt may have failed
            # because the model wasn't available or timed out — a later
            # run might succeed. _extract_music is cheap on the happy
            # path (gap fallback is <1s); MDX-Net has its own 120s cap.
            transcript = Transcript(
                text=analysis.transcript_text,
                words=[TranscriptWord(**w) for w in (cached.get("transcript_words") or [])]
                if isinstance(cached.get("transcript_words"), list)
                else [],
            )
            music_path = _extract_music(
                reference_path, transcript, analysis.duration, work_dir / "reference", settings,
            )
            if music_path is not None:
                analysis.music_path = music_path
                analysis.music_volume_db = _mean_volume_db(music_path, settings)
            return analysis
        except Exception:
            try:
                cache_file.unlink()
            except OSError:
                pass

    duration = probe_duration(reference_path, settings)
    ref_dir = work_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Group A: transcribe (CPU-bound Whisper) and extract_reference_frames
    # (I/O-bound ffmpeg) both consume the reference file but don't share
    # outputs — run them concurrently.
    with ThreadPoolExecutor(max_workers=2) as _executor:
        _transcript_future = _executor.submit(
            transcribe,
            reference_path, 0.0, duration, ref_dir, settings,
            model_size=REFERENCE_WHISPER_MODEL,
        )
        _frames_future = _executor.submit(
            extract_reference_frames,
            reference_path, min(duration, MAX_ANALYZE_SECONDS), ref_dir / "frames", settings,
        )
        transcript = _transcript_future.result()
        frames = _frames_future.result()

    spans = detect_broll_spans(frames)

    # Group B: describe_spans (cloud vision over frames) and _read_title
    # (cloud vision over title frame) are independent — run concurrently.
    # _read_title is cached by reference file hash, so a cache hit returns
    # instantly without entering the network/vision stage.
    with ThreadPoolExecutor(max_workers=2) as _executor:
        _describe_future = _executor.submit(describe_spans, spans, frames, settings)
        _title_future = _executor.submit(_read_title, reference_path, ref_dir, settings)
        span_profiles = _describe_future.result()
        title_text = _title_future.result()
    broll_spans = [(profile.start, profile.end, profile.query) for profile in span_profiles]
    broll_span_tags = [
        {
            "subjects": profile.subjects,
            "setting": profile.setting,
            "action": profile.action,
            "category": profile.category,
        }
        for profile in span_profiles
    ]
    broll_query_source = ["vision" if profile.query else "" for profile in span_profiles]

    music_path = _extract_music(reference_path, transcript, duration, ref_dir, settings)
    music_volume_db = _mean_volume_db(music_path, settings) if music_path else None

    analysis = ReferenceAnalysis(
        duration=duration,
        title_text=title_text,
        transcript_text=transcript.text,
        broll_spans=broll_spans,
        broll_span_tags=broll_span_tags,
        broll_query_source=broll_query_source,
        music_path=music_path,
        music_volume_db=music_volume_db,
    )

    # Persist the analysis to cache. transcript.words is stored so a
    # cache hit can re-run _extract_music (which needs the word timestamps
    # to find speech-free gaps). music_path itself is NOT cached — it
    # lives in the prior job's work_dir and isn't safe to point at across
    # jobs. We do cache a "music_tried" flag so cache hits know whether
    # the prior run successfully produced a music track or not.
    try:
        import json as _json
        cache_file.write_text(_json.dumps({
            "duration": analysis.duration,
            "title_text": analysis.title_text,
            "transcript_text": analysis.transcript_text,
            "transcript_words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in transcript.words
            ],
            "broll_spans": [list(span) for span in analysis.broll_spans],
            "broll_span_tags": analysis.broll_span_tags,
            "broll_query_source": analysis.broll_query_source,
            "music_volume_db": analysis.music_volume_db,
            "music_tried": music_path is not None,
        }, indent=2), encoding="utf-8")
    except OSError:
        logger.debug("Could not write reference analysis cache for %s", reference_path)

    return analysis


def _read_title(reference_path: Path, ref_dir: Path, settings: Settings) -> str:
    """Read the styled on-screen title from an early frame (top 40% of the canvas).

    The cloud-vision call here is ~15s on a third-party router. Cached by
    the reference file's content hash so a re-uploaded reference skips both
    the ffmpeg title-frame extraction and the vision call entirely. The
    cache key mixes size + a partial hash (first 1MB + last 1MB) so it stays
    O(constant) instead of streaming the whole upload.
    """
    cache_dir = settings.data_dir / "cache" / "reference_titles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _reference_cache_key(reference_path)
    cache_file = cache_dir / f"{cache_key}.txt"
    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    title_frame = ref_dir / "title_frame.jpg"
    result = subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-ss", "1.0",
            "-i", str(reference_path),
            "-frames:v", "1",
            "-vf", "crop=iw:ih*0.4:0:0,scale=720:-1",
            str(title_frame),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or not title_frame.exists():
        try:
            cache_file.write_text("", encoding="utf-8")
        except OSError:
            pass
        return ""
    try:
        text = _broll_vision(
            title_frame,
            "This is the top part of a vertical short-form video. Read the styled on-screen "
            "TITLE text (the headline overlay). Reply with ONLY the exact title text on one "
            "line. If there is no title text, reply NONE.",
            settings,
        )
        text = " ".join(text.split())
        if not text or text.upper() == "NONE" or len(text) > 120:
            text_out = ""
        else:
            text_out = text
    except Exception:
        logger.exception("Title vision read failed")
        text_out = ""
    try:
        cache_file.write_text(text_out, encoding="utf-8")
    except OSError:
        pass
    return text_out


def _reference_cache_key(reference_path: Path) -> str:
    """Stable per-content hash for the reference file. Uses file size plus
    the first and last 1MB so re-uploads of the same file collide without
    streaming the whole upload through the hash. md5 is fine here — this
    is just a cache key, not a security primitive.
    """
    import hashlib
    h = hashlib.md5()
    try:
        size = reference_path.stat().st_size
        h.update(f"size:{size}\n".encode("utf-8"))
        with reference_path.open("rb") as f:
            head = f.read(1024 * 1024)
            h.update(head)
            if size > 1024 * 1024:
                f.seek(max(0, size - 1024 * 1024))
                tail = f.read(1024 * 1024)
                h.update(tail)
    except OSError:
        pass
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------


def _extract_music(
    reference_path: Path,
    transcript: Transcript,
    duration: float,
    ref_dir: Path,
    settings: Settings,
) -> Path | None:
    """The reference's music WITHOUT its voiceover, timestamps preserved.

    Primary path: MDX-Net vocal separation (small ONNX model, CPU) over the full
    reference audio — the instrumental keeps the exact same timeline, so music
    hits land at the same timestamps as the reference. Fallback when separation
    is unavailable/fails: the old longest-speech-free-gap loop.
    """
    separated = _separate_instrumental(reference_path, ref_dir, settings)
    if separated is not None:
        return separated

    if not transcript.words:
        # No speech at all: the whole audio track is effectively music.
        gap = (0.0, min(duration, 20.0))
    else:
        gaps: list[tuple[float, float]] = []
        previous_end = 0.0
        for word in transcript.words:
            if word.start - previous_end >= MIN_MUSIC_GAP:
                gaps.append((previous_end, word.start))
            previous_end = max(previous_end, word.end)
        if duration - previous_end >= MIN_MUSIC_GAP:
            gaps.append((previous_end, duration))
        if not gaps:
            return None
        gap = max(gaps, key=lambda g: g[1] - g[0])

    music_path = ref_dir / "music.m4a"
    result = subprocess.run(
        [
            settings.ffmpeg_path,
            "-y",
            "-ss", f"{gap[0]:.3f}",
            "-i", str(reference_path),
            "-t", f"{gap[1] - gap[0]:.3f}",
            "-vn",
            "-c:a", "aac",
            "-b:a", "160k",
            str(music_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not music_path.exists() or music_path.stat().st_size < 5_000:
        return _extract_reference_audio_as_music(reference_path, ref_dir, settings)
    return music_path


def _extract_reference_audio_as_music(
    reference_path: Path, ref_dir: Path, settings: Settings,
) -> Path | None:
    """Last-resort music bed: the reference's own audio (with voice) at -18 dB,
    normalized and trimmed. Used only when both MDX-Net and the gap-fallback
    return nothing — e.g. a continuous-speech reference with no 1.5s+ pause.
    The user gets the reference's own audio bed in the output (with the voice
    faintly audible underneath the new edit) rather than total silence.

    This is intentionally weaker than a real instrumental — but silent
    videos are a worse failure mode than a faint voice under the music.
    """
    fallback_path = ref_dir / "music_fallback.m4a"
    if fallback_path.exists() and fallback_path.stat().st_size > 5_000:
        return fallback_path
    try:
        result = subprocess.run(
            [
                settings.ffmpeg_path,
                "-y",
                "-i", str(reference_path),
                "-vn",
                "-af", "volume=-18dB,aresample=44100",
                "-c:a", "aac",
                "-b:a", "128k",
                str(fallback_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return None
    if result.returncode != 0 or not fallback_path.exists() or fallback_path.stat().st_size < 5_000:
        return None
    return fallback_path


def _separate_instrumental(reference_path: Path, ref_dir: Path, settings: Settings) -> Path | None:
    """MDX-Net (ONNX, CPU) instrumental extraction of the FULL reference audio.

    Returns an m4a whose timeline is 1:1 with the reference, or None when the
    package/model is unavailable, separation fails, or the instrumental is
    basically silence (reference had no real music under the voice).
    """
    if not settings.music_separation:
        return None
    try:
        import audio_separator  # noqa: F401
    except ImportError:
        logger.warning("audio-separator not installed - music falls back to gap extraction")
        return None

    music_path = ref_dir / "music_instrumental.m4a"
    if music_path.exists() and music_path.stat().st_size > 5_000:
        return music_path
    try:
        # Skip the slow MDX-Net path for short references. MDX-Net on CPU
        # Note: we used to skip MDX-Net for refs ≤30s on the theory that
        # the gap fallback would be just as good. It isn't — a continuous-
        # speech reference (e.g. a 21s reel with the speaker talking
        # start-to-finish) has no 3s+ gap, so the fallback returns None
        # and the rendered video plays without music. MDX-Net actually
        # produces a usable instrumental for these refs, so we now always
        # try it (capped at MDX_NET_SUBPROCESS_TIMEOUT to keep the budget
        # bounded).
        wav = ref_dir / "ref_audio.wav"
        # Don't re-extract if a previous run already left a usable WAV on
        # disk — extraction takes 5-30s for an MPEG on Windows and is
        # identical for the same reference.
        if not wav.exists() or wav.stat().st_size < 5_000:
            extract = subprocess.run(
                [
                    settings.ffmpeg_path,
                    "-y",
                    "-i", str(reference_path),
                    "-vn", "-ac", "2", "-ar", "44100",
                    str(wav),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if extract.returncode != 0 or not wav.exists():
                return None

        separated_dir = ref_dir / "separated"
        separated_dir.mkdir(exist_ok=True)
        # Run the actual separation in its own subprocess, one at a time.
        # In-process ONNX Runtime sessions were observed to fail with
        # "BFCArena::AllocateRawInternal Failed to allocate memory" on
        # back-to-back calls within the same long-lived server process -
        # a lock alone wasn't enough because the prior session's arena
        # memory wasn't reliably released before the next one started. A
        # subprocess gets a clean memory slate every time; the lock keeps
        # peak memory bounded to one separation at a time on this machine.
        with _SEPARATION_LOCK:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "app._music_separator_worker",
                    str(wav),
                    str(settings.data_dir / "models"),
                    str(separated_dir),
                    "UVR-MDX-NET-Inst_HQ_3.onnx",
                ],
                capture_output=True,
                text=True,
                timeout=MDX_NET_SUBPROCESS_TIMEOUT,
            )
        if proc.returncode != 0:
            logger.warning("Music separation subprocess failed: %s", proc.stderr[-800:])
            return None
        instrumental_name = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        instrumental = separated_dir / instrumental_name if instrumental_name else None
        if instrumental is None or not instrumental.exists():
            return None
        if _mean_volume_db(instrumental, settings) < -45.0:
            logger.info("Instrumental is near-silent - reference has no real music")
            return None
        encode = subprocess.run(
            [
                settings.ffmpeg_path,
                "-y",
                "-i", str(instrumental),
                "-c:a", "aac",
                "-b:a", "160k",
                str(music_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if encode.returncode != 0 or not music_path.exists() or music_path.stat().st_size < 5_000:
            return None
        logger.info("Extracted instrumental music track (timestamps preserved)")
        return music_path
    except Exception:
        logger.exception("Music separation failed - falling back to gap extraction")
        return None


def _mean_volume_db(audio_path: Path, settings: Settings) -> float:
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
    match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", result.stderr)
    return float(match.group(1)) if match else -100.0


