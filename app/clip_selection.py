"""AI-assisted viral clip selection from a full-video transcript."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.config import Settings
from app.models import Transcript, TranscriptSegment, TranscriptWord
from app.viral_profile import ViralProfile, load_viral_profile

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a short-form video editor choosing one clip from a transcript.

Pick the single segment with the strongest viral potential: a clear hook, strong claim,
surprise, conflict, useful insight, money/number, famous person, contrarian opinion, or
emotionally charged moment.

Rules:
- Return STRICT JSON only, no markdown and no prose.
- Choose one continuous segment.
- The first 2-4 seconds must feel like a viral hook by themselves.
- Prefer a segment near the requested target duration.
- Never invent timestamps. Use only the transcript timing.
- Make the segment self-contained enough to work as a short.

Schema:
{"start": 12.3, "end": 27.8, "reason": "short reason"}"""

_HOOK_WORDS = {
    "secret",
    "mistake",
    "kill",
    "kills",
    "never",
    "why",
    "how",
    "money",
    "million",
    "billion",
    "trillion",
    "dollar",
    "ai",
    "best",
    "worst",
    "crazy",
    "insane",
    "truth",
    "actually",
    "problem",
    "failed",
    "failure",
    "success",
    "startup",
    "company",
    "business",
    "bezos",
    "musk",
    "jobs",
    "ford",
}


@dataclass(frozen=True)
class ClipSelection:
    start: float
    end: float
    reason: str


def select_viral_clip(transcript: Transcript, duration: float, settings: Settings) -> ClipSelection:
    """Choose a render range using the LLM provider ladder, with a deterministic fallback."""
    from app.title_generation import provider_ladder

    duration = max(0.0, duration)
    profile = load_viral_profile(settings)
    if transcript.text.strip():
        for provider in provider_ladder(settings):
            try:
                return _ai_select_viral_clip(transcript, duration, settings, profile, provider)
            except Exception:
                logger.exception("Viral clip selection with %s failed", provider[2])
    return fallback_viral_clip(transcript, duration, settings, profile)


def fallback_viral_clip(
    transcript: Transcript,
    duration: float,
    settings: Settings,
    profile: ViralProfile | None = None,
) -> ClipSelection:
    profile = profile or load_viral_profile(settings)
    target = _target_duration(duration, settings)
    if duration <= 0:
        return ClipSelection(0.0, target, "Defaulted to first clip")
    if not transcript.words and not transcript.segments:
        return ClipSelection(0.0, min(duration, target), "No transcript available")

    best_start = 0.0
    best_score = float("-inf")
    starts = _candidate_starts(transcript, duration, target)
    for start in starts:
        end = min(duration, start + target)
        score = _window_score(transcript, start, end, profile)
        if score > best_score:
            best_start = start
            best_score = score

    selection = _normalize_selection(best_start, best_start + target, duration, settings)
    return ClipSelection(selection.start, selection.end, "Picked highest-scoring transcript window")


def slice_transcript(transcript: Transcript, start: float, end: float) -> Transcript:
    """Return transcript text/timestamps relative to the selected clip."""
    clip_duration = max(0.0, end - start)
    words: list[TranscriptWord] = []
    for word in transcript.words:
        if word.end <= start or word.start >= end:
            continue
        words.append(
            TranscriptWord(
                word=word.word,
                start=max(0.0, word.start - start),
                end=min(clip_duration, word.end - start),
            )
        )

    segments: list[TranscriptSegment] = []
    for segment in transcript.segments:
        if segment.end <= start or segment.start >= end:
            continue
        segments.append(
            TranscriptSegment(
                text=segment.text,
                start=max(0.0, segment.start - start),
                end=min(clip_duration, segment.end - start),
            )
        )
    return Transcript(words=words, segments=segments)


def _ai_select_viral_clip(
    transcript: Transcript,
    duration: float,
    settings: Settings,
    profile: ViralProfile,
    provider: tuple[str, str, str] | None = None,
) -> ClipSelection:
    from openai import OpenAI

    base_url, api_key, model = provider or (
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
    )
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30,
        max_retries=0,
    )
    target = _target_duration(duration, settings)
    outline = _transcript_outline(transcript)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Video duration available: {duration:.2f}s\n"
                    f"Target clip duration: {target:.2f}s\n"
                    f"Minimum clip duration: {settings.auto_clip_min_seconds:.2f}s\n"
                    f"Maximum clip duration: {settings.auto_clip_max_seconds:.2f}s\n\n"
                    "Learned from the user's June viral shorts:\n"
                    f"- Strong hook terms: {', '.join(list(profile.hook_terms)[:30])}\n"
                    f"- Repeated viral phrases: {', '.join(list(profile.phrases)[:18])}\n\n"
                    "Opening-hook pattern from those viral shorts:\n"
                    f"- Opening terms: {', '.join(list(profile.opening_terms)[:24])}\n"
                    f"- Opening phrases: {', '.join(list(profile.opening_phrases)[:12])}\n\n"
                    "Pick a clip whose FIRST sentence or first 2-4 seconds has the strongest hook. "
                    "Do not pick a window that only becomes interesting later.\n\n"
                    f"Transcript:\n{outline}\n\nReturn strict JSON only."
                ),
            },
        ],
        temperature=0.35,
    )
    choice = response.choices[0] if response.choices else None
    content = (choice.message.content if choice and choice.message else None) or ""
    if not content.strip():
        raise ValueError("Clip selector returned empty content")
    payload = _extract_json(content)
    selection = _normalize_selection(
        float(payload.get("start", 0.0)),
        float(payload.get("end", 0.0)),
        duration,
        settings,
    )
    reason = str(payload.get("reason", "")).strip()[:160] or "AI-selected viral moment"
    return ClipSelection(selection.start, selection.end, reason)


def _normalize_selection(start: float, end: float, duration: float, settings: Settings) -> ClipSelection:
    target = _target_duration(duration, settings)
    min_duration = min(settings.auto_clip_min_seconds, target, duration or target)
    max_duration = min(settings.auto_clip_max_seconds, duration or settings.auto_clip_max_seconds)

    start = max(0.0, min(start, max(0.0, duration - min_duration)))
    end = max(start, min(end, duration))
    length = end - start

    if length < min_duration:
        center = start + length / 2
        start = max(0.0, center - target / 2)
        end = min(duration, start + target)
        start = max(0.0, end - target)
    elif length > max_duration:
        end = min(duration, start + max_duration)

    if end <= start:
        end = min(duration, start + target)
    return ClipSelection(round(start, 2), round(end, 2), "")


def _target_duration(duration: float, settings: Settings) -> float:
    target = max(settings.auto_clip_min_seconds, settings.auto_clip_target_seconds)
    target = min(target, settings.auto_clip_max_seconds)
    return min(target, duration) if duration > 0 else target


def _transcript_outline(transcript: Transcript, max_chars: int = 12000) -> str:
    lines: list[str] = []
    if transcript.segments:
        for segment in transcript.segments:
            lines.append(f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text.strip()}")
    else:
        for start in _candidate_starts(transcript, transcript.words[-1].end if transcript.words else 0.0, 8.0):
            end = start + 8.0
            text = _window_text(transcript, start, end)
            if text:
                lines.append(f"[{start:.2f}-{end:.2f}] {text}")

    outline = "\n".join(lines)
    if len(outline) <= max_chars:
        return outline
    return outline[:max_chars] + "\n[Transcript truncated for selection]"


def _candidate_starts(transcript: Transcript, duration: float, target: float) -> list[float]:
    starts = [0.0]
    starts.extend(segment.start for segment in transcript.segments)
    starts.extend(word.start for word in transcript.words[:: max(1, len(transcript.words) // 80)])
    max_start = max(0.0, duration - target)
    return sorted({round(max(0.0, min(start, max_start)), 2) for start in starts})


def _window_text(transcript: Transcript, start: float, end: float) -> str:
    if transcript.words:
        parts = [
            word.word.strip()
            for word in transcript.words
            if word.end > start and word.start < end and word.word.strip()
        ]
    else:
        parts = [
            segment.text.strip()
            for segment in transcript.segments
            if segment.end > start and segment.start < end and segment.text.strip()
        ]
    return " ".join(parts)


def _window_score(transcript: Transcript, start: float, end: float, profile: ViralProfile | None) -> float:
    text = _window_text(transcript, start, end)
    opening_end = min(end, start + max(2.5, min(4.0, (end - start) * 0.35)))
    opening = _window_text(transcript, start, opening_end)
    score = _viral_score(text, profile)
    # June shorts consistently open with the hook. Weight the opening hard so a
    # window with a buried good line loses to one that starts with pressure.
    score += _viral_score(opening, profile) * 1.15
    if profile:
        score += profile.score_opening(opening) * 1.65
    if _starts_with_hook(opening, profile):
        score += 4.0
    return score


def _viral_score(text: str, profile: ViralProfile | None = None) -> float:
    normalized = re.sub(r"[^a-z0-9$%+ ]", " ", text.lower())
    tokens = normalized.split()
    score = float(len(tokens)) / 50.0
    score += sum(2.0 for token in tokens if token in _HOOK_WORDS)
    score += len(re.findall(r"\$?\d[\d,.]*(?:k|m|b|%|x|\+)?", normalized, flags=re.IGNORECASE)) * 2.5
    score += text.count("?") * 1.5
    score += text.count("!") * 1.0
    if profile:
        score += profile.score_text(text)
    return score


def _starts_with_hook(text: str, profile: ViralProfile | None = None) -> bool:
    normalized = re.sub(r"[^a-z0-9$%+ ]", " ", text.lower())
    tokens = normalized.split()[:8]
    if not tokens:
        return False
    if any(token in _HOOK_WORDS for token in tokens):
        return True
    if re.search(r"\$?\d[\d,.]*(?:k|m|b|%|x|\+)?", " ".join(tokens), flags=re.IGNORECASE):
        return True
    if profile and any(token in profile.opening_terms for token in tokens):
        return True
    return False


def _extract_json(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return json.loads(match.group(0) if match else cleaned)
