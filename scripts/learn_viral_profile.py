"""Build data/viral_profile.json from the user's finished June shorts.

This intentionally keeps the learned data simple and inspectable: transcribe a
sample of local viral shorts, count repeated hook terms/phrases, and write a
profile consumed by app.clip_selection.

Run from C:/campeditor:
    .\.venv\Scripts\python.exe scripts\learn_viral_profile.py --source "C:\50 days\june" --limit 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.rendering import probe_duration
from app.transcription import transcribe

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "like",
    "not",
    "of",
    "on",
    "or",
    "people",
    "she",
    "so",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

HOOK_SEEDS = {
    "secret",
    "rule",
    "technique",
    "mistake",
    "hidden",
    "revealed",
    "never",
    "wrong",
    "truth",
    "bullshit",
    "kill",
    "kills",
    "ai",
    "money",
    "million",
    "billion",
    "trillion",
    "startup",
    "founder",
    "company",
    "amazon",
    "apple",
    "iphone",
    "bezos",
    "musk",
    "jobs",
    "ford",
}

TOKEN_RE = re.compile(r"[a-z0-9$%+-]+")
NUMBER_RE = re.compile(r"\$?\d[\d,.]*(?:k|m|b|%|x|\+)?", flags=re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"C:\50 days\june")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=45.0)
    parser.add_argument("--output", default="data/viral_profile.json")
    args = parser.parse_args()

    settings = get_settings()
    source = Path(args.source)
    output = Path(args.output)
    cache_root = settings.data_dir / "viral_learning"
    cache_root.mkdir(parents=True, exist_ok=True)

    videos = _select_videos(source, args.limit)
    records = []
    term_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    opening_term_counts: Counter[str] = Counter()
    opening_phrase_counts: Counter[str] = Counter()

    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video}")
        duration = min(probe_duration(video, settings), args.max_seconds)
        transcript_text = _cached_transcript(video, duration, cache_root, settings)
        if not transcript_text:
            transcript_text = _filename_text(video)
        tokens = _tokens(transcript_text)
        opening = " ".join(transcript_text.split()[:28])
        opening_tokens = _tokens(opening)
        term_counts.update(token for token in tokens if token not in STOPWORDS)
        phrase_counts.update(_phrases(tokens, 2))
        phrase_counts.update(_phrases(tokens, 3))
        opening_term_counts.update(token for token in opening_tokens if token not in STOPWORDS)
        opening_phrase_counts.update(_phrases(opening_tokens, 2))
        opening_phrase_counts.update(_phrases(opening_tokens, 3))
        records.append(
            {
                "path": str(video),
                "duration_analyzed": round(duration, 2),
                "word_count": len(tokens),
                "opening": opening,
            }
        )

    hook_terms = _weighted_terms(term_counts)
    phrases = _weighted_phrases(phrase_counts)
    opening_terms = _weighted_opening_terms(opening_term_counts)
    opening_phrases = _weighted_opening_phrases(opening_phrase_counts)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "videos_analyzed": len(records),
        "records": records,
        "hook_terms": hook_terms,
        "phrases": phrases,
        "opening_terms": opening_terms,
        "opening_phrases": opening_phrases,
        "notes": [
            "Built from local June shorts using Groq transcription when available.",
            "Used by app.clip_selection fallback scoring, opening-hook scoring, and AI selector prompt context.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output} with {len(hook_terms)} hook terms and {len(phrases)} phrases")


def _select_videos(source: Path, limit: int) -> list[Path]:
    videos = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS and "color graded" not in str(path).lower()
    ]
    videos.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    seen_stems: set[str] = set()
    selected: list[Path] = []
    for video in videos:
        stem = re.sub(r"\W+", " ", video.stem.lower()).strip()
        dedupe_key = " ".join(stem.split()[:4])
        if dedupe_key in seen_stems and len(selected) >= limit // 2:
            continue
        seen_stems.add(dedupe_key)
        selected.append(video)
        if len(selected) >= limit:
            break
    return selected


def _cached_transcript(video: Path, duration: float, cache_root: Path, settings) -> str:
    digest = hashlib.md5(str(video).lower().encode()).hexdigest()[:14]
    work_dir = cache_root / digest
    work_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = work_dir / "transcript.json"
    if transcript_path.exists():
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        return str(payload.get("text", ""))
    transcript = transcribe(video, 0.0, duration, work_dir, settings)
    payload = {
        "path": str(video),
        "duration": duration,
        "text": transcript.text,
        "segments": [segment.model_dump() for segment in transcript.segments],
        "words": [word.model_dump() for word in transcript.words],
    }
    transcript_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return transcript.text


def _filename_text(video: Path) -> str:
    return " ".join(re.sub(r"[^a-zA-Z0-9$%+.-]+", " ", video.stem).split())


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _phrases(tokens: list[str], size: int) -> list[str]:
    phrases: list[str] = []
    for index in range(0, len(tokens) - size + 1):
        phrase_tokens = tokens[index : index + size]
        if all(token in STOPWORDS for token in phrase_tokens):
            continue
        if phrase_tokens[0] in STOPWORDS or phrase_tokens[-1] in STOPWORDS:
            continue
        phrases.append(" ".join(phrase_tokens))
    return phrases


def _weighted_terms(counts: Counter[str]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for term, count in counts.most_common(80):
        if len(term) <= 2 and not NUMBER_RE.fullmatch(term):
            continue
        base = 1.5 + min(count, 8) * 0.35
        if term in HOOK_SEEDS or NUMBER_RE.fullmatch(term):
            base += 2.5
        scored[term] = round(base, 2)
    return dict(sorted(scored.items(), key=lambda item: item[1], reverse=True)[:45])


def _weighted_phrases(counts: Counter[str]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for phrase, count in counts.most_common(120):
        if count < 2 and not any(seed in phrase for seed in HOOK_SEEDS):
            continue
        score = 2.0 + min(count, 6) * 0.5
        if any(seed in phrase for seed in HOOK_SEEDS):
            score += 2.0
        scored[phrase] = round(score, 2)
    return dict(sorted(scored.items(), key=lambda item: item[1], reverse=True)[:35])


def _weighted_opening_terms(counts: Counter[str]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for term, count in counts.most_common(80):
        if len(term) <= 2 and not NUMBER_RE.fullmatch(term):
            continue
        base = 1.2 + min(count, 8) * 0.45
        if term in HOOK_SEEDS or NUMBER_RE.fullmatch(term):
            base += 1.8
        scored[term] = round(base, 2)
    return dict(sorted(scored.items(), key=lambda item: item[1], reverse=True)[:35])


def _weighted_opening_phrases(counts: Counter[str]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for phrase, count in counts.most_common(100):
        if count < 2 and not any(seed in phrase for seed in HOOK_SEEDS):
            continue
        score = 1.8 + min(count, 6) * 0.55
        if any(seed in phrase for seed in HOOK_SEEDS):
            score += 1.6
        scored[phrase] = round(score, 2)
    return dict(sorted(scored.items(), key=lambda item: item[1], reverse=True)[:25])


if __name__ == "__main__":
    main()
