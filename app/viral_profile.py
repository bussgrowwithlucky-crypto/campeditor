"""Learned viral-short scoring profile.

The profile is generated from the user's own finished shorts by
``scripts/learn_viral_profile.py`` and loaded by clip selection. The app still
has a strong built-in default so auto-selection works before the profile exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter
from typing import Any

from app.config import Settings

DEFAULT_HOOK_TERMS: dict[str, float] = {
    "secret": 4.0,
    "rule": 3.5,
    "mistake": 3.5,
    "revealed": 3.0,
    "hidden": 3.0,
    "truth": 2.8,
    "wrong": 2.5,
    "never": 2.5,
    "why": 2.0,
    "how": 2.0,
    "bullshit": 3.5,
    "kill": 3.0,
    "kills": 3.0,
    "trillion": 3.0,
    "billion": 2.5,
    "million": 2.0,
    "ai": 2.5,
    "amazon": 2.5,
    "apple": 2.5,
    "iphone": 2.5,
    "bezos": 2.5,
    "musk": 2.5,
    "jobs": 2.5,
    "ford": 2.5,
    "founder": 2.0,
    "company": 1.8,
    "startup": 1.8,
}

DEFAULT_PHRASES: dict[str, float] = {
    "work life balance": 4.0,
    "work life harmony": 4.5,
    "trillion dollar company": 4.0,
    "secret rule": 3.5,
    "secret technique": 3.5,
    "faster horses": 3.5,
    "built amazon": 3.0,
    "built apple": 3.0,
    "built the iphone": 3.0,
    "don't ask customers": 3.0,
}

_WORD_RE = re.compile(r"[a-z0-9$%+-]+")
_NUMBER_RE = re.compile(r"\$?\d[\d,.]*(?:k|m|b|%|x|\+)?", flags=re.IGNORECASE)

_STOPWORDS = {
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


class ViralProfile:
    def __init__(
        self,
        hook_terms: dict[str, float],
        phrases: dict[str, float],
        opening_terms: dict[str, float] | None = None,
        opening_phrases: dict[str, float] | None = None,
    ):
        self.hook_terms = {k.lower(): float(v) for k, v in hook_terms.items()}
        self.phrases = {k.lower(): float(v) for k, v in phrases.items()}
        self.opening_terms = {k.lower(): float(v) for k, v in (opening_terms or {}).items()}
        self.opening_phrases = {k.lower(): float(v) for k, v in (opening_phrases or {}).items()}

    def score_text(self, text: str) -> float:
        normalized = _normalize(text)
        tokens = _WORD_RE.findall(normalized)
        if not tokens:
            return 0.0
        score = min(len(tokens), 80) / 55.0
        score += sum(self.hook_terms.get(token, 0.0) for token in tokens)
        score += sum(weight for phrase, weight in self.phrases.items() if phrase in normalized)
        score += len(_NUMBER_RE.findall(normalized)) * 2.8
        score += text.count("?") * 1.5
        score += text.count("!") * 1.0
        if any(token in tokens for token in ("but", "however", "actually", "instead")):
            score += 1.6
        if any(token in tokens for token in ("don't", "never", "wrong", "bullshit")):
            score += 2.2
        return score

    def score_opening(self, text: str) -> float:
        normalized = _normalize(text)
        tokens = _WORD_RE.findall(normalized)
        if not tokens:
            return 0.0
        score = sum(self.opening_terms.get(token, 0.0) for token in tokens)
        score += sum(weight for phrase, weight in self.opening_phrases.items() if phrase in normalized)
        # A viral selection has to start with pressure: number, name, conflict,
        # or a clear promise. Reuse the full-window profile at a lower weight.
        score += self.score_text(text) * 0.65
        return score


def load_viral_profile(settings: Settings) -> ViralProfile:
    path = settings.data_dir / "viral_profile.json"
    hook_terms = dict(DEFAULT_HOOK_TERMS)
    phrases = dict(DEFAULT_PHRASES)
    opening_terms: dict[str, float] = {}
    opening_phrases: dict[str, float] = {}
    if path.exists():
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            hook_terms.update(_float_map(payload.get("hook_terms", {})))
            phrases.update(_float_map(payload.get("phrases", {})))
            opening_terms.update(_float_map(payload.get("opening_terms", {})))
            opening_phrases.update(_float_map(payload.get("opening_phrases", {})))
            if not opening_terms and not opening_phrases:
                learned_terms, learned_phrases = _opening_profile_from_records(payload.get("records", []))
                opening_terms.update(learned_terms)
                opening_phrases.update(learned_phrases)
        except Exception:
            # Selection should never fail just because the learned profile is
            # malformed or mid-write.
            pass
    return ViralProfile(
        hook_terms=hook_terms,
        phrases=phrases,
        opening_terms=opening_terms,
        opening_phrases=opening_phrases,
    )


def _opening_profile_from_records(records: Any) -> tuple[dict[str, float], dict[str, float]]:
    if not isinstance(records, list):
        return {}, {}
    term_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        opening = str(record.get("opening", ""))
        tokens = _WORD_RE.findall(_normalize(opening))
        term_counts.update(token for token in tokens if token not in _STOPWORDS)
        for size in (2, 3):
            for index in range(0, len(tokens) - size + 1):
                phrase_tokens = tokens[index : index + size]
                if phrase_tokens[0] in _STOPWORDS or phrase_tokens[-1] in _STOPWORDS:
                    continue
                phrase_counts[" ".join(phrase_tokens)] += 1
    return _weighted_opening_terms(term_counts), _weighted_opening_phrases(phrase_counts)


def _weighted_opening_terms(counts: Counter[str]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for term, count in counts.most_common(80):
        if len(term) <= 2 and not _NUMBER_RE.fullmatch(term):
            continue
        score = 1.2 + min(count, 8) * 0.45
        if term in DEFAULT_HOOK_TERMS or _NUMBER_RE.fullmatch(term):
            score += 1.8
        scored[term] = round(score, 2)
    return dict(sorted(scored.items(), key=lambda item: item[1], reverse=True)[:35])


def _weighted_opening_phrases(counts: Counter[str]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for phrase, count in counts.most_common(80):
        if count < 2 and not any(seed in phrase for seed in DEFAULT_HOOK_TERMS):
            continue
        score = 1.8 + min(count, 6) * 0.55
        if any(seed in phrase for seed in DEFAULT_HOOK_TERMS):
            score += 1.6
        scored[phrase] = round(score, 2)
    return dict(sorted(scored.items(), key=lambda item: item[1], reverse=True)[:25])


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key).lower()] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%+.' -]", " ", text.lower())).strip()
