"""LLM chat ladder for broll_intelligence — text-only completions.

Mirrors the Groq -> NVIDIA (up to 4 keys) -> Gemini -> local Ollama chain
from `app/broll.py::_chat` WITHOUT importing from `app.*`. The package must
stay standalone so it can be A/B tested against the existing pipeline
without risking drift.

Provider rotation + cooldown semantics are intentionally identical to the
vision ladder in `broll_intelligence.vision_ladder`:

  * 15s timeout (per call, default — caller may pass a tighter one)
  * 120s timeout for the local Ollama rung (slower CPU inference)
  * 90s cooldown on any provider that returns 429 / RateLimitError
  * Empty / non-JSON response from a rung is treated as "no answer" and we
    try the next rung

Single entry point: `chat(prompt, settings) -> str`. Empty string ==
"every provider failed". Never raises.

JSON helpers (`extract_json_object`, `strip_code_fence`) are exposed so the
search module can call the ladder with a JSON-only prompt and still get a
clean dict back even when the model wraps its answer in a markdown fence.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

CHAT_TIMEOUT = 15.0
OLLAMA_TIMEOUT = 120.0
RATE_LIMIT_COOLDOWN_SECONDS = 90.0

# Module-level cooldown dict, mirroring vision_ladder. Tests reset it via
# reset_cooldowns() so a 429 in one test doesn't poison the next.
_provider_cooldowns: dict[str, float] = {}


def reset_cooldowns() -> None:
    """Clear the provider cooldown dict. Test helper."""
    _provider_cooldowns.clear()


def _cooling(provider_key: str) -> bool:
    return time.monotonic() < _provider_cooldowns.get(provider_key, 0.0)


def _cool_provider_if_rate_limited(provider_key: str, error: Exception) -> None:
    """Same policy as vision_ladder: a 429 / RateLimitError parks that
    provider for RATE_LIMIT_COOLDOWN_SECONDS so we don't burn budget on a
    known-exhausted rung."""
    if type(error).__name__ == "RateLimitError" or "429" in str(error):
        _provider_cooldowns[provider_key] = time.monotonic() + RATE_LIMIT_COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# OpenAI-compatible chat call (all 4 providers speak the same OpenAI shape)
# ---------------------------------------------------------------------------


def _try_chat(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    *,
    timeout: float,
    temperature: float = 0.4,
    max_tokens: int = 200,
    **extra: Any,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )
    choice = response.choices[0] if response.choices else None
    return ((choice.message.content if choice and choice.message else None) or "").strip()


# ---------------------------------------------------------------------------
# Public chat ladder
# ---------------------------------------------------------------------------


def chat(
    prompt: str,
    settings: Settings,
    *,
    timeout: float = CHAT_TIMEOUT,
    temperature: float = 0.4,
    max_tokens: int = 200,
) -> str:
    """Run the Groq -> NVIDIA -> Gemini -> Ollama ladder against `prompt`.

    Returns the first non-empty model text. Returns "" when every provider
    failed (no key, 429, exception, empty response). Never raises.

    Defaults are tuned for the search-query-generation use case
    (200 tokens / 0.4 temp / 15s); callers can tighten or loosen via the
    keyword args without forking the ladder.
    """
    if not prompt:
        return ""

    # ---- Groq ----
    if settings.groq_api_key and not _cooling("groq"):
        try:
            result = _try_chat(
                settings.groq_api_key,
                "https://api.groq.com/openai/v1",
                settings.groq_vision_model,
                prompt,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if result:
                return result
        except Exception as exc:
            logger.warning("Groq chat failed (%s): %s", type(exc).__name__, exc)
            _cool_provider_if_rate_limited("groq", exc)

    # ---- NVIDIA (rotates through up to 4 keys) ----
    nvidia_keys = settings.nvidia_keys()
    for index, nvidia_key in enumerate(nvidia_keys):
        provider_key = f"nvidia{index}"
        if _cooling(provider_key):
            continue
        try:
            result = _try_chat(
                nvidia_key,
                settings.nvidia_base_url,
                # NVIDIA's text models share the same config slot as vision
                # (we don't have a dedicated chat model name). Falls back
                # to the vision-model name when unset.
                settings.nvidia_vision_model,
                prompt,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if result:
                return result
        except Exception as exc:
            logger.warning(
                "NVIDIA chat key #%d failed (%s): %s", index, type(exc).__name__, exc
            )
            _cool_provider_if_rate_limited(provider_key, exc)

    # ---- Gemini ----
    if settings.gemini_api_key and not _cooling("gemini"):
        try:
            result = _try_chat(
                settings.gemini_api_key,
                settings.gemini_base_url,
                settings.gemini_vision_model,
                prompt,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="none",
            )
            if result:
                return result
        except Exception as exc:
            logger.warning("Gemini chat failed (%s): %s", type(exc).__name__, exc)
            _cool_provider_if_rate_limited("gemini", exc)

    # ---- Ollama (local last-resort) ----
    if settings.ollama_vision_model and not _cooling("ollama"):
        try:
            result = _try_chat(
                "ollama",
                settings.ollama_base_url,
                settings.ollama_vision_model,
                prompt,
                timeout=OLLAMA_TIMEOUT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if result:
                return result
        except Exception as exc:
            logger.warning("Ollama local chat failed (%s): %s", type(exc).__name__, exc)
            _provider_cooldowns["ollama"] = (
                time.monotonic() + RATE_LIMIT_COOLDOWN_SECONDS
            )

    return ""


# ---------------------------------------------------------------------------
# JSON helpers — shared with search.py so the prompt + parse live together
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove the leading ```json (or bare ```) fence and the trailing ```,
    if present. Idempotent on text that isn't fenced."""
    if not text:
        return ""
    out = text.strip()
    out = _FENCE_RE.sub("", out, count=1).lstrip()
    if out.endswith("```"):
        out = out[:-3].rstrip()
    return out


def extract_json_object(raw: str) -> dict[str, Any]:
    """Forgiving JSON-object extractor.

    Accepts bare JSON, ```json ... ``` fenced JSON, prose-prefixed /
    -suffixed JSON, or anything with a single {...} span somewhere inside.
    Returns {} on any failure.
    """
    if not raw:
        return {}
    text = strip_code_fence(raw)
    match = _BRACE_RE.search(text)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def generate_search_queries(
    reference_prompt: str,
    settings: Settings,
    *,
    timeout: float = CHAT_TIMEOUT,
) -> list[str]:
    """Convenience wrapper: call the chat ladder with the supplied prompt
    and return whatever the model produced. Caller is responsible for
    validation + truncation + fallback; this function never raises and
    always returns a list (possibly empty) so callers can fallback
    deterministically."""
    raw = chat(reference_prompt, settings, timeout=timeout)
    obj = extract_json_object(raw)
    queries = obj.get("queries") if isinstance(obj, dict) else None
    if not isinstance(queries, list):
        return []
    out: list[str] = []
    for q in queries:
        if isinstance(q, str):
            s = " ".join(q.split()).strip()
            if s:
                out.append(s)
    return out


__all__ = [
    "chat",
    "extract_json_object",
    "strip_code_fence",
    "generate_search_queries",
    "reset_cooldowns",
    "CHAT_TIMEOUT",
    "OLLAMA_TIMEOUT",
    "RATE_LIMIT_COOLDOWN_SECONDS",
]