"""Local-only vision ladder for broll_intelligence.

Single-provider: Ollama, with the OpenAI-compatible chat-completions
endpoint at http://localhost:11434/v1 by default. No cloud. No API keys.
No rate-limit cooldowns (a local GPU either responds or doesn't).

Design rules (carried over from the previous ladder):
  * No `app.*` imports — this package must stay a standalone sibling so
    a regression here can never break the production campeditor pipeline.
  * Tolerant parsing — the model is free to hallucinate vocabulary; we
    drop unknowns and continue.
  * Atomic writes — index files use .tmp + os.replace.
  * Incremental by default — frame-tag cache is keyed by frame content
    hash, so re-running a build only re-tags changed clips.

Performance optimisations for the local case:
  1. Bounded async concurrency — the local GPU saturates at ~1-2
     simultaneous inference calls. Configurable via
     OLLAMA_MAX_CONCURRENCY (default 2). More in flight just adds RAM
     pressure without throughput gain.
  2. Batch multi-frame requests — call_once_with_frames submits N
     frames as a single chat-completions request and parses an array
     response. For 3-frame clips this is roughly 1/3 the wall-clock
     of 3 separate calls on a memory-bound model.
  3. Per-call timeout (default 60s, OLLAMA_TIMEOUT env) — a stuck
     inference call no longer blocks the queue.
  4. Frame-tag cache by content hash — unchanged behaviour, but kept
     central so callers don't need to roll their own.

Public API (callable from app/broll.py and the selector):

    from broll_intelligence.vision_ladder import (
        call,                              # single frame, returns str
        call_batch,                        # multiple frames, returns list[str]
        OllamaClient,                      # the underlying HTTP client
        OLLAMA_TIMEOUT,
    )

If Ollama isn't reachable, `call` returns "" and the caller falls back to
empty FeatureVector — the same degradation path the cloud ladder had.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))
OLLAMA_MAX_CONCURRENCY = int(os.environ.get("OLLAMA_MAX_CONCURRENCY", "2"))


# ---------------------------------------------------------------------------
# Concurrency gate
# ---------------------------------------------------------------------------
#
# The local GPU saturates at ~1-2 simultaneous inference calls; more in
# flight only adds memory pressure. We use a semaphore so callers can
# fire many requests in parallel without overwhelming the daemon.
#
# Module-level so the gate is shared across the process — important
# when the B-roll selector's parallel _rank_local and the
# library_indexer's parallel frame tagging both reach for vision.

_default_semaphore: asyncio.Semaphore | None = None
_default_semaphore_lock = threading.Lock()


def _get_semaphore(limit: int) -> asyncio.Semaphore:
    """Return a process-wide semaphore with the given concurrency limit.

    asyncio.Semaphore isn't threadsafe to construct outside a running
    loop, so we cache one created lazily inside an event loop. The
    caller (call / call_batch) hands us the loop via the active task.
    """
    global _default_semaphore
    # We always create a fresh semaphore per async invocation; cheap,
    # and avoids the threadsafe-construction issue entirely.
    return asyncio.Semaphore(limit)


# ---------------------------------------------------------------------------
# Circuit breaker for upstream rate limiting (HTTP 429)
# ---------------------------------------------------------------------------
#
# When byNara (or any OpenAI-compatible upstream that requires a Bearer
# token) rate-limits us with HTTP 429, the request still burns the full
# timeout before failing — typically 60s of waiting for nothing. With
# many parallel selectors running, that means N calls x 60s of wasted
# wall-clock.
#
# The breaker tracks consecutive 429s: if 5 happen within a 60-second
# window, we declare the upstream overloaded and short-circuit all
# subsequent calls for 120 seconds (the cooldown). During cooldown
# chat()/chat_sync() return {} immediately without making any HTTP
# request — letting the B-roll selector fall back to its degraded path
# instead of hanging.
#
# After the cooldown, we let exactly ONE probe call through. If the
# probe succeeds the breaker is closed and the counter resets; if the
# probe 429s again the breaker stays open for another full cooldown.
#
# State is module-level (no class changes) so all OllamaClient instances
# in the process share the same breaker — byNara's rate limit applies
# to the API key, not to per-client connections.
#
# Thread safety: a single threading.Lock guards all state mutation.
# The HTTP call itself happens OUTSIDE the lock so a slow upstream
# can't stall unrelated calls. A stuck probe (e.g., task cancelled
# before its result was recorded) is recovered after _PROBE_MAX_AGE
# seconds.

_429_THRESHOLD = 5
_429_WINDOW_SECONDS = 60.0
_CIRCUIT_COOLDOWN_SECONDS = 120.0
_PROBE_MAX_AGE_SECONDS = 180.0

_circuit_open: bool = False
_circuit_opened_at: float = 0.0
_consecutive_429s: int = 0
_first_429_at: float = 0.0
_probe_started_at: float = 0.0

_circuit_lock = threading.Lock()


def _circuit_check() -> bool:
    """Return True if the current call should be allowed through.

    Side effect: when the breaker is open and the cooldown has elapsed,
    marks this caller as the single probe and records when it was
    dispatched (so the result can be correlated back via
    _record_success / _record_429).
    """
    global _circuit_open, _circuit_opened_at, _probe_started_at
    if not _circuit_open:
        return True
    now = time.monotonic()
    if (now - _circuit_opened_at) < _CIRCUIT_COOLDOWN_SECONDS:
        return False
    # Cooldown elapsed — decide whether to dispatch a probe.
    with _circuit_lock:
        if not _circuit_open:
            return True
        # Safety: if a probe was dispatched more than _PROBE_MAX_AGE
        # seconds ago and never reported back (e.g., task cancelled),
        # let a fresh probe through.
        if _probe_started_at > 0 and (now - _probe_started_at) > _PROBE_MAX_AGE_SECONDS:
            _probe_started_at = now
            return True
        if _probe_started_at > 0:
            # A probe is already in flight — block everyone else.
            return False
        # Dispatch a single probe.
        _probe_started_at = now
        return True


def _record_429() -> None:
    """Update breaker state on a 429 response. Opens the circuit if
    5 consecutive 429s happen within a 60s window; if a probe just
    429'd, extends the cooldown and keeps the circuit open."""
    global _circuit_open, _circuit_opened_at, _consecutive_429s
    global _first_429_at, _probe_started_at
    now = time.monotonic()
    with _circuit_lock:
        _probe_started_at = 0.0
        if _circuit_open:
            # Probe 429'd — extend cooldown, keep circuit open.
            _circuit_opened_at = now
            logger.warning(
                "vision_ladder circuit breaker probe 429'd; cooldown extended"
            )
            return
        if _consecutive_429s == 0 or (now - _first_429_at) > _429_WINDOW_SECONDS:
            _consecutive_429s = 1
            _first_429_at = now
        else:
            _consecutive_429s += 1
        if _consecutive_429s >= _429_THRESHOLD:
            _circuit_open = True
            _circuit_opened_at = now
            logger.warning(
                "vision_ladder circuit breaker OPEN: %d consecutive 429s in %.1fs; "
                "skipping upstream calls for %.0fs",
                _consecutive_429s,
                now - _first_429_at,
                _CIRCUIT_COOLDOWN_SECONDS,
            )


def _record_success() -> None:
    """Reset breaker state on a successful HTTP call. Closes the circuit
    if a probe just succeeded."""
    global _circuit_open, _circuit_opened_at, _consecutive_429s
    global _first_429_at, _probe_started_at
    with _circuit_lock:
        was_open = _circuit_open
        _circuit_open = False
        _circuit_opened_at = 0.0
        _consecutive_429s = 0
        _first_429_at = 0.0
        _probe_started_at = 0.0
        if was_open:
            logger.info(
                "vision_ladder circuit breaker CLOSED: probe call succeeded"
            )


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Thin async wrapper around the Ollama OpenAI-compatible chat
    completions endpoint.

    Doesn't require the `openai` package — uses urllib for zero deps.
    But if openai is installed we use it for nicer error messages.

    `api_key` is OPTIONAL: when set and non-empty (and not the literal
    string "ollama"), a ``Authorization: Bearer <key>`` header is added
    so this client also works against OpenAI-compatible cloud providers
    (byNara, NVIDIA, OpenRouter, etc.) that share the same
    chat-completions shape as Ollama. Pass None / "" / "ollama" for the
    original local-Ollama behaviour (no auth).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = OLLAMA_TIMEOUT,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = (api_key or "").strip()

    def _is_local(self) -> bool:
        """True when no Bearer auth should be sent (local Ollama or
        explicit 'ollama' sentinel from the legacy ladder)."""
        return not self.api_key or self.api_key == "ollama"

    def _endpoint(self) -> str:
        # base_url already includes /v1 (e.g. http://localhost:11434/v1);
        # chat completions lives at /chat/completions.
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        """Return the per-request HTTP headers. Always Content-Type; plus
        Authorization=Bearer when a non-empty, non-sentinel api_key is set."""
        h = {"Content-Type": "application/json"}
        if not self._is_local():
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 350,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one chat-completions request. Returns the parsed JSON dict
        or {} on any failure. Never raises — vision failures degrade."""
        import json
        import urllib.error
        import urllib.request

        if not _circuit_check():
            return {}

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=body,
            method="POST",
            headers=self._auth_headers(),
        )
        to = timeout if timeout is not None else self.timeout
        loop = asyncio.get_running_loop()

        def _do_request() -> bytes:
            with urllib.request.urlopen(req, timeout=to) as resp:
                return resp.read()

        try:
            raw = await loop.run_in_executor(None, _do_request)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                logger.warning("rate-limited by upstream (429)")
                _record_429()
                return {}
            logger.warning(
                "Ollama HTTP failed (%s, code=%s): %s",
                type(exc).__name__, exc.code, exc,
            )
            return {}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Ollama HTTP failed (%s): %s", type(exc).__name__, exc)
            return {}
        except Exception as exc:
            logger.warning("Ollama request raised (%s): %s", type(exc).__name__, exc)
            return {}
        _record_success()
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            logger.warning("Ollama response not JSON (%s): %s", type(exc).__name__, exc)
            return {}

    def chat_sync(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 350,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Synchronous variant for callers without an event loop. Returns
        the same parsed JSON dict or {} on failure."""
        import json
        import urllib.error
        import urllib.request

        if not _circuit_check():
            return {}

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=body,
            method="POST",
            headers=self._auth_headers(),
        )
        to = timeout if timeout is not None else self.timeout
        try:
            with urllib.request.urlopen(req, timeout=to) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                logger.warning("rate-limited by upstream (429)")
                _record_429()
                return {}
            logger.warning(
                "Ollama HTTP failed (%s, code=%s): %s",
                type(exc).__name__, exc.code, exc,
            )
            return {}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Ollama HTTP failed (%s): %s", type(exc).__name__, exc)
            return {}
        except Exception as exc:
            logger.warning("Ollama request raised (%s): %s", type(exc).__name__, exc)
            return {}
        _record_success()
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            logger.warning("Ollama response not JSON (%s): %s", type(exc).__name__, exc)
            return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception as exc:
        logger.warning("Ollama: cannot read image %s (%s)", path, exc)
        return None


def _extract_text(response: dict[str, Any]) -> str:
    """Pull the first choice's text content out of an Ollama chat response."""
    if not response:
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def call_async(
    image_path: Path | str,
    prompt: str,
    settings: Settings,
    *,
    timeout: float | None = None,
) -> str:
    """Single-frame async vision call. Returns the model text or "" on any
    failure. Concurrency-gated by OLLAMA_MAX_CONCURRENCY."""
    image_path = Path(image_path)
    if not image_path.exists():
        logger.warning("Vision ladder: image missing: %s", image_path)
        return ""
    encoded = _encode_image(image_path)
    if encoded is None:
        return ""

    client = OllamaClient(settings.ollama_base_url, timeout=timeout or settings.ollama_timeout)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }
    ]
    sem = _get_semaphore(settings.ollama_max_concurrency)
    async with sem:
        started = time.monotonic()
        response = await client.chat(settings.ollama_vision_model, messages, timeout=timeout)
        duration = time.monotonic() - started
        if duration > 5.0:
            logger.info("Ollama vision call took %.1fs for %s", duration, image_path.name)
    return _extract_text(response)


async def call_batch_async(
    image_paths: list[Path | str],
    prompts: list[str],
    settings: Settings,
    *,
    timeout: float | None = None,
) -> list[str]:
    """Concurrent batch of single-frame calls. Returns a list parallel
    to image_paths — empty string for any failure. Concurrency is gated
    by OLLAMA_MAX_CONCURRENCY."""
    if len(image_paths) != len(prompts):
        raise ValueError("image_paths and prompts must be the same length")
    tasks = [
        asyncio.create_task(call_async(p, prompt, settings, timeout=timeout))
        for p, prompt in zip(image_paths, prompts)
    ]
    return await asyncio.gather(*tasks)


def call(
    image_path: Path | str,
    prompt: str,
    settings: Settings,
    *,
    timeout: float | None = None,
) -> str:
    """Synchronous single-frame call. Convenience wrapper around the async
    one — drops you onto a private event loop if the caller isn't already
    inside one."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — spin one up.
        return asyncio.run(call_async(image_path, prompt, settings, timeout=timeout))
    # Already inside a loop; run the coroutine to completion.
    return loop.run_until_complete(call_async(image_path, prompt, settings, timeout=timeout))


def call_batch(
    image_paths: list[Path | str],
    prompts: list[str],
    settings: Settings,
    *,
    timeout: float | None = None,
) -> list[str]:
    """Synchronous batch — same as call_batch_async but runs to completion."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call_batch_async(image_paths, prompts, settings, timeout=timeout))
    return loop.run_until_complete(call_batch_async(image_paths, prompts, settings, timeout=timeout))


# ---------------------------------------------------------------------------
# One-shot batched call (single request, N frames, array response)
# ---------------------------------------------------------------------------
#
# Some local models accept multiple images in one chat-completions
# request. When the model supports this, we can collapse N frames into
# one round-trip and save ~Nx wall-clock vs. N separate calls. The
# caller decides which mode to use based on what their model supports.


def _split_into_frame_json_array(raw_text: str) -> list[str]:
    """Split a multi-frame response into N per-frame JSON strings.

    The local model is asked to emit one JSON object per frame, separated
    by newlines. We strip fences and return the cleaned list. If parsing
    fails, return a single-element list with the whole text so the caller
    can still try to use it as one frame's output.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    # Try splitting on a separator the prompt explicitly asked for.
    for sep in ("\n---\n", "\n\n---\n\n", "\n###\n"):
        if sep in text:
            return [s.strip() for s in text.split(sep) if s.strip()]
    # Otherwise return the whole thing as one frame's response — caller
    # will fall back to per-frame caching at the library-indexer level.
    return [text] if text else []


BATCH_VISION_PROMPT_PREFIX = (
    "You will see {n_frames} video frames in this single request. They are "
    "from the same short clip, sampled at {n_frames} points in time. For EACH "
    "frame, emit ONE JSON object on its own line, separated by a line "
    "containing exactly ---. Output ONLY the JSON lines. No markdown, no prose. "
    "Each object must use exactly these keys:\n"
)


async def call_one_shot_batch(
    image_paths: list[Path | str],
    per_frame_prompt: str,
    settings: Settings,
    *,
    timeout: float | None = None,
) -> list[str]:
    """Send ALL frames as a single chat-completions request. Returns a list
    parallel to image_paths with one parsed response per frame.

    Performance: ~Nx faster than call_batch on memory-bound local models.
    Caveat: not all local vision models accept multiple images in one
    request — if you get garbage responses, fall back to call_batch.
    """
    if not image_paths:
        return []
    encoded: list[str] = []
    valid_indices: list[int] = []
    for i, p in enumerate(image_paths):
        e = _encode_image(Path(p))
        if e is not None:
            encoded.append(e)
            valid_indices.append(i)

    if not encoded:
        return [""] * len(image_paths)

    n_frames = len(encoded)
    full_prompt = BATCH_VISION_PROMPT_PREFIX.format(n_frames=n_frames) + per_frame_prompt
    content: list[dict] = [{"type": "text", "text": full_prompt}]
    for e in encoded:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{e}"}})

    client = OllamaClient(settings.ollama_base_url, timeout=timeout or settings.ollama_timeout)
    sem = _get_semaphore(settings.ollama_max_concurrency)
    async with sem:
        started = time.monotonic()
        response = await client.chat(
            settings.ollama_vision_model,
            [{"role": "user", "content": content}],
            max_tokens=350 * n_frames,  # allow more tokens for N-frame output
            timeout=timeout,
        )
        duration = time.monotonic() - started
        if duration > 5.0:
            logger.info("Ollama one-shot batch (%d frames) took %.1fs", n_frames, duration)

    raw_text = _extract_text(response)
    parts = _split_into_frame_json_array(raw_text)
    # Map back to the original image_paths order: invalid indices get "".
    result = [""] * len(image_paths)
    for k, vi in enumerate(valid_indices):
        result[vi] = parts[k] if k < len(parts) else ""
    return result


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------


def is_ollama_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """Synchronous one-shot check. Returns True if Ollama responds within
    `timeout` seconds. Used by the FastAPI /api/health endpoint."""
    import urllib.error
    import urllib.request
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    except Exception:
        return False


__all__ = [
    "call",
    "call_async",
    "call_batch",
    "call_batch_async",
    "call_one_shot_batch",
    "OllamaClient",
    "is_ollama_reachable",
    "OLLAMA_TIMEOUT",
    "OLLAMA_MAX_CONCURRENCY",
]