"""Quick health scan of every AI provider/key the pipeline uses.

Run:  .venv\\Scripts\\python scripts\\check_providers.py

Each probe is a tiny request (1 short completion / 1 tiny image / 1 metadata
search) so the scan itself costs almost nothing. Exit code 0 when at least one
vision AND one text provider work; 1 otherwise.
"""
import base64
import io
import json
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def _tiny_png_b64() -> str:
    """A 8x8 red PNG, generated in-process so the scan needs no asset files."""
    width = height = 8
    row = b"\x00" + b"\xff\x00\x00" * width
    raw = row * height
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def _probe_chat(base_url: str, api_key: str, model: str, image: bool = False, timeout: float = 20.0) -> tuple[bool, str]:
    from openai import OpenAI

    if image:
        content = [
            {"type": "text", "text": "What color is this image? One word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_tiny_png_b64()}"}},
        ]
    else:
        content = "Reply with the single word OK."
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=10,
            temperature=0,
        )
        choice = response.choices[0] if response.choices else None
        text = ((choice.message.content if choice and choice.message else None) or "").strip()
        return (bool(text), text[:60] or "(empty response)")
    except Exception as e:
        return (False, f"{type(e).__name__}: {str(e)[:120]}")


def _probe_youtube_key(key: str, timeout: float = 15.0) -> tuple[bool, str]:
    url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(
        {"part": "snippet", "q": "test", "type": "video", "maxResults": 1, "key": key}
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read())
        return (bool(data.get("items")), "ok")
    except urllib.error.HTTPError as e:
        return (False, f"HTTP {e.code}")
    except Exception as e:
        return (False, f"{type(e).__name__}: {str(e)[:80]}")


def main() -> int:
    settings = get_settings()
    results: dict[str, tuple[bool, str]] = {}

    print("Scanning providers (tiny probe requests)...\n")

    # ── Vision providers ──────────────────────────────────────────────
    if settings.groq_api_key:
        results["vision/groq"] = _probe_chat(
            "https://api.groq.com/openai/v1", settings.groq_api_key, settings.groq_vision_model, image=True)
    nvidia_keys = [
        ("vision/nvidia#0", settings.nvidia_api_key),
        ("vision/nvidia#1", settings.nvidia_fallback_api_key),
        ("vision/nvidia#2", settings.nvidia_fallback_api_key_2),
        ("vision/nvidia#3", settings.nvidia_fallback_api_key_3),
    ]
    for name, key in nvidia_keys:
        if key:
            results[name] = _probe_chat(settings.nvidia_base_url, key, settings.nvidia_vision_model, image=True)
    if settings.gemini_api_key:
        results["vision/gemini"] = _probe_chat(
            settings.gemini_base_url, settings.gemini_api_key, settings.gemini_vision_model, image=True)
    if settings.ollama_base_url and settings.ollama_vision_model:
        results["vision/ollama(local)"] = _probe_chat(
            settings.ollama_base_url, "ollama", settings.ollama_vision_model, image=True, timeout=120.0)

    # ── Text providers ────────────────────────────────────────────────
    if settings.llm_api_key:
        results["text/llm-router"] = _probe_chat(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    if settings.groq_api_key:
        results["text/groq"] = _probe_chat(
            "https://api.groq.com/openai/v1", settings.groq_api_key, settings.groq_chat_fallback_model)
    if settings.ollama_base_url and settings.ollama_text_model:
        results["text/ollama(local)"] = _probe_chat(
            settings.ollama_base_url, "ollama", settings.ollama_text_model, timeout=120.0)

    # ── YouTube Data API keys ─────────────────────────────────────────
    for index, key in enumerate(settings.youtube_data_api_keys):
        results[f"youtube/data-api#{index}"] = _probe_youtube_key(key)

    width = max(len(name) for name in results) if results else 20
    vision_ok = text_ok = False
    for name, (ok, detail) in results.items():
        status = "OK    " if ok else "FAIL  "
        print(f"  {status}{name.ljust(width + 2)}{detail}")
        if ok and name.startswith("vision/"):
            vision_ok = True
        if ok and name.startswith("text/"):
            text_ok = True

    print()
    print(f"vision: {'available' if vision_ok else 'ALL DOWN'}   text: {'available' if text_ok else 'ALL DOWN'}")
    return 0 if (vision_ok and text_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
