"""Local YouTube cookie storage for yt-dlp.

The app avoids scraping Chrome's locked cookie database during renders. Users can
export YouTube cookies from a browser extension and save them once as a Netscape
cookies.txt file that yt-dlp can reuse.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.config import Settings

AUTH_COOKIE_NAMES = {
    "SID",
    "HSID",
    "SSID",
    "APISID",
    "SAPISID",
    "LOGIN_INFO",
    "__Secure-1PSID",
    "__Secure-3PSID",
    "__Secure-1PAPISID",
    "__Secure-3PAPISID",
    "__Secure-1PSIDTS",
    "__Secure-3PSIDTS",
    "__Secure-3PSIDCC",
}


def save_youtube_cookies(raw: str, settings: Settings) -> dict[str, int | str]:
    content, cookie_count, auth_count = normalize_youtube_cookies(raw)
    if auth_count == 0:
        raise ValueError("No logged-in YouTube auth cookies found")
    target = settings.ytdlp_cookies_file or (settings.data_dir / "youtube-cookies.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(target)
    return {"path": str(target), "cookies": cookie_count, "auth_cookies": auth_count}


def normalize_youtube_cookies(raw: str) -> tuple[str, int, int]:
    text = raw.strip()
    if not text:
        raise ValueError("Cookie input is empty")
    if text.startswith("[") or text.startswith("{"):
        cookies = _json_cookies(text)
        return _json_to_netscape(cookies)
    if _looks_like_netscape(text):
        return _clean_netscape(text)
    return _header_to_netscape(text)


def _json_cookies(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, dict) and isinstance(payload.get("cookies"), list):
        payload = payload["cookies"]
    if not isinstance(payload, list):
        raise ValueError("Cookie JSON must be a list, or an object with a cookies list")
    return [cookie for cookie in payload if isinstance(cookie, dict)]


def _json_to_netscape(cookies: list[dict[str, Any]]) -> tuple[str, int, int]:
    lines = _header_lines()
    cookie_count = 0
    auth_count = 0
    for cookie in cookies:
        domain = str(cookie.get("domain") or "")
        name = str(cookie.get("name") or "")
        value = str(cookie.get("value") or "")
        if not domain or not name or not value or "youtube.com" not in domain:
            continue
        include_subdomains = "TRUE" if domain.startswith(".") or not cookie.get("hostOnly", False) else "FALSE"
        path = str(cookie.get("path") or "/")
        secure = "TRUE" if cookie.get("secure", False) else "FALSE"
        expires = int(float(cookie.get("expirationDate") or _default_expiry()))
        lines.append("\t".join([domain, include_subdomains, path, secure, str(expires), name, value]))
        cookie_count += 1
        auth_count += int(name in AUTH_COOKIE_NAMES)
    return "\n".join(lines) + "\n", cookie_count, auth_count


def _looks_like_netscape(text: str) -> bool:
    return "Netscape HTTP Cookie File" in text or any(len(line.split("\t")) >= 7 for line in text.splitlines())


def _clean_netscape(text: str) -> tuple[str, int, int]:
    lines = _header_lines()
    cookie_count = 0
    auth_count = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, include_subdomains, path, secure, expires, name, value = parts[:7]
        if "youtube.com" not in domain or not name or not value:
            continue
        lines.append("\t".join([domain, include_subdomains, path, secure, expires, name, value]))
        cookie_count += 1
        auth_count += int(name in AUTH_COOKIE_NAMES)
    return "\n".join(lines) + "\n", cookie_count, auth_count


def _header_to_netscape(text: str) -> tuple[str, int, int]:
    lines = _header_lines()
    cookie_count = 0
    auth_count = 0
    for part in text.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            continue
        lines.append(
            "\t".join([".youtube.com", "TRUE", "/", "TRUE", str(_default_expiry()), name, value])
        )
        cookie_count += 1
        auth_count += int(name in AUTH_COOKIE_NAMES)
    return "\n".join(lines) + "\n", cookie_count, auth_count


def _header_lines() -> list[str]:
    return [
        "# Netscape HTTP Cookie File",
        "# Generated locally for yt-dlp. Treat this file as a secret.",
    ]


def _default_expiry() -> int:
    return int(time.time()) + 365 * 24 * 60 * 60
