"""Convert browser-extension JSON cookies to yt-dlp's Netscape cookies.txt.

Usage:
    .\.venv\Scripts\python.exe scripts\cookies_json_to_netscape.py input.json data\youtube-cookies.txt

You can also pipe JSON through stdin:
    Get-Content input.json | .\.venv\Scripts\python.exe scripts\cookies_json_to_netscape.py - data\youtube-cookies.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: cookies_json_to_netscape.py <input.json|-> <output.txt>")
    raw = sys.stdin.read() if sys.argv[1] == "-" else Path(sys.argv[1]).read_text(encoding="utf-8")
    cookies = json.loads(raw)
    if not isinstance(cookies, list):
        raise SystemExit("Cookie JSON must be a list")
    output = Path(sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_to_netscape(cookies), encoding="utf-8")
    print(f"Wrote {output}")


def _to_netscape(cookies: list[dict[str, Any]]) -> str:
    lines = [
        "# Netscape HTTP Cookie File",
        "# Generated locally for yt-dlp. Treat this file as a secret.",
    ]
    for cookie in cookies:
        domain = str(cookie.get("domain") or "")
        name = str(cookie.get("name") or "")
        value = str(cookie.get("value") or "")
        if not domain or not name:
            continue
        include_subdomains = "TRUE" if domain.startswith(".") or not cookie.get("hostOnly", False) else "FALSE"
        path = str(cookie.get("path") or "/")
        secure = "TRUE" if cookie.get("secure", False) else "FALSE"
        expires = int(float(cookie.get("expirationDate") or 0))
        lines.append("\t".join([domain, include_subdomains, path, secure, str(expires), name, value]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
