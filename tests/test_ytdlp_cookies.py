from pathlib import Path

from app.config import Settings
from app.replicate import _yt_dlp_command


def test_yt_dlp_uses_configured_cookie_file(tmp_path: Path) -> None:
    cookies = tmp_path / "youtube-cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    settings = Settings(groq_api_key="", llm_api_key="")
    settings.ytdlp_cookies_file = cookies

    command = _yt_dlp_command("https://youtube.com/shorts/example", tmp_path / "reference.mp4", settings)

    assert "--cookies" in command
    assert str(cookies) in command
    assert "--cookies-from-browser" not in command


def test_yt_dlp_falls_back_to_browser_cookies(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    settings = Settings(groq_api_key="", llm_api_key="")
    settings.ytdlp_cookies_file = missing
    settings.ytdlp_cookies_from_browser = "chrome"

    command = _yt_dlp_command("https://youtube.com/shorts/example", tmp_path / "reference.mp4", settings)

    assert "--cookies-from-browser" in command
    assert "chrome" in command
