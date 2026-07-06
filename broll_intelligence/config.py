"""Settings for the broll_intelligence package.

Local-only as of the all-local rewrite. Mirrors app/config.py so the same
.env keys (CAMPEDITOR_DATA_DIR, BROLL_INTELLIGENCE_*, OLLAMA_*) are read
out of C:\\campeditor\\.env without any dependency on app.*.

Package-specific keys:
  * BROLL_INTELLIGENCE_LIBRARY_DIR — root folder of the local B-roll library
    to index. Defaults to <data_dir>/broll_library.
  * BROLL_INTELLIGENCE_INDEX_PATH — absolute path of the JSON cache file.
    Defaults to <data_dir>/cache/broll_intelligence_index.json.
  * BROLL_INTELLIGENCE_FFMPEG_PATH / FFPROBE_PATH — explicit binaries. If
    unset, falls back to FFMPEG_PATH / FFPROBE_PATH and finally to
    "ffmpeg"/"ffprobe" on PATH.

The rest (Ollama URL / model / timeout / concurrency) is intentionally
identical to app/config.py so the standalone package can be A/B-tested
against the production pipeline without duplicating configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_LIBRARY_DIR = _DEFAULT_DATA_DIR / "broll_library"
_DEFAULT_INDEX_PATH = _DEFAULT_DATA_DIR / "cache" / "broll_intelligence_index.json"


class Settings(BaseSettings):
    """Pydantic-settings — reads CWD/.env (same .env as campeditor)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---------- package-specific knobs ----------
    data_dir: Path = Field(default=_DEFAULT_DATA_DIR, validation_alias="CAMPEDITOR_DATA_DIR")
    library_dir: Path = Field(
        default=_DEFAULT_LIBRARY_DIR, validation_alias="BROLL_INTELLIGENCE_LIBRARY_DIR"
    )
    index_path: Path = Field(
        default=_DEFAULT_INDEX_PATH, validation_alias="BROLL_INTELLIGENCE_INDEX_PATH"
    )
    ffmpeg_path: str = Field(default="", validation_alias="BROLL_INTELLIGENCE_FFMPEG_PATH")
    ffprobe_path: str = Field(default="", validation_alias="BROLL_INTELLIGENCE_FFPROBE_PATH")

    # ---------- Ollama (the only vision provider now) ----------
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_vision_model: str = Field(
        default="llava:13b", validation_alias="OLLAMA_VISION_MODEL"
    )
    ollama_timeout: float = Field(default=60.0, validation_alias="OLLAMA_TIMEOUT")
    ollama_max_concurrency: int = Field(
        default=2, validation_alias="OLLAMA_MAX_CONCURRENCY", ge=1, le=8
    )

    def resolved_ffmpeg(self) -> str:
        """BROLL_INTELLIGENCE_* first, else FFMPEG_PATH from env, else 'ffmpeg'."""
        if self.ffmpeg_path.strip():
            return self.ffmpeg_path
        import os
        return os.environ.get("FFMPEG_PATH", "ffmpeg")

    def resolved_ffprobe(self) -> str:
        if self.ffprobe_path.strip():
            return self.ffprobe_path
        import os
        return os.environ.get("FFPROBE_PATH", "ffprobe")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()