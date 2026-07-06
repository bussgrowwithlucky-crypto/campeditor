from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """campeditor settings.

    Vision + text routing now has TWO rungs:

      1. LLM provider (cloud, OpenAI-compatible) — controlled by LLM_API_KEY,
         LLM_BASE_URL, LLM_MODEL (text) and LLM_VISION_MODEL (vision). The
         same key + base URL drives both. This is the PRIMARY rung. Empty
         api_key disables the cloud rung.
      2. Ollama (LOCAL fallback, optional) — controlled by
         OLLAMA_BASE_URL, OLLAMA_VISION_MODEL and OLLAMA_TEXT_MODEL. Empty
         model names disable that local rung.

    When only cloud is configured (LLM_* set, OLLAMA_* empty) the cloud
    rung is the sole rung — failures surface immediately and the pipeline
    degrades to no-output rather than silently retrying locally.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---- Groq — Whisper transcription (audio -> text) ----
    # Cloud-only; no local faster-whisper on the server. The audio file is
    # POSTed to https://api.groq.com/openai/v1/audio/transcriptions and the
    # response is mapped into the campeditor Transcript schema. Empty api_key
    # disables transcription (the pipeline renders without captions).
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_api_key_2: str = Field(default="", validation_alias="GROQ_API_KEY_2")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1", validation_alias="GROQ_BASE_URL"
    )
    # whisper-large-v3-turbo is the fastest Whisper model on Groq; whisper-large-v3
    # is the higher-quality non-turbo sibling.
    groq_transcription_model: str = Field(
        default="whisper-large-v3-turbo", validation_alias="GROQ_TRANSCRIPTION_MODEL"
    )

    def groq_api_keys(self) -> list[str]:
        """Return the non-empty Groq API keys in priority order.

        Caller iterates and rotates on auth (401/403) or rate-limit (429)
        responses. Returns an empty list when neither key is set, which
        short-circuits the transcription ladder to a no-op.
        """
        return [k for k in (self.groq_api_key, self.groq_api_key_2) if k]

    # ---- Cloud LLM provider (title generation, clip selection, vision tagging) ----
    # Any OpenAI-compatible chat-completions endpoint. The same base_url +
    # api_key drive BOTH the text model (LLM_MODEL) and, when set, the
    # vision model (LLM_VISION_MODEL). Used as the PRIMARY rung of both the
    # text and vision ladders — Ollama below only fires if both are left
    # empty.
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", validation_alias="LLM_BASE_URL"
    )
    # Text model — title generation + clip selection.
    llm_model: str = Field(default="mistral-large", validation_alias="LLM_MODEL")
    # Vision model — frame tagging (B-roll scoring, replicate title reads).
    # Empty = no cloud vision rung (the ladder falls straight to Ollama).
    llm_vision_model: str = Field(default="", validation_alias="LLM_VISION_MODEL")

    # ---- Ollama (LOCAL fallback, OPTIONAL) ----
    # Ollama exposes an OpenAI-compatible chat-completions endpoint at
    # http://localhost:11434/v1. Pull a vision-capable model
    # (`ollama pull llava:13b` or `ollama pull llama3.2-vision:11b`) and a
    # text model (`ollama pull llama3.1:8b`). The vision + text models
    # can be the same if it's a multimodal checkpoint. Leave either model
    # empty to disable that rung entirely.
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_vision_model: str = Field(
        default="", validation_alias="OLLAMA_VISION_MODEL"
    )
    ollama_text_model: str = Field(
        default="", validation_alias="OLLAMA_TEXT_MODEL"
    )
    # How long a single Ollama HTTP call may take before we give up.
    # 20s is intentionally short — rate-limited providers (HTTP 429) would
    # otherwise burn the full timeout per retry; this fails fast so the
    # ladder can fall through to the next rung. Bump if you're on CPU-only
    # or running a larger vision model than the 13B default.
    ollama_timeout: float = Field(default=20.0, validation_alias="OLLAMA_TIMEOUT")
    # Maximum number of concurrent in-flight Ollama requests. The local
    # GPU saturates at ~1-2 simultaneous inference calls; running more
    # just adds memory pressure without throughput gain. The cloud-vision
    # ladder (byNara / OpenRouter / NVIDIA) scales much better — a higher
    # ceiling here also speeds up describe_spans and YouTube preview
    # scoring, which were the long poles on multi-span references.
    ollama_max_concurrency: int = Field(
        default=4, validation_alias="OLLAMA_MAX_CONCURRENCY", ge=1, le=8
    )
    # Whether to auto-pull the configured models on first start. Default
    # false because pulling 8 GB takes minutes and shouldn't surprise
    # anyone running the daemon cold. Set true for one-shot deploys
    # where you want the model ready before the first request.
    ollama_auto_pull: bool = Field(default=False, validation_alias="OLLAMA_AUTO_PULL")

    # ---- Local filesystem + ffmpeg ----
    ffmpeg_path: str = Field(default="ffmpeg", validation_alias="FFMPEG_PATH")
    ffprobe_path: str = Field(default="ffprobe", validation_alias="FFPROBE_PATH")
    ytdlp_cookies_file: Path | None = Field(default=None, validation_alias="YTDLP_COOKIES_FILE")
    ytdlp_cookies_from_browser: str = Field(
        default="chrome", validation_alias="YTDLP_COOKIES_FROM_BROWSER"
    )
    broll_library_dir: Path = Field(
        default=Path("data/broll_library"), validation_alias="BROLL_LIBRARY_DIR"
    )

    # ---- YouTube search ----
    # Pure yt-dlp ytsearch — no Google API key needed. The Data API path
    # was removed with the cloud rewrite.
    broll_query_max_words: int = Field(
        default=8, validation_alias="BROLL_QUERY_MAX_WORDS"
    )
    broll_local_match_threshold: float = Field(
        default=0.35, validation_alias="BROLL_LOCAL_MATCH_THRESHOLD"
    )

    # ---- Music + B-roll learning ----
    music_separation: bool = Field(default=True, validation_alias="MUSIC_SEPARATION")
    broll_learning_enabled: bool = Field(
        default=True, validation_alias="BROLL_LEARNING_ENABLED"
    )

    # ---- HTTP / runtime ----
    cors_allow_origins: str = Field(default="*", validation_alias="CORS_ALLOW_ORIGINS")
    data_dir: Path = Field(default=Path("data"), validation_alias="CAMPEDITOR_DATA_DIR")
    max_upload_mb: int = Field(default=500, validation_alias="CAMPEDITOR_MAX_UPLOAD_MB")
    worker_count: int = Field(default=2, validation_alias="CAMPEDITOR_WORKER_COUNT")
    variation_count: int = Field(default=1, validation_alias="VARIATION_COUNT")
    max_bulk_pairs: int = Field(default=100, validation_alias="MAX_BULK_PAIRS")
    auto_clip_scan_seconds: float = Field(
        default=600.0, validation_alias="CAMPEDITOR_AUTO_CLIP_SCAN_SECONDS"
    )
    auto_clip_target_seconds: float = Field(
        default=15.0, validation_alias="CAMPEDITOR_AUTO_CLIP_TARGET_SECONDS"
    )
    auto_clip_min_seconds: float = Field(
        default=8.0, validation_alias="CAMPEDITOR_AUTO_CLIP_MIN_SECONDS"
    )
    auto_clip_max_seconds: float = Field(
        default=30.0, validation_alias="CAMPEDITOR_AUTO_CLIP_MAX_SECONDS"
    )

    # ---- Intelligent selector v2 knobs (SPEC §12) ----
    intelligent_cinema_floor: float = Field(
        default=0.18, validation_alias="CAMPEDITOR_INTELLIGENT_CINEMA_FLOOR",
        ge=0.05, le=0.40,
    )
    intelligent_continuity_penalty_max: float = Field(
        default=-0.08, validation_alias="CAMPEDITOR_INTELLIGENT_CONTINUITY_PENALTY_MAX",
        ge=-0.20, le=0.0,
    )
    intelligent_continuity_cosine_threshold: float = Field(
        default=0.92, validation_alias="CAMPEDITOR_INTELLIGENT_CONTINUITY_COSINE_THRESHOLD",
        ge=0.70, le=0.99,
    )
    intelligent_frame_tag_prompt_version: int = Field(
        default=2, validation_alias="CAMPEDITOR_INTELLIGENT_FRAME_TAG_PROMPT_VERSION",
        ge=1,
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.broll_library_dir.mkdir(parents=True, exist_ok=True)
    return settings