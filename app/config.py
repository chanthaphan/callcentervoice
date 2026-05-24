from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    watch_folder: Path = Path("voice")
    data_dir: Path = Path("data")
    max_parallel_files: int = Field(default=3, ge=1, le=16)
    auto_process_on_start: bool = True
    scan_interval_seconds: int = Field(default=30, ge=5)

    transcribe_provider: str = "mock"
    openai_api_key: str | None = None
    openai_transcribe_model: str = "whisper-1"
    openai_realtime_transcribe_model: str = "gpt-realtime-whisper"
    openai_realtime_transcription_delay: str = "minimal"
    openai_chunking_strategy: str = "auto"
    openai_transcribe_language: str | None = None
    openai_request_timeout_seconds: int = Field(default=600, ge=30)
    audio_transcode_bitrate: str = "64k"
    local_diarization_strategy: str = "alternating"
    realtime_sample_rate: int = Field(default=24000, ge=8000)
    realtime_chunk_seconds: int = Field(default=20, ge=1, le=60)
    realtime_chunk_timeout_seconds: int = Field(default=60, ge=5)
    realtime_partial_update_every_segments: int = Field(default=3, ge=0, le=100)

    llm_provider: str = "mock"
    analysis_language: str = "English"
    llm_request_timeout_seconds: int = Field(default=300, ge=30)
    openai_model: str = "gpt-4.1-mini"
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-12-01-preview"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"

    def absolute_watch_folder(self) -> Path:
        return self.watch_folder.expanduser().resolve()

    def absolute_data_dir(self) -> Path:
        return self.data_dir.expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
