from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fields that can be changed at runtime via /api/config
CONFIGURABLE_FIELDS: tuple[str, ...] = (
    "transcribe_provider",
    "openai_transcribe_model",
    "openai_transcribe_language",
    "openai_chunking_strategy",
    "audio_transcode_bitrate",
    "azure_speech_language",
    "azure_speech_diarization",
    "azure_speech_max_speakers",
    "llm_provider",
    "openai_model",
    "analysis_language",
    "auto_process_on_start",
    "max_parallel_files",
    "azure_language_enabled",
    "azure_language_pii_redaction",
    "azure_language_sentiment",
    "azure_language_summarization",
    "llm_pii_redaction",
    "kb_verification",
)


def apply_overrides(settings: "Settings", overrides: dict[str, Any]) -> None:
    # Mutates the process-wide Settings singleton in place. Worker threads read these
    # attributes while a PATCH /api/config runs; we rely on the GIL making each scalar
    # assignment atomic rather than locking. A concurrent multi-field PATCH may be seen
    # half-applied by an in-flight job, which is acceptable for these tuning knobs.
    for key, value in overrides.items():
        if key not in CONFIGURABLE_FIELDS or not hasattr(settings, key):
            continue
        current = getattr(settings, key)
        try:
            if isinstance(current, bool):
                if isinstance(value, str):
                    setattr(settings, key, value.lower() not in ("false", "0", "no", ""))
                else:
                    setattr(settings, key, bool(value))
            elif isinstance(current, int):
                setattr(settings, key, int(value))
            else:
                setattr(settings, key, value)
        except (TypeError, ValueError):
            pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    # Optional shared secret. When set, all /api/* requests must present it via the
    # X-API-Key header (or ?api_key= for browser-native audio/download URLs). Blank = open.
    app_api_key: str | None = None
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
    analysis_language: str = "Thai"
    # BBL credit-card product knowledge base; analysis scopes products to these.
    product_kb_dir: str = "data/prod_kb/Credit-Cards"
    # Verify the staff's factual statements against the product KB (extra LLM call).
    kb_verification: bool = False
    llm_request_timeout_seconds: int = Field(default=300, ge=30)
    openai_model: str = "gpt-4.1-mini"
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-12-01-preview"
    # Separate Azure endpoint for audio/transcriptions (can differ from chat completions endpoint)
    azure_openai_transcribe_endpoint: str | None = None
    azure_openai_transcribe_api_version: str = "2025-03-01-preview"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"

    # Azure Language Service (post-processing enrichment)
    azure_language_enabled: bool = False
    azure_language_pii_redaction: bool = True
    azure_language_sentiment: bool = True
    azure_language_summarization: bool = True
    # Leave blank to reuse azure_openai_api_key / azure_openai_transcribe_endpoint
    azure_language_endpoint: str | None = None
    azure_language_api_key: str | None = None

    # LLM-based PII redaction: after the deterministic regex floor, ask the analysis
    # LLM to locate context-dependent PII (names, addresses, birth dates) and mask it
    # before diarization/analysis. Uses the configured LLM provider; off by default.
    llm_pii_redaction: bool = False

    # Azure AI Speech Service (TRANSCRIBE_PROVIDER=azure_speech)
    azure_speech_api_key: str | None = None
    azure_speech_region: str = "eastus2"
    # Full resource endpoint from the portal's "Keys and Endpoint" page, e.g.
    # https://<resource>.cognitiveservices.azure.com/ . Required for custom-subdomain
    # resources. If blank, falls back to the regional host {region}.stt.speech.microsoft.com.
    azure_speech_endpoint: str | None = None
    azure_speech_language: str = "th-TH"
    # Speaker separation strategy for azure_speech:
    #   diarization — AI-based (works on mono/mixed audio, labels Speaker 1 / Speaker 2)
    #   channel     — stereo-channel split (agent on ch0, customer on ch1; requires stereo recording)
    #   none        — no speaker separation (single "Unknown" speaker, LLM assigns roles later)
    azure_speech_diarization: str = "diarization"
    # Comma-separated BCP-47 locales for language identification, e.g. "th-TH,en-US".
    # When set, overrides azure_speech_language; the service detects the locale per phrase.
    azure_speech_language_candidates: str | None = None
    # Max speakers the diarizer may distinguish (used when azure_speech_diarization=diarization).
    # For agent+customer calls, 2 is correct. Azure supports 2–36.
    azure_speech_max_speakers: int = Field(default=2, ge=2, le=36)

    def absolute_watch_folder(self) -> Path:
        return self.watch_folder.expanduser().resolve()

    def absolute_data_dir(self) -> Path:
        return self.data_dir.expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
