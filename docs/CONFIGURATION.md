# Configuration

All settings are read from `.env` at startup (via `pydantic-settings`). Every value has a default and the service starts with no keys set (using `mock` providers). Variable names are case-insensitive.

A subset of settings is **also editable at runtime** from the Settings panel in the UI or `PATCH /api/config`; those changes persist to `DATA_DIR/config_overrides.json` and are re-applied on the next start. See [Runtime-configurable settings](#runtime-configurable-settings).

---

## Server

| Variable | Default | Description |
|---|---|---|
| `APP_HOST` | `127.0.0.1` | Documented bind address. The actual bind is set by how you launch uvicorn (`uvicorn app.main:app --host … --port …`). |
| `APP_PORT` | `8000` | Documented port (see above). |
| `APP_API_KEY` | — | Optional shared secret. When set, every `/api/*` request must present it via the `X-API-Key` header or an `?api_key=` query param. Leave blank for local-only use. |
| `WATCH_FOLDER` | `voice` | Folder scanned for audio files, **recursively** (subfolders included). |
| `DATA_DIR` | `data` | Directory for the SQLite store (`calls.db`) and `config_overrides.json`. |
| `MAX_PARALLEL_FILES` | `3` | Concurrent processing workers (1–16). Applied at startup; change + restart, or `--workers` in the CLI. |
| `AUTO_PROCESS_ON_START` | `true` | If `true`, discovered files are processed automatically. If `false`, they are added as **pending** and you start them manually. |
| `SCAN_INTERVAL_SECONDS` | `30` | Background folder re-scan interval (≥5). |

## Transcription

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_PROVIDER` | `mock` | `mock`, `openai`, `openai_realtime`, or `azure_speech`. |
| `AUDIO_TRANSCODE_BITRATE` | `64k` | ffmpeg MP3 bitrate when converting/compressing. |
| `LOCAL_DIARIZATION_STRATEGY` | `alternating` | For the `openai`/`realtime` providers when the model gives no speakers: `alternating` (A/B by turn), `channel` (stereo split), `none`. |

### `openai` / Azure OpenAI batch

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key (public API). |
| `OPENAI_TRANSCRIBE_MODEL` | `whisper-1` | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`. |
| `OPENAI_TRANSCRIBE_LANGUAGE` | `th` | BCP-47/ISO code, or blank/`auto` to detect. |
| `OPENAI_CHUNKING_STRATEGY` | `auto` | Server-side chunking hint for the diarize model (`auto`/`server`). |
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | `600` | Request timeout. |
| `AZURE_OPENAI_TRANSCRIBE_ENDPOINT` | — | If set (with `AZURE_OPENAI_API_KEY`), routes the `openai` provider to **Azure OpenAI** audio transcriptions instead of public OpenAI. |
| `AZURE_OPENAI_TRANSCRIBE_API_VERSION` | `2025-03-01-preview` | API version for the Azure OpenAI transcription endpoint. |

Per-model request params are sent automatically: `whisper-1` → `verbose_json` + segment timestamps; `*-transcribe-diarize` → `diarized_json` + chunking; `gpt-4o[-mini]-transcribe` → plain `json` (text only, no timestamps → weak diarization).

### `openai_realtime`

| Variable | Default | Description |
|---|---|---|
| `OPENAI_REALTIME_TRANSCRIBE_MODEL` | `gpt-realtime-whisper` | Realtime model. |
| `OPENAI_REALTIME_TRANSCRIPTION_DELAY` | `minimal` | Latency hint. |
| `REALTIME_SAMPLE_RATE` | `24000` | PCM sample rate streamed to the socket. |
| `REALTIME_CHUNK_SECONDS` | `20` | Audio chunk size. |
| `REALTIME_CHUNK_TIMEOUT_SECONDS` | `60` | Per-chunk response timeout. |
| `REALTIME_PARTIAL_UPDATE_EVERY_SEGMENTS` | `3` | How often partial transcript updates are pushed. |

### `azure_speech` — Azure AI Speech Fast Transcription

Synchronous Fast Transcription API (no polling). Recommended for Thai diarized calls.

| Variable | Default | Description |
|---|---|---|
| `AZURE_SPEECH_API_KEY` | — | Speech/Cognitive Services resource key. |
| `AZURE_SPEECH_REGION` | `eastus2` | Region (used to build the fallback host). |
| `AZURE_SPEECH_ENDPOINT` | — | Full endpoint from the portal's *Keys and Endpoint* page, e.g. `https://<resource>.cognitiveservices.azure.com/`. **Required for custom-subdomain resources** (the regional `*.stt.speech.microsoft.com` host 404s for them). If blank, falls back to `https://<region>.stt.speech.microsoft.com`. |
| `AZURE_SPEECH_LANGUAGE` | `th-TH` | Primary BCP-47 locale. |
| `AZURE_SPEECH_DIARIZATION` | `diarization` | `diarization` (AI splits speakers on mono/mixed audio), `channel` (stereo: agent = ch0, customer = ch1), or `none`. |
| `AZURE_SPEECH_LANGUAGE_CANDIDATES` | — | Comma-separated locales for per-phrase language ID, e.g. `th-TH,en-US`. Overrides `AZURE_SPEECH_LANGUAGE`. |
| `AZURE_SPEECH_MAX_SPEAKERS` | `2` | Max speakers the diarizer may distinguish (2–36). Agent + customer = 2. |

Requests are retried with exponential backoff on HTTP 429/5xx and timeouts.

## LLM Analysis

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, `openai`, `azure_openai`, `anthropic`. |
| `ANALYSIS_LANGUAGE` | `Thai` | Language for all free-text output (summary, topics, etc.). |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `300` | LLM call timeout. |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model for the `openai` provider. |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI key. |
| `AZURE_OPENAI_ENDPOINT` | — | e.g. `https://<resource>.openai.azure.com/`. |
| `AZURE_OPENAI_DEPLOYMENT` | — | Deployment name (used as the model). |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | Chat-completions API version. |
| `ANTHROPIC_API_KEY` | — | Anthropic key. |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-latest` | `claude-haiku-4-5-20251001` for throughput, `claude-sonnet-4-6` for quality. |

GPT-5 / o-series models only accept the default temperature, which is handled automatically; other models use `temperature=0`. LLM calls use `max_retries=4`.

## Analysis add-ons — products, PII & KB verification

| Variable | Default | Description |
|---|---|---|
| `PRODUCT_KB_DIR` | `data/prod_kb/Credit-Cards` | BBL product knowledge base. Credit-card products are scoped to these titles; non-BBL cards bucket to `อื่นๆ`. |
| `LLM_PII_REDACTION` | `false` | After the regex floor, use the LLM to mask context PII (names, addresses, birth dates) before diarization/analysis. One extra LLM call. |
| `KB_VERIFICATION` | `false` | Verify the staff's factual claims against the product KB page (`supported` / `not_found` / `contradicts`). One extra LLM call; runs only when a BBL product is detected. |

All three need a real `LLM_PROVIDER` (no-ops under `mock`) and apply to future analyses — **Re-analyze** existing calls to apply them. Full details: **[ANALYTICS_AND_KB.md](ANALYTICS_AND_KB.md)**.

## Azure AI Language enrichment (optional)

Runs after the LLM step to add name-level PII redaction, per-utterance sentiment, and conversation summarization. **Requires a separate Azure AI Language resource.** Best-effort — if it fails or isn't configured, the pipeline continues.

| Variable | Default | Description |
|---|---|---|
| `AZURE_LANGUAGE_ENABLED` | `true` | Master toggle. Skipped entirely if no endpoint/key resolves. |
| `AZURE_LANGUAGE_ENDPOINT` | — | Language resource endpoint, `https://<language-resource>.cognitiveservices.azure.com/`. If blank, reuses `AZURE_OPENAI_TRANSCRIBE_ENDPOINT`. |
| `AZURE_LANGUAGE_API_KEY` | — | Language resource key. If blank, reuses `AZURE_OPENAI_API_KEY`. |
| `AZURE_LANGUAGE_PII_REDACTION` | `true` | Redact names/phones/IDs in stored transcripts. |
| `AZURE_LANGUAGE_SENTIMENT` | `true` | Per-utterance positive/negative/neutral. |
| `AZURE_LANGUAGE_SUMMARIZATION` | `true` | Issue / resolution / narrative / follow-up tasks. English fully supported; Thai degrades gracefully. |

> **PII redaction layers.** The local regex floor (always on, no dependency) masks 9+ digit runs **and emails** inside transcription. For names/addresses/birth dates, enable either the local LLM layer (`LLM_PII_REDACTION`) or Azure Language PII (`AZURE_LANGUAGE_PII_REDACTION`). With neither enabled, names are not masked in the transcript (the "Call Summary" card's detected-PII line still lists entities the LLM noticed).

## Runtime-configurable settings

These can be changed live from the Settings panel or `PATCH /api/config` (no restart), and persist to `DATA_DIR/config_overrides.json`:

`transcribe_provider`, `openai_transcribe_model`, `openai_transcribe_language`, `openai_chunking_strategy`, `audio_transcode_bitrate`, `azure_speech_language`, `azure_speech_diarization`, `azure_speech_max_speakers`, `llm_provider`, `openai_model`, `analysis_language`, `auto_process_on_start`, `max_parallel_files`, `azure_language_enabled`, `azure_language_pii_redaction`, `azure_language_sentiment`, `azure_language_summarization`, `llm_pii_redaction`, `kb_verification`.

Notes:
- **API keys/endpoints are never runtime-editable** and never sent to the browser — set them in `.env`.
- `max_parallel_files` is applied at startup (the worker pool is created once); change it then restart.
- If `config_overrides.json` exists, it takes precedence over `.env` for the fields above. To return to pure `.env` config, delete that file. Pick one workflow (env *or* UI) to avoid drift.
