# Call Center Voice Analysis

A FastAPI service that transcribes, diarizes, and analyzes recorded call-center audio. Drop audio files into a folder (or point it at one) and it produces a structured post-call report — speaker-separated transcript, sentiment, emotion journey, critical flags, credit-card products, risks, and next actions — viewable in a browser UI or generated headless for large batches.

Built for a Thai bank's call-center QA, but provider-agnostic and language-agnostic.

## Highlights

- **Multiple transcription backends** — `mock`, OpenAI/Azure OpenAI batch (Whisper, GPT-4o-transcribe, GPT-4o-transcribe-diarize), OpenAI Realtime (streaming), and **Azure AI Speech Fast Transcription** with diarization.
- **LLM analysis** — `mock`, OpenAI, Azure OpenAI, or Anthropic. Produces a strict structured report and assigns speaker roles (customer vs. call-center staff), even for named speakers.
- **Two ways to run** — a browser UI for review, and a **headless batch CLI** for unattended bulk processing (designed for 10k+ files).
- **Recursive discovery** — scans nested folders (e.g. date folders `voice/2026-05-01/…`) and keeps the subpath in the display name.
- **PII redaction** — masks ID/phone/card numbers in stored transcripts; optional Azure AI Language enrichment adds name-level PII redaction, per-utterance sentiment, and summarization.
- **Resilient & scalable** — SQLite store (no full in-memory load), exponential-backoff retries on throttling/timeouts, resumable processing, configurable concurrency.
- **Runtime config** — change providers/models/options from the Settings panel (or `PATCH /api/config`) without editing files.

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) — required for non-WAV formats, oversized-file chunking, and realtime PCM conversion.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The defaults use `mock` providers — no API keys needed. Put audio in `voice/`, click **Refresh** to list them, then **▶ Start** (or **Process** for the whole folder).

To use real providers, edit `.env` (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)) and restart.

## Running

### Web UI

`uvicorn app.main:app --reload` (add `--host 0.0.0.0 --port 8000` to expose it). The UI lists sessions (paginated), shows a live processing stage strip, an audio timeline with speaker lanes, the speaker-separated transcript, profiles, sentiment/flags, and an emotion journey. A gear icon opens provider-aware Settings.

### Headless batch (no UI)

For bulk/unattended runs — e.g. 10,000+ recordings organized in date folders:

```bash
python -m app.batch /path/to/recordings --workers 8
```

| Flag | Purpose |
|---|---|
| `folder` | Folder to scan recursively (default: `WATCH_FOLDER`). |
| `--workers N` | Parallel workers (overrides `MAX_PARALLEL_FILES`). |
| `--reprocess` | Re-transcribe **and** re-analyze even if complete. |
| `--reanalyze` | Re-run analysis on the existing transcript only. |
| `--poll SECONDS` | Progress-report interval (default 5). |

It reuses the same config and store as the web app, logs progress + a final tally, **skips already-completed files** (resumable), and on `Ctrl+C` cancels queued files while letting in-flight ones finish. See [docs/BATCH.md](docs/BATCH.md).

## How it works

```
audio file ─▶ transcribe ─▶ diarize ─▶ analyze ─▶ [Azure Language] ─▶ redact PII ─▶ store
              (ASR)         (LLM)      (LLM, structured)  (optional)   (numbers)    (SQLite)
```

Each file flows through stages `queued → transcribing → diarizing → analyzing → enriching → complete` (or `failed`). The `BatchProcessor` runs files concurrently in a thread pool. Full architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Configuration

All settings come from `.env` (each has a default; the service starts with no keys set). A subset is also editable at runtime from the Settings panel / `PATCH /api/config` (persisted to `data/config_overrides.json`).

The most common knobs:

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_PROVIDER` | `mock` | `mock`, `openai`, `openai_realtime`, `azure_speech` |
| `LLM_PROVIDER` | `mock` | `mock`, `openai`, `azure_openai`, `anthropic` |
| `WATCH_FOLDER` | `voice` | Folder scanned for audio (recursive) |
| `MAX_PARALLEL_FILES` | `3` | Concurrent jobs (1–16) |
| `AUTO_PROCESS_ON_START` | `true` | Auto-process discovered files; if `false`, files appear as *pending* until started |
| `ANALYSIS_LANGUAGE` | `Thai` | Language for free-text output |
| `APP_API_KEY` | — | Optional shared secret guarding `/api/*` (see Security) |

**Full reference for every variable** (server, transcription, Azure Speech, LLM, Azure Language): **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

### Transcription providers (at a glance)

- **`azure_speech`** — Azure AI Speech Fast Transcription. Best for Thai diarization. Set `AZURE_SPEECH_API_KEY`, `AZURE_SPEECH_ENDPOINT` (custom-domain `https://<resource>.cognitiveservices.azure.com/`), `AZURE_SPEECH_DIARIZATION=diarization`.
- **`openai`** — OpenAI **or** Azure OpenAI batch. Supports `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`. Route to Azure by setting `AZURE_OPENAI_TRANSCRIBE_ENDPOINT` + `AZURE_OPENAI_API_KEY`.
- **`openai_realtime`** — streaming WebSocket; useful for live progress and channel-split telephony.
- **`mock`** — fake transcript for local testing.

> Note: `gpt-4o-transcribe`/`-mini-transcribe` return a single text block (no timestamps), so speaker separation is weak — prefer `whisper-1`, `gpt-4o-transcribe-diarize`, or `azure_speech` for diarized calls.

### PII redaction

Local redaction always masks 9+ digit runs (Thai national IDs, phone, card/account numbers → `[ปกปิด]`) in stored transcripts. For **name-level** PII plus per-utterance sentiment and summarization, enable Azure AI Language (`AZURE_LANGUAGE_ENABLED=true` + a Language resource). Details in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## API

Summary below; full reference with examples: **[docs/API.md](docs/API.md)**.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/calls?limit&offset&status` | Paginated list of lightweight summaries `{items, total, …}` |
| `GET` | `/api/calls/{id}` | One full record (transcript + analysis) |
| `GET` | `/api/calls/{id}/audio` | Stream the original audio |
| `GET` | `/api/calls/{id}/download.txt` | Text report (named after the audio file) |
| `POST` | `/api/process-folder` | Scan a folder and process (`{folder?, force?, reanalyze?}`) |
| `POST` | `/api/discover-folder` | Scan a folder and add files as *pending* (no processing) |
| `POST` | `/api/calls/{id}/start` · `/reprocess` · `/reanalyze` | Single-call actions |
| `POST` | `/api/calls/bulk` | Bulk `start` / `reanalyze` / `reprocess` / `delete` |
| `DELETE` | `/api/calls/{id}` | Delete a record |
| `GET`/`PATCH` | `/api/config` | Read / update runtime-configurable settings |

## Security

By default the app binds to `127.0.0.1` (local only). If you expose it (`--host 0.0.0.0`), set `APP_API_KEY`: every `/api/*` request must then send it via the `X-API-Key` header (the UI prompts once and remembers it) or an `?api_key=` query param (used for audio/download links). The UI shell and static assets stay open so the page can load and prompt.

## Storage

Records live in a SQLite database at `DATA_DIR/calls.db` (WAL mode), read lazily by id — so it scales to large datasets without loading everything into memory. On first run it auto-imports any legacy `DATA_DIR/calls/*.json` files. `DATA_DIR` and the audio in `voice/` are gitignored.

## Project layout

```
app/
  main.py        FastAPI app, routes, API-key guard, startup
  batch.py       Headless CLI (python -m app.batch)
  processor.py   BatchProcessor — discovery, queue, per-file pipeline
  audio.py       TranscriptionService — mock/openai/realtime/azure_speech
  agent.py       PostCallAgent — LLM diarization + structured analysis
  language.py    AzureLanguageService — optional enrichment
  enrichment.py  Maps roles/sentiment onto transcript segments
  redaction.py   Local number-PII masking
  retry.py       Exponential-backoff helper
  storage.py     SQLite CallStore
  models.py      Pydantic models
  config.py      Settings + runtime override handling
  static/index.html   Single-file browser UI
docs/            Architecture, configuration, API, and batch guides
tests/           Pytest suite
```

## Testing

```bash
pytest -q
```

## Supported audio formats

Direct upload (no conversion): `.flac` `.m4a` `.mp3` `.mp4` `.mpeg` `.mpga` `.ogg` `.wav` `.webm`

Auto-converted to MP3 via ffmpeg: `.3g2` `.3gp` `.aac` `.aif` `.aiff` `.amr` `.au` `.caf` `.m4b` `.m4p` `.m4r` `.mka` `.mkv` `.mov` `.mp2` `.mpg` `.oga` `.ogv` `.opus` `.wma`

Files over 25 MB are automatically chunked (10-minute segments) for the OpenAI batch provider.
