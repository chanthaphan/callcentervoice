# Call Center Voice Analysis

FastAPI service that transcribes, diarizes, and analyzes recorded call center audio. Drop audio files into a folder, and the service produces a structured post-call report — sentiment, emotion journey, critical flags, credit card products, risks, and next actions — viewable in a browser UI.

## Features

- **Batch processing** — watches a folder and processes all audio files automatically on startup
- **Parallel execution** — configurable concurrency via `MAX_PARALLEL_FILES`
- **Live progress** — UI shows per-file stage (transcribing → analyzing → complete) with segment count updates
- **Three transcription providers** — `mock`, `openai` (Whisper batch), `openai_realtime` (streaming WebSocket)
- **Three LLM analysis providers** — `mock`, `openai`, `azure_openai`, `anthropic`
- **Broad audio support** — native WAV/MP3/M4A/OGG/WebM; other formats auto-converted via ffmpeg
- **Speaker diarization** — alternating or channel-based strategies; LLM classifies speakers as customer / staff
- **Download** — export any call as a plain-text report
- **In-memory record cache** — fast API reads regardless of the number of stored calls

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) — required for non-WAV formats and realtime PCM conversion

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The default `.env.example` uses `mock` providers — no API keys needed. Put audio files in `voice/` and click **Process**.

## Configuration

All settings are read from `.env`. Every value has a default and the service starts without any key set.

### Server

| Variable | Default | Description |
|---|---|---|
| `APP_HOST` | `127.0.0.1` | Bind address |
| `APP_PORT` | `8000` | Port |
| `WATCH_FOLDER` | `voice` | Folder scanned for audio files |
| `DATA_DIR` | `data` | Where call JSON records are stored |
| `MAX_PARALLEL_FILES` | `3` | Concurrent transcription jobs (1–16) |
| `AUTO_PROCESS_ON_START` | `true` | Scan and process on startup |
| `SCAN_INTERVAL_SECONDS` | `30` | Re-scan interval when auto-process is on |

### Transcription

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_PROVIDER` | `mock` | `mock`, `openai`, or `openai_realtime` |
| `OPENAI_API_KEY` | — | Required for `openai` and `openai_realtime` |
| `OPENAI_TRANSCRIBE_MODEL` | `whisper-1` | Model for the `openai` batch provider |
| `OPENAI_TRANSCRIBE_LANGUAGE` | `auto` | ISO 639-1 code, or `auto` to detect |
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | `600` | Batch transcription timeout |
| `AUDIO_TRANSCODE_BITRATE` | `64k` | ffmpeg bitrate when converting to MP3 |
| `LOCAL_DIARIZATION_STRATEGY` | `alternating` | `alternating`, `channel`, or `none` |
| `OPENAI_CHUNKING_STRATEGY` | `auto` | Chunking hint for diarization models |

#### Realtime provider

| Variable | Default | Description |
|---|---|---|
| `OPENAI_REALTIME_TRANSCRIBE_MODEL` | `gpt-realtime-whisper` | Realtime model name |
| `OPENAI_REALTIME_TRANSCRIPTION_DELAY` | `minimal` | Latency hint |
| `REALTIME_SAMPLE_RATE` | `24000` | PCM sample rate sent to the WebSocket |
| `REALTIME_CHUNK_SECONDS` | `5` | Size of each audio chunk in seconds |
| `REALTIME_CHUNK_TIMEOUT_SECONDS` | `60` | Per-chunk response timeout |
| `REALTIME_PARTIAL_UPDATE_EVERY_SEGMENTS` | `3` | How often to push partial transcript updates |

### Analysis (LLM)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, `openai`, `azure_openai`, or `anthropic` |
| `ANALYSIS_LANGUAGE` | `Thai` | Language for all free-text output fields |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `300` | LLM call timeout |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model for `openai` provider |
| `AZURE_OPENAI_API_KEY` | — | Azure credential |
| `AZURE_OPENAI_ENDPOINT` | — | Azure resource endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | — | Azure deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | Azure API version |
| `ANTHROPIC_API_KEY` | — | Anthropic credential |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-latest` | Model for `anthropic` provider |

## Transcription Providers

### `openai` — Whisper batch (fast, simple)

```env
TRANSCRIBE_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_TRANSCRIBE_MODEL=whisper-1
OPENAI_TRANSCRIBE_LANGUAGE=th
LOCAL_DIARIZATION_STRATEGY=alternating
```

Uploads the file in one request and receives timestamped segments. Speaker labels are assigned by the `LOCAL_DIARIZATION_STRATEGY`. For built-in diarization, switch to a diarization model:

```env
OPENAI_TRANSCRIBE_MODEL=gpt-4o-transcribe-diarize
OPENAI_CHUNKING_STRATEGY=auto
```

### `openai_realtime` — Streaming WebSocket (live progress)

```env
TRANSCRIBE_PROVIDER=openai_realtime
OPENAI_API_KEY=sk-...
OPENAI_REALTIME_TRANSCRIBE_MODEL=gpt-realtime-whisper
OPENAI_TRANSCRIBE_LANGUAGE=th
REALTIME_CHUNK_SECONDS=5
LOCAL_DIARIZATION_STRATEGY=alternating
```

Converts audio to PCM16 with ffmpeg and streams it through the OpenAI Realtime WebSocket in fixed-size chunks. Each chunk becomes a timestamped segment. If the file has two channels and `LOCAL_DIARIZATION_STRATEGY=channel`, the channels are transcribed in parallel and merged — useful when the call center telephony splits agent and customer onto separate channels.

## LLM Analysis Providers

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
```

### Azure OpenAI

```env
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### Anthropic

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

Use `claude-haiku-4-5-20251001` for maximum throughput on high-volume batches. Switch to `claude-sonnet-4-6` when output quality matters more than speed.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/calls` | List all call records |
| `GET` | `/api/calls/{id}` | Get one call record |
| `GET` | `/api/calls/{id}/audio` | Stream the original audio file |
| `GET` | `/api/calls/{id}/download.txt` | Download transcript + analysis as plain text |
| `POST` | `/api/process-folder` | Queue all audio files in a folder |
| `POST` | `/api/calls/{id}/reprocess` | Re-run transcription and analysis for one call |

### Process a folder

```bash
curl -X POST http://127.0.0.1:8000/api/process-folder \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

`force: true` reprocesses files that are already complete. Omit `folder` to use the configured `WATCH_FOLDER`.

### Reprocess one call

```bash
curl -X POST http://127.0.0.1:8000/api/calls/{call_id}/reprocess
```

## Supported Audio Formats

Direct upload (no conversion): `.flac` `.m4a` `.mp3` `.mp4` `.mpeg` `.mpga` `.ogg` `.wav` `.webm`

Auto-converted to MP3 via ffmpeg: `.3g2` `.3gp` `.aac` `.aif` `.aiff` `.amr` `.au` `.caf` `.m4b` `.m4p` `.m4r` `.mka` `.mkv` `.mov` `.mp2` `.mpg` `.oga` `.ogv` `.opus` `.wma`
