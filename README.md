# Call Center Voice Analysis

Small FastAPI service for processing recorded call center WAV files.

It scans a folder, transcribes and diarizes each recording, runs a LangGraph post-call analysis agent, and shows the results in a browser UI with transcript timeline, sentiment flags, emotion journey, risks, topics, and next actions.

## Features

- Batch processing for every `.wav` or `.wave` file in a folder.
- Parallel execution controlled by `MAX_PARALLEL_FILES`.
- Pluggable transcription provider:
  - `mock` for local testing.
  - `openai` for OpenAI audio transcription.
- Broad audio intake:
  - Direct OpenAI upload formats: `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`.
  - Other common audio/video containers are converted to MP3 with `ffmpeg` before upload.
- Pluggable LangChain/LangGraph analysis model:
  - `mock`
  - `openai`
  - `azure_openai`
  - `anthropic`
- Static web UI at `/`.
- API results stored as JSON in `data/calls`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`, then click `Process`.

The default `.env.example` uses mock providers, so the service works without API keys. Put WAV files in `voice/` to simulate call batches.

## Cloud Configuration

OpenAI transcription, faster file mode:

```bash
TRANSCRIBE_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_TRANSCRIBE_MODEL=whisper-1
OPENAI_TRANSCRIBE_LANGUAGE=th
LOCAL_DIARIZATION_STRATEGY=alternating
```

`whisper-1` returns segment timestamps with `verbose_json`; the app labels speakers locally with a simple alternating speaker strategy. This is faster than `gpt-4o-transcribe-diarize`, but speaker labels are heuristic.

OpenAI Realtime Whisper transcription:

```bash
TRANSCRIBE_PROVIDER=openai_realtime
OPENAI_API_KEY=...
OPENAI_REALTIME_TRANSCRIBE_MODEL=gpt-realtime-whisper
OPENAI_TRANSCRIBE_LANGUAGE=th
REALTIME_CHUNK_SECONDS=10
LOCAL_DIARIZATION_STRATEGY=alternating
```

This streams each stored audio file through a Realtime transcription WebSocket. The app converts audio to PCM16 with `ffmpeg`, commits fixed-size chunks manually, and uses each chunk as a timestamped transcript segment.

OpenAI transcription, slower diarization mode:

```bash
TRANSCRIBE_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_TRANSCRIBE_MODEL=gpt-4o-transcribe-diarize
OPENAI_CHUNKING_STRATEGY=auto
OPENAI_TRANSCRIBE_LANGUAGE=th
```

OpenAI post-call agent:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

Azure OpenAI post-call agent:

```bash
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-chat-deployment
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

Anthropic post-call agent:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

## API

- `GET /api/calls` lists all processed calls.
- `GET /api/calls/{call_id}` returns one call result.
- `POST /api/process-folder` queues every WAV in a folder.
- `POST /api/calls/{call_id}/reprocess` reprocesses one recording.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/process-folder \
  -H 'Content-Type: application/json' \
  -d '{"folder":"voice","force":true}'
```

## Notes

The mock transcription and analysis are deterministic placeholders. They let you test the batch pipeline and UI immediately. For production, set `TRANSCRIBE_PROVIDER=openai` and choose one of the LLM providers for the LangGraph agent.
