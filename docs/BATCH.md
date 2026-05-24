# Headless batch processing

For unattended bulk runs (e.g. 10,000+ recordings) without the web UI. The CLI reuses the same `.env` configuration, pipeline, and SQLite store as the web app.

## Usage

```bash
python -m app.batch [folder] [--workers N] [--reprocess] [--reanalyze] [--poll SECONDS]
```

| Argument | Default | Description |
|---|---|---|
| `folder` | `WATCH_FOLDER` | Folder to scan **recursively** (subfolders included). |
| `--workers N` | `MAX_PARALLEL_FILES` | Parallel workers. Tune to your provider's quota. |
| `--reprocess` | off | Re-transcribe **and** re-analyze even if already complete. |
| `--reanalyze` | off | Re-run analysis on the existing transcript only (no re-transcription). |
| `--poll SECONDS` | `5` | Progress-report interval. |

## Example

Recordings organized by date:

```
recordings/
  2026-05-01/
    call_0001.wav
    call_0002.wav
  2026-05-02/
    call_0003.wav
  …
```

```bash
python -m app.batch /data/recordings --workers 8
```

Output:

```
INFO batch: Scanning /data/recordings  (transcribe=azure_speech, llm=azure_openai, workers=8)
INFO batch: Enqueued 9824 file(s).
INFO batch: progress — complete=0 failed=0 remaining=9824
INFO batch: progress — complete=312 failed=2 remaining=9510
…
INFO batch: Done. 9810 complete, 14 failed, of 9824 enqueued.
WARNING batch: FAILED 2026-05-02/call_7731.wav: <error message>
```

Records show the subfolder in their name (`2026-05-01/call_0001.wav`) so same-named files across dates stay distinct.

## Behavior

- **Recursive** — every supported audio file under `folder` is discovered.
- **Resumable** — already-complete files are skipped (unless `--reprocess`/`--reanalyze`). Interrupted runs continue on the next invocation.
- **Retries** — transient throttling/timeouts (HTTP 429/5xx) are retried with exponential backoff per file; the LLM client also retries. A file only ends `failed` after retries are exhausted.
- **`Ctrl+C`** — cancels not-yet-started files and lets in-flight ones finish, then exits promptly. Re-run to resume the rest.
- **Same data as the UI** — results land in `DATA_DIR/calls.db`; start the web app afterward to review them.

## Scheduling

Run periodically with cron / systemd timers / Airflow, etc. Because it's resumable and idempotent, overlapping or repeated runs are safe:

```cron
# every night at 01:00, process the day's drop folder
0 1 * * *  cd /srv/callcentervoice && .venv/bin/python -m app.batch /data/recordings --workers 8 >> /var/log/cc-batch.log 2>&1
```

## Cost & throttling at scale

Each file costs one transcription call plus (typically) two LLM calls (diarization + analysis), and optionally Azure Language calls. For 10k files this is substantial spend and will hit provider rate limits. Recommendations:

- Set `--workers` to match your Azure/OpenAI quota (start conservative, e.g. 4–8).
- Keep retries on (default) so transient `429`s don't permanently fail files.
- Run in waves by pointing at per-day subfolders rather than the whole tree at once.
- Monitor `failed` count in the summary; re-run to retry failures (they're skipped once complete).

## Provider notes for bulk Thai calls

- **`azure_speech`** with `AZURE_SPEECH_DIARIZATION=diarization` gives the best Thai speaker separation. Set `AZURE_SPEECH_ENDPOINT` to the custom-domain endpoint.
- Avoid `gpt-4o-transcribe`/`-mini-transcribe` for diarized calls — they return a single text block with no timestamps. Use `whisper-1`, `gpt-4o-transcribe-diarize`, or `azure_speech`.
