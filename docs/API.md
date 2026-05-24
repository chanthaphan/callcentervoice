# HTTP API

Base URL: `http://127.0.0.1:8000` (default). All endpoints under `/api` return JSON.

## Authentication

If `APP_API_KEY` is set, every `/api/*` request must include the secret as either:
- header `X-API-Key: <key>`, or
- query param `?api_key=<key>` (used by the UI for `<audio>` and download links, which can't send headers).

A missing/wrong key returns `401`. When `APP_API_KEY` is unset (default), the API is open. The UI shell (`/`, `/static/*`) is never guarded so the page can load and prompt for the key.

---

## Calls

### `GET /api/calls`
Paginated list of **lightweight summaries** (no transcript/analysis bodies — fast at any scale).

Query params: `limit` (default 50, max 500), `offset` (default 0), `status` (optional filter: `pending`/`queued`/`processing`/`complete`/`failed`).

```bash
curl "http://127.0.0.1:8000/api/calls?limit=50&offset=0&status=complete"
```
```json
{
  "items": [
    {
      "id": "f586da5bbcd15b55",
      "file_name": "2026-05-01/call_a.wav",
      "file_path": "/abs/path/voice/2026-05-01/call_a.wav",
      "status": "complete",
      "stage": "complete",
      "progress_message": "Complete",
      "error": null,
      "session_topic": "ลูกค้าขออายัดบัตร…",
      "created_at": "2026-05-24T…",
      "updated_at": "2026-05-24T…",
      "file_exists": true
    }
  ],
  "total": 7, "limit": 50, "offset": 0
}
```

### `GET /api/calls/{id}`
One **full** record — includes `transcript` (segments) and `analysis`, plus `file_exists`. This is what the UI fetches when you open a session.

### `GET /api/calls/{id}/audio`
Streams the original audio (supports range requests). `404` if the file was moved/deleted.

### `GET /api/calls/{id}/download.txt`
Plain-text report (summary, profiles, topics, flags, risks, next actions, speaker-separated transcript). Downloaded filename is derived from the **audio file name** (e.g. `call_a.txt`).

---

## Processing actions

### `POST /api/process-folder`
Scan a folder recursively and **process** every supported audio file.
```bash
curl -X POST http://127.0.0.1:8000/api/process-folder \
  -H "Content-Type: application/json" \
  -d '{"folder": "voice", "force": false, "reanalyze": false}'
```
- `folder` — omit to use `WATCH_FOLDER`.
- `force` — re-transcribe + re-analyze even if complete.
- `reanalyze` — re-run analysis on the existing transcript only.

### `POST /api/discover-folder`
Scan a folder recursively and **add files as `pending`** without processing (used by the UI's **Refresh** button). Same body shape (`{folder?}`).

### `POST /api/calls/{id}/start`
Start a `pending` file.

### `POST /api/calls/{id}/reprocess`
Re-transcribe **and** re-analyze one call (`force`).

### `POST /api/calls/{id}/reanalyze`
Re-run analysis (and role-aware re-diarization) on the existing transcript — no re-transcription.

### `POST /api/calls/bulk`
Apply one action to many records.
```bash
curl -X POST http://127.0.0.1:8000/api/calls/bulk \
  -H "Content-Type: application/json" \
  -d '{"call_ids": ["id1","id2"], "action": "reprocess"}'
```
`action` ∈ `start` | `reanalyze` | `reprocess` | `delete`. Returns a per-id result list.

### `DELETE /api/calls/{id}`
Delete one record (the audio file on disk is **not** removed).

---

## Configuration

### `GET /api/config`
Returns the current values of the runtime-configurable fields (never API keys).

### `PATCH /api/config`
Update one or more runtime-configurable fields; persisted to `DATA_DIR/config_overrides.json`.
```bash
curl -X PATCH http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"changes": {"transcribe_provider": "azure_speech", "azure_speech_max_speakers": 2}}'
```
Unknown or non-configurable keys are ignored. See the runtime-configurable list in [CONFIGURATION.md](CONFIGURATION.md#runtime-configurable-settings).

---

## Static

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | The browser UI (single-page `index.html`). |
| `GET` | `/static/*` | Static assets. |
