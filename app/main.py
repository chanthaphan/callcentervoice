import asyncio
import json
import logging
import mimetypes
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import PostCallAgent
from app.audio import TranscriptionService
from app.config import CONFIGURABLE_FIELDS, apply_overrides, get_settings
from app.language import AzureLanguageService
from app.processor import BatchProcessor
from app.redaction import redact_transcript
from app.storage import CallStore

logger = logging.getLogger(__name__)

settings = get_settings()

# Apply any saved runtime config overrides before creating service components
_data_dir = settings.absolute_data_dir()
_data_dir.mkdir(parents=True, exist_ok=True)
_config_overrides_path = _data_dir / "config_overrides.json"
if _config_overrides_path.exists():
    try:
        apply_overrides(settings, json.loads(_config_overrides_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        logger.warning("Could not load config overrides from %s", _config_overrides_path, exc_info=True)

def _redact_stored_transcripts(call_store: CallStore) -> None:
    """Mask PII in already-stored transcripts. Idempotent — masked text has no
    digit runs left to match, so re-running on later boots is a no-op."""
    for record in call_store.iter_records():
        if not record.transcript:
            continue
        redacted = redact_transcript(record.transcript)
        if redacted is not record.transcript:
            record.transcript = redacted
            call_store.save(record)


store = CallStore(settings.absolute_data_dir())
_redact_stored_transcripts(store)
processor = BatchProcessor(
    store=store,
    transcription=TranscriptionService(settings),
    agent=PostCallAgent(settings),
    language_service=AzureLanguageService(settings),
    max_workers=settings.max_parallel_files,
)

app = FastAPI(title="Call Center Voice Analysis")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    # No-op unless APP_API_KEY is configured. Only /api/* is guarded so the UI shell
    # can still load and prompt for the key. Browser-native requests (audio, downloads)
    # can't set headers, so a ?api_key= query param is also accepted.
    expected = settings.app_api_key
    if expected and request.url.path.startswith("/api"):
        provided = request.headers.get("x-api-key") or request.query_params.get("api_key") or ""
        if not secrets.compare_digest(provided, expected):
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
    return await call_next(request)


def _value(item) -> str:
    return getattr(item, "value", str(item))


class ProcessFolderRequest(BaseModel):
    folder: str | None = None
    force: bool = False
    reanalyze: bool = False


class BulkActionRequest(BaseModel):
    call_ids: list[str]
    action: str  # "start" | "reanalyze" | "reprocess" | "delete"


class ConfigPatch(BaseModel):
    changes: dict[str, Any]


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_scan_loop())


async def _scan_loop() -> None:
    while True:
        folder = settings.absolute_watch_folder()
        if settings.auto_process_on_start:
            await asyncio.to_thread(processor.process_folder, folder, False)
        else:
            await asyncio.to_thread(processor.discover_folder, folder)
        await asyncio.sleep(settings.scan_interval_seconds)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/config")
def get_config():
    return {field: getattr(settings, field, None) for field in CONFIGURABLE_FIELDS}


@app.patch("/api/config")
def patch_config(payload: ConfigPatch):
    apply_overrides(settings, payload.changes)
    current = {field: getattr(settings, field, None) for field in CONFIGURABLE_FIELDS}
    _config_overrides_path.write_text(
        json.dumps(current, indent=2, default=str), encoding="utf-8"
    )
    return current


@app.get("/api/calls")
def list_calls(limit: int = 50, offset: int = 0, status: str | None = None):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    items = store.list_summaries(limit=limit, offset=offset, status=status)
    for item in items:
        item["file_exists"] = Path(item["file_path"]).is_file()
    return {"items": items, "total": store.count(status), "limit": limit, "offset": offset}


@app.post("/api/calls/bulk")
def bulk_action(payload: BulkActionRequest):
    results: list[dict] = []
    for call_id in payload.call_ids:
        if payload.action == "delete":
            deleted = store.delete(call_id)
            results.append({"id": call_id, "deleted": deleted})
        else:
            record = store.get(call_id)
            if not record:
                results.append({"id": call_id, "error": "not found"})
                continue
            if payload.action == "start":
                updated = processor.enqueue(Path(record.file_path))
            elif payload.action == "reanalyze":
                updated = processor.enqueue(Path(record.file_path), reanalyze=True)
            elif payload.action == "reprocess":
                updated = processor.enqueue(Path(record.file_path), force=True)
            else:
                results.append({"id": call_id, "error": f"unknown action {payload.action}"})
                continue
            results.append({"id": call_id, "status": _value(updated.status)})
    return results


@app.delete("/api/calls/{call_id}")
def delete_call(call_id: str):
    if not store.delete(call_id):
        raise HTTPException(status_code=404, detail="Call not found")
    return {"deleted": True}


@app.get("/api/calls/{call_id}")
def get_call(call_id: str):
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    data = record.model_dump(mode="json")
    data["file_exists"] = Path(record.file_path).is_file()
    return data


@app.get("/api/calls/{call_id}/audio")
def get_call_audio(call_id: str) -> FileResponse:
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    path = Path(record.file_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=record.file_name)


@app.get("/api/calls/{call_id}/download.txt")
def download_call_text(call_id: str) -> PlainTextResponse:
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    lines = [
        f"File: {record.file_name}",
        f"Status: {_value(record.status)}",
        f"Stage: {_value(record.stage)}",
        "",
    ]
    if record.analysis:
        a = record.analysis
        lines.extend(["Summary", a.summary, ""])
        if a.session_topic:
            lines.extend(["Session topic", a.session_topic, ""])
        ap = a.agent_profile
        if ap and (ap.name or ap.persona):
            lines.extend(["Agent profile"])
            if ap.name:
                lines.append(f"  Name: {ap.name}")
            lines.append(f"  Gender: {_value(ap.gender)}")
            if ap.persona:
                lines.append(f"  Persona: {ap.persona}")
            lines.append("")
        cp = a.customer_profile
        if cp and (cp.name or cp.persona):
            lines.extend(["Customer profile"])
            if cp.name:
                lines.append(f"  Name: {cp.name}")
            lines.append(f"  Gender: {_value(cp.gender)}")
            if cp.persona:
                lines.append(f"  Persona: {cp.persona}")
            lines.append("")
        lines.extend([
            "Key topics", *[f"- {item}" for item in a.key_topics], "",
            "Critical flags", *[f"- {item}" for item in a.critical_flags], "",
            "Credit card products", *[f"- {item}" for item in a.credit_card_products], "",
            "Risks", *[f"- {item}" for item in a.risks], "",
            "Next actions", *[f"- {item}" for item in a.next_actions], "",
        ])
        if a.quality_notes:
            lines.extend(["Quality notes", *[f"- {item}" for item in a.quality_notes], ""])
        if a.speaker_classifications:
            lines.extend(["Speaker roles"])
            lines.extend(
                f"- {item.speaker}: {_value(item.role)}"
                + (f" ({item.display_name})" if item.display_name else "")
                for item in a.speaker_classifications
            )
            lines.append("")
    if record.transcript:
        lines.append("Transcript")
        for segment in record.transcript.segments:
            lines.append(
                f"[{segment.start:0.2f}-{segment.end:0.2f}] "
                f"{segment.speaker} ({_value(segment.sentiment)}): {segment.text}"
            )
    content = "\n".join(lines).strip() + "\n"
    stem = Path(record.file_name).stem
    filename = f"{stem}.txt"
    # ASCII fallback (some browsers ignore filename*) — derive from the audio name, not the id
    ascii_stem = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in stem).strip("_")
    fallback_name = f"{ascii_stem or record.id}.txt"
    disposition = f"attachment; filename=\"{fallback_name}\"; filename*=UTF-8''{quote(filename)}"
    return PlainTextResponse(
        content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": disposition},
    )


@app.post("/api/process-folder")
def process_folder(payload: ProcessFolderRequest):
    folder = Path(payload.folder) if payload.folder else settings.absolute_watch_folder()
    return processor.process_folder(folder, force=payload.force, reanalyze=payload.reanalyze)


@app.post("/api/discover-folder")
def discover_folder(payload: ProcessFolderRequest):
    folder = Path(payload.folder) if payload.folder else settings.absolute_watch_folder()
    return processor.discover_folder(folder)


@app.post("/api/calls/{call_id}/start")
def start_call(call_id: str):
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    return processor.enqueue(Path(record.file_path))


@app.post("/api/calls/{call_id}/reprocess")
def reprocess_call(call_id: str):
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    return processor.enqueue(Path(record.file_path), force=True)


@app.post("/api/calls/{call_id}/reanalyze")
def reanalyze_call(call_id: str):
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    return processor.enqueue(Path(record.file_path), reanalyze=True)
