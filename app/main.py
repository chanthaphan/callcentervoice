import asyncio
import mimetypes
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import PostCallAgent
from app.audio import TranscriptionService
from app.config import get_settings
from app.processor import BatchProcessor
from app.storage import CallStore

settings = get_settings()
store = CallStore(settings.absolute_data_dir())
processor = BatchProcessor(
    store=store,
    transcription=TranscriptionService(settings),
    agent=PostCallAgent(settings),
    max_workers=settings.max_parallel_files,
)

app = FastAPI(title="Call Center Voice Analysis")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def _value(item) -> str:
    return getattr(item, "value", str(item))


class ProcessFolderRequest(BaseModel):
    folder: str | None = None
    force: bool = False


@app.on_event("startup")
async def startup() -> None:
    if settings.auto_process_on_start:
        asyncio.create_task(_scan_loop())


async def _scan_loop() -> None:
    while True:
        await asyncio.to_thread(processor.process_folder, settings.absolute_watch_folder(), False)
        await asyncio.sleep(settings.scan_interval_seconds)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/calls")
def list_calls():
    return store.list()


@app.get("/api/calls/{call_id}")
def get_call(call_id: str):
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    return record


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
        lines.extend(
            [
                "Summary",
                record.analysis.summary,
                "",
                "Key topics",
                *[f"- {item}" for item in record.analysis.key_topics],
                "",
                "Critical flags",
                *[f"- {item}" for item in record.analysis.critical_flags],
                "",
                "Credit card products",
                *[f"- {item}" for item in record.analysis.credit_card_products],
                "",
                "Risks",
                *[f"- {item}" for item in record.analysis.risks],
                "",
                "Next actions",
                *[f"- {item}" for item in record.analysis.next_actions],
                "",
            ]
        )
        if record.analysis.speaker_classifications:
            lines.extend(["Speaker roles"])
            lines.extend(
                f"- {item.speaker}: {_value(item.role)}"
                + (f" ({item.display_name})" if item.display_name else "")
                for item in record.analysis.speaker_classifications
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
    filename = f"{Path(record.file_name).stem}.txt"
    fallback_name = f"{record.id}.txt"
    disposition = f"attachment; filename=\"{fallback_name}\"; filename*=UTF-8''{quote(filename)}"
    return PlainTextResponse(
        content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": disposition},
    )


@app.post("/api/process-folder")
def process_folder(payload: ProcessFolderRequest):
    folder = Path(payload.folder) if payload.folder else settings.absolute_watch_folder()
    return processor.process_folder(folder, force=payload.force)


@app.post("/api/calls/{call_id}/reprocess")
def reprocess_call(call_id: str):
    record = store.get(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    return processor.enqueue(Path(record.file_path), force=True)
