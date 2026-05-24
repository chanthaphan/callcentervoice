import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from app.agent import PostCallAgent
from app.audio import SUPPORTED_AUDIO_EXTENSIONS, TranscriptionService
from app.enrichment import enrich_transcript_with_analysis
from app.models import CallRecord, JobStatus, ProcessingStage, TranscriptResult
from app.storage import CallStore


SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS


class BatchProcessor:
    def __init__(
        self,
        store: CallStore,
        transcription: TranscriptionService,
        agent: PostCallAgent,
        max_workers: int = 3,
    ):
        self.store = store
        self.transcription = transcription
        self.agent = agent
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def process_folder(self, folder: Path, force: bool = False) -> list[CallRecord]:
        folder = folder.expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        records: list[CallRecord] = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
                continue
            record = self.enqueue(path, force=force)
            records.append(record)
        return records

    def enqueue(self, path: Path, force: bool = False) -> CallRecord:
        path = path.expanduser().resolve()
        call_id = self.call_id_for(path)
        existing = self.store.get(call_id)
        if existing and existing.status == JobStatus.complete and not force:
            return existing

        with self._lock:
            if call_id in self._active:
                return existing or CallRecord.from_path(path, call_id)
            record = existing or CallRecord.from_path(path, call_id)
            record.status = JobStatus.queued
            record.stage = ProcessingStage.queued
            record.progress_message = "Waiting to process"
            record.error = None
            if force:
                record.transcript = None
                record.analysis = None
            record.updated_at = datetime.now(UTC)
            self.store.save(record)
            self._active.add(call_id)
            self.executor.submit(self._process_one, record)
            return record

    def _process_one(self, record: CallRecord) -> None:
        try:
            record.status = JobStatus.processing
            self._save_stage(record, ProcessingStage.transcribing, "Transcribing audio")

            def save_partial(transcript: TranscriptResult) -> None:
                record.transcript = transcript
                self._save_stage(
                    record,
                    ProcessingStage.transcribing,
                    f"Transcribed {len(transcript.segments)} segment(s)",
                )

            transcript = self.transcription.transcribe(Path(record.file_path), on_partial=save_partial)
            record.transcript = transcript
            self._save_stage(record, ProcessingStage.diarizing, "Applying speaker labels")

            self._save_stage(record, ProcessingStage.analyzing, "Running post-call analysis")
            analysis = self.agent.analyze(transcript)
            record.analysis = analysis
            record.transcript = enrich_transcript_with_analysis(transcript, analysis)
            record.status = JobStatus.complete
            record.stage = ProcessingStage.complete
            record.progress_message = "Complete"
            record.error = None
        except Exception as exc:
            record.status = JobStatus.failed
            record.stage = ProcessingStage.failed
            record.progress_message = "Failed"
            record.error = str(exc)
        finally:
            record.updated_at = datetime.now(UTC)
            self.store.save(record)
            with self._lock:
                self._active.discard(record.id)

    def _save_stage(self, record: CallRecord, stage: ProcessingStage, message: str) -> None:
        record.stage = stage
        record.progress_message = message
        record.updated_at = datetime.now(UTC)
        self.store.save(record)

    @staticmethod
    def call_id_for(path: Path) -> str:
        stat = path.stat()
        fingerprint = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
        return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
