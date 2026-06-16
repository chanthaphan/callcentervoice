import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from app.agent import PostCallAgent
from app.audio import SUPPORTED_AUDIO_EXTENSIONS, TranscriptionService
from app.enrichment import enrich_transcript_with_analysis
from app.language import AzureLanguageService
from app.models import CallRecord, JobStatus, PostCallAnalysis, ProcessingStage, TranscriptResult
from app.redaction import mask_literal_spans
from app.storage import CallStore


SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS


class BatchProcessor:
    def __init__(
        self,
        store: CallStore,
        transcription: TranscriptionService,
        agent: PostCallAgent,
        language_service: AzureLanguageService | None = None,
        max_workers: int = 3,
    ):
        self.store = store
        self.transcription = transcription
        self.agent = agent
        self.language_service = language_service
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _audio_files(folder: Path) -> list[Path]:
        """All supported audio files under folder, recursing into subfolders."""
        return sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def discover(self, path: Path, root: Path | None = None) -> CallRecord:
        """Add file to store as pending without starting processing. No-op if already known."""
        path = path.expanduser().resolve()
        call_id = self.call_id_for(path)
        existing = self.store.get(call_id)
        if existing:
            return existing
        record = CallRecord.from_path(path, call_id, root=root)
        record.status = JobStatus.pending
        record.stage = ProcessingStage.queued
        record.progress_message = "Ready to start"
        record.updated_at = datetime.now(UTC)
        self.store.save(record)
        return record

    def discover_folder(self, folder: Path) -> list[CallRecord]:
        folder = folder.expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        return [self.discover(path, root=folder) for path in self._audio_files(folder)]

    def process_folder(self, folder: Path, force: bool = False, reanalyze: bool = False) -> list[CallRecord]:
        folder = folder.expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        return [
            self.enqueue(path, force=force, reanalyze=reanalyze, root=folder)
            for path in self._audio_files(folder)
        ]

    def enqueue(self, path: Path, force: bool = False, reanalyze: bool = False, root: Path | None = None) -> CallRecord:
        path = path.expanduser().resolve()
        call_id = self.call_id_for(path)
        existing = self.store.get(call_id)
        if existing and existing.status == JobStatus.complete and not force and not reanalyze:
            return existing

        with self._lock:
            if call_id in self._active:
                return existing or CallRecord.from_path(path, call_id, root=root)
            record = existing or CallRecord.from_path(path, call_id, root=root)
            profile_context = self._build_profile_context(record.analysis) if reanalyze and record.analysis else None
            record.status = JobStatus.queued
            record.stage = ProcessingStage.queued
            record.progress_message = "Waiting to process"
            record.error = None
            if force:
                record.transcript = None
                record.analysis = None
            elif reanalyze:
                record.analysis = None
            record.updated_at = datetime.now(UTC)
            self.store.save(record)
            self._active.add(call_id)
            self.executor.submit(self._process_one, record, profile_context)
            return record

    def _process_one(self, record: CallRecord, profile_context: str | None = None) -> None:
        try:
            record.status = JobStatus.processing
            if record.transcript is None:
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
                # The regex floor inside transcribe() has already masked numeric PII.
                # Optionally layer LLM detection on top for context-dependent PII
                # (names, addresses, birth dates) before any analysis sees the text.
                if self.agent.settings.llm_pii_redaction:
                    self._save_stage(record, ProcessingStage.transcribing, "Redacting PII")
                    spans = self.agent.detect_pii_spans(transcript)
                    transcript = mask_literal_spans(transcript, spans)
                    record.transcript = transcript
                self._save_stage(record, ProcessingStage.diarizing, "Running speaker diarization")
                transcript = self.agent.diarize_transcript(transcript)
                record.transcript = transcript
                self._save_stage(record, ProcessingStage.diarizing, f"Diarized {len(transcript.segments)} segment(s)")
            else:
                transcript = record.transcript
                if profile_context:
                    self._save_stage(record, ProcessingStage.diarizing, "Re-running speaker diarization with profile context")
                    transcript = self.agent.diarize_transcript(transcript, profile_context=profile_context)
                    record.transcript = transcript
                    self._save_stage(record, ProcessingStage.diarizing, f"Re-diarized {len(transcript.segments)} segment(s)")

            self._save_stage(record, ProcessingStage.analyzing, "Running post-call analysis")
            analysis = self.agent.analyze(transcript)

            if self.language_service:
                self._save_stage(record, ProcessingStage.enriching, "Running Azure Language enrichment")
                transcript, analysis = self.language_service.enrich(transcript, analysis)

            if self.agent.settings.kb_verification:
                self._save_stage(record, ProcessingStage.analyzing, "Verifying staff statements against KB")
                analysis.kb_checks = self.agent.verify_against_kb(transcript, analysis)

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

    @staticmethod
    def _build_profile_context(analysis: PostCallAnalysis) -> str | None:
        parts: list[str] = []
        if analysis.session_topic:
            parts.append(f"Call topic: {analysis.session_topic}")
        ap = analysis.agent_profile
        if ap and (ap.name or ap.persona):
            lines = ["Agent profile:"]
            if ap.name:
                lines.append(f"  Name: {ap.name}")
            if ap.gender and ap.gender.value != "Not sure":
                lines.append(f"  Gender: {ap.gender.value}")
            if ap.persona:
                lines.append(f"  Description: {ap.persona}")
            parts.append("\n".join(lines))
        cp = analysis.customer_profile
        if cp and (cp.name or cp.persona):
            lines = ["Customer profile:"]
            if cp.name:
                lines.append(f"  Name: {cp.name}")
            if cp.gender and cp.gender.value != "Not sure":
                lines.append(f"  Gender: {cp.gender.value}")
            if cp.persona:
                lines.append(f"  Description: {cp.persona}")
            parts.append("\n".join(lines))
        for sc in analysis.speaker_classifications:
            if sc.display_name and sc.speaker:
                parts.append(f"Speaker label '{sc.speaker}' = {sc.role.value} ({sc.display_name})")
        return "\n\n".join(parts) if parts else None
