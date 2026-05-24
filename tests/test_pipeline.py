from pathlib import Path

from app.agent import PostCallAgent
from app.audio import TranscriptionService
from app.config import Settings
from app.enrichment import enrich_transcript_with_analysis
from app.models import (
    JobStatus,
    JourneyMoment,
    PostCallAnalysis,
    Sentiment,
    SpeakerClassification,
    SpeakerRole,
    ToneFlag,
    TranscriptResult,
    TranscriptSegment,
)
from app.processor import BatchProcessor, SUPPORTED_EXTENSIONS
from app.storage import CallStore


def test_mock_pipeline_processes_folder(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voice"
    data_dir = tmp_path / "data"
    voice_dir.mkdir()
    source = Path("voice/creditcard_call_center.WAV")
    target = voice_dir / "call.WAV"
    target.write_bytes(source.read_bytes())

    settings = Settings(
        watch_folder=voice_dir,
        data_dir=data_dir,
        transcribe_provider="mock",
        llm_provider="mock",
        auto_process_on_start=False,
    )
    store = CallStore(settings.absolute_data_dir())
    processor = BatchProcessor(
        store=store,
        transcription=TranscriptionService(settings),
        agent=PostCallAgent(settings),
        max_workers=1,
    )

    records = processor.process_folder(settings.absolute_watch_folder(), force=True)
    assert len(records) == 1

    processor.executor.shutdown(wait=True)
    processed = store.get(records[0].id)
    assert processed is not None
    assert processed.status == JobStatus.complete
    assert processed.transcript is not None
    assert len(processed.transcript.segments) >= 2
    assert processed.analysis is not None
    assert processed.analysis.next_actions


def test_batch_processor_accepts_common_audio_extensions() -> None:
    assert ".mp3" in SUPPORTED_EXTENSIONS
    assert ".m4a" in SUPPORTED_EXTENSIONS
    assert ".ogg" in SUPPORTED_EXTENSIONS
    assert ".webm" in SUPPORTED_EXTENSIONS
    assert ".flac" in SUPPORTED_EXTENSIONS
    assert ".wma" in SUPPORTED_EXTENSIONS


def test_language_auto_does_not_force_transcription_language(tmp_path: Path) -> None:
    service = TranscriptionService(Settings(openai_transcribe_language="auto", data_dir=tmp_path))
    assert service._transcribe_language() is None

    service = TranscriptionService(Settings(openai_transcribe_language="multilingual", data_dir=tmp_path))
    assert service._transcribe_language() is None

    service = TranscriptionService(Settings(openai_transcribe_language="th", data_dir=tmp_path))
    assert service._transcribe_language() == "th"


def test_enrichment_relabels_speakers_and_aligns_journey_sentiment() -> None:
    transcript = TranscriptResult(
        language="th",
        duration_seconds=20,
        full_text="",
        segments=[
            TranscriptSegment(start=0, end=10, speaker="Speaker A", text="แจ้งปัญหา"),
            TranscriptSegment(start=10, end=20, speaker="Speaker B", text="ขอตรวจสอบข้อมูล"),
        ],
    )
    analysis = PostCallAnalysis(
        summary="ลูกค้าติดต่อเรื่องปัญหา",
        speaker_classifications=[
            SpeakerClassification(speaker="Speaker A", role=SpeakerRole.customer, display_name="ลูกค้า"),
            SpeakerClassification(speaker="Speaker B", role=SpeakerRole.call_center_staff, display_name="เจ้าหน้าที่"),
        ],
        customer_sentiment=Sentiment.negative,
        agent_sentiment=Sentiment.neutral,
        customer_tone_flags=[ToneFlag.frustrated],
        agent_tone_flags=[ToneFlag.procedural],
        customer_emotion_journey=[
            JourneyMoment(time_seconds=0, label="ลูกค้าเริ่มหงุดหงิด", description="แจ้งปัญหา", sentiment=Sentiment.negative)
        ],
        agent_emotion_journey=[
            JourneyMoment(time_seconds=10, label="เจ้าหน้าที่รับเรื่อง", description="ตรวจสอบข้อมูล", sentiment=Sentiment.neutral)
        ],
        key_topics=["ปัญหา"],
        risks=[],
        next_actions=["ติดตามผล"],
    )

    enriched = enrich_transcript_with_analysis(transcript, analysis)

    assert enriched.segments[0].speaker == "ลูกค้า"
    assert enriched.segments[0].sentiment == Sentiment.negative
    assert enriched.segments[0].tone_flags == [ToneFlag.frustrated]
    assert enriched.segments[1].speaker == "เจ้าหน้าที่"
    assert enriched.segments[1].sentiment == Sentiment.neutral
    assert enriched.segments[1].tone_flags == [ToneFlag.procedural]
