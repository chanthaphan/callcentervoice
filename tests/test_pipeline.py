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


def test_redact_pii_masks_numbers_but_keeps_short_values() -> None:
    from app.redaction import MASK, redact_pii, redact_transcript

    assert redact_pii("โทร 081-234-5678 นะครับ") == f"โทร {MASK} นะครับ"
    assert redact_pii("บัตรเลข 4111 1111 1111 1111") == f"บัตรเลข {MASK}"
    assert redact_pii("เลขบัตรประชาชน 1 2345 67890 12 3") == f"เลขบัตรประชาชน {MASK}"
    # short numbers (amounts, dates) and decimals are left intact
    assert redact_pii("ยอด 2,500 บาท") == "ยอด 2,500 บาท"
    assert redact_pii("วันที่ 2024-05-24") == "วันที่ 2024-05-24"
    assert redact_pii("อัตรา 3.14159265 หน่วย") == "อัตรา 3.14159265 หน่วย"

    transcript = TranscriptResult(
        language="th",
        duration_seconds=10,
        segments=[
            TranscriptSegment(start=0, end=2, speaker="ลูกค้า", text="เบอร์ผม 0812345678 ครับ"),
            TranscriptSegment(start=2, end=4, speaker="นาลินี", text="ได้ค่ะ"),
        ],
        full_text="ลูกค้า: เบอร์ผม 0812345678 ครับ\nนาลินี: ได้ค่ะ",
    )
    redacted = redact_transcript(transcript)
    assert MASK in redacted.segments[0].text
    assert "0812345678" not in redacted.full_text
    # idempotent
    assert redact_transcript(redacted) is redacted

    # emails are part of the deterministic floor too
    assert redact_pii("ส่งมาที่ john.doe@example.com ครับ") == f"ส่งมาที่ {MASK} ครับ"


def test_mask_literal_spans_masks_given_substrings() -> None:
    from app.redaction import MASK, mask_literal_spans

    transcript = TranscriptResult(
        language="th",
        duration_seconds=10,
        segments=[
            TranscriptSegment(start=0, end=2, speaker="เจ้าหน้าที่", text="คุณสมชาย ใจดี เกิดวันที่ 05/01/2530 ใช่ไหมคะ"),
            TranscriptSegment(start=2, end=4, speaker="ลูกค้า", text="ใช่ครับ"),
        ],
        full_text="เจ้าหน้าที่: คุณสมชาย ใจดี เกิดวันที่ 05/01/2530 ใช่ไหมคะ\nลูกค้า: ใช่ครับ",
    )
    # Longer span first proves longest-first ordering doesn't matter to the caller.
    masked = mask_literal_spans(transcript, ["05/01/2530", "สมชาย ใจดี"])
    assert "สมชาย ใจดี" not in masked.full_text
    assert "05/01/2530" not in masked.full_text
    assert masked.segments[0].text.count(MASK) == 2

    # idempotent / no-op when nothing matches
    assert mask_literal_spans(masked, ["ไม่มีอยู่จริง"]) is masked
    assert mask_literal_spans(transcript, []) is transcript


def test_detect_pii_spans_returns_empty_for_mock_provider(tmp_path: Path) -> None:
    agent = PostCallAgent(Settings(llm_provider="mock", data_dir=tmp_path))
    transcript = TranscriptResult(
        language="th",
        duration_seconds=4,
        segments=[TranscriptSegment(start=0, end=2, speaker="ลูกค้า", text="สวัสดีครับ")],
        full_text="ลูกค้า: สวัสดีครับ",
    )
    assert agent.detect_pii_spans(transcript) == []


def test_llm_pii_redaction_masks_spans_before_analysis(tmp_path: Path) -> None:
    """When llm_pii_redaction is on, detected spans are masked before analysis sees them."""
    from app.redaction import MASK

    raw = TranscriptResult(
        language="th",
        duration_seconds=4,
        segments=[
            TranscriptSegment(start=0, end=2, speaker="Speaker A", text="ผมชื่อสมชาย ใจดี ครับ"),
            TranscriptSegment(start=2, end=4, speaker="Speaker B", text="รับทราบค่ะ"),
        ],
        full_text="Speaker A: ผมชื่อสมชาย ใจดี ครับ\nSpeaker B: รับทราบค่ะ",
    )

    class StubTranscription:
        def transcribe(self, path: Path, on_partial=None) -> TranscriptResult:
            return raw

    seen: dict[str, TranscriptResult] = {}

    class SpyAgent:
        settings = Settings(llm_pii_redaction=True, data_dir=tmp_path)

        def detect_pii_spans(self, transcript: TranscriptResult) -> list[str]:
            return ["สมชาย ใจดี"]

        def diarize_transcript(self, transcript: TranscriptResult, profile_context: str | None = None) -> TranscriptResult:
            return transcript

        def analyze(self, transcript: TranscriptResult) -> PostCallAnalysis:
            seen["analyze"] = transcript
            return PostCallAnalysis(
                summary="ok",
                customer_sentiment=Sentiment.neutral,
                agent_sentiment=Sentiment.neutral,
            )

    store = CallStore(tmp_path / "data")
    processor = BatchProcessor(
        store=store,
        transcription=StubTranscription(),  # type: ignore[arg-type]
        agent=SpyAgent(),  # type: ignore[arg-type]
        max_workers=1,
    )

    from app.models import CallRecord

    record = CallRecord.from_path(tmp_path / "call.wav", "call-1")
    processor._process_one(record)

    assert "สมชาย ใจดี" not in seen["analyze"].full_text
    assert MASK in seen["analyze"].segments[0].text
    assert record.status == JobStatus.complete


def test_transcribe_redacts_output_and_partials(tmp_path: Path, monkeypatch) -> None:
    """TranscriptionService.transcribe() redacts its final output and every streamed partial."""
    from app.redaction import MASK

    raw = TranscriptResult(
        language="th",
        duration_seconds=4,
        segments=[
            TranscriptSegment(start=0, end=2, speaker="Speaker A", text="เบอร์ผม 0812345678 ครับ"),
            TranscriptSegment(start=2, end=4, speaker="Speaker B", text="รับทราบค่ะ"),
        ],
        full_text="Speaker A: เบอร์ผม 0812345678 ครับ\nSpeaker B: รับทราบค่ะ",
    )
    service = TranscriptionService(Settings(transcribe_provider="openai_realtime", data_dir=tmp_path))

    def fake_realtime(path: Path, on_partial=None) -> TranscriptResult:
        if on_partial:
            on_partial(raw)
        return raw

    monkeypatch.setattr(service, "_openai_realtime_transcribe", fake_realtime)

    seen: list[TranscriptResult] = []
    result = service.transcribe(Path("x.wav"), on_partial=seen.append)

    # final output redacted
    assert MASK in result.segments[0].text
    assert "0812345678" not in result.full_text
    # streamed partial redacted before the callback sees it
    assert MASK in seen[0].segments[0].text
    assert "0812345678" not in seen[0].full_text


def test_compute_stats_aggregates_periods_and_insights() -> None:
    from datetime import UTC, datetime

    from app.models import CallRecord, ProcessingStage
    from app.stats import call_date, compute_stats

    def rec(call_id, fname, created, status=JobStatus.complete, analysis=None, duration=None):
        return CallRecord(
            id=call_id,
            file_name=fname,
            file_path=f"/voice/{fname}",
            status=status,
            stage=ProcessingStage.complete,
            created_at=created,
            updated_at=created,
            transcript=(
                TranscriptResult(language="th", duration_seconds=duration, segments=[], full_text="")
                if duration is not None
                else None
            ),
            analysis=analysis,
        )

    a1 = PostCallAnalysis(
        summary="x",
        customer_sentiment=Sentiment.negative,
        agent_sentiment=Sentiment.neutral,
        customer_tone_flags=[ToneFlag.frustrated],
        key_topics=["บัตรหาย"],
        critical_flags=["fraud"],
        credit_card_products=["Platinum"],
    )
    a2 = PostCallAnalysis(
        summary="y",
        customer_sentiment=Sentiment.positive,
        agent_sentiment=Sentiment.neutral,
        key_topics=["ยอดเงิน"],
    )
    records = [
        rec("1", "2026-05-01/a.wav", datetime(2026, 5, 1, tzinfo=UTC), analysis=a1, duration=120),
        rec("2", "2026-05-02/b.wav", datetime(2026, 5, 2, tzinfo=UTC), analysis=a2, duration=60),
        rec("3", "callC.wav", datetime(2026, 6, 10, tzinfo=UTC), status=JobStatus.failed),
    ]

    # call_date: parsed from filename, else created_at fallback
    assert call_date(records[0]).isoformat() == "2026-05-01"
    assert call_date(records[2]).isoformat() == "2026-06-10"

    stats = compute_stats(records, digest="monthly")
    assert stats["totals"]["total"] == 3
    assert stats["totals"]["completed"] == 2
    timeline = {row["period"]: row for row in stats["timeline"]}
    assert timeline["2026-05"]["total"] == 2
    assert timeline["2026-06"]["failed"] == 1
    assert stats["sentiment"]["negative"] == 1 and stats["sentiment"]["positive"] == 1
    assert stats["kpis"]["negative_pct"] == 50.0
    assert stats["kpis"]["avg_duration_seconds"] == 90.0
    assert {t["label"] for t in stats["tone_flags"]} == {"frustrated"}
    assert {t["label"] for t in stats["topics"]} == {"บัตรหาย", "ยอดเงิน"}
    assert stats["critical_flags"][0]["label"] == "fraud"
    assert len(stats["monthly"]) == 2

    # daily digest buckets by exact date
    daily = {row["period"]: row for row in compute_stats(records, digest="daily")["timeline"]}
    assert daily["2026-05-01"]["total"] == 1 and daily["2026-05-02"]["total"] == 1


def test_product_kb_loads_titles_and_buckets_non_bbl(tmp_path: Path) -> None:
    import json

    from app.product_kb import OTHER, load_bbl_products, normalize_products

    base = tmp_path / "Credit-Cards"
    for folder, title in [
        ("Bangkok-Bank-Visa-Infinite-Card", "บัตรอินฟินิท ธนาคารกรุงเทพ"),
        ("Bangkok-Bank-Titanium-Credit-Card", "บัตรเครดิตไทเทเนียม ธนาคารกรุงเทพ"),
    ]:
        (base / folder).mkdir(parents=True)
        (base / folder / "product.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    # non-card folder must be ignored (doesn't match Bangkok-Bank-*)
    (base / "Rewards").mkdir(parents=True)
    (base / "Rewards" / "product.json").write_text(json.dumps({"title": "Rewards"}), encoding="utf-8")

    kb = load_bbl_products(str(base))
    assert len(kb) == 2
    assert "บัตรอินฟินิท ธนาคารกรุงเทพ" in kb

    out = normalize_products(["บัตรอินฟินิท", "Visa", "KTC", ""], kb)
    assert "บัตรอินฟินิท ธนาคารกรุงเทพ" in out  # short form maps to canonical
    assert out.count(OTHER) == 1  # Visa + KTC collapse to a single Other
    # the generic term "credit card" must NOT match a specific card
    assert normalize_products(["บัตรเครดิต"], kb) == [OTHER]


def test_stats_filtering_and_product_bucketing() -> None:
    from datetime import UTC, date, datetime

    from app.models import CallRecord, ProcessingStage
    from app.product_kb import OTHER
    from app.stats import compute_stats, filter_records

    kb = ["บัตรอินฟินิท ธนาคารกรุงเทพ"]

    def rec(call_id, fname, created, sent, products):
        return CallRecord(
            id=call_id, file_name=fname, file_path=f"/v/{fname}",
            status=JobStatus.complete, stage=ProcessingStage.complete,
            created_at=created, updated_at=created,
            transcript=TranscriptResult(language="th", duration_seconds=60, segments=[], full_text=""),
            analysis=PostCallAnalysis(
                summary="s", customer_sentiment=sent, agent_sentiment=Sentiment.neutral,
                credit_card_products=products,
            ),
        )

    records = [
        rec("1", "2026-05-01/a.wav", datetime(2026, 5, 1, tzinfo=UTC), Sentiment.negative, ["บัตรอินฟินิท"]),
        rec("2", "2026-05-02/b.wav", datetime(2026, 5, 2, tzinfo=UTC), Sentiment.positive, ["KTC"]),
    ]

    stats = compute_stats(records, digest="daily", kb=kb)
    labels = {p["label"] for p in stats["products"]}
    assert "บัตรอินฟินิท ธนาคารกรุงเทพ" in labels and OTHER in labels
    assert len(stats["sentiment_trend"]) == 2
    assert any(t["negative"] == 1 for t in stats["sentiment_trend"])
    assert stats["kpis"]["positive_pct"] == 50.0

    assert [r.id for r in filter_records(records, sentiments={"negative"}, kb=kb)] == ["1"]
    assert [r.id for r in filter_records(records, products={"บัตรอินฟินิท ธนาคารกรุงเทพ"}, kb=kb)] == ["1"]
    assert [r.id for r in filter_records(records, date_from=date(2026, 5, 2), kb=kb)] == ["2"]


def test_verify_against_kb_empty_for_mock_and_without_products(tmp_path: Path) -> None:
    agent = PostCallAgent(Settings(llm_provider="mock", data_dir=tmp_path))
    transcript = TranscriptResult(
        language="th", duration_seconds=4,
        segments=[TranscriptSegment(start=0, end=2, speaker="เจ้าหน้าที่", text="สวัสดีครับ")],
        full_text="เจ้าหน้าที่: สวัสดีครับ",
    )
    analysis = PostCallAnalysis(
        summary="s", customer_sentiment=Sentiment.neutral, agent_sentiment=Sentiment.neutral,
        credit_card_products=["บัตรอินฟินิท ธนาคารกรุงเทพ"],
    )
    # mock provider → no verification
    assert agent.verify_against_kb(transcript, analysis) == []


def test_stats_aggregates_kb_issue_rate() -> None:
    from datetime import UTC, datetime

    from app.models import CallRecord, KbCheck, KbVerdict, ProcessingStage
    from app.stats import compute_stats

    def rec(call_id, created, checks):
        return CallRecord(
            id=call_id, file_name=f"2026-05-01/{call_id}.wav", file_path=f"/v/{call_id}",
            status=JobStatus.complete, stage=ProcessingStage.complete,
            created_at=created, updated_at=created,
            analysis=PostCallAnalysis(
                summary="s", customer_sentiment=Sentiment.neutral, agent_sentiment=Sentiment.neutral,
                kb_checks=checks,
            ),
        )

    records = [
        rec("1", datetime(2026, 5, 1, tzinfo=UTC), [KbCheck(claim="a", verdict=KbVerdict.supported)]),
        rec("2", datetime(2026, 5, 1, tzinfo=UTC), [KbCheck(claim="b", verdict=KbVerdict.contradicts)]),
    ]
    stats = compute_stats(records)
    assert stats["kb"]["checked"] == 2
    assert stats["kb"]["verdicts"] == {"supported": 1, "contradicts": 1}
    assert stats["kpis"]["kb_issue_pct"] == 50.0  # one of two calls has a non-supported claim


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
