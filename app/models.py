from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    pending = "pending"
    queued = "queued"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class ProcessingStage(str, Enum):
    queued = "queued"
    transcribing = "transcribing"
    diarizing = "diarizing"
    analyzing = "analyzing"
    enriching = "enriching"
    complete = "complete"
    failed = "failed"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    mixed = "mixed"


class ToneFlag(str, Enum):
    calm = "calm"
    frustrated = "frustrated"
    angry = "angry"
    confused = "confused"
    urgent = "urgent"
    satisfied = "satisfied"
    empathetic = "empathetic"
    procedural = "procedural"


class SpeakerRole(str, Enum):
    customer = "customer"
    call_center_staff = "call_center_staff"
    unknown = "unknown"


class Gender(str, Enum):
    M = "M"
    F = "F"
    not_sure = "Not sure"


class SpeakerClassification(BaseModel):
    speaker: str
    role: SpeakerRole = SpeakerRole.unknown
    display_name: str | None = None
    rationale: str | None = None


class PersonProfile(BaseModel):
    name: str | None = None
    gender: Gender = Gender.not_sure
    persona: str | None = None


class TranscriptSegment(BaseModel):
    start: float = 0
    end: float = 0
    speaker: str = "Unknown"
    text: str
    sentiment: Sentiment = Sentiment.neutral
    tone_flags: list[ToneFlag] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[TranscriptSegment]
    full_text: str


class JourneyMoment(BaseModel):
    time_seconds: float = 0
    label: str
    description: str
    sentiment: Sentiment = Sentiment.neutral


class PostCallAnalysis(BaseModel):
    summary: str
    session_topic: str | None = None
    speaker_classifications: list[SpeakerClassification] = Field(default_factory=list)
    agent_profile: PersonProfile = Field(default_factory=PersonProfile)
    customer_profile: PersonProfile = Field(default_factory=PersonProfile)
    customer_sentiment: Sentiment
    agent_sentiment: Sentiment
    customer_tone_flags: list[ToneFlag] = Field(default_factory=list)
    agent_tone_flags: list[ToneFlag] = Field(default_factory=list)
    customer_emotion_journey: list[JourneyMoment] = Field(default_factory=list)
    agent_emotion_journey: list[JourneyMoment] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    critical_flags: list[str] = Field(default_factory=list)
    credit_card_products: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    # Azure Language Service enrichment (populated when azure_language_enabled=true)
    azure_summary_issue: str | None = None
    azure_summary_resolution: str | None = None
    azure_summary_narrative: str | None = None
    azure_follow_up_items: list[str] = Field(default_factory=list)
    pii_entities_found: list[str] = Field(default_factory=list)


class CallRecord(BaseModel):
    id: str
    file_name: str
    file_path: str
    status: JobStatus = JobStatus.queued
    stage: ProcessingStage = ProcessingStage.queued
    progress_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    transcript: TranscriptResult | None = None
    analysis: PostCallAnalysis | None = None

    @classmethod
    def from_path(cls, path: Path, call_id: str) -> "CallRecord":
        return cls(id=call_id, file_name=path.name, file_path=str(path.resolve()))
