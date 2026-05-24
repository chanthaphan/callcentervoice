import json
import time
import urllib.error
import urllib.request
from typing import Any

from app.config import Settings
from app.models import PostCallAnalysis, Sentiment, TranscriptResult

CONVERSATION_API_VERSION = "2024-05-01-preview"
MAX_POLL_SECONDS = 120
POLL_INTERVAL_SECONDS = 3


class AzureLanguageService:
    """Post-processes transcripts and analysis using Azure Language Service.

    Provides three optional enrichments applied after the LLM analysis step:
    - ConversationPIIRedaction   — redacts PII in stored segment text
    - ConversationSentimentAnalysis — adds per-utterance sentiment scores
    - ConversationSummarization  — adds issue, resolution, narrative, follow-up
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def enrich(
        self,
        transcript: TranscriptResult,
        analysis: PostCallAnalysis,
    ) -> tuple[TranscriptResult, PostCallAnalysis]:
        if not self._is_enabled():
            return transcript, analysis

        tasks = self._build_tasks()
        if not tasks:
            return transcript, analysis

        items = self._build_conversation_items(transcript)
        if not items:
            return transcript, analysis

        language = transcript.language or "th"
        try:
            results = self._submit_and_poll(items, tasks, language)
        except Exception:
            # Azure Language is best-effort; never block the pipeline
            return transcript, analysis

        return self._apply_results(transcript, analysis, results)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _is_enabled(self) -> bool:
        return bool(
            self.settings.azure_language_enabled
            and self._effective_endpoint()
            and self._effective_api_key()
        )

    def _effective_endpoint(self) -> str | None:
        return (
            self.settings.azure_language_endpoint
            or self.settings.azure_openai_transcribe_endpoint
        )

    def _effective_api_key(self) -> str | None:
        return (
            self.settings.azure_language_api_key
            or self.settings.azure_openai_api_key
        )

    def _build_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        if self.settings.azure_language_pii_redaction:
            tasks.append({"kind": "ConversationPIIRedaction", "taskName": "pii"})
        if self.settings.azure_language_sentiment:
            tasks.append({"kind": "ConversationSentimentAnalysis", "taskName": "sentiment"})
        if self.settings.azure_language_summarization:
            tasks.append({
                "kind": "ConversationSummarization",
                "taskName": "summary",
                "parameters": {
                    "summaryAspects": ["issue", "resolution", "narrative", "followUpTasks"],
                },
            })
        return tasks

    def _build_conversation_items(self, transcript: TranscriptResult) -> list[dict[str, Any]]:
        return [
            {
                "id": str(i),
                "participantId": seg.speaker,
                "text": seg.text,
                "role": self._speaker_to_role(seg.speaker),
            }
            for i, seg in enumerate(transcript.segments)
            if seg.text.strip()
        ]

    def _speaker_to_role(self, speaker: str) -> str:
        lower = speaker.lower()
        if any(kw in lower for kw in ("agent", "staff", "เจ้าหน้าที่")):
            return "Agent"
        if any(kw in lower for kw in ("customer", "caller", "ลูกค้า")):
            return "Customer"
        return "Generic"

    def _submit_and_poll(
        self,
        items: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
        language: str,
    ) -> dict[str, Any]:
        endpoint = self._effective_endpoint().rstrip("/")
        api_key = self._effective_api_key()
        submit_url = (
            f"{endpoint}/language/analyze-conversations/jobs"
            f"?api-version={CONVERSATION_API_VERSION}"
        )

        participants = list({item["participantId"] for item in items})
        body = {
            "analysisInput": {
                "conversations": [{
                    "id": "call-1",
                    "language": language,
                    "modality": "text",
                    "participants": [
                        {"id": p, "role": self._speaker_to_role(p)}
                        for p in participants
                    ],
                    "conversationItems": items,
                }]
            },
            "tasks": tasks,
        }

        req = urllib.request.Request(
            submit_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            operation_url = resp.headers.get("operation-location")
        if not operation_url:
            raise RuntimeError("Azure Language API: missing operation-location header")

        poll_req = urllib.request.Request(
            operation_url,
            headers={"Ocp-Apim-Subscription-Key": api_key},
        )
        deadline = time.monotonic() + MAX_POLL_SECONDS
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_SECONDS)
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                result = json.loads(resp.read())
            status = result.get("status", "").lower()
            if status == "succeeded":
                return result
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"Azure Language job {status}: {result.get('errors')}")

        raise RuntimeError("Azure Language Service timed out")

    def _apply_results(
        self,
        transcript: TranscriptResult,
        analysis: PostCallAnalysis,
        results: dict[str, Any],
    ) -> tuple[TranscriptResult, PostCallAnalysis]:
        pii_text_by_id: dict[str, str] = {}
        pii_entity_types: set[str] = set()
        sentiment_by_id: dict[str, Sentiment] = {}
        summary_issue: str | None = None
        summary_resolution: str | None = None
        summary_narrative: str | None = None
        follow_up_items: list[str] = []

        for task in results.get("tasks", {}).get("items", []):
            if task.get("status") != "succeeded":
                continue
            task_name = task.get("taskName", "")
            conv = (task.get("results", {}).get("conversations") or [{}])[0]

            if task_name == "pii":
                for item in conv.get("conversationItems", []):
                    redacted = (item.get("redactedContent") or {}).get("text")
                    if redacted:
                        pii_text_by_id[item["id"]] = redacted
                    for entity in item.get("entities", []):
                        cat = entity.get("category")
                        if cat:
                            pii_entity_types.add(cat)

            elif task_name == "sentiment":
                for item in conv.get("conversationItems", []):
                    sent_str = item.get("sentiment", "")
                    if sent_str:
                        sentiment_by_id[item["id"]] = self._map_sentiment(sent_str)

            elif task_name == "summary":
                for s in conv.get("summaries", []):
                    aspect = s.get("aspect", "")
                    text = s.get("text", "").strip()
                    if not text:
                        continue
                    if aspect == "issue":
                        summary_issue = text
                    elif aspect == "resolution":
                        summary_resolution = text
                    elif aspect == "narrative":
                        summary_narrative = text
                    elif aspect == "followUpTasks":
                        follow_up_items = [
                            line.lstrip("•-*123456789. ").strip()
                            for line in text.splitlines()
                            if line.strip()
                        ]

        # Apply PII redaction and per-utterance sentiment to segments
        new_segments = []
        for i, seg in enumerate(transcript.segments):
            updates: dict[str, Any] = {}
            if str(i) in pii_text_by_id:
                updates["text"] = pii_text_by_id[str(i)]
            if str(i) in sentiment_by_id:
                updates["sentiment"] = sentiment_by_id[str(i)]
            new_segments.append(seg.model_copy(update=updates) if updates else seg)

        new_transcript = transcript.model_copy(
            update={
                "segments": new_segments,
                "full_text": "\n".join(f"{s.speaker}: {s.text}" for s in new_segments),
            }
        )

        analysis_updates: dict[str, Any] = {}
        if summary_issue:
            analysis_updates["azure_summary_issue"] = summary_issue
        if summary_resolution:
            analysis_updates["azure_summary_resolution"] = summary_resolution
        if summary_narrative:
            analysis_updates["azure_summary_narrative"] = summary_narrative
        if follow_up_items:
            analysis_updates["azure_follow_up_items"] = follow_up_items
        if pii_entity_types:
            analysis_updates["pii_entities_found"] = sorted(pii_entity_types)

        new_analysis = (
            analysis.model_copy(update=analysis_updates) if analysis_updates else analysis
        )
        return new_transcript, new_analysis

    @staticmethod
    def _map_sentiment(value: str) -> Sentiment:
        return {
            "positive": Sentiment.positive,
            "negative": Sentiment.negative,
            "neutral": Sentiment.neutral,
            "mixed": Sentiment.mixed,
        }.get(value.lower(), Sentiment.neutral)
