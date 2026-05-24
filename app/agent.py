from app.config import Settings
from app.models import (
    JourneyMoment,
    PostCallAnalysis,
    Sentiment,
    ToneFlag,
    TranscriptResult,
)


SYSTEM_PROMPT = """You are a post-call quality analyst for a call center.
Return strict structured analysis. Focus on customer sentiment, agent handling,
emotion journey, risks, and concrete next actions.
Classify every transcript speaker as either customer or call_center_staff.
Use behavioral evidence: staff usually explains policies, verifies identity, confirms procedures, and closes service; customers usually ask for help, dispute charges, express concerns, or request action.
Highlight critical flags such as fraud, scam, unauthorized charge, chargeback, dispute, card blocking, missing refund, compliance risk, angry escalation, or repeated failed resolution.
Classify any credit card product, network, issuer, or branded card mentioned, such as Visa, Mastercard, JCB, UnionPay, American Express, SCB, KTC, Krungsri, First Choice, or generic credit card."""


class PostCallAgent:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, transcript: TranscriptResult) -> PostCallAnalysis:
        provider = self.settings.llm_provider.lower()
        if provider == "mock":
            return self._mock_analysis(transcript)
        return self._llm_analysis(transcript)

    def _llm_analysis(self, transcript: TranscriptResult) -> PostCallAnalysis:
        llm = self._build_llm().with_structured_output(PostCallAnalysis)
        language = self.settings.analysis_language
        messages = [
            ("system", f"{SYSTEM_PROMPT}\nWrite all free-text fields in {language}."),
            (
                "human",
                "Analyze this diarized transcript:\n\n"
                f"{transcript.full_text}\n\n"
                "Use segment timestamps to build the customer and agent journeys. "
                "Fill speaker_classifications for every speaker label in the transcript. "
                "Fill critical_flags with short business-critical flags and credit_card_products with product/network/issuer names mentioned. "
                "Treat the transcript as an initial multilingual ASR result; keep Thai, English, and product names as spoken, and use the overall context to normalize obvious product names and flags. "
                f"Return summaries, labels, risks, topics, notes, and next actions in {language}.",
            ),
        ]
        result = llm.invoke(messages)
        if isinstance(result, PostCallAnalysis):
            return result
        return PostCallAnalysis.model_validate(result)

    def _build_llm(self):
        provider = self.settings.llm_provider.lower()
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                temperature=0,
                request_timeout=self.settings.llm_request_timeout_seconds,
            )
        if provider == "azure_openai":
            missing = [
                name
                for name, value in {
                    "AZURE_OPENAI_API_KEY": self.settings.azure_openai_api_key,
                    "AZURE_OPENAI_ENDPOINT": self.settings.azure_openai_endpoint,
                    "AZURE_OPENAI_DEPLOYMENT": self.settings.azure_openai_deployment,
                }.items()
                if not value
            ]
            if missing:
                raise RuntimeError(f"{', '.join(missing)} required when LLM_PROVIDER=azure_openai")
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI(
                api_key=self.settings.azure_openai_api_key,
                azure_endpoint=self.settings.azure_openai_endpoint,
                azure_deployment=self.settings.azure_openai_deployment,
                api_version=self.settings.azure_openai_api_version,
                temperature=0,
                request_timeout=self.settings.llm_request_timeout_seconds,
            )
        if provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.settings.anthropic_model,
                api_key=self.settings.anthropic_api_key,
                temperature=0,
                timeout=self.settings.llm_request_timeout_seconds,
            )
        raise RuntimeError(f"Unsupported LLM_PROVIDER={self.settings.llm_provider}")

    def _mock_analysis(self, transcript: TranscriptResult) -> PostCallAnalysis:
        text = transcript.full_text.lower()
        negative = any(word in text for word in ["frustrated", "worried", "angry", "complaint", "suspicious"])
        customer_sentiment = Sentiment.negative if negative else Sentiment.neutral
        first_customer = next((item for item in transcript.segments if "customer" in item.speaker.lower()), None)
        last_customer = next(
            (item for item in reversed(transcript.segments) if "customer" in item.speaker.lower()),
            first_customer,
        )
        return PostCallAnalysis(
            summary=(
                "Customer contacted the call center about a suspicious credit card charge. "
                "The agent verified the concern, explained next steps, and the customer requested confirmation."
            ),
            speaker_classifications=[],
            customer_sentiment=customer_sentiment,
            agent_sentiment=Sentiment.neutral,
            customer_tone_flags=[ToneFlag.frustrated, ToneFlag.urgent] if negative else [ToneFlag.calm],
            agent_tone_flags=[ToneFlag.empathetic, ToneFlag.procedural],
            customer_emotion_journey=[
                JourneyMoment(
                    time_seconds=first_customer.start if first_customer else 0,
                    label="Concern raised",
                    description="Customer opens with worry about an unfamiliar charge.",
                    sentiment=Sentiment.negative if negative else Sentiment.neutral,
                ),
                JourneyMoment(
                    time_seconds=last_customer.start if last_customer else 0,
                    label="Action requested",
                    description="Customer asks for blocking or confirmation after guidance.",
                    sentiment=Sentiment.neutral,
                ),
            ],
            agent_emotion_journey=[
                JourneyMoment(
                    time_seconds=0,
                    label="Verification",
                    description="Agent uses a procedural, helpful tone to verify the issue.",
                    sentiment=Sentiment.neutral,
                )
            ],
            key_topics=["credit card", "suspicious charge", "verification", "confirmation"],
            critical_flags=["suspicious charge", "possible fraud"],
            credit_card_products=["credit card"],
            risks=["Possible duplicate or suspicious authorization requires follow-up."],
            next_actions=[
                "Confirm whether the disputed charge was blocked or escalated.",
                "Send the customer a written confirmation with the case reference.",
                "Review duplicate authorization handling for compliance.",
            ],
            quality_notes=["Agent acknowledged the concern and moved toward resolution."],
        )
