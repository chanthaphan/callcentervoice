import logging

from pydantic import BaseModel

from app.config import Settings
from app.product_kb import OTHER, load_bbl_products, load_page_text, normalize_products
from app.models import (
    Gender,
    JourneyMoment,
    KbCheck,
    PersonProfile,
    PostCallAnalysis,
    Sentiment,
    SpeakerRole,
    ToneFlag,
    TranscriptResult,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


class _DiarizeSegment(BaseModel):
    speaker: str  # exactly "customer" or "call_center_staff"


class _DiarizeResult(BaseModel):
    segments: list[_DiarizeSegment]


class _PIISpan(BaseModel):
    text: str  # the PII substring exactly as it appears in the transcript
    category: str  # e.g. person_name, address, birth_date, account_number, card_number


class _PIIResult(BaseModel):
    spans: list[_PIISpan]


class _KbVerifyResult(BaseModel):
    checks: list[KbCheck]


KB_VERIFY_PROMPT = """You are a QA reviewer for a Thai bank call center.
You are given the official product knowledge base (KB) and a call transcript.
Examine ONLY the statements made by the bank's call-center staff/agent about the product(s).
For each distinct factual claim the staff makes (benefits, fees, interest, conditions,
eligibility, points, promotions, procedures), output one item with:
- claim: the staff's claim, concisely, in Thai
- verdict: exactly one of
    "supported"   — the KB clearly backs the claim
    "not_found"   — the KB does not mention this, so it cannot be confirmed
    "contradicts" — the claim conflicts with what the KB states
- evidence: a short Thai quote/paraphrase from the KB the verdict is based on (null if not_found)
- product: which product the claim concerns
Judge ONLY factual product claims by the staff — ignore greetings, identity verification,
small talk, and anything the customer says. If the staff made no factual product claims,
return an empty list. Write claim and evidence in Thai."""


PII_REDACTION_PROMPT = """You are a PII detection specialist for a Thai bank call center.
Read the transcript and return every span of personal data that should be redacted.

Redact (return the verbatim substring exactly as written, including Thai text):
- Personal names of customers or staff (ชื่อ-นามสกุล)
- Home/mailing addresses
- Birth dates / dates of birth (วันเกิด), in any format
- Account numbers, card numbers, national ID numbers, reference/case numbers
- Phone numbers and email addresses

Do NOT return:
- Monetary amounts, balances, interest rates, or generic dates that are not birth dates
- Product names, bank/branch names, or generic job titles
- Whole sentences — return only the minimal PII substring itself

Return each detected item as it appears verbatim in the text so it can be matched
exactly. If there is no PII, return an empty list."""


DIARIZE_PROMPT = """You are a call center diarization specialist for a Thai bank.
Your only task: read the full transcript and assign each segment to exactly one of:
  "customer"         — the person calling for help
  "call_center_staff" — the bank employee answering

Rules:
- Read ALL segments before assigning. Do NOT assume speakers alternate.
- The same speaker may occupy many consecutive segments.

call_center_staff — any one is sufficient:
- Opens with a company or service greeting ("ธนาคาร...", "ยินดีให้บริการ", "สวัสดีครับ/ค่ะ [company]")
- Asks identity-verification questions (ชื่อ, เลขบัตรประชาชน, วันเกิด, เบอร์โทร)
- Speaks in formal, consistently polite service language
- Explains bank procedures, policies, or eligibility criteria
- Provides solutions, case/reference numbers, or escalates on behalf of the caller
- Closes with a service farewell ("ขอบคุณที่ใช้บริการ", "มีอะไรให้ช่วยอีกไหมครับ/ค่ะ")

customer — any one is sufficient:
- Describes a problem they personally experienced with their own account or card
- Asks about their own transactions, charges, balance, or card status
- Requests action: block card, cancel, refund, dispute a charge
- Expresses frustration, urgency, or confusion about their personal situation
- Uses informal language, filler sounds (เอ่อ, อ่า, อ้าว), or incomplete sentences
- Asks WHY / HOW / WHEN from a first-person caller perspective

TIEBREAKER: the first speaker to use a service greeting = call_center_staff.
The speaker who opens by describing a personal problem = customer."""


SYSTEM_PROMPT = """You are a post-call quality analyst for a call center.
Return strict structured analysis. Focus on customer sentiment, agent handling,
emotion journey, risks, and concrete next actions.

Speaker classification rules:
Read the ENTIRE transcript before assigning roles. Classify every speaker label as either customer or call_center_staff.
The same speaker label must receive the same role throughout — never flip roles mid-call.
Speakers do NOT alternate strictly; a customer or agent may occupy many consecutive segments.

call_center_staff strong signals (any one is sufficient):
- Opens the call with a company greeting (e.g. "ธนาคาร...", "บริการลูกค้า...", "สวัสดีครับ/ค่ะ ยินดีให้บริการ")
- Verifies the caller's identity (name, ID card, phone number, date of birth)
- Uses formal service language and polite sentence-ending particles consistently
- Explains bank procedures, policies, or eligibility criteria
- Offers solutions, provides reference numbers, or escalates on behalf of the caller
- Closes the call with a service farewell ("ขอบคุณที่ใช้บริการ", "มีอะไรให้ช่วยเหลืออีกไหมครับ/ค่ะ")
- Speaks more calmly, formally, and procedurally than the other party

customer strong signals:
- Initiates the problem, complaint, or inquiry ("ผม/หนูโทรมาเพราะ...", "มีปัญหาเรื่อง...")
- Asks questions about their own account, card balance, transactions, or charges
- Disputes a charge, requests a refund, or asks for a block/cancellation
- Expresses frustration, urgency, or confusion about their situation
- Asks "ทำไม", "ได้ไหม", "นานแค่ไหน", "ต้องทำยังไง" — caller-perspective questions
- May use informal language, incomplete sentences, or filler sounds ("เอ่อ", "อ่า", "อ้าว")

Tiebreaker: the party who speaks FIRST in the call and uses a service greeting is almost always call_center_staff. The party who describes a problem they are experiencing is almost always the customer.

Sentiment scoring — judge the customer's EMOTIONAL TONE, not the call topic. Calling about a
problem (a lost card, a disputed charge, a question, a blocked account) is the REASON for the
call, NOT by itself evidence of negative sentiment. Most routine service calls are neutral.
Default to neutral and only move off it when the customer's own words clearly carry emotion.
- customer_sentiment neutral (DEFAULT): customer reports an issue or asks questions in a calm,
  matter-of-fact, cooperative way — even when the topic is a dispute, fraud, or a blocked card,
  and even if they sound mildly concerned. This is the expected tone for the majority of calls.
- customer_sentiment negative: the wording shows REAL, sustained emotional negativity —
  clear frustration or anger, repeated impatience, panic/distress that is not quickly reassured,
  blaming the bank, or explicit dissatisfaction with the service or outcome. A single worried
  phrase, or simply having a serious problem, is NOT enough on its own.
- customer_sentiment positive: customer explicitly expresses relief, gratitude, or satisfaction.
- customer_sentiment mixed: the emotional arc clearly shifts — e.g. starts frustrated or anxious
  and ends relieved/satisfied once the issue is handled.
- When genuinely torn between neutral and negative, choose neutral.
- agent_sentiment neutral: agent is calm and procedural (default for well-handled calls).
- agent_sentiment positive: agent expresses warmth or celebrates a resolution.
- agent_sentiment negative: agent is dismissive, curt, or unhelpful.

ToneFlag assignment (only assign if clearly evidenced in the text):
- frustrated: customer repeats themselves, uses phrases like "still not resolved", "how many times", or shows impatience.
- angry: strong negative language, raised-tone markers, threats to escalate or close account.
- urgent: customer uses time-pressure words — "right now", "immediately", "today", "ASAP", "ด่วน", "เดี๋ยวนี้".
- confused: customer asks for clarification, says they don't understand, or asks the same question twice.
- satisfied: customer thanks the agent or confirms the issue is resolved.
- calm: customer is cooperative and shows no signs of distress.
- empathetic: agent explicitly acknowledges the customer's feelings ("I understand", "ผมเข้าใจ", "ดิฉันเข้าใจ", "ขอโทษที่ทำให้รอ").
- procedural: agent follows verification scripts, explains policy steps, or reads from a process.

Highlight critical flags such as fraud, scam, unauthorized charge, chargeback, dispute, card blocking, missing refund, compliance risk, angry escalation, or repeated failed resolution.
Classify credit card products into credit_card_products following the rule given in the per-call instructions (scope to the known BBL products; bucket anything else as the "other" label provided).

Person profile extraction (fill every field; use null only when there is genuinely no evidence):
- session_topic: one concise Thai sentence capturing the core reason for this call (e.g. "ลูกค้าโต้แย้งรายการที่ไม่รู้จักจำนวน 2,500 บาทบนบัตรเครดิต").
- agent_profile.name: agent's name if spoken aloud. null if not mentioned.
- agent_profile.gender: detect from Thai gendered particles used by the agent —
    Male particles: ครับ, ค้าบ, ฮะ, ผม → "M"
    Female particles: ค่ะ, คะ, ค้า, จ้า, ดิฉัน → "F"
    If both types appear for the same speaker (code-switching or ASR error), use the majority.
    If no particles are clear → "Not sure".
- agent_profile.persona: 1–2 sentences on the agent's communication style, formality, and tone.
- customer_profile.name: customer's name if mentioned or verified during identity check. null if unknown.
- customer_profile.gender: same particle detection rules as agent — look for ครับ/ค้าบ/ฮะ/ผม (M) or ค่ะ/คะ/ค้า/จ้า/ดิฉัน (F).
- customer_profile.persona: 1–2 sentences on the customer's emotional state, primary concern, and how they communicate."""


class PostCallAgent:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, transcript: TranscriptResult) -> PostCallAnalysis:
        provider = self.settings.llm_provider.lower()
        if provider == "mock":
            return self._mock_analysis(transcript)
        return self._llm_analysis(transcript)

    def diarize_transcript(self, transcript: TranscriptResult, profile_context: str | None = None) -> TranscriptResult:
        provider = self.settings.llm_provider.lower()
        if provider == "mock" or not transcript.segments:
            return transcript
        llm = self._build_llm().with_structured_output(_DiarizeResult)
        numbered = "\n".join(f"[{i}] {seg.text}" for i, seg in enumerate(transcript.segments))
        profile_hint = (
            f"\n\nContext from previous analysis:\n{profile_context}" if profile_context else ""
        )
        result = llm.invoke([
            ("system", DIARIZE_PROMPT + profile_hint),
            (
                "human",
                f"Assign speakers for {len(transcript.segments)} segments. "
                f"Return exactly {len(transcript.segments)} entries.\n\n{numbered}",
            ),
        ])
        diarized = result if isinstance(result, _DiarizeResult) else _DiarizeResult.model_validate(result)
        if len(diarized.segments) != len(transcript.segments):
            logger.warning(
                "Diarization count mismatch (got %d, expected %d); keeping original speaker labels",
                len(diarized.segments),
                len(transcript.segments),
            )
            return transcript
        new_segments = []
        for seg, da in zip(transcript.segments, diarized.segments):
            speaker = "Customer" if da.speaker == "customer" else "Agent"
            new_segments.append(seg.model_copy(update={"speaker": speaker}))
        return TranscriptResult(
            language=transcript.language,
            duration_seconds=transcript.duration_seconds,
            segments=new_segments,
            full_text="\n".join(f"{s.speaker}: {s.text}" for s in new_segments),
        )

    def detect_pii_spans(self, transcript: TranscriptResult) -> list[str]:
        """Return verbatim PII substrings the LLM finds (names, addresses, birth dates, …).

        The caller masks these deterministically via ``mask_literal_spans`` — the LLM
        only locates PII, it never rewrites the transcript. Returns [] for the mock
        provider or an empty transcript.
        """
        provider = self.settings.llm_provider.lower()
        if provider == "mock" or not transcript.segments:
            return []
        llm = self._build_llm().with_structured_output(_PIIResult)
        numbered = "\n".join(f"[{i}] {seg.text}" for i, seg in enumerate(transcript.segments))
        result = llm.invoke([
            ("system", PII_REDACTION_PROMPT),
            ("human", f"Find all PII spans in this transcript:\n\n{numbered}"),
        ])
        parsed = result if isinstance(result, _PIIResult) else _PIIResult.model_validate(result)
        return [span.text for span in parsed.spans if span.text and span.text.strip()]

    def verify_against_kb(self, transcript: TranscriptResult, analysis: PostCallAnalysis) -> list[KbCheck]:
        """Check the staff's factual product claims against the product KB.

        Returns [] for the mock provider, when no BBL product was discussed, or when
        no KB page is available. Reads each discussed product's page.md as the ground
        truth (capped to the two most relevant products to bound context/cost).
        """
        provider = self.settings.llm_provider.lower()
        if provider == "mock" or not transcript.segments:
            return []
        bbl = [p for p in analysis.credit_card_products if p != OTHER]
        if not bbl:
            return []
        blocks = []
        for product in bbl[:2]:
            text = load_page_text(product, self.settings.product_kb_dir)
            if text:
                blocks.append(f"### {product}\n{text}")
        if not blocks:
            return []
        kb_context = "\n\n".join(blocks)
        llm = self._build_llm().with_structured_output(_KbVerifyResult)
        result = llm.invoke([
            ("system", KB_VERIFY_PROMPT),
            (
                "human",
                f"Product knowledge base:\n{kb_context}\n\nCall transcript:\n{transcript.full_text}\n\n"
                "Return the bank staff's factual product claims, each with a verdict.",
            ),
        ])
        parsed = result if isinstance(result, _KbVerifyResult) else _KbVerifyResult.model_validate(result)
        checks: list[KbCheck] = []
        for check in parsed.checks:
            if not check.claim or not check.claim.strip():
                continue
            if not check.product and len(bbl) == 1:
                check.product = bbl[0]
            checks.append(check)
        return checks

    def _llm_analysis(self, transcript: TranscriptResult) -> PostCallAnalysis:
        llm = self._build_llm().with_structured_output(PostCallAnalysis)
        language = self.settings.analysis_language
        bbl_products = load_bbl_products(self.settings.product_kb_dir)
        product_rule = (
            "Known BBL credit-card products:\n- " + "\n- ".join(bbl_products) + "\n"
            f"For credit_card_products, name a product ONLY if it is one of these (use its exact "
            f"name above); if a credit card that is NOT a BBL product is mentioned (another bank, a "
            f"bare network, or a generic card), output '{OTHER}'. Include only cards actually mentioned."
            if bbl_products
            else "Fill credit_card_products with product/network/issuer names mentioned."
        )
        unique_speakers = sorted({s.speaker for s in transcript.segments})
        speakers_str = ", ".join(f'"{s}"' for s in unique_speakers)
        messages = [
            ("system", f"{SYSTEM_PROMPT}\nWrite all free-text fields in {language}."),
            (
                "human",
                f"Analyze this transcript. The speaker labels present are: {speakers_str}.\n\n"
                f"{transcript.full_text}\n\n"
                "Instructions:\n"
                f"1. Fill speaker_classifications for every label in [{speakers_str}], assigning each the correct role "
                "(customer or call_center_staff) based on behavioral evidence across the full transcript. "
                "Use exactly the same label string as it appears in the transcript.\n"
                "2. Use segment timestamps to build the emotion journeys.\n"
                "3. Fill critical_flags with short business-critical flags.\n"
                f"4. {product_rule}\n"
                f"5. Return all free-text fields in {language}.",
            ),
        ]
        result = llm.invoke(messages)
        analysis = result if isinstance(result, PostCallAnalysis) else PostCallAnalysis.model_validate(result)
        analysis.credit_card_products = normalize_products(analysis.credit_card_products, bbl_products)
        return self._align_speaker_labels(analysis, unique_speakers)

    def _align_speaker_labels(self, analysis: PostCallAnalysis, transcript_speakers: list[str]) -> PostCallAnalysis:
        sc_speakers = {item.speaker for item in analysis.speaker_classifications}
        if sc_speakers and sc_speakers.issubset(set(transcript_speakers)):
            return analysis

        _customer_kw = {"ลูกค้า", "customer", "cust", "caller"}
        _staff_kw = {"เจ้าหน้าที่", "staff", "agent", "พนักงาน", "ธนาคาร"}
        customer_label = next(
            (s for s in transcript_speakers if any(kw in s.lower() for kw in _customer_kw)), None
        ) or (transcript_speakers[0] if transcript_speakers else None)
        staff_label = next(
            (s for s in transcript_speakers if any(kw in s.lower() for kw in _staff_kw)), None
        ) or (transcript_speakers[1] if len(transcript_speakers) > 1 else customer_label)

        new_sc = []
        for item in analysis.speaker_classifications:
            if item.speaker in transcript_speakers:
                new_sc.append(item)
            elif item.role == SpeakerRole.customer and customer_label:
                new_sc.append(item.model_copy(update={"speaker": customer_label}))
            elif item.role == SpeakerRole.call_center_staff and staff_label:
                new_sc.append(item.model_copy(update={"speaker": staff_label}))
            else:
                new_sc.append(item)
        return analysis.model_copy(update={"speaker_classifications": new_sc})

    @staticmethod
    def _temperature_kwargs(model_name: str | None) -> dict:
        # GPT-5 series and o-series reasoning models only accept the default temperature (1);
        # langchain otherwise injects 0.7, which they reject. Other models get 0 for
        # deterministic analysis.
        name = (model_name or "").lower()
        if name.startswith(("gpt-5", "o1", "o3", "o4")) or "gpt-5" in name:
            return {"temperature": 1}
        return {"temperature": 0}

    def _build_llm(self):
        provider = self.settings.llm_provider.lower()
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                request_timeout=self.settings.llm_request_timeout_seconds,
                max_retries=4,
                **self._temperature_kwargs(self.settings.openai_model),
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
                request_timeout=self.settings.llm_request_timeout_seconds,
                max_retries=4,
                **self._temperature_kwargs(self.settings.azure_openai_deployment or self.settings.openai_model),
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
                max_retries=4,
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
            session_topic="Customer reports a suspicious credit card charge and requests verification.",
            agent_profile=PersonProfile(
                name=None,
                gender=Gender.not_sure,
                persona="Calm and procedural. Follows verification scripts and explains next steps clearly.",
            ),
            customer_profile=PersonProfile(
                name=None,
                gender=Gender.not_sure,
                persona="Concerned about an unrecognized charge. Communicates urgently and requests immediate action.",
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
