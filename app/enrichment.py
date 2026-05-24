from app.models import PostCallAnalysis, Sentiment, SpeakerRole, TranscriptResult, TranscriptSegment


CRITICAL_TERMS = {
    "fraud": "fraud",
    "scam": "scam",
    "unauthorized": "unauthorized",
    "chargeback": "chargeback",
    "dispute": "dispute",
    "block card": "block card",
    "refund": "refund",
    "ฟรอด": "fraud",
    "ทุจริต": "fraud",
    "มิจฉาชีพ": "scam",
    "ชาร์จแบ็ค": "chargeback",
    "ปฏิเสธรายการ": "dispute",
    "เงินคืน": "refund",
}

PRODUCT_TERMS = {
    "visa": "Visa",
    "วีซ่า": "Visa",
    "mastercard": "Mastercard",
    "master card": "Mastercard",
    "มาสเตอร์การ์ด": "Mastercard",
    "american express": "American Express",
    "amex": "American Express",
    "jcb": "JCB",
    "unionpay": "UnionPay",
    "union pay": "UnionPay",
    "scb": "SCB",
    "ktc": "KTC",
    "krungsri": "Krungsri",
    "first choice": "First Choice",
    "บัตรเครดิต": "บัตรเครดิต",
}


def enrich_transcript_with_analysis(
    transcript: TranscriptResult,
    analysis: PostCallAnalysis,
) -> TranscriptResult:
    role_by_speaker = {
        item.speaker: item.role
        for item in analysis.speaker_classifications
        if item.speaker
    }
    display_by_speaker = {
        item.speaker: item.display_name
        for item in analysis.speaker_classifications
        if item.speaker and item.display_name
    }

    enriched: list[TranscriptSegment] = []
    for segment in transcript.segments:
        data = segment.model_copy(deep=True)
        role = role_by_speaker.get(segment.speaker)
        if role == SpeakerRole.customer:
            data.speaker = display_by_speaker.get(segment.speaker) or "ลูกค้า"
            data.sentiment = _nearest_sentiment(segment.start, analysis.customer_emotion_journey, analysis.customer_sentiment)
            data.tone_flags = analysis.customer_tone_flags
        elif role == SpeakerRole.call_center_staff:
            data.speaker = display_by_speaker.get(segment.speaker) or "เจ้าหน้าที่"
            data.sentiment = _nearest_sentiment(segment.start, analysis.agent_emotion_journey, analysis.agent_sentiment)
            data.tone_flags = analysis.agent_tone_flags
        else:
            data.sentiment = _overall_sentiment(segment.start, analysis)
        data.keywords = _keywords_for_segment(segment.text, analysis)
        enriched.append(data)

    return TranscriptResult(
        language=transcript.language,
        duration_seconds=transcript.duration_seconds,
        segments=enriched,
        full_text="\n".join(f"{item.speaker}: {item.text}" for item in enriched),
    )


def _nearest_sentiment(time_seconds: float, moments, fallback: Sentiment) -> Sentiment:
    if not moments:
        return fallback
    nearest = min(moments, key=lambda item: abs(item.time_seconds - time_seconds))
    return nearest.sentiment


def _overall_sentiment(time_seconds: float, analysis: PostCallAnalysis) -> Sentiment:
    moments = [*analysis.customer_emotion_journey, *analysis.agent_emotion_journey]
    return _nearest_sentiment(time_seconds, moments, Sentiment.neutral)


def _keywords_for_segment(text: str, analysis: PostCallAnalysis) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    for topic in analysis.key_topics + analysis.critical_flags + analysis.credit_card_products:
        if topic and topic.lower() in lowered:
            matches.append(topic)
    for needle, label in CRITICAL_TERMS.items():
        if needle in lowered:
            matches.append(label)
    for needle, label in PRODUCT_TERMS.items():
        if needle in lowered:
            matches.append(label)
    return list(dict.fromkeys(matches))[:6]
