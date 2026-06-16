import re

from app.models import TranscriptResult

# Mask any run of 9 or more digits (Arabic or Thai numerals), allowing single
# space/dash separators. Catches Thai national IDs (13), phone numbers (9–10),
# and card/account numbers (13–19). Dot is NOT a separator, so decimals like
# 3.14159265 and short numbers like amounts/years are left intact.
_DIGIT = r"[0-9๐-๙]"
_PII_RE = re.compile(rf"{_DIGIT}(?:[ \-]?{_DIGIT}){{8,}}")
# Email addresses are unambiguous and have no digit signature, so they belong in
# the deterministic floor rather than relying on the LLM layer to catch them.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
MASK = "[ปกปิด]"


def redact_pii(text: str | None) -> str | None:
    if not text:
        return text
    text = _EMAIL_RE.sub(MASK, text)
    return _PII_RE.sub(MASK, text)


def redact_transcript(transcript: TranscriptResult | None) -> TranscriptResult | None:
    """Return a transcript with PII masked in segment text and full_text.

    Returns the same object unchanged when there is nothing to redact (so callers
    can cheaply detect whether a re-save is needed).
    """
    if not transcript or not transcript.segments:
        return transcript
    new_segments = [
        seg.model_copy(update={"text": redact_pii(seg.text)}) if seg.text else seg
        for seg in transcript.segments
    ]
    new_full = redact_pii(transcript.full_text)
    changed = new_full != transcript.full_text or any(
        n.text != o.text for n, o in zip(new_segments, transcript.segments)
    )
    if not changed:
        return transcript
    return transcript.model_copy(update={"segments": new_segments, "full_text": new_full})


def mask_literal_spans(
    transcript: TranscriptResult | None,
    spans: list[str],
) -> TranscriptResult | None:
    """Mask exact PII substrings (e.g. detected by the LLM layer) in the transcript.

    Masking is done deterministically in Python — the LLM only *locates* PII, it
    never rewrites the text. Longer spans are masked first so a shorter span that
    is a substring of a longer one cannot partially corrupt it. Like
    ``redact_transcript`` this returns the same object when nothing changes.
    """
    cleaned = sorted({s.strip() for s in spans if s and s.strip()}, key=len, reverse=True)
    if not transcript or not transcript.segments or not cleaned:
        return transcript

    def mask(text: str | None) -> str | None:
        if not text:
            return text
        for span in cleaned:
            text = text.replace(span, MASK)
        return text

    new_segments = [
        seg.model_copy(update={"text": mask(seg.text)}) if seg.text else seg
        for seg in transcript.segments
    ]
    new_full = mask(transcript.full_text)
    changed = new_full != transcript.full_text or any(
        n.text != o.text for n, o in zip(new_segments, transcript.segments)
    )
    if not changed:
        return transcript
    return transcript.model_copy(update={"segments": new_segments, "full_text": new_full})
