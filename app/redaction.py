import re

from app.models import TranscriptResult

# Mask any run of 9 or more digits (Arabic or Thai numerals), allowing single
# space/dash separators. Catches Thai national IDs (13), phone numbers (9–10),
# and card/account numbers (13–19). Dot is NOT a separator, so decimals like
# 3.14159265 and short numbers like amounts/years are left intact.
_DIGIT = r"[0-9๐-๙]"
_PII_RE = re.compile(rf"{_DIGIT}(?:[ \-]?{_DIGIT}){{8,}}")
MASK = "[ปกปิด]"


def redact_pii(text: str | None) -> str | None:
    if not text:
        return text
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
