"""Aggregate statistics across all processed calls for the dashboard.

Pure functions over an iterable of CallRecord — no I/O, no clock — so the
results are deterministic and easy to test. The HTTP layer streams records in
via ``store.iter_records()`` and supplies the BBL product KB for scoping.
"""

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import date, timedelta

from app.models import CallRecord, JobStatus, Sentiment
from app.product_kb import normalize_products

# A date embedded in the file name / folder, e.g. "2026-05-01/call.wav",
# "call_20260501.wav", "2026/05/01-...". Year anchored to 20xx with valid
# month/day so stray digit runs (IDs, durations) don't false-match.
_DATE_RE = re.compile(r"(20\d{2})[-_/]?(0[1-9]|1[0-2])[-_/]?(0[1-9]|[12]\d|3[01])")

_VALID_DIGESTS = ("daily", "weekly", "monthly")
_SENTIMENTS = ("positive", "neutral", "negative", "mixed")


def call_date(record: CallRecord) -> date:
    """Best-effort call date: parse from the file name, else fall back to created_at."""
    match = _DATE_RE.search(record.file_name or "")
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    return record.created_at.date()


def _record_products(record: CallRecord, kb: tuple[str, ...] | list[str]) -> list[str]:
    """Normalized (KB-scoped) products for a record, or [] when no analysis."""
    if not record.analysis:
        return []
    return normalize_products(record.analysis.credit_card_products, kb)


def filter_records(
    records: Iterable[CallRecord],
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    statuses: set[str] | None = None,
    sentiments: set[str] | None = None,
    products: set[str] | None = None,
    kb: tuple[str, ...] | list[str] = (),
) -> Iterator[CallRecord]:
    """Yield records passing ALL active filters (empty/None filter = no constraint)."""
    for record in records:
        if statuses and record.status.value not in statuses:
            continue
        day = call_date(record)
        if date_from and day < date_from:
            continue
        if date_to and day > date_to:
            continue
        analysis = record.analysis
        if sentiments and (not analysis or analysis.customer_sentiment.value not in sentiments):
            continue
        if products and not (set(_record_products(record, kb)) & products):
            continue
        yield record


def _period_key(day: date, digest: str) -> str:
    if digest == "monthly":
        return f"{day.year:04d}-{day.month:02d}"
    if digest == "weekly":
        return (day - timedelta(days=day.weekday())).isoformat()  # Monday of that week
    return day.isoformat()


def _top(counter: Counter, limit: int = 8) -> list[dict]:
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def compute_stats(
    records: Iterable[CallRecord],
    digest: str = "daily",
    kb: tuple[str, ...] | list[str] = (),
) -> dict:
    """Return dashboard aggregates. ``digest`` controls the volume-timeline granularity
    (daily/weekly/monthly); the holistic table is always grouped by month. ``kb`` is the
    BBL product knowledge base used to scope products (non-BBL → the 'other' bucket)."""
    if digest not in _VALID_DIGESTS:
        digest = "daily"

    status_counts: Counter = Counter()
    timeline: dict[str, dict] = {}
    trend: dict[str, Counter] = {}  # period -> per-sentiment counts (completed calls)
    monthly: dict[str, dict] = {}
    sentiment: Counter = Counter()
    tone: Counter = Counter()
    topics: Counter = Counter()
    products: Counter = Counter()
    critical: Counter = Counter()

    kb_verdicts: Counter = Counter()
    durations: list[float] = []
    completed = 0
    negative = 0
    positive = 0
    critical_calls = 0
    kb_checked = 0
    kb_issue_calls = 0

    for record in records:
        status_counts[record.status.value] += 1
        day = call_date(record)
        period = _period_key(day, digest)

        bucket = timeline.setdefault(period, {"period": period, "total": 0, "complete": 0, "failed": 0})
        bucket["total"] += 1
        if record.status == JobStatus.complete:
            bucket["complete"] += 1
        elif record.status == JobStatus.failed:
            bucket["failed"] += 1

        month_key = _period_key(day, "monthly")
        month = monthly.setdefault(
            month_key,
            {"month": month_key, "total": 0, "complete": 0, "duration": 0.0, "dur_n": 0,
             "negative": 0, "critical": 0, "topics": Counter()},
        )
        month["total"] += 1

        analysis = record.analysis
        if record.status == JobStatus.complete and analysis:
            completed += 1
            month["complete"] += 1
            cs = analysis.customer_sentiment.value
            sentiment[cs] += 1
            trend.setdefault(period, Counter())[cs] += 1
            if analysis.customer_sentiment == Sentiment.negative:
                negative += 1
                month["negative"] += 1
            elif analysis.customer_sentiment == Sentiment.positive:
                positive += 1
            for flag in analysis.customer_tone_flags:
                tone[flag.value] += 1
            for topic in analysis.key_topics:
                topics[topic] += 1
                month["topics"][topic] += 1
            for product in normalize_products(analysis.credit_card_products, kb):
                products[product] += 1
            if analysis.critical_flags:
                critical_calls += 1
                month["critical"] += 1
            for flag in analysis.critical_flags:
                critical[flag] += 1
            if analysis.kb_checks:
                kb_checked += 1
                if any(c.verdict.value != "supported" for c in analysis.kb_checks):
                    kb_issue_calls += 1
                for c in analysis.kb_checks:
                    kb_verdicts[c.verdict.value] += 1

        seconds = record.transcript.duration_seconds if record.transcript else None
        if seconds:
            durations.append(seconds)
            month["duration"] += seconds
            month["dur_n"] += 1

    total = sum(status_counts.values())

    monthly_rows = []
    for month_key in sorted(monthly):
        m = monthly[month_key]
        top_topic = m["topics"].most_common(1)
        monthly_rows.append({
            "month": month_key,
            "total": m["total"],
            "complete": m["complete"],
            "avg_duration": round(m["duration"] / m["dur_n"], 1) if m["dur_n"] else 0,
            "negative_pct": round(100 * m["negative"] / m["complete"], 1) if m["complete"] else 0,
            "critical_pct": round(100 * m["critical"] / m["complete"], 1) if m["complete"] else 0,
            "top_topic": top_topic[0][0] if top_topic else None,
        })

    sentiment_trend = [
        {"period": period, **{s: trend.get(period, Counter()).get(s, 0) for s in _SENTIMENTS}}
        for period in sorted(timeline)
    ]

    return {
        "digest": digest,
        "totals": {"total": total, "completed": completed, "by_status": dict(status_counts)},
        "kpis": {
            "avg_duration_seconds": round(sum(durations) / len(durations), 1) if durations else 0,
            "total_duration_seconds": round(sum(durations)),
            "negative_pct": round(100 * negative / completed, 1) if completed else 0,
            "positive_pct": round(100 * positive / completed, 1) if completed else 0,
            "critical_pct": round(100 * critical_calls / completed, 1) if completed else 0,
            "kb_issue_pct": round(100 * kb_issue_calls / kb_checked, 1) if kb_checked else 0,
        },
        "kb": {"checked": kb_checked, "verdicts": dict(kb_verdicts)},
        "timeline": [timeline[key] for key in sorted(timeline)],
        "sentiment_trend": sentiment_trend,
        "sentiment": dict(sentiment),
        "tone_flags": _top(tone),
        "topics": _top(topics),
        "products": _top(products, 16),
        "critical_flags": _top(critical),
        "monthly": monthly_rows,
    }
