# Analytics, Product KB & Verification

This guide covers the QA-and-insights layer on top of the core transcription/analysis
pipeline: the **analytics dashboard**, **knowledge-base–scoped products**, **KB
verification** of staff statements, the **layered PII redaction**, and **Thai output**.

For the base pipeline see [ARCHITECTURE.md](ARCHITECTURE.md); for every env var see
[CONFIGURATION.md](CONFIGURATION.md); for the REST surface see [API.md](API.md).

---

## 1. Layered PII redaction

Redaction is a **guaranteed property of `TranscriptionService.transcribe()`** — every
transcript it returns, and every streamed partial, is masked *before* diarization, the
LLM analysis, or any enrichment ever sees it. There are two layers:

| Layer | Where | Covers | Cost |
|---|---|---|---|
| **Regex floor** (always on) | `app/redaction.py`, inside `transcribe()` | 9+ digit runs → Thai national IDs, phone, card/account numbers; **email addresses** | free, deterministic |
| **LLM span detection** (opt-in) | `app/agent.py::detect_pii_spans`, in the processor before diarization | context PII the regex can't catch — **names, addresses, birth dates** | one extra LLM call |

The LLM layer only *locates* PII (returns verbatim spans); masking is then done
deterministically in Python (`mask_literal_spans`), so the model can't rewrite or
hallucinate transcript text. Masked text uses the marker `[ปกปิด]`.

Enable the LLM layer with `LLM_PII_REDACTION=true` (off by default). Azure AI Language's
`ConversationPIIRedaction` remains available as an alternative/additional pass when
`AZURE_LANGUAGE_ENABLED=true`.

> Because the regex floor runs first, the optional LLM PII / analysis calls only ever see
> already-digit-redacted text.

---

## 2. Products scoped to the BBL knowledge base

Raw LLM product extraction is noisy (it names any network/issuer it hears). Instead,
products are scoped to a **Bangkok Bank product knowledge base** on disk:

```
data/prod_kb/Credit-Cards/
  Bangkok-Bank-Visa-Infinite-Card/
    product.json        ← { "title": "บัตรอินฟินิท ธนาคารกรุงเทพ", ... }
    page.md             ← product facts (benefits, fees, conditions) in Thai
  Bangkok-Bank-Titanium-Credit-Card/
  …                     ← 15 BBL card products
```

- `app/product_kb.py::load_bbl_products(dir)` reads the canonical Thai `title` of every
  `Bangkok-Bank-*/product.json`.
- The analysis prompt is constrained to those titles; a deterministic post-step
  (`normalize_products`) maps each mentioned card to a canonical BBL title or, for any
  non-BBL card (other banks, bare networks, generic "บัตรเครดิต"), the single bucket
  **`อื่นๆ` (Other)**. This also re-buckets historical records at read time.

Configure the KB location with `PRODUCT_KB_DIR` (default `data/prod_kb/Credit-Cards`).

---

## 3. KB verification of staff statements

When enabled, each call's **staff factual claims** are checked against the discussed
product's KB page (`page.md`) and tagged:

| Verdict | Meaning | UI |
|---|---|---|
| `supported` | the KB backs the claim | ✅ ตรงตาม KB (green) |
| `not_found` | the KB doesn't cover it — unconfirmed | ⚠️ ไม่พบใน KB (amber) |
| `contradicts` | the claim conflicts with the KB | ❌ ขัดแย้งกับ KB (red) |

Each check carries the `claim`, `verdict`, a short KB `evidence` quote, and the `product`.
It is stored on `PostCallAnalysis.kb_checks` and shown as a per-call card plus a dashboard
KPI ("% of verified calls with a non-supported claim").

Enable with `KB_VERIFICATION=true` (off by default — one extra LLM call). It runs only
when a BBL product is detected, and v1 reads each product's `page.md` only (not the PDF
brochures). Requires a real LLM provider; the `mock` provider returns no checks.

---

## 4. Thai analysis output

`ANALYSIS_LANGUAGE` defaults to `Thai`, so all free-text fields (summary, session topic,
personas, key topics, critical flags, risks, next actions, emotion-journey labels, KB
evidence) come back in Thai. Enum fields (sentiment, tone flags) stay as enums and are
rendered with Thai labels in the UI.

> Changing the language affects **future** analyses. Re-run **Re-analyze** on existing
> calls to refresh them.

---

## 5. Call dating & date-folder structure

The dashboard buckets calls by **call date**, derived by `app/stats.py::call_date`:

1. a date parsed from the file path/name — `voice/2026-05-01/call.wav`, `call_20260501.wav`
   (pattern `20YY[-_/]?MM[-_/]?DD`), else
2. the record's `created_at`.

Organising audio into `voice/YYYY-MM-DD/…` folders therefore gives meaningful
daily/weekly/monthly digests. The recursive scan keeps the date subpath in the display
name automatically.

---

## 6. Analytics dashboard

A **Dashboard** view (toggle in the header) renders entirely with self-contained SVG/CSS
— no external chart library, so it works offline/air-gapped. It shows:

- **KPIs**: total calls, analyzed, avg duration, total talk time, % negative, % positive,
  % with critical flags, and (when KB verification has run) % of calls with KB issues.
- **Call volume** over time (bar) and a **stacked customer-sentiment trend**.
- **Customer sentiment** and **top customer tone** distributions.
- **Top topics** and **critical flags**.
- **Credit-card products** with a BBL-vs-`อื่นๆ` split.
- **Monthly overview** table (volume, analyzed, avg duration, negative %, critical %, top topic).
- **KB verification** verdict breakdown (when enabled).

All KPIs and charts react to the **filter bar**: date range, customer sentiment, product
(BBL + `อื่นๆ`), and call status. Filtering is done server-side.

### `GET /api/stats`

Query params (all optional):

| Param | Example | Meaning |
|---|---|---|
| `digest` | `daily` \| `weekly` \| `monthly` | volume/trend bucketing (default `daily`) |
| `date_from`, `date_to` | `2026-05-01` | inclusive call-date range (ISO) |
| `status` | `complete,failed` | comma-separated `JobStatus` values |
| `sentiment` | `negative,mixed` | comma-separated customer-sentiment values |
| `product` | `บัตรอินฟินิท ธนาคารกรุงเทพ,อื่นๆ` | comma-separated canonical products / `อื่นๆ` |

Response shape:

```jsonc
{
  "digest": "daily",
  "totals": { "total": 8, "completed": 8, "by_status": { "complete": 8 } },
  "kpis": {
    "avg_duration_seconds": 213.0, "total_duration_seconds": 1707,
    "negative_pct": 87.5, "positive_pct": 0.0, "critical_pct": 100.0,
    "kb_issue_pct": 0.0
  },
  "kb": { "checked": 0, "verdicts": {} },
  "timeline":        [{ "period": "2026-05-05", "total": 1, "complete": 1, "failed": 0 }],
  "sentiment_trend": [{ "period": "2026-05-05", "positive": 0, "neutral": 0, "negative": 1, "mixed": 0 }],
  "sentiment":       { "negative": 7, "mixed": 1 },
  "tone_flags":      [{ "label": "frustrated", "count": 2 }],
  "topics":          [{ "label": "อายัดบัตร", "count": 5 }],
  "products":        [{ "label": "อื่นๆ", "count": 8 }],
  "critical_flags":  [{ "label": "card blocking", "count": 6 }],
  "monthly":         [{ "month": "2026-05", "total": 4, "complete": 4, "avg_duration": 213.0,
                        "negative_pct": 100.0, "critical_pct": 100.0, "top_topic": "อายัดบัตร" }],
  "facets": {
    "products":   ["…15 BBL titles…", "อื่นๆ"],
    "sentiments": ["positive", "neutral", "negative", "mixed"],
    "statuses":   ["pending", "queued", "processing", "complete", "failed"]
  }
}
```

`facets` is filter-independent (built from the KB + enums) so the UI can populate filter
controls regardless of the current selection.

---

## Configuration summary

| Variable | Default | Description |
|---|---|---|
| `ANALYSIS_LANGUAGE` | `Thai` | language for free-text analysis output |
| `PRODUCT_KB_DIR` | `data/prod_kb/Credit-Cards` | BBL product knowledge base directory |
| `LLM_PII_REDACTION` | `false` | LLM span-detection for names/addresses/birth dates |
| `KB_VERIFICATION` | `false` | verify staff claims against the product KB |

The three booleans are also editable at runtime from the Settings panel /
`PATCH /api/config`. The LLM-backed features need a real `LLM_PROVIDER` (the `mock`
provider is a no-op for them) and only affect future analyses — **Re-analyze** existing
calls to apply them.
