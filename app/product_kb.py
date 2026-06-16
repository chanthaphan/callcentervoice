"""Bangkok Bank credit-card product knowledge base.

Loads the canonical BBL product names from ``data/prod_kb/Credit-Cards`` and
normalizes free-form product mentions (from the LLM or historical records) to
either a canonical BBL product or the single "other" bucket. This keeps the
dashboard scoped to real BBL cards instead of arbitrary networks/issuers.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

OTHER = "อื่นๆ"  # bucket for any credit card that is not a BBL product

# Generic terms that are a prefix/substring of many BBL titles ("บัตรเครดิต" =
# "credit card") but carry no specific product — these must go to OTHER, never
# match a specific card.
_GENERIC = {
    "บัตรเครดิต", "บัตร", "เครดิต", "บัตรเดบิต", "เดบิต", "ธนาคารกรุงเทพ", "ธนาคาร",
    "credit card", "credit", "card", "debit", "debit card", "bbl", "bangkok bank",
}


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


@lru_cache(maxsize=8)
def _title_to_folder(kb_dir: str) -> dict[str, str]:
    """Map canonical Thai title -> product folder path. Cached (KB is read-only)."""
    base = Path(kb_dir)
    if not base.is_dir():
        return {}
    mapping: dict[str, str] = {}
    for product_json in base.glob("Bangkok-Bank-*/product.json"):
        try:
            title = json.loads(product_json.read_text(encoding="utf-8")).get("title")
        except (json.JSONDecodeError, OSError):
            continue
        if title and title.strip():
            mapping[title.strip()] = str(product_json.parent)
    return mapping


def load_bbl_products(kb_dir: str) -> tuple[str, ...]:
    """Canonical Thai product titles. Empty tuple if the KB dir is missing."""
    return tuple(sorted(_title_to_folder(kb_dir)))


_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # markdown images
_URL_RE = re.compile(r"https?://\S+")
_BLANK_RE = re.compile(r"\n{3,}")


@lru_cache(maxsize=32)
def load_page_text(title: str, kb_dir: str, max_chars: int = 20000) -> str:
    """Cleaned ``page.md`` text for a product title (images/URLs stripped, capped).

    Returns "" when the product or its page is missing. Used to ground the
    KB-verification check on the official product facts.
    """
    folder = _title_to_folder(kb_dir).get(title.strip())
    if not folder:
        return ""
    page = Path(folder) / "page.md"
    if not page.is_file():
        return ""
    text = page.read_text(encoding="utf-8")
    text = _IMG_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _BLANK_RE.sub("\n\n", text).strip()
    return text[:max_chars]


def normalize_products(raw: list[str], kb: tuple[str, ...] | list[str]) -> list[str]:
    """Map each mentioned product to a canonical BBL title, else to ``OTHER``.

    Matching is case-insensitive, whitespace-collapsed, and bidirectional-substring
    (so "บัตรอินฟินิท" matches "บัตรอินฟินิท ธนาคารกรุงเทพ"). Order is preserved and
    duplicates collapsed; ``OTHER`` appears at most once.
    """
    kb_norm = [(_norm(name), name) for name in kb]
    result: list[str] = []
    for item in raw:
        if not item or not item.strip():
            continue
        needle = _norm(item)
        if needle in _GENERIC:
            canonical = OTHER
        else:
            canonical = next(
                (name for nrm, name in kb_norm if needle == nrm or needle in nrm or nrm in needle),
                OTHER,
            )
        if canonical not in result:
            result.append(canonical)
    return result
