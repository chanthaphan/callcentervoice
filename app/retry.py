import logging
import time
import urllib.error
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP = {429, 500, 502, 503, 504}
T = TypeVar("T")


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_HTTP
    # URLError covers connection resets/DNS; TimeoutError covers socket timeouts
    return isinstance(exc, (urllib.error.URLError, TimeoutError))


def _retry_after_seconds(exc: Exception) -> float | None:
    if isinstance(exc, urllib.error.HTTPError) and exc.headers:
        raw = exc.headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def retry_request(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    label: str = "request",
) -> T:
    """Call fn(), retrying transient HTTP 429/5xx and network/timeout errors with
    exponential backoff (honoring Retry-After when present). Non-transient errors
    and the final attempt propagate immediately."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below unless transient
            if attempt >= attempts or not _is_transient(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = max(delay, _retry_after_seconds(exc) or 0.0)
            logger.warning(
                "%s failed (%s); retrying in %.1fs (attempt %d/%d)",
                label, exc, delay, attempt, attempts,
            )
            time.sleep(delay)
    raise RuntimeError(f"{label} exhausted retries")  # unreachable
