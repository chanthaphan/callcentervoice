import json
import threading
from pathlib import Path

from app.models import CallRecord


class CallStore:
    def __init__(self, data_dir: Path):
        self.calls_dir = data_dir / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, CallRecord] = {}
        self._load_all()

    def _path(self, call_id: str) -> Path:
        return self.calls_dir / f"{call_id}.json"

    def _load_all(self) -> None:
        for path in self.calls_dir.glob("*.json"):
            try:
                record = CallRecord.model_validate_json(path.read_text(encoding="utf-8"))
                self._cache[record.id] = record
            except (json.JSONDecodeError, ValueError):
                continue

    def save(self, record: CallRecord) -> CallRecord:
        with self._lock:
            self._cache[record.id] = record
            self._path(record.id).write_text(
                record.model_dump_json(indent=2),
                encoding="utf-8",
            )
        return record

    def get(self, call_id: str) -> CallRecord | None:
        return self._cache.get(call_id)

    def delete(self, call_id: str) -> bool:
        with self._lock:
            if call_id not in self._cache:
                return False
            del self._cache[call_id]
            path = self._path(call_id)
            if path.exists():
                path.unlink()
        return True

    def list(self) -> list[CallRecord]:
        with self._lock:
            records = list(self._cache.values())
        return sorted(records, key=lambda item: item.updated_at, reverse=True)
