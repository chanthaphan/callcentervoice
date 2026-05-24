from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

from app.models import CallRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id              TEXT PRIMARY KEY,
    file_name       TEXT,
    file_path       TEXT,
    status          TEXT,
    stage           TEXT,
    progress_message TEXT,
    error           TEXT,
    session_topic   TEXT,
    created_at      TEXT,
    updated_at      TEXT,
    data            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_updated ON calls(updated_at);
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_file_path ON calls(file_path);
"""

_SUMMARY_COLS = (
    "id, file_name, file_path, status, stage, progress_message, error, "
    "session_topic, created_at, updated_at"
)


class CallStore:
    """SQLite-backed store. Records are read lazily by id; nothing is held in
    memory, so it scales to large datasets. Thread-safe via a single guarded
    connection (DB ops are sub-millisecond next to the transcription/LLM work)."""

    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.calls_dir = data_dir / "calls"  # legacy JSON location (for one-time import)
        self._db_path = data_dir / "calls.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;"
        )
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._import_legacy_json()

    # ── internal ─────────────────────────────────────────────────────────────
    def _import_legacy_json(self) -> None:
        if not self.calls_dir.is_dir():
            return
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            if count:
                return
            for path in self.calls_dir.glob("*.json"):
                try:
                    record = CallRecord.model_validate_json(path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                self._upsert(record)
            self._conn.commit()

    def _upsert(self, record: CallRecord) -> None:
        self._conn.execute(
            """INSERT INTO calls
                (id, file_name, file_path, status, stage, progress_message, error,
                 session_topic, created_at, updated_at, data)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 file_name=excluded.file_name, file_path=excluded.file_path,
                 status=excluded.status, stage=excluded.stage,
                 progress_message=excluded.progress_message, error=excluded.error,
                 session_topic=excluded.session_topic, created_at=excluded.created_at,
                 updated_at=excluded.updated_at, data=excluded.data""",
            (
                record.id,
                record.file_name,
                record.file_path,
                getattr(record.status, "value", record.status),
                getattr(record.stage, "value", record.stage),
                record.progress_message,
                record.error,
                record.analysis.session_topic if record.analysis else None,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.model_dump_json(),
            ),
        )

    # ── public API ───────────────────────────────────────────────────────────
    def save(self, record: CallRecord) -> CallRecord:
        with self._lock:
            self._upsert(record)
            self._conn.commit()
        return record

    def get(self, call_id: str) -> CallRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM calls WHERE id = ?", (call_id,)).fetchone()
        return CallRecord.model_validate_json(row["data"]) if row else None

    def delete(self, call_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list(self) -> list[CallRecord]:
        """Full records, newest first. Loads everything — prefer list_summaries()/
        iter_records() for large datasets."""
        with self._lock:
            rows = self._conn.execute("SELECT data FROM calls ORDER BY updated_at DESC").fetchall()
        return [CallRecord.model_validate_json(r["data"]) for r in rows]

    def iter_records(self) -> Iterator[CallRecord]:
        """Stream full records one at a time (no full in-memory list)."""
        with self._lock:
            rows = self._conn.execute("SELECT id FROM calls ORDER BY updated_at DESC").fetchall()
        for row in rows:
            record = self.get(row["id"])
            if record:
                yield record

    def list_summaries(
        self, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[dict]:
        """Lightweight rows (no transcript/analysis) for the sidebar list."""
        where = "WHERE status = ?" if status else ""
        params: list = [status] if status else []
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_SUMMARY_COLS} FROM calls {where} "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self, status: str | None = None) -> int:
        with self._lock:
            if status:
                row = self._conn.execute("SELECT COUNT(*) FROM calls WHERE status = ?", (status,)).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()
        return row[0]
