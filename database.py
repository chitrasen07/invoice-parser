"""SQLite storage shared by the deterministic and LLM extractors.

Rows are unique on (source_file, extraction_method), so a deterministic run
and an LLM run of the same PDF both remain stored.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    status TEXT NOT NULL,
    invoice_id TEXT,
    invoice_date TEXT,
    seller_name TEXT,
    buyer_name TEXT,
    grand_total REAL,
    currency TEXT,
    payload_json TEXT,
    warnings_json TEXT,
    error TEXT,
    extraction_time_ms INTEGER,
    model TEXT,
    extracted_at TEXT NOT NULL,
    UNIQUE (source_file, extraction_method)
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_extractions_method
ON extractions (extraction_method)
"""


@dataclass
class ExtractionRecord:
    source_file: str
    extraction_method: str
    status: str
    payload: Optional[dict[str, Any]]
    warnings: list[str]
    error: Optional[str]
    extraction_time_ms: Optional[int]
    model: Optional[str] = None


class ExtractionStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(CREATE_SQL)
        self._conn.execute(CREATE_INDEX_SQL)
        self._conn.commit()

    def save(self, record: ExtractionRecord) -> None:
        if not record.source_file.strip():
            raise ValueError("source_file is required")
        if not record.extraction_method.strip():
            raise ValueError("extraction_method is required")

        payload = record.payload or {}
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO extractions (
                source_file, extraction_method, status, invoice_id, invoice_date,
                seller_name, buyer_name, grand_total, currency, payload_json,
                warnings_json, error, extraction_time_ms, model, extracted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_file, extraction_method) DO UPDATE SET
                status = excluded.status,
                invoice_id = excluded.invoice_id,
                invoice_date = excluded.invoice_date,
                seller_name = excluded.seller_name,
                buyer_name = excluded.buyer_name,
                grand_total = excluded.grand_total,
                currency = excluded.currency,
                payload_json = excluded.payload_json,
                warnings_json = excluded.warnings_json,
                error = excluded.error,
                extraction_time_ms = excluded.extraction_time_ms,
                model = excluded.model,
                extracted_at = excluded.extracted_at
            """,
            (
                record.source_file,
                record.extraction_method,
                record.status,
                payload.get("invoice_id"),
                payload.get("invoice_date"),
                payload.get("seller_name"),
                payload.get("buyer_name"),
                payload.get("grand_total"),
                payload.get("currency"),
                json.dumps(record.payload, ensure_ascii=False) if record.payload is not None else None,
                json.dumps(record.warnings, ensure_ascii=False),
                record.error,
                record.extraction_time_ms,
                record.model,
                now,
            ),
        )
        self._conn.commit()

    def get(self, source_file: str, extraction_method: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM extractions
            WHERE source_file = ? AND extraction_method = ?
            """,
            (source_file, extraction_method),
        ).fetchone()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExtractionStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
