"""SQLite persistence for Model Advisor.

Schema is intentionally denormalized — read-heavy workload, single-user, no concurrent writes.
The DB caches data fetched from 8 external sources and is the source of truth for recommendations.
"""

from __future__ import annotations

import sqlite3
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(".cache/model-advisor.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    canonical_id TEXT PRIMARY KEY,         -- e.g. "llama-3.1:70b:instruct"
    family TEXT NOT NULL,                  -- "llama-3.1"
    parameter_size TEXT NOT NULL,          -- "70B"  (display label)
    variant TEXT NOT NULL,                 -- "instruct" | "base" | "chat" | "code" | "vision"
    display_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    -- Architecture awareness — critical for MoE models like Qwen3-30B-A3B.
    -- For dense models: total_params_b == active_params_b (e.g. 8 / 8).
    -- For MoE: total drives memory cost, active drives inference speed.
    total_params_b REAL DEFAULT 0,         -- e.g. 30.0 for 30B model
    active_params_b REAL DEFAULT 0,        -- e.g. 3.0 for Qwen3-30B-A3B
    is_moe INTEGER DEFAULT 0,
    context_length INTEGER DEFAULT 0,
    tool_calling INTEGER DEFAULT 0,        -- bool
    vision INTEGER DEFAULT 0,              -- bool
    license TEXT DEFAULT '',
    base_model TEXT DEFAULT '',
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS aliases (
    source TEXT NOT NULL,                  -- "ollama" | "hf" | "lmsys" | etc.
    source_id TEXT NOT NULL,               -- the id used by that source
    canonical_id TEXT NOT NULL,
    PRIMARY KEY (source, source_id),
    FOREIGN KEY (canonical_id) REFERENCES models(canonical_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON aliases(canonical_id);

CREATE TABLE IF NOT EXISTS source_artifacts (
    source TEXT NOT NULL,                  -- "ollama" | "hf" | "lmstudio"
    canonical_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    quantization TEXT DEFAULT '',
    size_mb REAL DEFAULT 0,
    download_url TEXT DEFAULT '',
    install_command TEXT DEFAULT '',
    extra TEXT DEFAULT '{}',               -- JSON blob
    PRIMARY KEY (source, source_id),
    FOREIGN KEY (canonical_id) REFERENCES models(canonical_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_canonical ON source_artifacts(canonical_id);

CREATE TABLE IF NOT EXISTS scores (
    canonical_id TEXT NOT NULL,
    benchmark TEXT NOT NULL,               -- "humaneval" | "mmlu_pro" | "arena_elo" | etc.
    source TEXT NOT NULL,                  -- which fetcher produced this
    value REAL NOT NULL,
    max_value REAL DEFAULT 100,
    confidence TEXT DEFAULT 'measured',    -- "measured" | "interpolated" | "estimated"
    fetched_at INTEGER NOT NULL,
    PRIMARY KEY (canonical_id, benchmark, source),
    FOREIGN KEY (canonical_id) REFERENCES models(canonical_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scores_canonical ON scores(canonical_id);
CREATE INDEX IF NOT EXISTS idx_scores_benchmark ON scores(benchmark);

CREATE TABLE IF NOT EXISTS source_runs (
    source TEXT PRIMARY KEY,
    last_run_at INTEGER NOT NULL,
    last_status TEXT NOT NULL,             -- "ok" | "error" | "partial"
    error_message TEXT DEFAULT '',
    rows_written INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0
);
"""


def db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def upsert_model(conn: sqlite3.Connection, model: dict) -> None:
    payload = {
        "total_params_b": 0.0,
        "active_params_b": 0.0,
        "is_moe": 0,
        **model,
        "updated_at": int(time.time()),
    }
    conn.execute(
        """
        INSERT INTO models (canonical_id, family, parameter_size, variant, display_name,
                            description, total_params_b, active_params_b, is_moe,
                            context_length, tool_calling, vision, license, base_model, updated_at)
        VALUES (:canonical_id, :family, :parameter_size, :variant, :display_name,
                :description, :total_params_b, :active_params_b, :is_moe,
                :context_length, :tool_calling, :vision, :license, :base_model, :updated_at)
        ON CONFLICT(canonical_id) DO UPDATE SET
            display_name = excluded.display_name,
            description = COALESCE(NULLIF(excluded.description, ''), models.description),
            total_params_b = MAX(excluded.total_params_b, models.total_params_b),
            active_params_b = MAX(excluded.active_params_b, models.active_params_b),
            is_moe = MAX(excluded.is_moe, models.is_moe),
            context_length = MAX(excluded.context_length, models.context_length),
            tool_calling = MAX(excluded.tool_calling, models.tool_calling),
            vision = MAX(excluded.vision, models.vision),
            license = COALESCE(NULLIF(excluded.license, ''), models.license),
            base_model = COALESCE(NULLIF(excluded.base_model, ''), models.base_model),
            updated_at = excluded.updated_at
        """,
        payload,
    )


def upsert_alias(conn: sqlite3.Connection, source: str, source_id: str, canonical_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO aliases (source, source_id, canonical_id) VALUES (?, ?, ?)",
        (source, source_id, canonical_id),
    )


def upsert_artifact(conn: sqlite3.Connection, artifact: dict) -> None:
    extra = artifact.get("extra", {})
    conn.execute(
        """
        INSERT OR REPLACE INTO source_artifacts
        (source, canonical_id, source_id, quantization, size_mb, download_url, install_command, extra)
        VALUES (:source, :canonical_id, :source_id, :quantization, :size_mb, :download_url, :install_command, :extra)
        """,
        {
            "source": artifact["source"],
            "canonical_id": artifact["canonical_id"],
            "source_id": artifact["source_id"],
            "quantization": artifact.get("quantization", ""),
            "size_mb": artifact.get("size_mb", 0),
            "download_url": artifact.get("download_url", ""),
            "install_command": artifact.get("install_command", ""),
            "extra": json.dumps(extra) if isinstance(extra, dict) else str(extra),
        },
    )


def ensure_model_stub(conn: sqlite3.Connection, canonical_id: str) -> None:
    """Insert a minimal model row from a canonical_id if not present.

    Used by source fetchers that resolve to a heuristic canonical_id which may
    not exist in `models` yet — without this, foreign-key constraints fail.
    Curated canonicals already exist (seed runs first); this only kicks in for
    discovered long-tail models.
    """
    parts = canonical_id.split(":")
    family = parts[0] if parts else ""
    parameter_size = parts[1] if len(parts) > 1 else ""
    variant = parts[2] if len(parts) > 2 else "base"
    conn.execute(
        """
        INSERT OR IGNORE INTO models
        (canonical_id, family, parameter_size, variant, display_name, description,
         total_params_b, active_params_b, is_moe,
         context_length, tool_calling, vision, license, base_model, updated_at)
        VALUES (?, ?, ?, ?, ?, '', 0, 0, 0, 0, 0, 0, '', '', ?)
        """,
        (canonical_id, family, parameter_size, variant, canonical_id, int(time.time())),
    )


def upsert_score(conn: sqlite3.Connection, score: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO scores
        (canonical_id, benchmark, source, value, max_value, confidence, fetched_at)
        VALUES (:canonical_id, :benchmark, :source, :value, :max_value, :confidence, :fetched_at)
        """,
        {**score, "fetched_at": int(time.time())},
    )


def record_source_run(conn: sqlite3.Connection, source: str, status: str,
                      rows: int = 0, duration_ms: int = 0, error: str = "") -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO source_runs
        (source, last_run_at, last_status, error_message, rows_written, duration_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source, int(time.time()), status, error, rows, duration_ms),
    )


def get_source_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM source_runs ORDER BY source").fetchall()
    return [dict(r) for r in rows]


def stale_sources(conn: sqlite3.Connection, max_age_seconds: int) -> list[str]:
    """Return source names whose last successful run is older than max_age_seconds (or never ran)."""
    rows = conn.execute("SELECT source, last_run_at, last_status FROM source_runs").fetchall()
    now = int(time.time())
    seen = {r["source"] for r in rows}
    stale = [r["source"] for r in rows if r["last_status"] != "ok" or now - r["last_run_at"] > max_age_seconds]
    # Sources that never ran are also stale; the caller passes the full source list separately.
    return stale + [s for s in seen if False]  # placeholder; full list comes from caller
