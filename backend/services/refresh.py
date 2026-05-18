"""Orchestrates running all source fetchers (plus seed).

Artificial Analysis is registered only when MODEL_ADVISOR_AA_API_KEY is set —
their data is auth-only, so the source is genuinely unusable without a key
and we don't pretend otherwise.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from backend.db import connect
from backend.services.sources import SourceRunResult, SOURCE_NAMES
from backend.services.sources import seed
from backend.services.sources import (
    ollama, hf_metadata, hf_leaderboard, bigcode, lmsys,
    eqbench, artificial_analysis, lm_studio,
)


def _build_fetchers() -> dict:
    fetchers = {
        "seed": seed.fetch_and_store,
        "ollama": ollama.fetch_and_store,
        "hf_metadata": hf_metadata.fetch_and_store,
        "hf_leaderboard": hf_leaderboard.fetch_and_store,
        "bigcode": bigcode.fetch_and_store,
        "lmsys": lmsys.fetch_and_store,
        "eqbench": eqbench.fetch_and_store,
        "lm_studio": lm_studio.fetch_and_store,
    }
    if os.environ.get("MODEL_ADVISOR_AA_API_KEY", "").strip():
        fetchers["artificial_analysis"] = artificial_analysis.fetch_and_store
    return fetchers


FETCHERS = _build_fetchers()


def purge_inactive_sources() -> None:
    """Drop source_runs rows for sources that aren't currently registered.

    Keeps the Sources page honest — only shows what's actually active.
    """
    active = set(FETCHERS.keys())
    with connect() as conn:
        rows = conn.execute("SELECT source FROM source_runs").fetchall()
        for r in rows:
            if r["source"] not in active:
                conn.execute("DELETE FROM source_runs WHERE source = ?", (r["source"],))


async def refresh_all(only: Optional[list[str]] = None) -> list[SourceRunResult]:
    """Run all fetchers concurrently. Always runs `seed` first."""
    targets = only or list(FETCHERS.keys())
    if "seed" in targets:
        await FETCHERS["seed"]()
        targets = [t for t in targets if t != "seed"]

    coros = [FETCHERS[name]() for name in targets if name in FETCHERS]
    return await asyncio.gather(*coros)


async def refresh_one(source: str) -> SourceRunResult:
    fn = FETCHERS.get(source)
    if not fn:
        return SourceRunResult(source=source, status="error", error=f"unknown source: {source}")
    return await fn()
