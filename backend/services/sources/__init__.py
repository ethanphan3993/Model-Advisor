"""Eight data source clients. Each fetches from an external API and writes to SQLite.

Sources:
  - ollama:               Ollama public catalog (artifacts only)
  - hf_metadata:          HuggingFace per-model siblings → real GGUF size, context, capabilities
  - hf_leaderboard:       Open LLM Leaderboard (IFEval, BBH, MMLU-PRO, GPQA, MUSR, MATH)
  - bigcode:              BigCode Models Leaderboard (HumanEval, MultiPL-E, BigCodeBench)
  - lmsys:                LMSYS Chatbot Arena leaderboard (ELO)
  - eqbench:              EQ-Bench (creative writing, emotional intelligence)
  - artificial_analysis:  Artificial Analysis API cross-validation
  - lm_studio:            HF filter for lmstudio-community org

All clients implement: `async def fetch_and_store() -> SourceRunResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class SourceRunResult:
    source: str
    status: str  # "ok" | "error" | "partial"
    rows_written: int = 0
    duration_ms: int = 0
    error: str = ""


async def timed_run(source: str, fn: Callable[[], Awaitable[tuple[int, str, str]]]) -> SourceRunResult:
    """Wrap a fetcher: time it, catch errors, normalize the result, and persist
    the real duration into source_runs (which the inner fn records before it
    knows how long it took).
    """
    from backend.db import connect

    start = time.time()
    try:
        rows, status, err = await fn()
        duration_ms = int((time.time() - start) * 1000)
        result = SourceRunResult(source=source, status=status, rows_written=rows,
                                 duration_ms=duration_ms, error=err)
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        result = SourceRunResult(source=source, status="error", rows_written=0,
                                 duration_ms=duration_ms, error=str(e)[:500])

    # Backfill duration into source_runs row if it exists
    try:
        with connect() as conn:
            conn.execute("UPDATE source_runs SET duration_ms = ? WHERE source = ?",
                         (duration_ms, source))
    except Exception:
        pass
    return result


SOURCE_NAMES = [
    "ollama", "hf_metadata", "hf_leaderboard", "bigcode",
    "lmsys", "eqbench", "artificial_analysis", "lm_studio",
]
