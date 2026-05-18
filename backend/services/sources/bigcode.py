"""BigCode Models Leaderboard.

Source: huggingface.co/spaces/bigcode/bigcode-models-leaderboard
Data file: data/code_eval_board.csv

CSV columns:
    T,Model,Size (B),Win Rate,Throughput (tokens/s),Seq_length,#Languages,
    humaneval-python,java,javascript,cpp,php,julia,d,Average score,
    lua,r,racket,rust,swift,Throughput (tokens/s) bs=50,Peak Memory (MB),
    models_query,Links,Submission PR

The "Links" column points to the HuggingFace model. We resolve via the HF alias map.
"""

from __future__ import annotations

import csv
import io
import re

import httpx

from backend.db import connect, ensure_model_stub, record_source_run, upsert_score
from backend.services.identity import resolve_alias, heuristic_canonical_id
from backend.services.sources import SourceRunResult, timed_run

CSV_URL = "https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/raw/main/data/code_eval_board.csv"

HF_ID_RE = re.compile(r"huggingface\.co/([^/]+/[^/?#]+)", re.IGNORECASE)


def _hf_id_from_link(link: str) -> str:
    if not link:
        return ""
    m = HF_ID_RE.search(link)
    return m.group(1) if m else ""


def _to_float(s: str) -> float | None:
    try:
        v = float(s)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


async def _fetch() -> tuple[int, str, str]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(CSV_URL)
        if resp.status_code != 200:
            with connect() as conn:
                record_source_run(conn, "bigcode", "error", 0, error=f"HTTP {resp.status_code}")
            return 0, "error", f"HTTP {resp.status_code}"
        text = resp.text

    rows_written = 0
    reader = csv.DictReader(io.StringIO(text))
    with connect() as conn:
        for row in reader:
            link = row.get("Links", "") or row.get("link", "")
            hf_id = _hf_id_from_link(link)
            model_name = row.get("Model", "") or row.get("models_query", "")
            cid = (resolve_alias("hf", hf_id) if hf_id else None) \
                  or resolve_alias("bigcode", model_name) \
                  or heuristic_canonical_id(model_name)
            if not cid:
                continue
            ensure_model_stub(conn, cid)

            # Map columns to our benchmark vocabulary
            mappings = {
                "humaneval": row.get("humaneval-python", ""),
                "bigcodebench": row.get("Win Rate", ""),    # closest proxy in this CSV
                "multipl_e": row.get("Average score", ""),  # avg across languages
            }
            for benchmark, raw in mappings.items():
                v = _to_float(raw)
                if v is None:
                    continue
                upsert_score(conn, {
                    "canonical_id": cid,
                    "benchmark": benchmark,
                    "source": "bigcode",
                    "value": v,
                    "max_value": 100.0,
                    "confidence": "measured",
                })
                rows_written += 1

        record_source_run(conn, "bigcode", "ok" if rows_written else "partial", rows_written)
    return rows_written, ("ok" if rows_written else "partial"), ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("bigcode", _fetch)
