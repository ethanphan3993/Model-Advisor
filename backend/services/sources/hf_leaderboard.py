"""HuggingFace Open LLM Leaderboard.

Uses the datasets-server rows API (no `datasets` lib needed):
  https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fcontents&config=default&split=train&offset=N&length=100

Schema fields of interest (subject to upstream change):
  - model.url  / fullname / id
  - IFEval, BBH, MMLU-PRO, GPQA, MUSR, MATH

We're tolerant of schema drift — we map flexibly and skip what we can't parse.
"""

from __future__ import annotations

import httpx

from backend.db import connect, ensure_model_stub, record_source_run, upsert_score
from backend.services.identity import heuristic_canonical_id, resolve_alias
from backend.services.sources import SourceRunResult, timed_run

DATASET_ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "open-llm-leaderboard/contents"

BENCHMARK_KEYS = {
    "ifeval": ["IFEval", "ifeval"],
    "bbh": ["BBH", "bbh"],
    "mmlu_pro": ["MMLU-PRO", "MMLU_PRO", "mmlu_pro"],
    "gpqa": ["GPQA", "gpqa"],
    "musr": ["MUSR", "musr"],
    "math": ["MATH", "math", "MATH Lvl 5"],
}


def _pick(row: dict, keys: list[str]) -> float | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _model_id_from_row(row: dict) -> str | None:
    for k in ("fullname", "model", "id", "Model"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict) and "name" in v:
            return str(v["name"])
    return None


async def _fetch() -> tuple[int, str, str]:
    rows_written = 0
    last_status: int | None = None
    error_msg = ""
    async with httpx.AsyncClient(timeout=30.0) as client:
        offset = 0
        page_size = 100
        while True:
            resp = await client.get(DATASET_ROWS, params={
                "dataset": DATASET, "config": "default", "split": "train",
                "offset": offset, "length": page_size,
            })
            last_status = resp.status_code
            if resp.status_code != 200:
                # Surface the actual HTTP error rather than reporting "partial 0 rows".
                # 429 = HF datasets-server rate limit (throttles to ~1-2 req/sec per IP).
                error_msg = f"HTTP {resp.status_code}"
                if resp.status_code == 429:
                    error_msg += " (HF datasets-server rate limit — wait 60s before retrying)"
                break
            payload = resp.json()
            rows = [r.get("row", {}) for r in payload.get("rows", []) or []]
            if not rows:
                break

            with connect() as conn:
                for row in rows:
                    model_id = _model_id_from_row(row)
                    if not model_id:
                        continue
                    cid = resolve_alias("hf", model_id) or heuristic_canonical_id(model_id)
                    if not cid:
                        continue
                    ensure_model_stub(conn, cid)
                    for benchmark, keys in BENCHMARK_KEYS.items():
                        v = _pick(row, keys)
                        if v is None:
                            continue
                        upsert_score(conn, {
                            "canonical_id": cid,
                            "benchmark": benchmark,
                            "source": "hf_leaderboard",
                            "value": float(v),
                            "max_value": 100.0,
                            "confidence": "measured",
                        })
                        rows_written += 1

            if len(rows) < page_size:
                break
            offset += page_size
            if offset > 5000:  # safety cap
                break

    if rows_written > 0:
        status = "ok"
    elif last_status and last_status != 200:
        status = "error"
    else:
        status = "partial"

    with connect() as conn:
        record_source_run(conn, "hf_leaderboard", status, rows_written, error=error_msg)
    return rows_written, status, error_msg


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("hf_leaderboard", _fetch)
