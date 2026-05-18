"""HuggingFace Open LLM Leaderboard.

Fetches benchmark scores via datasets-server's paginated rows API:
  https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fcontents&...

Rate-limit handling:
  - HF datasets-server throttles aggressive scrapers to ~1-2 req/s per IP.
  - We pace requests at ~250ms between pages to stay under that comfortably.
  - On 429 we honor Retry-After (or fall back to a 60s wait).
  - On 5xx we use bounded exponential backoff (1s, 2s, 4s, max 3 retries).

Schema fields of interest (subject to upstream change):
  - model.url / fullname / id
  - IFEval, BBH, MMLU-PRO, GPQA, MUSR, MATH

We're tolerant of schema drift — we map flexibly and skip what we can't parse.
"""

from __future__ import annotations

import asyncio

import httpx

from backend.db import connect, ensure_model_stub, record_source_run, upsert_score
from backend.services.identity import heuristic_canonical_id, resolve_alias
from backend.services.sources import SourceRunResult, timed_run

DATASET_ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "open-llm-leaderboard/contents"

PAGE_SIZE = 100
PAGE_PACE_SECONDS = 0.25         # polite delay between successful pages
HARD_CAP = 50000                  # safety ceiling — full leaderboard is ~10-15K models
MAX_RETRIES = 3
DEFAULT_RETRY_AFTER = 60          # fallback if Retry-After header is missing on 429

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


async def _get_with_backoff(client: httpx.AsyncClient, params: dict) -> tuple[httpx.Response | None, str]:
    """Fetch one page with retry. Returns (response, error_message).

    On 429: honors Retry-After then retries (up to MAX_RETRIES times).
    On 5xx: exponential backoff (1s, 2s, 4s) then retries.
    On other errors: returns immediately with the error.
    """
    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.get(DATASET_ROWS, params=params)
        except httpx.HTTPError as e:
            if attempt >= MAX_RETRIES:
                return None, f"network error: {e}"
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8)
            continue

        if resp.status_code == 200:
            return resp, ""
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else DEFAULT_RETRY_AFTER
            if attempt >= MAX_RETRIES:
                return resp, f"rate-limited (HTTP 429) after {MAX_RETRIES} retries"
            await asyncio.sleep(min(wait, 120))
            continue
        if 500 <= resp.status_code < 600:
            if attempt >= MAX_RETRIES:
                return resp, f"upstream error HTTP {resp.status_code} after {MAX_RETRIES} retries"
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8)
            continue
        # Non-retryable status code
        return resp, f"HTTP {resp.status_code}"

    return None, "exhausted retries"


async def _fetch() -> tuple[int, str, str]:
    rows_written = 0
    pages_fetched = 0
    error_msg = ""
    last_status: int | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        offset = 0
        while offset < HARD_CAP:
            resp, err = await _get_with_backoff(client, {
                "dataset": DATASET, "config": "default", "split": "train",
                "offset": offset, "length": PAGE_SIZE,
            })
            if resp is None:
                error_msg = err
                break
            last_status = resp.status_code
            if resp.status_code != 200:
                error_msg = err or f"HTTP {resp.status_code}"
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

            pages_fetched += 1
            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            # Pace ourselves between pages to stay polite
            await asyncio.sleep(PAGE_PACE_SECONDS)

    if rows_written > 0 and not error_msg:
        status = "ok"
    elif rows_written > 0:
        status = "partial"
    elif last_status and last_status != 200:
        status = "error"
    else:
        status = "partial"

    if pages_fetched > 0 and not error_msg:
        error_msg = f"fetched {pages_fetched} pages"

    with connect() as conn:
        record_source_run(conn, "hf_leaderboard", status, rows_written, error=error_msg)
    return rows_written, status, error_msg


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("hf_leaderboard", _fetch)
