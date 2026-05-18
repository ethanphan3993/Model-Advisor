"""Artificial Analysis API.

Requires a free API key. If MODEL_ADVISOR_AA_API_KEY is unset, this source skips
silently. Curated AA-style quality scores are not currently seeded; AA is a
cross-validation source rather than a primary one.
"""

from __future__ import annotations

import os

import httpx

from backend.db import connect, record_source_run, upsert_score
from backend.services.identity import resolve_alias, heuristic_canonical_id
from backend.services.sources import SourceRunResult, timed_run

AA_MODELS_URL = "https://api.artificialanalysis.ai/v2/data/llms/models"


async def _fetch() -> tuple[int, str, str]:
    api_key = os.environ.get("MODEL_ADVISOR_AA_API_KEY", "").strip()
    if not api_key:
        msg = "set MODEL_ADVISOR_AA_API_KEY to enable Artificial Analysis cross-validation"
        with connect() as conn:
            record_source_run(conn, "artificial_analysis", "skipped", 0, error=msg)
        return 0, "skipped", msg

    rows_written = 0
    headers = {"x-api-key": api_key}
    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        try:
            resp = await client.get(AA_MODELS_URL)
            if resp.status_code != 200:
                with connect() as conn:
                    record_source_run(conn, "artificial_analysis", "error", 0,
                                      error=f"HTTP {resp.status_code}")
                return 0, "error", f"HTTP {resp.status_code}"
            data = resp.json()
        except httpx.HTTPError as e:
            with connect() as conn:
                record_source_run(conn, "artificial_analysis", "error", 0, error=str(e)[:500])
            return 0, "error", str(e)[:500]

    models = data.get("data") if isinstance(data, dict) else data
    if not isinstance(models, list):
        with connect() as conn:
            record_source_run(conn, "artificial_analysis", "partial", 0,
                              error="unexpected response shape")
        return 0, "partial", "unexpected response shape"

    with connect() as conn:
        for m in models:
            if not isinstance(m, dict):
                continue
            name = m.get("model_id") or m.get("name") or m.get("model")
            quality = m.get("quality_index") or m.get("artificial_analysis_quality_index")
            if not isinstance(name, str) or not isinstance(quality, (int, float)):
                continue
            cid = (resolve_alias("artificial_analysis", name)
                   or resolve_alias("hf", name)
                   or heuristic_canonical_id(name))
            if not cid:
                continue
            upsert_score(conn, {
                "canonical_id": cid,
                "benchmark": "aa_quality",
                "source": "artificial_analysis",
                "value": float(quality),
                "max_value": 100.0,
                "confidence": "measured",
            })
            rows_written += 1

        record_source_run(conn, "artificial_analysis",
                          "ok" if rows_written else "partial", rows_written)
    return rows_written, ("ok" if rows_written else "partial"), ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("artificial_analysis", _fetch)
