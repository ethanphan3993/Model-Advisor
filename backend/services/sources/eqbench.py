"""EQ-Bench v3 — emotional intelligence + creative writing.

eqbench.com publishes the leaderboard data as a JS file:
  https://eqbench.com/eqbench3_chartdata.js
contains: `const chartData = { <model>: {<axis>: ..., absoluteRadar: {labels, values}}, ... };`

We strip the `const chartData = ` prefix and trailing `;` to get JSON, then
extract a composite creative-writing score per model.
"""

from __future__ import annotations

import json
import re

import httpx

from backend.db import connect, ensure_model_stub, record_source_run, upsert_score
from backend.services.identity import resolve_alias, heuristic_canonical_id
from backend.services.sources import SourceRunResult, timed_run

JS_URL = "https://eqbench.com/eqbench3_chartdata.js"

# The EQ-Bench v3 axes that capture "creative warmth" vs analytical:
WARMTH_AXES = {"warmth", "demonstrated_empathy", "humanlike", "validating", "conversational"}
DEPTH_AXES = {"depth_of_insight", "emotional_reasoning", "social_dexterity", "pragmatic_ei"}


def _strip_js(text: str) -> str:
    """Convert `const chartData = {...};` to JSON."""
    s = text.strip()
    s = re.sub(r"^\s*(const|var|let)\s+\w+\s*=\s*", "", s)
    s = s.rstrip().rstrip(";")
    return s


def _composite_creative(entry: dict) -> float | None:
    """Average the warmth + depth axes from absoluteRadar — these track creative writing quality."""
    radar = entry.get("absoluteRadar") if isinstance(entry, dict) else None
    if not isinstance(radar, dict):
        return None
    labels = radar.get("labels") or []
    values = radar.get("values") or []
    if len(labels) != len(values):
        return None

    target_axes = WARMTH_AXES | DEPTH_AXES
    selected = [v for l, v in zip(labels, values)
                if isinstance(v, (int, float)) and l in target_axes]
    if not selected:
        return None
    # Scale: raw values are 0..~20; multiply by 5 to get 0..100 range
    return min(100.0, sum(selected) / len(selected) * 5)


async def _fetch() -> tuple[int, str, str]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(JS_URL)
        if resp.status_code != 200:
            with connect() as conn:
                record_source_run(conn, "eqbench", "error", 0, error=f"HTTP {resp.status_code}")
            return 0, "error", f"HTTP {resp.status_code}"
        text = resp.text

    try:
        payload = json.loads(_strip_js(text))
    except (json.JSONDecodeError, ValueError) as e:
        with connect() as conn:
            record_source_run(conn, "eqbench", "error", 0, error=f"parse: {e}")
        return 0, "error", f"parse: {e}"

    rows_written = 0
    with connect() as conn:
        for model_name, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            score = _composite_creative(entry)
            if score is None:
                continue
            cid = (resolve_alias("eqbench", model_name)
                   or resolve_alias("hf", model_name)
                   or heuristic_canonical_id(model_name))
            if not cid:
                continue
            ensure_model_stub(conn, cid)
            upsert_score(conn, {
                "canonical_id": cid,
                "benchmark": "eqbench_creative",
                "source": "eqbench",
                "value": float(score),
                "max_value": 100.0,
                "confidence": "measured",
            })
            rows_written += 1

        record_source_run(conn, "eqbench", "ok" if rows_written else "partial", rows_written)
    return rows_written, ("ok" if rows_written else "partial"), ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("eqbench", _fetch)
