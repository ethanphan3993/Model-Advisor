"""LMSYS / lmarena.ai leaderboard.

The original lmsys/chatbot-arena-leaderboard Space was renamed to lmarena-ai/arena-leaderboard.
Their published CSV (`leaderboard_table_<date>.csv`) contains MT-bench + MMLU per model.
Arena ELO itself lives in pickle files which we don't load (security + no pickle dep);
curated arena_elo values in benchmarks.yaml cover the top models.

The Space publishes a new CSV daily; we discover the latest filename via the tree API.
"""

from __future__ import annotations

import csv
import io
import re

import httpx

from backend.db import connect, ensure_model_stub, record_source_run, upsert_score
from backend.services.identity import resolve_alias, heuristic_canonical_id
from backend.services.sources import SourceRunResult, timed_run

SPACE = "lmarena-ai/arena-leaderboard"
TREE_URL = f"https://huggingface.co/api/spaces/{SPACE}/tree/main"
RAW_BASE = f"https://huggingface.co/spaces/{SPACE}/raw/main"

LEADERBOARD_RE = re.compile(r"^leaderboard_table_(\d{8})\.csv$")


async def _latest_csv_path(client: httpx.AsyncClient) -> str | None:
    resp = await client.get(TREE_URL)
    if resp.status_code != 200:
        return None
    files = resp.json()
    dates: list[tuple[str, str]] = []
    for f in files:
        path = f.get("path", "")
        name = path.rsplit("/", 1)[-1]
        m = LEADERBOARD_RE.match(name)
        if m:
            dates.append((m.group(1), path))
    if not dates:
        return None
    dates.sort(reverse=True)
    return dates[0][1]


def _to_float(s: str) -> float | None:
    try:
        v = float(s)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


async def _fetch() -> tuple[int, str, str]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        path = await _latest_csv_path(client)
        if not path:
            with connect() as conn:
                record_source_run(conn, "lmsys", "error", 0, error="no leaderboard CSV found")
            return 0, "error", "no leaderboard CSV found"
        url = f"{RAW_BASE}/{path}"
        resp = await client.get(url)
        if resp.status_code != 200:
            with connect() as conn:
                record_source_run(conn, "lmsys", "error", 0, error=f"HTTP {resp.status_code}")
            return 0, "error", f"HTTP {resp.status_code}"
        text = resp.text

    rows_written = 0
    reader = csv.DictReader(io.StringIO(text))
    with connect() as conn:
        for row in reader:
            key = row.get("key", "").strip()
            display = row.get("Model", "").strip()
            link = row.get("Link", "")

            cid = (resolve_alias("lmsys", key)
                   or resolve_alias("lmsys", display)
                   or _resolve_via_hf_link(link)
                   or heuristic_canonical_id(display))
            if not cid:
                continue
            ensure_model_stub(conn, cid)

            # Multiple benchmark columns
            mt = _to_float(row.get("MT-bench (score)", ""))
            if mt is not None:
                upsert_score(conn, {
                    "canonical_id": cid, "benchmark": "mt_bench", "source": "lmsys",
                    "value": mt, "max_value": 10.0, "confidence": "measured",
                })
                rows_written += 1
            mmlu = _to_float(row.get("MMLU", ""))
            if mmlu is not None:
                # Stored 0..1 in this CSV; upscale to 0..100 to match our scale convention
                upsert_score(conn, {
                    "canonical_id": cid, "benchmark": "mmlu", "source": "lmsys",
                    "value": mmlu * 100 if mmlu <= 1.0 else mmlu, "max_value": 100.0,
                    "confidence": "measured",
                })
                rows_written += 1

        record_source_run(conn, "lmsys", "ok" if rows_written else "partial", rows_written)
    return rows_written, ("ok" if rows_written else "partial"), ""


def _resolve_via_hf_link(link: str) -> str | None:
    if not link or "huggingface.co/" not in link:
        return None
    hf_id = link.split("huggingface.co/", 1)[1].rstrip("/")
    return resolve_alias("hf", hf_id)


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("lmsys", _fetch)
