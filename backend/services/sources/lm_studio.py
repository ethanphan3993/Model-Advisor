"""LM Studio catalog — what users actually see in LM Studio's model browser.

LM Studio's GUI searches all HuggingFace GGUFs at runtime. To match that
experience, we pull from the major GGUF-publishing HF orgs:

  - lmstudio-community   — LM Studio's official curated set (highest quality bar)
  - bartowski            — most prolific community quanter; near-complete
                           coverage of popular models
  - mradermacher         — very large catalog, many obscure / niche models
  - unsloth              — fast-quants of frontier releases

Models are recorded with two source labels:
  - "lmstudio-community" → for the LM Studio harness filter (highest trust)
  - "huggingface_gguf"   → for everything else (broader coverage)

The LM Studio harness filter accepts both, so any GGUF on HF that we've
discovered will be installable.
"""

from __future__ import annotations

import asyncio

import httpx

from backend.config import settings
from backend.db import connect, record_source_run, upsert_alias, upsert_artifact, upsert_model
from backend.services.identity import get_canonical, heuristic_canonical_id, resolve_alias
from backend.services.sources import SourceRunResult, timed_run

HF_API = settings.huggingface_api_base + "/models"

# Each entry: (HF org, source label, fetch limit)
# Limits are per-org caps — keep ingestion bounded. Users can lift later.
PUBLISHERS = [
    ("lmstudio-community",  "lmstudio-community",  500),   # Highest trust — LM Studio's curated org
    ("bartowski",           "huggingface_gguf",   1000),   # Most prolific quanter; top of LM Studio search
    ("unsloth",             "huggingface_gguf",    500),   # Fast quants of frontier releases
    ("mradermacher",        "huggingface_gguf",   1000),   # Large catalog, niche coverage
]


async def _fetch_publisher(client: httpx.AsyncClient, author: str, limit: int) -> list[dict]:
    """Pull up to `limit` GGUF models from one HF org. Returns [] on error
    (per-org failures don't kill the whole source run)."""
    try:
        resp = await client.get(HF_API, params={
            "author": author,
            "filter": "gguf",
            "limit": limit,
            "sort": "downloads",
            "direction": "-1",
        })
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data if isinstance(data, list) else []
    except httpx.HTTPError:
        return []


async def _fetch() -> tuple[int, str, str]:
    rows_written = 0
    per_publisher: dict[str, int] = {}
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for author, source_label, limit in PUBLISHERS:
            models = await _fetch_publisher(client, author, limit)
            if not models:
                errors.append(f"{author}: 0")
                continue

            count_for_pub = 0
            with connect() as conn:
                for m in models:
                    hf_id = m.get("id") or m.get("modelId")
                    if not isinstance(hf_id, str):
                        continue
                    display = hf_id.rsplit("/", 1)[-1]

                    # Try multiple alias paths for cross-publisher dedup.
                    cid = (resolve_alias("lmstudio", hf_id)
                           or resolve_alias("hf", hf_id)
                           or heuristic_canonical_id(display))
                    if not cid:
                        continue

                    canonical = get_canonical(cid)
                    family, param_size, variant = (cid.split(":", 2) + ["", "", ""])[:3]
                    upsert_model(conn, {
                        "canonical_id": cid,
                        "family": canonical.family if canonical else family,
                        "parameter_size": canonical.parameter_size if canonical else param_size,
                        "variant": canonical.variant if canonical else variant,
                        "display_name": canonical.display_name if canonical else display,
                        "description": "",
                        "context_length": 0,
                        "tool_calling": 0,
                        "vision": int(canonical.vision) if canonical else 0,
                        "license": "",
                        "base_model": "",
                    })

                    # Record the source-specific alias and artifact.
                    upsert_alias(conn, "lmstudio" if source_label == "lmstudio-community" else "hf",
                                 hf_id, cid)
                    install_cmd = (f"Open LM Studio → search '{display}'"
                                   if source_label == "lmstudio-community"
                                   else f"Open LM Studio → search '{display}' or huggingface-cli download {hf_id}")
                    upsert_artifact(conn, {
                        "source": source_label,
                        "canonical_id": cid,
                        "source_id": hf_id,
                        "quantization": "",
                        "size_mb": 0,
                        "download_url": f"https://huggingface.co/{hf_id}",
                        "install_command": install_cmd,
                        "extra": {"author": author},
                    })
                    count_for_pub += 1
                    rows_written += 1

            per_publisher[author] = count_for_pub
            # Polite pacing between orgs (HF rate-limits aggressive bulk fetches)
            await asyncio.sleep(0.5)

    summary = ", ".join(f"{a}={c}" for a, c in per_publisher.items()) or ""
    if errors:
        summary += f" (errors: {','.join(errors)})"
    status = "ok" if rows_written > 0 else ("partial" if not per_publisher else "error")
    with connect() as conn:
        record_source_run(conn, "lm_studio", status, rows_written, error=summary)
    return rows_written, status, summary


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("lm_studio", _fetch)
