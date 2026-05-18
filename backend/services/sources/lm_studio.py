"""LM Studio catalog — proxied via the lmstudio-community HF org.

LM Studio doesn't publish a model API. They curate models on HF under one org.
We list that org's models and tag them as `available_in: lmstudio-community`.
"""

from __future__ import annotations

import httpx

from backend.config import settings
from backend.db import connect, record_source_run, upsert_alias, upsert_artifact, upsert_model
from backend.services.identity import get_canonical, heuristic_canonical_id, resolve_alias
from backend.services.sources import SourceRunResult, timed_run

HF_API = settings.huggingface_api_base + "/models"
LMS_AUTHOR = "lmstudio-community"


async def _fetch() -> tuple[int, str, str]:
    rows_written = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(HF_API, params={
                "author": LMS_AUTHOR,
                "limit": 500,
                "full": "true",
            })
            if resp.status_code != 200:
                with connect() as conn:
                    record_source_run(conn, "lm_studio", "error", 0,
                                      error=f"HTTP {resp.status_code}")
                return 0, "error", f"HTTP {resp.status_code}"
            models = resp.json()
        except httpx.HTTPError as e:
            with connect() as conn:
                record_source_run(conn, "lm_studio", "error", 0, error=str(e)[:500])
            return 0, "error", str(e)[:500]

    if not isinstance(models, list):
        with connect() as conn:
            record_source_run(conn, "lm_studio", "partial", 0, error="unexpected response shape")
        return 0, "partial", "unexpected response"

    with connect() as conn:
        for m in models:
            hf_id = m.get("id") or m.get("modelId")
            if not isinstance(hf_id, str):
                continue
            display = hf_id.rsplit("/", 1)[-1]
            cid = resolve_alias("lmstudio", hf_id) or heuristic_canonical_id(display)
            if not cid:
                continue

            # Ensure parent model row exists before writing alias/artifact (FK constraint).
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

            upsert_alias(conn, "lmstudio", hf_id, cid)
            upsert_artifact(conn, {
                "source": "lmstudio-community",
                "canonical_id": cid,
                "source_id": hf_id,
                "quantization": "",
                "size_mb": 0,
                "download_url": f"https://huggingface.co/{hf_id}",
                "install_command": f"Open LM Studio → search '{display}'",
                "extra": {"author": LMS_AUTHOR},
            })
            rows_written += 1

        record_source_run(conn, "lm_studio",
                          "ok" if rows_written else "partial", rows_written)
    return rows_written, ("ok" if rows_written else "partial"), ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("lm_studio", _fetch)
