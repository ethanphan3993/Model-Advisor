"""Ollama public catalog. Provides install artifacts, not benchmark scores."""

from __future__ import annotations

import httpx

from backend.config import settings
from backend.db import connect, upsert_alias, upsert_artifact, upsert_model, record_source_run
from backend.services.identity import resolve_or_heuristic, get_canonical
from backend.services.sources import SourceRunResult, timed_run

OLLAMA_TAGS_URL = settings.ollama_api_base + "/tags"


async def _fetch() -> tuple[int, str, str]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(OLLAMA_TAGS_URL)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", []) or []

    rows = 0
    with connect() as conn:
        for raw in models:
            name = raw.get("name") or ""
            if not name:
                continue
            details = raw.get("details") or {}
            display_name = name
            cid = resolve_or_heuristic("ollama", name, display_name)
            if not cid:
                continue

            # Ensure model exists (may be created here for the first time)
            canonical = get_canonical(cid)
            family, param_size, variant = cid.split(":", 2) if cid.count(":") >= 2 else ("", "", "")
            upsert_model(conn, {
                "canonical_id": cid,
                "family": canonical.family if canonical else family,
                "parameter_size": canonical.parameter_size if canonical else param_size,
                "variant": canonical.variant if canonical else variant,
                "display_name": canonical.display_name if canonical else display_name,
                "description": "",
                "context_length": 0,
                "tool_calling": 0,
                "vision": int(canonical.vision) if canonical else 0,
                "license": "",
                "base_model": "",
            })

            upsert_alias(conn, "ollama", name, cid)

            size_mb = round((raw.get("size", 0) or 0) / (1024 * 1024), 2)
            quant = details.get("quantization_level", "")
            upsert_artifact(conn, {
                "source": "ollama",
                "canonical_id": cid,
                "source_id": name,
                "quantization": quant,
                "size_mb": size_mb,
                "download_url": f"https://ollama.com/library/{name.split(':')[0]}",
                "install_command": f"ollama pull {name}",
                "extra": {
                    "parent_model": details.get("parent_model", ""),
                    "format": details.get("format", ""),
                    "parameter_size": details.get("parameter_size", ""),
                },
            })
            rows += 1

        record_source_run(conn, "ollama", "ok" if rows > 0 else "partial", rows)
    return rows, ("ok" if rows > 0 else "partial"), ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("ollama", _fetch)
