"""HuggingFace model metadata.

Calls /api/models/{id} for each curated canonical model to get:
  - real GGUF file sizes (from `siblings`)
  - context length, tool_calling, vision flags (from `cardData` / tags)
  - license, base_model

This fixes the bug where HF models had `total_size_mb: 0`.
We only enrich models that exist in our alias map (curated set) — pulling
metadata for the entire HF GGUF index would be 1000s of requests.
"""

from __future__ import annotations

import asyncio
import re

import httpx

from backend.config import settings
from backend.db import connect, upsert_alias, upsert_artifact, upsert_model, record_source_run
from backend.services.identity import all_canonical_models
from backend.services.sources import SourceRunResult, timed_run

HF_MODEL_URL = settings.huggingface_api_base + "/models/{model_id}"
GGUF_SIZE_RE = re.compile(r"\.q?(\d+)_k_?[ms]?\.gguf$|\.q?(\d+)_0\.gguf$|\.gguf$", re.IGNORECASE)


def _detect_capabilities(tags: list[str], card: dict) -> dict:
    tags_lower = [t.lower() for t in (tags or [])]
    text_blob = " ".join(tags_lower) + " " + str(card or "").lower()
    return {
        "tool_calling": int(any(k in text_blob for k in ["function-calling", "tool-use", "tools", "function_call"])),
        "vision": int(any(k in text_blob for k in ["multimodal", "vision", "image-text-to-text", "vqa"])),
    }


def _detect_context_length(card: dict, tags: list[str]) -> int:
    """Extract context length from cardData if available."""
    if not card:
        return 0
    cd = card if isinstance(card, dict) else {}
    for key in ("max_position_embeddings", "context_length", "max_seq_len", "max_position"):
        v = cd.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return 0


async def _fetch_one(client: httpx.AsyncClient, hf_id: str) -> dict | None:
    try:
        resp = await client.get(HF_MODEL_URL.format(model_id=hf_id))
        if resp.status_code != 200:
            return None
        return resp.json()
    except httpx.HTTPError:
        return None


async def _fetch() -> tuple[int, str, str]:
    canonicals = all_canonical_models()
    targets: list[tuple[str, str]] = []  # (canonical_id, hf_id)
    from backend.services.identity import _load_aliases
    _, alias_map = _load_aliases()
    for cid, source_id in [(v, k[1]) for k, v in alias_map.items() if k[0] == "hf"]:
        targets.append((cid, source_id))

    rows = 0
    async with httpx.AsyncClient(timeout=20.0) as client:
        # Concurrent but capped
        sem = asyncio.Semaphore(8)

        async def process(cid: str, hf_id: str) -> int:
            async with sem:
                data = await _fetch_one(client, hf_id)
            if not data:
                return 0
            siblings = data.get("siblings") or []
            tags = data.get("tags") or []
            card = data.get("cardData") or {}
            caps = _detect_capabilities(tags, card)
            ctx = _detect_context_length(card, tags)

            with connect() as conn:
                # Update model with discovered capabilities
                from backend.services.identity import get_canonical
                canon = get_canonical(cid)
                if canon:
                    def _str_or_first(v):
                        if isinstance(v, list):
                            return str(v[0]) if v else ""
                        return str(v) if v else ""

                    upsert_model(conn, {
                        "canonical_id": cid,
                        "family": canon.family,
                        "parameter_size": canon.parameter_size,
                        "variant": canon.variant,
                        "display_name": canon.display_name,
                        "description": _str_or_first(card.get("model_description") if isinstance(card, dict) else ""),
                        "total_params_b": canon.total_params_b,
                        "active_params_b": canon.active_params_b,
                        "is_moe": int(canon.is_moe),
                        "context_length": ctx,
                        "tool_calling": caps["tool_calling"],
                        "vision": int(canon.vision) | caps["vision"],
                        "license": _str_or_first(card.get("license") if isinstance(card, dict) else ""),
                        "base_model": _str_or_first(card.get("base_model") if isinstance(card, dict) else ""),
                    })
                upsert_alias(conn, "hf", hf_id, cid)

                # Pick the largest GGUF sibling as the canonical artifact
                gguf_files = [s for s in siblings if str(s.get("rfilename", "")).lower().endswith(".gguf")]
                if gguf_files:
                    # Largest = best quality quant we'd recommend by default
                    largest = max(gguf_files, key=lambda s: s.get("size", 0) or 0)
                    size_mb = round((largest.get("size", 0) or 0) / (1024 * 1024), 2)
                    fname = largest.get("rfilename", "")
                    upsert_artifact(conn, {
                        "source": "huggingface_gguf",
                        "canonical_id": cid,
                        "source_id": f"{hf_id}/{fname}",
                        "quantization": _quant_from_filename(fname),
                        "size_mb": size_mb,
                        "download_url": f"https://huggingface.co/{hf_id}/resolve/main/{fname}",
                        "install_command": f"huggingface-cli download {hf_id} {fname}",
                        "extra": {"all_quants": [s.get("rfilename") for s in gguf_files]},
                    })
            return 1

        results = await asyncio.gather(*[process(cid, hf_id) for cid, hf_id in targets])
        rows = sum(results)

    with connect() as conn:
        record_source_run(conn, "hf_metadata", "ok" if rows > 0 else "partial", rows)
    return rows, ("ok" if rows > 0 else "partial"), ""


def _quant_from_filename(fname: str) -> str:
    fname = fname.lower()
    for q in ["q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0", "q3_k_m", "q3_k_s", "q2_k", "iq4_xs", "iq3_xs"]:
        if q in fname:
            return q.upper()
    if "fp16" in fname or "f16" in fname:
        return "FP16"
    return ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("hf_metadata", _fetch)
