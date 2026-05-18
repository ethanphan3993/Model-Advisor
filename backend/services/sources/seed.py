"""Seed canonical models, artifacts, and curated benchmark scores into the DB.

Runs on every startup so the catalog and scores are queryable from t=0.
Live source fetchers later overwrite individual rows with fresh data.

This is the most important fetcher: with the curated benchmarks/artifacts seeded,
the recommender produces meaningful results even when external sources are down.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from backend.db import connect, upsert_alias, upsert_artifact, upsert_model, upsert_score, record_source_run
from backend.services.identity import all_canonical_models, _load_aliases
from backend.services.sources import SourceRunResult, timed_run

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_curated_benchmarks() -> dict[str, dict]:
    path = DATA_DIR / "benchmarks.yaml"
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("scores", {}) or {}


def _seed_artifact_for_alias(conn, source_label: str, source_id: str, canonical_id: str,
                             display_name: str) -> None:
    """Write a placeholder artifact row for a curated alias.

    Live sources later overwrite these with real sizes and metadata.
    The point is to make framework filters work even before live fetches succeed.
    """
    if source_label == "ollama":
        upsert_artifact(conn, {
            "source": "ollama",
            "canonical_id": canonical_id,
            "source_id": source_id,
            "quantization": "",
            "size_mb": 0,
            "download_url": f"https://ollama.com/library/{source_id.split(':')[0]}",
            "install_command": f"ollama pull {source_id}",
            "extra": {},
        })
    elif source_label == "lmstudio":
        display = source_id.rsplit("/", 1)[-1]
        upsert_artifact(conn, {
            "source": "lmstudio-community",
            "canonical_id": canonical_id,
            "source_id": source_id,
            "quantization": "",
            "size_mb": 0,
            "download_url": f"https://huggingface.co/{source_id}",
            "install_command": f"Open LM Studio → search '{display}'",
            "extra": {"author": "lmstudio-community"},
        })
    elif source_label == "hf":
        upsert_artifact(conn, {
            "source": "huggingface_gguf",
            "canonical_id": canonical_id,
            "source_id": source_id,
            "quantization": "",
            "size_mb": 0,
            "download_url": f"https://huggingface.co/{source_id}",
            "install_command": f"huggingface-cli download {source_id}",
            "extra": {},
        })


async def _fetch() -> tuple[int, str, str]:
    rows = 0
    canonicals = all_canonical_models()
    _, alias_map = _load_aliases()
    benchmarks = _load_curated_benchmarks()

    # Reverse lookup: canonical_id → list of (source, source_id) aliases
    by_canonical: dict[str, list[tuple[str, str]]] = {}
    for (source, source_id), cid in alias_map.items():
        by_canonical.setdefault(cid, []).append((source, source_id))

    with connect() as conn:
        for c in canonicals:
            cid = c.canonical_id
            bench = benchmarks.get(cid, {})
            ctx = int(bench.get("context_length", 0)) if isinstance(bench.get("context_length", 0), (int, float)) else 0
            tool_calling = bool(bench.get("tool_calling", False))
            vision = c.vision or bool(bench.get("vision", False))

            upsert_model(conn, {
                "canonical_id": cid,
                "family": c.family,
                "parameter_size": c.parameter_size,
                "variant": c.variant,
                "display_name": c.display_name,
                "description": "",
                "total_params_b": c.total_params_b,
                "active_params_b": c.active_params_b,
                "is_moe": int(c.is_moe),
                "context_length": ctx,
                "tool_calling": int(tool_calling),
                "vision": int(vision),
                "license": "",
                "base_model": "",
            })
            rows += 1

            # Aliases (every external id we know maps to this canonical)
            for source, source_id in by_canonical.get(cid, []):
                upsert_alias(conn, source, source_id, cid)
                _seed_artifact_for_alias(conn, source, source_id, cid, c.display_name)

            # Curated benchmark scores
            for benchmark, value in bench.items():
                if benchmark in ("context_length", "tool_calling", "vision"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                # arena_elo on its own scale; rest 0..100
                max_val = 1500.0 if benchmark == "arena_elo" else 100.0
                upsert_score(conn, {
                    "canonical_id": cid,
                    "benchmark": benchmark,
                    "source": "curated",
                    "value": float(value),
                    "max_value": max_val,
                    "confidence": "measured",
                })

        record_source_run(conn, "seed", "ok", rows)

    return rows, "ok", ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("seed", _fetch)
