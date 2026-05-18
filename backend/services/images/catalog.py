"""Image-generation catalog loader.

The image track is YAML-only — there's no equivalent of HF's Open LLM
Leaderboard for diffusion models, so we hand-curate ~20 entries with
cited sources rather than scraping. This module loads and indexes them,
plus the image-specific use cases and harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class ImageScore:
    benchmark: str
    value: float
    source: str
    confidence: str


@dataclass
class ImageModel:
    canonical_id: str
    family: str
    variant: str
    display_name: str
    architecture: str
    total_params_b: float
    default_steps: int
    vram_gb: dict[str, float]                  # fp16/q8/q4 -> GB resident
    time_per_step_seconds: dict[str, float]    # m3_max/m4_max reference times
    license: str
    supports: list[str]
    harnesses_compatible: list[str]
    scores: dict[str, ImageScore]
    hf_id: str = ""                            # Hugging Face repo ID (org/name)
    comfyui_folder: str = "checkpoints"        # subdir under ComfyUI/models/
    notes: str = ""


@lru_cache(maxsize=1)
def image_models() -> list[ImageModel]:
    raw = yaml.safe_load((DATA_DIR / "image_aliases.yaml").read_text())
    out: list[ImageModel] = []
    for entry in raw.get("image_models", []) or []:
        scores = {}
        for bench, payload in (entry.get("scores") or {}).items():
            scores[bench] = ImageScore(
                benchmark=bench,
                value=float(payload["value"]),
                source=str(payload.get("source", "")),
                confidence=str(payload.get("confidence", "estimated")),
            )
        out.append(ImageModel(
            canonical_id=entry["canonical_id"],
            family=entry["family"],
            variant=entry.get("variant", ""),
            display_name=entry["display_name"],
            architecture=entry.get("architecture", "diffusion"),
            total_params_b=float(entry["total_params_b"]),
            default_steps=int(entry.get("default_steps", 25)),
            vram_gb=dict(entry.get("vram_gb", {})),
            time_per_step_seconds=dict(entry.get("time_per_step_seconds", {})),
            license=entry.get("license", ""),
            supports=list(entry.get("supports", [])),
            harnesses_compatible=list(entry.get("harnesses_compatible", [])),
            scores=scores,
            hf_id=str(entry.get("hf_id", "")),
            comfyui_folder=str(entry.get("comfyui_folder", "checkpoints")),
            notes=entry.get("notes", ""),
        ))
    return out


@lru_cache(maxsize=1)
def image_use_cases() -> list[dict[str, Any]]:
    raw = yaml.safe_load((DATA_DIR / "image_use_cases.yaml").read_text())
    return raw.get("image_use_cases", [])


@lru_cache(maxsize=1)
def image_harnesses() -> list[dict[str, Any]]:
    raw = yaml.safe_load((DATA_DIR / "image_harnesses.yaml").read_text())
    return raw.get("image_harnesses", [])


def get_image_use_case(use_case_id: str) -> dict[str, Any] | None:
    for u in image_use_cases():
        if u["id"] == use_case_id:
            return u
    return None


def get_image_harness(harness_id: str) -> dict[str, Any] | None:
    for h in image_harnesses():
        if h["id"] == harness_id:
            return h
    return None


def get_image_model(canonical_id: str) -> ImageModel | None:
    for m in image_models():
        if m.canonical_id == canonical_id:
            return m
    return None
