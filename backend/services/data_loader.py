"""Load YAML config files (harnesses, use cases) with caching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def harnesses() -> list[dict[str, Any]]:
    """Agent harnesses — concrete agents/IDEs/runtimes that consume models.

    Modeled on OpenRouter's apps page. Replaces the old `frameworks` concept.
    """
    raw = yaml.safe_load((DATA_DIR / "harnesses.yaml").read_text())
    return raw.get("harnesses", [])


@lru_cache(maxsize=1)
def use_cases() -> list[dict[str, Any]]:
    raw = yaml.safe_load((DATA_DIR / "use_cases.yaml").read_text())
    return raw.get("use_cases", [])


def get_harness(harness_id: str) -> dict[str, Any] | None:
    for h in harnesses():
        if h["id"] == harness_id:
            return h
    return None


def get_use_case(use_case_id: str) -> dict[str, Any] | None:
    for u in use_cases():
        if u["id"] == use_case_id:
            return u
    return None
