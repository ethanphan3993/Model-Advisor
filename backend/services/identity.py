"""Cross-source model identity normalization.

Eight data sources name the same model differently:
  HF:        meta-llama/Meta-Llama-3.1-8B-Instruct
  Ollama:    llama3.1:8b
  LMSYS:     llama-3.1-8b-instruct
  LM Studio: lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF

We assign each model a canonical_id of the form `{family}:{parameter_size}:{variant}`
and route every external id back to it. The aliases.yaml file is the source of truth
for ~50 popular models; everything else is matched heuristically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass(frozen=True)
class CanonicalModel:
    canonical_id: str
    family: str
    parameter_size: str
    variant: str
    display_name: str
    total_params_b: float = 0.0
    active_params_b: float = 0.0
    is_moe: bool = False
    vision: bool = False
    uncensored: bool = False


@lru_cache(maxsize=1)
def _load_aliases() -> tuple[dict[str, CanonicalModel], dict[tuple[str, str], str]]:
    """Return (canonical_id → CanonicalModel, (source, source_id) → canonical_id)."""
    path = DATA_DIR / "aliases.yaml"
    if not path.exists():
        return {}, {}
    raw = yaml.safe_load(path.read_text())
    by_canonical: dict[str, CanonicalModel] = {}
    by_alias: dict[tuple[str, str], str] = {}
    for entry in raw.get("models", []):
        cid = entry["canonical_id"]
        total = float(entry.get("total_params_b", 0) or 0)
        active = float(entry.get("active_params_b", 0) or total)
        model = CanonicalModel(
            canonical_id=cid,
            family=entry["family"],
            parameter_size=entry["parameter_size"],
            variant=entry["variant"],
            display_name=entry["display_name"],
            total_params_b=total,
            active_params_b=active,
            is_moe=bool(entry.get("is_moe", False)),
            vision=bool(entry.get("vision", False)),
            uncensored=bool(entry.get("uncensored", False)),
        )
        by_canonical[cid] = model
        for source, source_id in (entry.get("aliases") or {}).items():
            if isinstance(source_id, str):
                by_alias[(source, source_id.lower())] = cid
    return by_canonical, by_alias


def all_canonical_models() -> list[CanonicalModel]:
    by_canonical, _ = _load_aliases()
    return list(by_canonical.values())


def get_canonical(canonical_id: str) -> Optional[CanonicalModel]:
    by_canonical, _ = _load_aliases()
    return by_canonical.get(canonical_id)


def resolve_alias(source: str, source_id: str) -> Optional[str]:
    """Return the canonical_id for an external (source, id) pair, or None."""
    _, by_alias = _load_aliases()
    return by_alias.get((source, source_id.lower()))


# ---------------------------------------------------------------------------
# Heuristic normalization for models without a curated alias entry
# ---------------------------------------------------------------------------

PARAM_RE = re.compile(r"(?<![a-zA-Z])(\d+(?:\.\d+)?)(b|m)\b", re.IGNORECASE)
MOE_RE = re.compile(r"(\d+)x(\d+(?:\.\d+)?)b", re.IGNORECASE)
VARIANT_HINTS = [
    ("instruct", "instruct"),
    ("-it", "instruct"),
    ("chat", "chat"),
    ("coder", "instruct"),  # coder models are typically instruction-tuned
    ("vision", "vision"),
    ("vl", "vision"),
    ("base", "base"),
]


def parse_parameter_size(text: str) -> str:
    """Extract '8B' / '70B' / '8x7B' / '3.8B' from a model name. Empty string if none found."""
    moe = MOE_RE.search(text)
    if moe:
        return f"{moe.group(1)}x{moe.group(2)}B"
    m = PARAM_RE.search(text)
    if m:
        unit = m.group(2).upper()
        return f"{m.group(1)}{unit}"
    return ""


def param_count_b(size: str) -> float:
    """'8B' → 8.0, '8x7B' → 56.0, '3.8B' → 3.8, '30B-A3B' → 30.0 (total).

    The single source of truth for converting a parameter_size string into
    a numeric count in billions. Used by ensure_model_stub and the recommender.
    """
    if not size:
        return 0.0
    s = size.upper().split("-A", 1)[0].rstrip("B")  # "30B-A3B" → "30"
    if "X" in s:
        a, b = s.split("X", 1)
        try:
            return float(a) * float(b)
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def active_param_count_b(size: str) -> float:
    """For MoE strings like '30B-A3B' returns 3.0; otherwise == param_count_b(size)."""
    if not size or "-A" not in size.upper():
        return param_count_b(size)
    after_a = size.upper().split("-A", 1)[1].rstrip("B")
    try:
        return float(after_a)
    except ValueError:
        return param_count_b(size)


def parse_variant(text: str) -> str:
    text_lower = text.lower()
    for needle, variant in VARIANT_HINTS:
        if needle in text_lower:
            return variant
    return "base"


def normalize_family(name: str) -> str:
    """Normalize a model name to a family slug.

    Examples:
        "Meta-Llama-3.1-8B-Instruct"     -> "llama-3.1"
        "Qwen2.5-Coder-7B-Instruct"      -> "qwen-2.5-coder"
        "Mixtral-8x7B-Instruct-v0.1"     -> "mixtral"
        "openai/gpt-oss-120b"            -> "gpt-oss"
    """
    s = name.lower()
    # Strip any HuggingFace org prefix (anything matching `<org>/`)
    if "/" in s:
        s = s.rsplit("/", 1)[1]
    # Then known prefix variants without the slash
    s = re.sub(r"^(meta-)", "", s)
    s = re.sub(r"-instruct.*$", "", s)
    s = re.sub(r"-it$", "", s)
    s = re.sub(r"-chat$", "", s)
    s = re.sub(r"-base$", "", s)
    s = re.sub(r"-gguf.*$", "", s)
    s = re.sub(r"-v\d+(\.\d+)*$", "", s)
    # Strip parameter size
    s = MOE_RE.sub("", s)
    s = PARAM_RE.sub("", s)
    # Tidy
    s = re.sub(r"-+", "-", s).strip("-")
    # Insert dot before version numbers attached to the family root (qwen2.5 -> qwen-2.5)
    s = re.sub(r"([a-z])(\d)", r"\1-\2", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def heuristic_canonical_id(name: str) -> Optional[str]:
    """Best-effort canonical_id for a model name without a curated alias.

    Returns None if we can't extract a parameter size — we don't want to score
    things we can't size.
    """
    family = normalize_family(name)
    size = parse_parameter_size(name)
    variant = parse_variant(name)
    if not family or not size:
        return None
    return f"{family}:{size.lower()}:{variant}"


def resolve_or_heuristic(source: str, source_id: str, display_name: str = "") -> Optional[str]:
    """Try alias map first, fall back to heuristic family/size/variant parsing."""
    cid = resolve_alias(source, source_id)
    if cid:
        return cid
    return heuristic_canonical_id(display_name or source_id)
