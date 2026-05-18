"""3-axis recommendation engine — physics-based hardware model.

fit_score = 0.55 · use_case_score
          + 0.30 · hardware_fit
          + 0.15 · harness_fit

Hardware fit is now grounded in the actual physics of LLM inference on
Apple Silicon:

  1. Memory cost of weights = total_params × bytes_per_param(quant) + KV cache.
     For MoE models, ALL experts must be resident — total_params drives memory.
  2. Decode speed (tokens/sec) is memory-bandwidth bound:
        TPS ≈ bandwidth_GB_per_sec / active_size_GB · efficiency
     For MoE models, only ACTIVE params are read per token — so a 30B-A3B
     model decodes ~10x faster than a dense 30B at the same total size.
  3. The Neural Engine is NOT used by llama.cpp / MLX / Ollama; it's irrelevant
     to inference TPS.

This is what was wrong with the previous version: a fake GPU-cores TPS table,
a NPU multiplier with no physical basis, no MoE awareness, no KV cache budget.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from backend.db import connect
from backend.services.data_loader import get_harness, get_use_case


# ---------------------------------------------------------------------------
# Apple Silicon memory bandwidth — what actually drives decode speed.
# Numbers are published by Apple / verified on llama.cpp + MLX benchmarks.
# Values in GB/s.
# ---------------------------------------------------------------------------

CHIP_BANDWIDTH_GB_S = {
    "Apple M1": 68,
    "Apple M1 Pro": 200,
    "Apple M1 Max": 400,
    "Apple M1 Ultra": 800,
    "Apple M2": 100,
    "Apple M2 Pro": 200,
    "Apple M2 Max": 400,
    "Apple M2 Ultra": 800,
    "Apple M3": 100,
    "Apple M3 Pro": 150,           # notably lower than M2 Pro — Apple cut bandwidth here
    "Apple M3 Max": 400,           # 16-core CPU bin (300 for 12-core, we use upper)
    "Apple M4": 120,
    "Apple M4 Pro": 273,
    "Apple M4 Max": 546,           # upper bin (410 for lower)
    "Apple M5": 153,
    "Apple M5 Pro": 273,           # estimate, similar to M4 Pro
    "Apple M5 Max": 546,           # estimate
}

# Realistic efficiency: llama.cpp/Ollama gets ~65-75% of theoretical bandwidth;
# MLX gets ~75-85%. We use 0.70 as a representative average.
BANDWIDTH_EFFICIENCY = 0.70


# ---------------------------------------------------------------------------
# Bytes-per-parameter for each quantization. These are real measured ratios
# from GGUF files: bytes_per_param ≈ avg_bits_per_weight / 8.
# Source: llama.cpp k-quants documentation.
# ---------------------------------------------------------------------------

QUANT_BYTES_PER_PARAM = {
    "FP16":   2.00,
    "Q8_0":   1.00,    # ~8.5 bits with overhead
    "Q6_K":   0.65,    # ~6.5 bpw
    "Q5_K_M": 0.56,
    "Q5_K_S": 0.55,
    "Q5_0":   0.625,
    "Q4_K_M": 0.47,    # the popular default — ~4.5 bpw
    "Q4_K_S": 0.45,
    "Q4_0":   0.50,
    "IQ4_XS": 0.42,
    "Q3_K_M": 0.39,
    "Q3_K_S": 0.36,
    "IQ3_XS": 0.34,
    "IQ3_XXS": 0.32,
    "Q2_K":   0.27,
}


# ---------------------------------------------------------------------------
# KV cache scaling — bytes per token of context, per parameter B.
# Rough rule: KV cache size scales with context_length × layers × hidden_dim.
# For a 7B model with 32 layers, KV at 8K is ~1 GB; at 32K it's ~4 GB; at 128K it's ~16 GB.
# We approximate this as: kv_gb ≈ context_K × params_B × 0.0125
# ---------------------------------------------------------------------------

def kv_cache_gb(context_tokens: int, total_params_b: float) -> float:
    if context_tokens <= 0 or total_params_b <= 0:
        return 0.0
    context_k = context_tokens / 1024
    return round(context_k * total_params_b * 0.0125, 2)


# ---------------------------------------------------------------------------
# Quantization candidates — ordered from highest to lowest quality.
# We exclude FP16 from the auto-selection (rarely what users want; if it fits,
# Q8_0 is indistinguishable at half the size).
# ---------------------------------------------------------------------------

QUANT_QUALITY_ORDER = [
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS",
    "Q3_K_M", "Q3_K_S", "IQ3_XS", "Q2_K",
]


@dataclass
class HardwareSnapshot:
    chip: str
    generation: str
    gpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    neural_engine_cores: int
    storage_free_gb: float

    def bandwidth_gb_s(self) -> float:
        # Try exact match, then prefix match, then fall back by chip generation
        if self.chip in CHIP_BANDWIDTH_GB_S:
            return CHIP_BANDWIDTH_GB_S[self.chip]
        for k, v in CHIP_BANDWIDTH_GB_S.items():
            if self.chip.startswith(k):
                return v
        # Conservative fallback by generation
        return {"gen5": 153, "gen4": 120, "gen3": 100, "gen2": 100, "gen1": 68}.get(self.generation, 100)


# ---------------------------------------------------------------------------
# Recommendation result types
# ---------------------------------------------------------------------------

@dataclass
class ScoreEvidence:
    benchmark: str
    value: float
    normalized: float
    source: str
    confidence: str


@dataclass
class Provenance:
    use_case_components: list[ScoreEvidence] = field(default_factory=list)
    hardware_components: dict[str, float] = field(default_factory=dict)
    harness_components: dict[str, float] = field(default_factory=dict)
    missing_data: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    rank: int
    canonical_id: str
    display_name: str
    family: str
    parameter_size: str
    variant: str
    is_moe: bool
    fit_score: float
    use_case_score: float
    hardware_fit: float
    harness_fit: float
    confidence: str
    quantization_recommended: str
    estimated_size_mb: float
    estimated_kv_cache_mb: float
    estimated_tokens_per_sec: tuple[int, int]
    install_options: list[dict]
    warnings: list[str]
    provenance: Provenance
    why: str


# ---------------------------------------------------------------------------
# Score normalization (per benchmark)
# ---------------------------------------------------------------------------

BENCHMARK_SCALE = {
    "humaneval": 100.0, "bigcodebench": 60.0, "multipl_e": 80.0,
    "ifeval": 90.0, "bbh": 80.0, "mmlu_pro": 70.0, "gpqa": 70.0,
    "math": 80.0, "musr": 70.0, "arena_elo": 1400.0,
    "eqbench": 80.0, "eqbench_creative": 80.0, "aa_quality": 80.0,
    "mt_bench": 10.0, "mmlu": 100.0,
}


@dataclass
class ModelRecord:
    canonical_id: str
    family: str
    parameter_size: str
    variant: str
    display_name: str
    total_params_b: float
    active_params_b: float
    is_moe: bool
    context_length: int
    tool_calling: bool
    vision: bool
    license: str
    artifacts: list[dict]
    scores: dict[str, list[dict]]


def load_all_models(conn: sqlite3.Connection) -> list[ModelRecord]:
    rows = conn.execute("SELECT * FROM models").fetchall()
    out: list[ModelRecord] = []
    for r in rows:
        cid = r["canonical_id"]
        artifacts = [dict(a) for a in conn.execute(
            "SELECT * FROM source_artifacts WHERE canonical_id = ?", (cid,)
        ).fetchall()]
        score_rows = conn.execute(
            "SELECT benchmark, value, max_value, source, confidence FROM scores WHERE canonical_id = ?", (cid,)
        ).fetchall()
        scores: dict[str, list[dict]] = {}
        for s in score_rows:
            scores.setdefault(s["benchmark"], []).append(dict(s))

        total = float(r["total_params_b"] or 0)
        active = float(r["active_params_b"] or 0)
        # If params not set, infer from parameter_size string
        if total == 0:
            total = _parse_param_count(r["parameter_size"])
        if active == 0:
            active = total

        out.append(ModelRecord(
            canonical_id=cid,
            family=r["family"],
            parameter_size=r["parameter_size"],
            variant=r["variant"],
            display_name=r["display_name"],
            total_params_b=total,
            active_params_b=active,
            is_moe=bool(r["is_moe"]),
            context_length=r["context_length"] or 0,
            tool_calling=bool(r["tool_calling"]),
            vision=bool(r["vision"]),
            license=r["license"] or "",
            artifacts=artifacts,
            scores=scores,
        ))
    return out


def normalized_score(model: ModelRecord, benchmark: str) -> Optional[ScoreEvidence]:
    candidates = model.scores.get(benchmark, [])
    if not candidates:
        return None
    rank = {"measured": 0, "interpolated": 1, "estimated": 2}
    best = sorted(candidates, key=lambda s: rank.get(s["confidence"], 3))[0]
    scale = BENCHMARK_SCALE.get(benchmark, 100.0)
    return ScoreEvidence(
        benchmark=benchmark,
        value=float(best["value"]),
        normalized=max(0.0, min(1.0, float(best["value"]) / scale)),
        source=best["source"],
        confidence=best["confidence"],
    )


# ---------------------------------------------------------------------------
# Use-case scoring
# ---------------------------------------------------------------------------

def score_use_case(model: ModelRecord, use_case_id: str, prov: Provenance,
                   harness_boost: float = 1.0) -> float:
    use_case = get_use_case(use_case_id)
    if not use_case:
        return 0.0
    benchmarks = use_case.get("benchmarks") or {}
    if not benchmarks:
        return 5.0

    total_weight = 0.0
    weighted_sum = 0.0
    for benchmark, weight in benchmarks.items():
        ev = normalized_score(model, benchmark)
        if ev is None:
            prov.missing_data.append(f"use_case:{benchmark}")
            continue
        prov.use_case_components.append(ev)
        weighted_sum += ev.normalized * float(weight)
        total_weight += float(weight)

    if total_weight == 0:
        return 5.0
    raw = weighted_sum / total_weight * 10
    # Don't cap before applying harness_boost — that would discard the ranking
    # signal between e.g. an 8.7 model and a 7.5 model when boost=1.5x. After
    # boosting, soft-cap at 12 so a single very high benchmark doesn't dominate.
    return round(min(12.0, raw * harness_boost), 2)


# ---------------------------------------------------------------------------
# Hardware fit — physics-grounded
# ---------------------------------------------------------------------------

def _parse_param_count(size: str) -> float:
    """'8B' → 8.0, '8x7B' → 56.0, '3.8B' → 3.8, '30B-A3B' → 30.0 (total)."""
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


def model_size_gb(params_b: float, quant: str) -> float:
    """Memory footprint of weights only (no KV cache)."""
    bpp = QUANT_BYTES_PER_PARAM.get(quant, 0.47)
    return params_b * bpp


def best_artifact_size_mb(model: ModelRecord) -> float:
    sizes = [a["size_mb"] for a in model.artifacts if (a["size_mb"] or 0) > 0]
    return min(sizes) if sizes else 0.0


def recommend_quantization(total_params_b: float, available_ram_gb: float,
                           kv_budget_gb: float, headroom: float = 0.85) -> str:
    """Pick the highest-quality quant that fits weights + KV cache + headroom."""
    if total_params_b <= 0 or available_ram_gb <= 0:
        return "Q4_K_M"
    budget_gb = available_ram_gb * headroom - kv_budget_gb
    if budget_gb <= 0:
        return "Q3_K_S"
    for q in QUANT_QUALITY_ORDER:
        bpp = QUANT_BYTES_PER_PARAM[q]
        if total_params_b * bpp <= budget_gb:
            return q
    return "Q2_K"


def estimate_decode_tps(active_params_b: float, quant: str, hw: HardwareSnapshot,
                        is_moe: bool) -> tuple[int, int]:
    """Memory-bandwidth-bound decode TPS.

    For dense models: bytes/token = total_params × bpp (we read all weights per token).
    For MoE: bytes/token = active_params × bpp (only active experts loaded per token).
    TPS ≈ bandwidth_bytes_per_sec / bytes_per_token × efficiency.
    """
    if active_params_b <= 0:
        return (1, 5)
    bpp = QUANT_BYTES_PER_PARAM.get(quant, 0.47)
    bytes_per_token_gb = active_params_b * bpp
    bandwidth = hw.bandwidth_gb_s()
    raw_tps = bandwidth / bytes_per_token_gb * BANDWIDTH_EFFICIENCY
    # Real-world variance: prompt processing slows the experience, batching helps
    return (max(1, int(raw_tps * 0.80)), max(2, int(raw_tps * 1.15)))


def score_hardware(model: ModelRecord, hw: HardwareSnapshot,
                   prov: Provenance) -> tuple[float, str, float, float, tuple[int, int]]:
    """Return (score 0..10, recommended_quant, weights_mb, kv_mb, tps_range).

    The score reflects three honest dimensions:
      1. Does it FIT (RAM budget for weights + KV cache)?
      2. Will it be FAST (bandwidth-bound TPS at chosen quant)?
      3. Will the disk download SUCCEED?
    """
    total = model.total_params_b or _parse_param_count(model.parameter_size)
    active = model.active_params_b or total

    # KV cache budget — assume user wants a "reasonable" 8K context for fitting
    target_ctx = min(model.context_length or 8192, 8192)
    kv_gb = kv_cache_gb(target_ctx, total)

    avail_gb = hw.available_memory_gb
    free_storage_gb = hw.storage_free_gb

    quant = recommend_quantization(total, avail_gb, kv_gb)
    weights_gb = model_size_gb(total, quant)
    total_resident_gb = weights_gb + kv_gb

    # Memory fit (out of 10): how much of available RAM does it use?
    mem_ratio = total_resident_gb / max(avail_gb, 0.5)
    if mem_ratio <= 0.40:
        mem_score = 10
    elif mem_ratio <= 0.60:
        mem_score = 9
    elif mem_ratio <= 0.75:
        mem_score = 8
    elif mem_ratio <= 0.85:
        mem_score = 6
    elif mem_ratio <= 0.95:
        mem_score = 4
    elif mem_ratio <= 1.10:
        mem_score = 2
    else:
        mem_score = 0
    prov.hardware_components["memory_score"] = mem_score
    prov.hardware_components["memory_ratio"] = round(mem_ratio, 2)
    prov.hardware_components["weights_gb"] = round(weights_gb, 2)
    prov.hardware_components["kv_cache_gb"] = round(kv_gb, 2)

    # Speed score — memory-bandwidth-bound, MoE-aware
    tps = estimate_decode_tps(active, quant, hw, model.is_moe)
    avg_tps = (tps[0] + tps[1]) / 2
    if avg_tps >= 60:
        speed_score = 10
    elif avg_tps >= 40:
        speed_score = 9
    elif avg_tps >= 25:
        speed_score = 8
    elif avg_tps >= 15:
        speed_score = 7
    elif avg_tps >= 10:
        speed_score = 5
    elif avg_tps >= 5:
        speed_score = 3
    else:
        speed_score = 1
    prov.hardware_components["speed_score"] = speed_score
    prov.hardware_components["tps_avg"] = round(avg_tps, 1)

    # Storage fit — full GGUF download size
    download_size_mb = best_artifact_size_mb(model)
    if download_size_mb == 0:
        download_size_mb = total * QUANT_BYTES_PER_PARAM["Q4_K_M"] * 1024
    storage_ratio = (download_size_mb / 1024) / max(free_storage_gb, 0.5)
    if storage_ratio <= 0.05:
        st_score = 10
    elif storage_ratio <= 0.15:
        st_score = 8
    elif storage_ratio <= 0.30:
        st_score = 6
    elif storage_ratio <= 0.60:
        st_score = 4
    else:
        st_score = 2
    prov.hardware_components["storage_score"] = st_score

    # Combine: memory fit and speed dominate (it has to fit AND be usable);
    # storage is a sanity check on whether the download will succeed.
    combined = 0.50 * mem_score + 0.40 * speed_score + 0.10 * st_score

    weights_mb = weights_gb * 1024
    kv_mb = kv_gb * 1024
    return round(combined, 2), quant, round(weights_mb, 1), round(kv_mb, 1), tps


# ---------------------------------------------------------------------------
# Use-case + harness filters and scoring
# ---------------------------------------------------------------------------

def filter_use_case(model: ModelRecord, use_case_id: str) -> tuple[bool, list[str]]:
    uc = get_use_case(use_case_id)
    if not uc:
        return True, []
    requires = uc.get("requires") or {}
    warns: list[str] = []
    if requires.get("vision") and not model.vision:
        return False, ["Not a vision model"]
    if requires.get("tool_calling") and not model.tool_calling:
        warns.append("No tool-calling capability detected")
    ctx_min = requires.get("context_length_min")
    if ctx_min and (model.context_length or 0) < ctx_min:
        warns.append(f"Short context ({model.context_length}) for agentic use")
    return True, warns


def score_harness(model: ModelRecord, harness_id: Optional[str],
                  prov: Provenance) -> tuple[float, bool, list[str], dict]:
    if not harness_id:
        return 7.0, True, [], {}
    h = get_harness(harness_id)
    if not h:
        return 7.0, True, [], {}

    requires = h.get("requires") or {}
    prefers = h.get("prefers") or {}
    warnings: list[str] = []
    passes = True

    ctx_min = requires.get("context_length_min")
    if ctx_min and (model.context_length or 0) < ctx_min:
        passes = False
        warnings.append(f"Context length {model.context_length or '?'} < {ctx_min} required by {h['name']}")
    if requires.get("tool_calling") and not model.tool_calling:
        passes = False
        warnings.append(f"{h['name']} needs tool calling support")

    avail_in = requires.get("available_in")
    if avail_in:
        sources_present = {a["source"] for a in model.artifacts}
        wanted = avail_in if isinstance(avail_in, list) else [avail_in]
        if not any(w in sources_present for w in wanted):
            passes = False
            warnings.append(f"Not available in {', '.join(wanted)}")

    family_in = requires.get("family_in")
    if family_in and model.family not in family_in:
        passes = False
        warnings.append(f"{h['name']} doesn't support family {model.family}")

    score = 8.0 if passes else 0.0
    family_bonus = prefers.get("family_bonus") or {}
    if isinstance(family_bonus, dict):
        bonus = family_bonus.get(model.family, 0.0)
        if bonus:
            score += float(bonus)
    if "tool_calling_bonus" in prefers and model.tool_calling:
        score += float(prefers["tool_calling_bonus"])
    if "reasoning_bonus" in prefers:
        gpqa = normalized_score(model, "gpqa")
        bbh = normalized_score(model, "bbh")
        if gpqa and bbh:
            score += float(prefers["reasoning_bonus"]) * (gpqa.normalized + bbh.normalized) / 2

    score = max(0.0, min(10.0, score))
    prov.harness_components["harness_pass"] = 1.0 if passes else 0.0
    prov.harness_components["harness_score"] = score

    return score, passes, warnings, prefers


def install_options_for(model: ModelRecord, harness_id: Optional[str]) -> list[dict]:
    out: list[dict] = []
    h = get_harness(harness_id) if harness_id else None
    h_name = h["name"] if h else None
    for art in model.artifacts:
        if h and h["id"] == "ollama" and art["source"] != "ollama":
            continue
        if h and h["id"] == "lm-studio" and "lmstudio" not in art["source"]:
            continue
        out.append({
            "harness": h_name,
            "source": art["source"],
            "source_id": art["source_id"],
            "command": art.get("install_command", ""),
            "url": art.get("download_url", ""),
            "size_mb": art.get("size_mb", 0),
            "quantization": art.get("quantization", ""),
        })
    return out


def confidence_label(prov: Provenance) -> str:
    if len(prov.use_case_components) >= 2 and len(prov.missing_data) == 0:
        return "high"
    if len(prov.use_case_components) >= 1:
        return "medium"
    return "low"


def build_why(rec_name: str, prov: Provenance, harness_name: Optional[str],
              is_moe: bool, tps_avg: float) -> str:
    parts: list[str] = []
    if prov.use_case_components:
        top = sorted(prov.use_case_components, key=lambda e: e.normalized, reverse=True)[:2]
        parts.append("strong on " + ", ".join(f"{e.benchmark} ({e.value:.1f})" for e in top))
    if is_moe:
        parts.append(f"MoE — runs at ~{int(tps_avg)} tok/s despite total size")
    if harness_name and prov.harness_components.get("harness_pass") == 1.0:
        parts.append(f"works with {harness_name}")
    if "memory_score" in prov.hardware_components:
        ms = prov.hardware_components["memory_score"]
        if ms >= 8:
            parts.append("fits comfortably in RAM")
        elif ms >= 4:
            parts.append("tight RAM fit")
        elif ms > 0:
            parts.append("RAM-constrained")
    return f"{rec_name}: " + "; ".join(parts) if parts else f"{rec_name}: limited benchmark data"


def recommend(
    use_case_id: str,
    harness_id: Optional[str],
    hw: HardwareSnapshot,
    limit: int = 10,
    include_too_big: bool = False,
) -> list[Recommendation]:
    with connect() as conn:
        models = load_all_models(conn)

    h_name = None
    if harness_id:
        h = get_harness(harness_id)
        h_name = h["name"] if h else None

    results: list[Recommendation] = []
    for m in models:
        passes_uc, uc_warns = filter_use_case(m, use_case_id)
        if not passes_uc:
            continue

        prov = Provenance()
        h_score, h_pass, h_warns, prefers = score_harness(m, harness_id, prov)
        if harness_id and not h_pass:
            continue

        boost = 1.0
        ucb = prefers.get("use_case_boost") if isinstance(prefers, dict) else None
        if isinstance(ucb, dict):
            boost = float(ucb.get(use_case_id, 1.0))

        uc_score = score_use_case(m, use_case_id, prov, harness_boost=boost)
        hw_score, quant, weights_mb, kv_mb, tps = score_hardware(m, hw, prov)

        if not include_too_big and prov.hardware_components.get("memory_ratio", 0) > 1.0:
            continue

        fit = 0.55 * uc_score + 0.30 * hw_score + 0.15 * h_score
        warns = uc_warns + h_warns
        if m.is_moe:
            warns = ["MoE: " + str(prov.hardware_components.get("weights_gb", 0)) + " GB resident, ~"
                    + str(int(prov.hardware_components.get("tps_avg", 0))) + " tok/s decode"] + warns

        results.append(Recommendation(
            rank=0,
            canonical_id=m.canonical_id,
            display_name=m.display_name,
            family=m.family,
            parameter_size=m.parameter_size,
            variant=m.variant,
            is_moe=m.is_moe,
            fit_score=round(fit, 2),
            use_case_score=uc_score,
            hardware_fit=hw_score,
            harness_fit=h_score,
            confidence=confidence_label(prov),
            quantization_recommended=quant,
            estimated_size_mb=weights_mb,
            estimated_kv_cache_mb=kv_mb,
            estimated_tokens_per_sec=tps,
            install_options=install_options_for(m, harness_id),
            warnings=warns,
            provenance=prov,
            why=build_why(m.display_name, prov, h_name, m.is_moe,
                          prov.hardware_components.get("tps_avg", 0)),
        ))

    results.sort(key=lambda r: r.fit_score, reverse=True)
    for i, r in enumerate(results[:limit]):
        r.rank = i + 1
    return results[:limit]
