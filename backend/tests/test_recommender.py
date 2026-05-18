"""Recommender end-to-end tests with synthetic data."""

import pytest

from backend.db import connect, upsert_artifact, upsert_model, upsert_score
from backend.services.recommender import (
    HardwareSnapshot, recommend_quantization, score_use_case, recommend, ModelRecord,
    Provenance, estimate_decode_tps,
)


HW_M3_36GB = HardwareSnapshot(
    chip="Apple M3 Max", generation="gen3", gpu_cores=30,
    total_memory_gb=36, available_memory_gb=22,
    neural_engine_cores=16, storage_free_gb=200,
)

HW_M5_PRO_64GB = HardwareSnapshot(
    chip="Apple M5 Pro", generation="gen5", gpu_cores=20,
    total_memory_gb=64, available_memory_gb=48,
    neural_engine_cores=16, storage_free_gb=200,
)


def _seed_model(canonical_id: str, family: str, size: str, *, ctx: int = 8000,
                tool_calling: bool = False, vision: bool = False, size_mb: float = 5000,
                total_b: float = 0, active_b: float = 0, is_moe: bool = False):
    with connect() as conn:
        upsert_model(conn, {
            "canonical_id": canonical_id, "family": family, "parameter_size": size,
            "variant": "instruct", "display_name": f"{family} {size}",
            "description": "",
            "total_params_b": total_b, "active_params_b": active_b, "is_moe": int(is_moe),
            "context_length": ctx, "tool_calling": int(tool_calling),
            "vision": int(vision), "license": "", "base_model": "",
        })
        upsert_artifact(conn, {
            "source": "ollama", "canonical_id": canonical_id, "source_id": f"{family}:{size}",
            "quantization": "Q4_K_M", "size_mb": size_mb,
            "download_url": "", "install_command": f"ollama pull {family}:{size}",
            "extra": {},
        })


def _seed_score(canonical_id: str, benchmark: str, value: float):
    with connect() as conn:
        upsert_score(conn, {
            "canonical_id": canonical_id, "benchmark": benchmark, "source": "test",
            "value": value, "max_value": 100.0, "confidence": "measured",
        })


def test_recommend_quantization_picks_largest_that_fits():
    # 7B model, 22 GB available, no KV budget → Q8_0 (1.0 bpp) = 7 GB easily fits.
    q = recommend_quantization(7.0, 22.0, kv_budget_gb=0)
    assert q == "Q8_0"


def test_recommend_quantization_drops_to_smaller_when_tight():
    # 14B model on 8 GB available with 1 GB KV cache → 7 GB budget. Q4_K_M (0.47 bpp) = 6.6 GB fits.
    q = recommend_quantization(14.0, 8.0, kv_budget_gb=1.0)
    from backend.services.recommender import QUANT_BYTES_PER_PARAM
    bpp = QUANT_BYTES_PER_PARAM[q]
    assert 14.0 * bpp + 1.0 <= 8.0 * 0.85 + 0.1
    assert q not in {"Q8_0", "Q6_K"}


def test_recommend_quantization_falls_back_when_nothing_fits():
    # 50B model in 8GB → nothing fits at any quant.
    q = recommend_quantization(50, 8, kv_budget_gb=0)
    assert q in {"Q2_K", "Q3_K_S"}


def test_moe_decodes_faster_than_dense_at_same_total_size():
    """A 30B-A3B MoE should decode ~10x faster than dense 30B (only active params read per token)."""
    dense_tps = estimate_decode_tps(30.0, "Q4_K_M", HW_M5_PRO_64GB, is_moe=False)
    moe_tps = estimate_decode_tps(3.0, "Q4_K_M", HW_M5_PRO_64GB, is_moe=True)
    # MoE active is 3B vs dense 30B → ~10x faster
    assert moe_tps[0] > dense_tps[1] * 5  # at least 5x


def test_moe_model_recommended_when_dense_would_be_too_slow():
    """Qwen3-30B-A3B should outrank a dense 32B on M5 Pro because it decodes ~10x faster."""
    _seed_model("dense:32b:instruct", "dense", "32B", ctx=128000, tool_calling=True,
                total_b=32, active_b=32, is_moe=False, size_mb=18000)
    _seed_model("moe:30b-a3b:instruct", "moe", "30B-A3B", ctx=128000, tool_calling=True,
                total_b=30, active_b=3, is_moe=True, size_mb=18000)
    # Identical benchmark scores so use_case_score is equal
    for cid in ("dense:32b:instruct", "moe:30b-a3b:instruct"):
        _seed_score(cid, "humaneval", 85.0)
        _seed_score(cid, "ifeval", 85.0)
        _seed_score(cid, "bbh", 80.0)

    recs = recommend("coding", "cline", HW_M5_PRO_64GB, limit=5)
    assert len(recs) >= 2
    # MoE should outrank dense at equal use_case_score because hardware_fit prefers fast decode
    moe = next(r for r in recs if r.canonical_id == "moe:30b-a3b:instruct")
    dense = next(r for r in recs if r.canonical_id == "dense:32b:instruct")
    assert moe.fit_score > dense.fit_score
    assert moe.estimated_tokens_per_sec[0] > dense.estimated_tokens_per_sec[1] * 3


def test_recommend_returns_ranked_list_with_use_case_filter():
    _seed_model("a:1b:instruct", "a", "1B", size_mb=800)
    _seed_model("b:7b:instruct", "b", "7B", size_mb=4500)
    _seed_score("a:1b:instruct", "humaneval", 30.0)
    _seed_score("b:7b:instruct", "humaneval", 75.0)
    _seed_score("a:1b:instruct", "ifeval", 50.0)
    _seed_score("b:7b:instruct", "ifeval", 80.0)

    recs = recommend("coding", harness_id=None, hw=HW_M3_36GB, limit=10)
    assert len(recs) == 2
    # Higher humaneval should outrank
    assert recs[0].canonical_id == "b:7b:instruct"
    assert recs[0].rank == 1
    assert recs[1].rank == 2


def test_vision_use_case_filters_non_vision():
    _seed_model("plain:7b:instruct", "plain", "7B", vision=False)
    _seed_model("seer:7b:instruct", "seer", "7B", vision=True)
    recs = recommend("vision", None, HW_M3_36GB, limit=10)
    assert all("seer" in r.canonical_id for r in recs)


def test_harness_filter_by_context_length():
    _seed_model("short:7b:instruct", "short", "7B", ctx=2000, tool_calling=True)
    _seed_model("long:7b:instruct", "long", "7B", ctx=128000, tool_calling=True)
    recs = recommend("agentic", "cline", HW_M3_36GB, limit=10)
    # Cline requires 32K+ context
    assert all("long" in r.canonical_id for r in recs)


def test_recommend_includes_too_big_when_flagged():
    _seed_model("huge:405b:instruct", "huge", "405B", size_mb=200000)
    recs = recommend("coding", None, HW_M3_36GB, limit=10, include_too_big=False)
    assert len(recs) == 0
    recs = recommend("coding", None, HW_M3_36GB, limit=10, include_too_big=True)
    assert len(recs) >= 1


def test_harness_use_case_boost_tilts_ranking():
    # Two equally capable coders; with cline harness, both pass filters; the
    # boost applies equally so order depends on the underlying scores.
    _seed_model("excellent:7b:instruct", "excellent", "7B", ctx=128000, tool_calling=True)
    _seed_model("ok:7b:instruct", "ok", "7B", ctx=128000, tool_calling=True)
    _seed_score("excellent:7b:instruct", "humaneval", 90.0)
    _seed_score("excellent:7b:instruct", "ifeval", 85.0)
    _seed_score("ok:7b:instruct", "humaneval", 50.0)
    _seed_score("ok:7b:instruct", "ifeval", 60.0)

    recs = recommend("coding", "cline", HW_M3_36GB, limit=10)
    assert len(recs) == 2
    assert recs[0].canonical_id == "excellent:7b:instruct"
    # use_case_score should be boosted by Cline's coding=1.5 multiplier
    assert recs[0].use_case_score > 7.0
