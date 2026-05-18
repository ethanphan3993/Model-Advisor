# Architecture

This document covers how Model Advisor is structured, why the scoring formula
takes its specific shape, and how to extend it. If you're contributing code,
read this first — it'll save you a lot of "why did you do it this way?" PR
comments.

## Data flow

```
                      ┌─────────────────────────────┐
                      │        7 data sources        │
                      ├─────────────────────────────┤
                      │  ollama       (catalog)      │
                      │  hf_metadata  (sizes, ctx)   │
                      │  hf_leaderboard  (Open LLM)  │
                      │  bigcode      (HumanEval)    │
                      │  lmsys        (MT-bench)     │
                      │  eqbench      (creative)     │
                      │  lm_studio    (curated GGUF) │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────┐
                  │        SQLite catalog           │
                  │   .cache/model-advisor.db       │
                  ├─────────────────────────────────┤
                  │  models      (canonical id +    │
                  │              params, ctx, MoE)  │
                  │  aliases     (cross-source ids) │
                  │  source_artifacts (install ops) │
                  │  scores      (per-benchmark)    │
                  │  source_runs (status, timing)   │
                  └──────────────┬──────────────────┘
                                 │
                                 ▼
   ┌──────────────┐    ┌─────────────────────────┐    ┌─────────────────┐
   │ Hardware     │    │      Recommender         │    │    FastAPI       │
   │ scan         │───▶│  3-axis fit_score        │───▶│  /api/recommend  │
   │ (macOS only) │    │  + provenance tracking   │    │  /api/models     │
   └──────────────┘    └─────────────────────────┘    │  /api/sources    │
                                                       └────────┬─────────┘
                                                                │
                                                                ▼
                                                       ┌─────────────────┐
                                                       │ React frontend  │
                                                       │  Wizard →       │
                                                       │  Results →      │
                                                       │  Why? panel     │
                                                       └─────────────────┘
```

Sources fetch independently. A single source failure never blocks recommendations —
the SQLite catalog is always queryable from the curated `seed` data even when
all live fetches fail.

## Why a local SQLite catalog instead of querying live?

- **Recommendations must be fast.** Top-10 ranking against 3,400 models with
  full join over scores + artifacts has to return in <100ms for the UI to feel
  responsive. Live API queries can't promise that.
- **Sources go down.** HuggingFace datasets-server rate-limits, EQ-Bench's
  static site occasionally blips. The catalog smooths over transient outages.
- **No per-user API keys are required** for the common case. Refresh once,
  recommend forever (until staleness becomes a problem, which is hours, not
  seconds).

## Scoring: why these weights?

```
fit_score = 0.55 × use_case_score   (benchmark match for the task)
          + 0.30 × hardware_fit      (RAM fit + decode speed)
          + 0.15 × harness_fit       (compatibility with chosen agent)
```

The weights came from this reasoning:

- **Use case is dominant (0.55)** because if a model can't do the task well, no
  amount of hardware fit or harness compatibility helps. A 70B model that
  can't code fluently is worse than a 7B coder for coding work.
- **Hardware fit (0.30)** is the hard constraint — a model that doesn't fit
  in RAM or decodes at 2 tok/s is unusable regardless of benchmark scores.
  Below ~4 on this axis, the model is essentially eliminated.
- **Harness fit (0.15)** is a soft signal — most models work in most harnesses.
  The harness mainly contributes through *use-case boosting* (e.g., Cline
  multiplies coding × 1.5), so its direct weight is intentionally smaller.

Earlier iterations tried a 4-axis (use-case + persona + hardware + framework)
formula at `0.40·uc + 0.25·persona + 0.25·hw + 0.10·fw`. The persona axis was
dropped after finding that "Claude-like" / "Pi-like" matching was both
subjective (no public benchmark for it) and dominated by other signals once
the catalog was full. Real users wanted "what works with Cline?", not "what
feels like Claude?". See [issue history] for the rationale.

### Sub-score caps

`use_case_score` returns 0–10 by default but **isn't capped** before the harness
boost is applied. Without this, two models with raw scores 7.5 and 8.7 would
both clamp to 10.0 after a 1.5× boost, losing the ranking signal between them.
After boosting, the score soft-caps at 12 — the headline number on the card
can therefore read up to ~12 for ideal matches. The UI explainer notes this.

## Hardware fit: physics, not vibes

The previous version of this scorer had a fake `GPU_BASE_TPS` lookup and a
"NPU multiplier" that contributed nothing because llama.cpp / MLX / Ollama
don't use the Neural Engine for inference. That's gone.

The current model is grounded in actual LLM-on-Mac performance:

```
TPS ≈ bandwidth_GB_per_sec / active_size_GB × 0.70
```

Where `0.70` is the typical efficiency of llama.cpp / MLX vs. theoretical
memory bandwidth (some workloads hit 0.85, some 0.65).

For dense models, `active_size = total_size`. For Mixture-of-Experts models,
**only the active experts are read per token**, so a 30B model with 3B active
params (Qwen3-30B-A3B) decodes ~10× faster than a dense 30B at the same memory
footprint.

### Apple Silicon bandwidths used

| Chip | Bandwidth (GB/s) |
|---|---|
| M1 | 68 |
| M1 Pro | 200 |
| M1 Max | 400 |
| M1 Ultra | 800 |
| M2 / M3 base | 100 |
| M2 Pro | 200 |
| M2 Max | 400 |
| M2 Ultra | 800 |
| M3 Pro | 150 *(notably lower than M2 Pro — Apple cut bandwidth here)* |
| M3 Max | 400 |
| M4 | 120 |
| M4 Pro | 273 |
| M4 Max | 546 |
| M5 / M5 Pro / M5 Max | 153 / 273 / 546 |

These are the published Apple specs verified against community benchmark threads.

## Memory budget: weights + KV cache

A model fits if `quant_size + kv_cache_size ≤ 0.85 × available_ram`. The 15%
headroom is for OS, browser, and other apps the user is running.

KV cache scales with context length and total params:
```
kv_gb ≈ context_K × total_params_B × 0.0125
```

A 14B model at 32K context needs ~5.6 GB *just for KV cache*, which is why the
recommender drops to a smaller quant when the user picks a memory-tight
config. We don't ignore this like older naive recommenders did.

## Cross-source identity normalization

The same model has different names across sources:

| Source | Name |
|---|---|
| HF | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Ollama | `llama3.1:8b` |
| LMSYS | `llama-3.1-8b-instruct` |
| BigCode | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| LM Studio | `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF` |

We assign each model a `canonical_id` of the form `{family}:{size}:{variant}`,
e.g. `llama-3.1:8b:instruct`. Two paths to resolve external ids to canonical:

1. **Curated alias map** in `backend/data/aliases.yaml` — explicit mappings for
   the top ~50 models. Source of truth, highest confidence.
2. **Heuristic fallback** in `services/identity.py` — parses family,
   parameter count, and variant from the model name string. Used for the
   long tail of leaderboard entries we haven't curated.

Curated wins when both apply. This means **adding to `aliases.yaml` is the
primary contribution path** — it upgrades a model from "auto-resolved with
unknown errors" to "high-confidence ranking".

## MoE awareness

Mixture-of-Experts changes everything about hardware fit. The schema stores:

- `total_params_b` — full model size (memory cost; you must load all experts)
- `active_params_b` — params read per forward pass (decode speed)
- `is_moe` — flag for UI badge + algorithm dispatch

For dense: `active_params == total_params`. For MoE: separate. This is the
single most important addition since the v0 algorithm — without it,
Qwen3-30B-A3B would either be filtered out (treated as dense 30B requiring
60+ GB) or under-ranked on speed (treated as decoding at dense-30B rate).
Neither is true.

## Source design rules

When adding a data source (`backend/services/sources/<name>.py`), follow these
rules — they're not optional, and they're enforced by code review:

1. **Be tolerant of schema drift.** External sites change. Use `.get()`,
   wrap parsing in try/except, return `"partial"` on shape mismatch and
   `"error"` on HTTP failure. Never `"ok"` with 0 rows — that hides real
   failures.

2. **Always `record_source_run`** — even on early returns. Sources that don't
   record runs become invisible on the Sources page and confuse users.

3. **Always `ensure_model_stub(conn, cid)`** before inserting scores or
   aliases for heuristic canonical_ids. The foreign key constraint will
   reject the row otherwise.

4. **Use `confidence` honestly.** `"measured"` for direct values from a
   benchmark run, `"interpolated"` for derived/aggregate scores,
   `"estimated"` for guesses. The recommender prefers measured > interpolated
   > estimated when the same benchmark is reported by multiple sources.

5. **Don't gate the source on auth without an env var escape hatch.**
   Artificial Analysis is the model: requires `MODEL_ADVISOR_AA_API_KEY`,
   gracefully skips when unset.

## Tests

The recommender tests in `backend/tests/test_recommender.py` are the most
load-bearing — they encode the *meaning* of the algorithm, not just its code:

- `test_moe_decodes_faster_than_dense_at_same_total_size` — asserts the
  MoE physics is implemented correctly (active params drive speed, not total)
- `test_moe_model_recommended_when_dense_would_be_too_slow` — at equal
  use-case scores, MoE ranks above dense on bandwidth-constrained hardware
- `test_harness_use_case_boost_tilts_ranking` — confirms harness boost
  affects ranking (not just display)
- `test_recommend_includes_too_big_when_flagged` — the "include models that
  won't fit" toggle actually works

Don't change the algorithm without updating these. If you do change the
algorithm, *do* update them — they're a contract with users about what
"recommended" means.
