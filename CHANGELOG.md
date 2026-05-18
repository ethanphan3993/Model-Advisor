# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Image generation track** (#16). Separate `/images` route covering local
  diffusion + flow-matching models. Curated catalog of ~20 entries
  (FLUX.1 dev/schnell/Kontext, SD 3.5 Large/Medium/Turbo, SDXL, Stable Cascade,
  HiDream-I1/E1, AuraFlow, OmniGen, Sana, Lumina-Next, PixArt-Σ, Kolors,
  InstructPix2Pix, SD 1.5) with GenEval / imagen-arena ELO / Emu-Edit scores
  cited per model. New compute-bound hardware-fit cost model: time-per-image
  scales with chip FP16 TFLOPS rather than memory bandwidth. New harness
  layer covering Draw Things, Mochi Diffusion, ComfyUI, InvokeAI, Forge,
  and Diffusers. Backend endpoints under `/api/images/*`. Doesn't touch the
  existing text-LLM recommender path.
- **gpt-oss-20b and gpt-oss-120b** (OpenAI's open-weight, August 2025), MoE
  with 3.6B / 5.1B active params and curated benchmark scores from the public
  announcement.
- Backfill migration: `total_params_b` populated from `parameter_size` for
  legacy rows. Catalog went from 38 → ~2,900 models with parsed param counts.
- Recommender response cache: ~250× speedup on warm hits, auto-invalidates on
  any source refresh.
- Hardware snapshot cache (15s TTL) at the API router — avoids re-running
  `system_profiler` on every recommend call.
- HF leaderboard fetcher: lifted hard cap from 5,000 → 50,000 rows; added
  `Retry-After`-aware 429 handling and 5xx exponential backoff; polite 250ms
  per-page pacing.
- Numeric `XX% confidence` chip on result cards (replaces high/medium/low).
- Quantization quality penalty: use_case_score multiplied by retention factor
  for the chosen quant (Q3_K_S → 0.86, Q2_K → 0.78, Q8_0 → 1.00).
- Always-visible 3-bar score breakdown on result cards (was hidden behind
  "Why?").
- Explicit RAM math and TPS math lines on every card.
- "vs #1" comparison line on rows 2+.
- "Limited data" filter behavior: unscored models excluded from default
  ranking; `include_unscored=true` surfaces them.

### Changed
- RAM budget anchor switched from `available_memory_gb` (transient) to
  `max(available, total × 0.70)` — recommends what you *could* run, not what
  fits this instant. `fits_currently_free` flag separately signals when you'd
  need to free RAM.
- `normalize_family()` strips any HF org prefix (was hardcoded to a small
  allowlist; missed `openai/`, `CohereForAI/`, etc.).
- Persona axis dropped from API and UI (was already stale; this removes
  remaining surface).

### Fixed
- `ensure_model_stub()` no longer hardcodes `total_params_b=0`. Heuristic-
  resolved leaderboard entries get accurate hardware fit.
- SPA fallback in single-port packaged build: deep-links like `/wizard/coding`
  now resolve correctly on direct navigation/refresh.
- HF leaderboard `partial 0 rows` was masking real HTTP errors. Now reports
  `error HTTP 429 (rate limit)` honestly.

## [0.1.0] — 2025-05-18

Initial public release.

### Added
- 3-axis recommender: `0.55 × use-case + 0.30 × hardware + 0.15 × harness`
- 7 working data sources (Ollama, HF metadata, HF Open LLM Leaderboard,
  BigCode, LMSYS, EQ-Bench, LM Studio) plus a curated seed
- ~30 hand-curated models in `aliases.yaml` with published benchmark scores
- ~3,400 long-tail models from leaderboard fetches
- **MoE awareness**: separate `total_params_b` / `active_params_b` fields;
  Qwen3-30B-A3B ranks at 3B-decode-speed, not 30B
- **Bandwidth-bound TPS estimate**: physics-grounded for Apple Silicon
  (M1 → M5) using published memory bandwidth
- **KV-cache budgeting**: total resident size = weights + KV at chosen context
- 16 agent harnesses (Cline, Claude Code, Roo Code, Kilo Code, Aider,
  Continue.dev, pi, Hermes Agent, OpenClaw, MCP clients, Open WebUI,
  SillyTavern, LibreChat, Ollama, LM Studio, MLX) with hard filters + soft
  reranking via `use_case_boost`
- Wizard flow: use case → harness → ranked recommendations
- Browse page: server-side search, filters (capabilities, size buckets, source,
  family), sort, infinite scroll, live facet counts
- Sources page with per-source refresh + status timing
- "Why?" expandable per recommendation: full benchmark provenance + copy-paste
  install commands per (model, harness)
- 5 sort modes on Results: best fit / highest benchmarks / fastest / smallest /
  best reasoning — headline number adapts to the active sort
- Score explainer panel (formula + bands)
- Single-port packaged build (`make package run`) with SPA fallback for
  client-side routes
- 20 pytest tests covering recommender, identity, API
- GitHub Actions CI: pytest + tsc + vite build on every PR

### Honest limitations (carried into 0.1.0)
- macOS-only for hardware scan (recommender works without one if hardware
  specs are passed manually via API)
- Curated benchmarks for only ~30 models out of ~3,400 in catalog
- HuggingFace gated models (Llama, Gemma) return 401 to anonymous metadata
  fetches; `HF_TOKEN` support not yet wired
- LMSYS Arena ELO lives in pickle files we don't deserialize; only MT-bench
  and MMLU are pulled from that source
- Artificial Analysis cross-validation requires `MODEL_ADVISOR_AA_API_KEY`
  (skipped silently otherwise)

[Unreleased]: https://github.com/ethanphan3993/Model-Advisor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ethanphan3993/Model-Advisor/releases/tag/v0.1.0
