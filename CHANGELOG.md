# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(nothing yet)

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
