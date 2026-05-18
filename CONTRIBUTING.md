# Contributing to Model Advisor

Thanks for your interest! The most useful contributions right now are:

1. **Adding models to the curated catalog** (5 min — just YAML)
2. **Adding agent harnesses** people are using locally (15 min — YAML + maybe a small recommender hint)
3. **Adding data sources** (an hour — implement a `fetch_and_store()` async function)
4. **Bug reports** with reproducible scenarios (especially scoring weirdness)

Everything below assumes you've cloned the repo and have it running locally.

## Dev setup

```bash
git clone https://github.com/ethanphan3993/Model-Advisor.git
cd Model-Advisor
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
(cd frontend && npm install)

make refresh   # one-time: populate cache (~25s)
make dev       # backend :8000 + frontend :5173
```

## Running tests

```bash
make test                                # backend pytest (20 tests)
(cd frontend && npx tsc --noEmit)        # frontend type check
(cd frontend && npm run build)           # frontend build (also runs tsc)
```

CI runs all three on every PR.

## Adding a model to the curated catalog

The curated catalog is `backend/data/aliases.yaml`. Adding an entry tells the
system how the same model is named across Ollama / HuggingFace / LM Studio /
LMSYS leaderboards, and (importantly) gives it the param-count metadata the
recommender needs.

1. **Open `backend/data/aliases.yaml`** and append a block in the relevant
   section (or start a new one). Schema:

   ```yaml
   - canonical_id: my-family:8b:instruct          # {family}:{size}:{variant}
     family: my-family                             # lowercase, hyphenated
     parameter_size: 8B                            # "8B" / "8x7B" / "30B-A3B" (display)
     variant: instruct                             # instruct | base | chat | code | vision
     display_name: My Family 8B Instruct
     total_params_b: 8                             # full param count (memory cost)
     active_params_b: 8                            # per-token (== total for dense)
     # is_moe: true                                # uncomment for MoE models
     # vision: true                                # uncomment for VLMs
     aliases:
       hf: someorg/MyFamily-8B-Instruct            # exact HF model id
       ollama: my-family:8b                        # ollama tag from ollama.com/library
       lmsys: my-family-8b-instruct                # name in lmarena leaderboard CSV (optional)
       lmstudio: lmstudio-community/MyFamily-8B-Instruct-GGUF   # if curated by LM Studio (optional)
   ```

2. **(Optional but recommended) add benchmark scores** in
   `backend/data/benchmarks.yaml`:

   ```yaml
   my-family:8b:instruct:
     humaneval: 75.0
     ifeval: 80.0
     mmlu_pro: 45.0
     gpqa: 35.0
     bbh: 70.0
     math: 65.0
     arena_elo: 1180          # optional — raw ELO
     context_length: 32768
     tool_calling: true
   ```

   Use real published numbers from the model card or leaderboard. If you
   don't have a value, omit it — partial coverage is fine.

3. **Run** `make refresh` to seed the new entry into the local DB.

4. **Verify**: `curl localhost:8000/api/models/my-family:8b:instruct | jq` —
   you should see the model with its benchmarks.

5. **Submit a PR** with just those two files changed.

### MoE models

Mixture-of-Experts models have **separate** `total_params_b` and
`active_params_b`. The recommender uses total for memory budgeting (you have
to load all experts) and active for decode speed. This is the whole reason
Qwen3-30B-A3B can run at 3B-speed on hardware that couldn't otherwise handle
a 30B model.

```yaml
- canonical_id: qwen-3:30b-a3b:instruct
  parameter_size: 30B-A3B           # display: "{total}B-A{active}B"
  total_params_b: 30                # 30B in memory
  active_params_b: 3                # 3B reads per token → 10× faster decode
  is_moe: true                      # ← important
```

## Adding an agent harness

Harnesses are concrete agents/IDEs/runtimes that consume models — Cline,
Cursor, Aider, Continue, LM Studio, etc. They appear in the wizard's step 2
and shape the recommendation.

Edit `backend/data/harnesses.yaml`:

```yaml
- id: my-cool-agent
  name: My Cool Agent
  category: coding-agent          # coding-agent | general-agent | tool-use | chat-ui | runtime
  description: One-line description shown in the wizard.
  homepage: https://example.com   # optional; renders as ↗ on the card

  requires:                        # HARD filters — model must satisfy all
    context_length_min: 32000      # ctx ≥ 32K
    tool_calling: true             # must support function calling
    # available_in: [ollama, lmstudio-community, huggingface_gguf]
    # family_in: [llama-3, qwen-2.5]   # restrict to specific families

  prefers:                         # SOFT signals — boost matching models
    use_case_boost:                # multiply use_case_score for these tasks
      coding: 1.5
      agentic: 1.3
    reasoning_bonus: 1.0           # adds (gpqa+bbh)/2 × this factor
    family_bonus:                  # add raw points if family matches
      hermes-3: 2.0
    tool_calling_bonus: 0.3        # add this if model has tool calling

  install_command_template: "Configure with model={display_name}"
```

Reference the existing entries (Cline, Aider, Hermes Agent) for examples of
each pattern.

No code changes needed — the YAML is read on every startup.

## Adding a data source

Sources fetch external data and write it to SQLite. Each lives in
`backend/services/sources/<name>.py` and exposes one async function:

```python
async def fetch_and_store() -> SourceRunResult:
    return await timed_run("my_source", _fetch)
```

Where `_fetch()` does the work and returns `(rows_written, status, error_msg)`.
Status is `"ok" | "partial" | "error" | "skipped"`.

A minimal scaffold:

```python
"""My new source — describe what it provides."""

from __future__ import annotations
import httpx
from backend.db import connect, ensure_model_stub, record_source_run, upsert_score
from backend.services.identity import resolve_alias, heuristic_canonical_id
from backend.services.sources import SourceRunResult, timed_run

URL = "https://example.com/leaderboard.json"


async def _fetch() -> tuple[int, str, str]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(URL)
        if resp.status_code != 200:
            with connect() as conn:
                record_source_run(conn, "my_source", "error", 0,
                                  error=f"HTTP {resp.status_code}")
            return 0, "error", f"HTTP {resp.status_code}"
        data = resp.json()

    rows = 0
    with connect() as conn:
        for entry in data.get("models", []):
            model_name = entry["model"]
            cid = (resolve_alias("hf", model_name)
                   or heuristic_canonical_id(model_name))
            if not cid:
                continue
            ensure_model_stub(conn, cid)        # creates parent row if missing

            upsert_score(conn, {
                "canonical_id": cid,
                "benchmark": "my_metric",
                "source": "my_source",
                "value": float(entry["score"]),
                "max_value": 100.0,
                "confidence": "measured",
            })
            rows += 1
        record_source_run(conn, "my_source", "ok" if rows else "partial", rows)
    return rows, ("ok" if rows else "partial"), ""


async def fetch_and_store() -> SourceRunResult:
    return await timed_run("my_source", _fetch)
```

Then register it in `backend/services/refresh.py`:

```python
from backend.services.sources import my_source

def _build_fetchers():
    fetchers = {
        ...
        "my_source": my_source.fetch_and_store,
    }
```

Run `make refresh` and verify the new source appears on `/api/sources`.

### Source design rules

1. **Be tolerant of schema drift.** Use `.get()` everywhere. Wrap parsing in
   try/except. Log status as `"partial"` if the response shape is wrong;
   `"error"` for HTTP failures.
2. **Always `record_source_run`** — even on early returns. The Sources page
   needs a row to show the source.
3. **Always `ensure_model_stub(conn, cid)`** before `upsert_score` or
   `upsert_alias` — heuristic canonical_ids may not have a parent row yet,
   and the foreign-key constraint will fail.
4. **Use `confidence: "measured"`** for direct benchmark numbers,
   `"interpolated"` for derived/aggregated scores, `"estimated"` for guesses.

## Code style

**Python:** follow the existing style. Type hints on public functions.
Dataclasses for structured data. f-strings, no `.format()`. No `from typing
import Optional` — use `X | None` (Python 3.11+).

**TypeScript:** existing style is Tailwind utility classes, functional
components, `useState`/`useEffect` (no state library). Match it.

**Commits:** imperative mood, ~50 char subject. Example:
```
Add Granite 4 family to curated catalog

Adds 8B and 32B variants with published HumanEval / MMLU-PRO scores
from the IBM model card.
```

## PR process

1. Fork → branch → push → open PR against `main`.
2. CI must pass (pytest + tsc + vite build).
3. Keep PRs focused — one model, one source, one feature per PR is ideal.
4. For non-trivial changes, open an issue first to discuss approach.

## Reporting bugs

Use the bug report template — it asks for hardware specs, relevant Sources
page status, and reproduction steps. Scoring complaints especially benefit
from the *exact* (use_case, harness) combination and what you'd expect.

## What I won't merge (probably)

- Anything that adds telemetry or phones home
- Vendoring closed/proprietary leaderboards
- Recommendations that aren't traceable to a public source
- Adding personality/persona-style abstractions back (we tried; they didn't
  work — concrete agents/use-cases beat vibes)

That last one is open to revisit if someone shows a clean way.

## Questions

Open an issue with the `question` label.
