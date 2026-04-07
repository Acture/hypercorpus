# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

hypercorpus is a research prototype for **query-time budgeted subgraph/corpus discovery** over naturally linked corpora. It is a **selector-first** system: the primary output is a selected corpus or subgraph under an explicit token budget, run *before* downstream RAG or GraphRAG. It is not an answerer-first system, not a production service, and not an end-to-end RAG pipeline.

## Commands

```bash
# Run fast synthetic test suite
uv run pytest -q

# Run a single test file
uv run pytest tests/test_selector.py -q

# Run a single test by name
uv run pytest -k "test_name" -q

# Lint and format
uv run ruff check .
uv run ruff format .

# CLI help
uv run hypercorpus --help
```

## Architecture

Two packages live under `src/`:

- **`hypercorpus`** — core library (selector algorithms, graph structures, evaluation, dataset adapters)
- **`hypercorpus_cli`** — Typer-based CLI wrapping the core library

### Core data flow

1. **Datasets** (`datasets/`) — fetch and normalize corpora (HotpotQA, IIRC, MuSiQue, 2WikiMultihopQA). Each dataset has an adapter (`BaseDatasetAdapter`) and a fetch function. Prepared stores (`PreparedDatasetStore`, `ShardedDocumentStore`) serve documents at query time.
2. **Graph** (`graph.py`) — `LinkContextGraph` holds document nodes and hyperlink edges with anchor text, sentence context, and reference metadata. Supports `topk_similar()` for dense-retrieval seeding.
3. **Selection** (`selector.py`, `selector_llm.py`, `walker.py`) — `CorpusSelector` runs budget-aware subgraph selection. Strategies include dense-started hyperlink walks, single/multipath traversal, and optional LLM-guided controllers. `WalkBudget` and `SelectorBudget` enforce token limits.
4. **Scoring** (`candidate/`) — pluggable scorers: overlap-based (`AnchorOverlapStepScorer`), embedding-based (`SentenceTransformerEmbedder`), and LLM-based (`LLMStepLinkScorer`, `LLMController`).
5. **Evaluation** (`eval.py`) — `Evaluator` computes metrics per case. Headline metric: `support_f1_zero_on_empty`. Also tracks budget adherence/utilization, node counts, and optional answer F1/EM.
6. **Experiments** (`experiments.py`) — orchestrates end-to-end runs per dataset. Outputs go to `runs/<exp-name>/chunks/<chunk-id>/` with `results.jsonl`, `selector_logs.jsonl`, `summary.json`, `summary_rows.csv`.
7. **Answering** (`answering.py`) — optional post-selection answer synthesis via `Answerer`/`LLMAnswerer` (secondary to selection quality).

### Key design patterns

- **Protocol-based interfaces**: `SupportsAnswer`, `TextEmbedder`, `EmbeddedGraphLike`, `GraphLike` — duck-typed contracts over concrete inheritance.
- **Budget as a first-class concept**: token budgets flow through selection, evaluation, and reporting. `EvaluationBudget` supports absolute and relative modes.
- **Checkpoint/resume**: `RunState` and `InterruptController` in `resume.py` allow experiment resumption from partial runs.

### External integrations

- **MDR baseline** via git submodule at `baselines/mdr` (fork of multihop_dense_retrieval). Wrapped by `baselines/mdr.py`.
- **LLM providers**: Anthropic (Claude), OpenAI, Google Gemini — used for LLM-based selectors and answering.
- **GraphRAG**: optional export of selection results for downstream GraphRAG processing.

## Build & Environment

- **Python ≥3.13**, managed with **uv** (≥0.5.26)
- Dev dependencies: pytest, ruff
- `.env` file holds API keys (never commit)
- Git submodule: `baselines/mdr` — run `git submodule update --init` after clone

## Current Phase: Paper Closure (Mar–May 2026, targeting CIKM 2026-05-25)

The project is in paper-closure mode, tracked in Linear (project: "Hypercorpus Paper Closure", ACT-5 through ACT-9). Current workstream status (2026-04-07):

- **WS1** (ACT-5, High): Pre-gate passed. 20-case controller pilot landed with positive signal (F1 = 0.46 vs dense 0.41). Remaining: 100-case canonical surface on `gcr-vm` + MDR go/no-go decision (~April 10).
- **WS2** (ACT-7, Done): Historical cleanup complete. 2Wiki retest merged, partial-IIRC deprecated.
- **WS3** (ACT-6, In Progress): Paper skeleton landed. Sections 2-3 drafted. Sections 1, 4-9 blocked on IIRC table.
- **WS4** (ACT-8, In Progress): Related work outline + venue packaging done. Prose expansion (C5/ACT-61) ready to write.
- **CTRL** (ACT-9, In Progress): Gate 1 passed. Gates 2-5 blocked on 100-case surface + MDR decision.

### Phase constraints

- No GraphRetriever or HippoRAG expansion before MDR decision is made.
- No broad superiority claims before full-IIRC paper-facing table is locked.
- No concurrent edits to main paper skeleton — hand off through Linear.
- CIKM 2026 is the target venue (abstract 2026-05-18, full paper 2026-05-25).

### Coordination

- **Linear** tracks execution details, blockers, and handoffs.
- **Notion** tracks milestones, key conclusions, and open decisions.
- Every handoff includes: latest commit/branch, artifact path, current blocker, next step.

## Paper Directory

`paper/` contains the paper scaffold (owned by WS3):

- `outline.md` — draft skeleton
- `claim-ledger.md` — claims and their evidence status
- `tables-and-figures.md` — figure/table plan
- `open-risks.md` — explicit risk list
- `related-work-outline.md` — related work structure (WS4)
- `venue-packaging.md` — venue-specific formatting notes (WS4)

## Research Documentation

Canonical reading order in `docs/`: `paper-positioning.md` → `phase-decisions.md` → `current-implementation.md` → `corpus-selection-literature.md` → `literature-map.md`. See `docs/README.md` for source-of-truth rules.
