# webwalker

Purpose: repo entrypoint, docs map, and source-of-truth guide.
Canonical for: project overview, reading order, documentation boundaries, and minimal commands.
Not for / See also: paper-facing claims live in `docs/phase-decisions.md`; active experiment plan/status lives in `docs/next-phase-experiments.md`; exploratory notes live under `docs/notes/`.

`webwalker` is a `query-time budgeted subgraph/corpus discovery` prototype for naturally linked corpora.

## What It Is

- A research prototype for discovering a compact evidence set or induced subgraph before downstream RAG or GraphRAG.
- A selector-first system: the primary output is a selected corpus or subgraph under an explicit token budget.
- A lightweight, training-free, budget-aware experimentation repo.

## What It Is Not

- Not a complete RAG system.
- Not an answerer-first system.
- Not the current round's learned retriever.
- Not a production service.

## Documentation Map

This repo is organized around one paper-facing question:

`how should webwalker be argued as query-time budgeted subgraph/corpus discovery over naturally linked corpora`

### Recommended Reading Order

1. [docs/paper-positioning.md](docs/paper-positioning.md)
2. [docs/phase-decisions.md](docs/phase-decisions.md)
3. [docs/current-implementation.md](docs/current-implementation.md)
4. [docs/next-phase-experiments.md](docs/next-phase-experiments.md)
5. [docs/corpus-selection-literature.md](docs/corpus-selection-literature.md)
6. [docs/literature-map.md](docs/literature-map.md)
7. [docs/notes/venue-strategy.md](docs/notes/venue-strategy.md)
8. [docs/notes/search-ideas.md](docs/notes/search-ideas.md)

### Canonical Docs

- [docs/paper-positioning.md](docs/paper-positioning.md): problem framing, thesis, safe claims, and novelty boundaries.
- [docs/phase-decisions.md](docs/phase-decisions.md): evidence-backed experiment conclusions, remaining ablations, and not-yet-compared systems.
- [docs/current-implementation.md](docs/current-implementation.md): implementation surface, evaluation contract, current gaps, and test coverage.
- [docs/next-phase-experiments.md](docs/next-phase-experiments.md): active experiment plan, execution order, local run state, and recovery commands.
- [docs/corpus-selection-literature.md](docs/corpus-selection-literature.md): paper-facing related-work guide and baseline priority.
- [docs/literature-map.md](docs/literature-map.md): full annotated literature inventory derived from `docs/literature-map.tsv`.

### Notes

- [docs/notes/venue-strategy.md](docs/notes/venue-strategy.md): venue-specific narrative deltas, fit criteria, and current risks.
- [docs/notes/search-ideas.md](docs/notes/search-ideas.md): speculative selector/search ideas that are not canonical claims.

## Source-Of-Truth Rules

- Problem framing belongs in `docs/paper-positioning.md`, not in ad hoc notes.
- Claim support belongs in `docs/phase-decisions.md`, not in narrative or strategy docs.
- Implementation status belongs in `docs/current-implementation.md`, not in experiment logs.
- Active execution order and run recovery belong in `docs/next-phase-experiments.md`.
- Bibliography truth lives in `docs/literature-map.tsv`; prose literature guidance lives in `docs/corpus-selection-literature.md`.
- Notes under `docs/notes/` may inform future work, but they are not canonical claim or implementation documents.

## Minimal Commands

Run the fast synthetic test suite:

```bash
uv run pytest -q
```

Run the current store-backed 2Wiki experiment flow:

```bash
uv run webwalker-cli experiments run-2wiki-store \
  --store /data/2wiki-store \
  --exp-name pilot-2wiki \
  --split dev \
  --chunk-size 100 \
  --chunk-index 0 \
  --selectors top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop,gold_support_context \
  --token-budgets 256 \
  --no-e2e \
  --no-export-graphrag-inputs
```

Chunk outputs are written under `runs/<exp-name>/chunks/<chunk-id>/` and include:

- `results.jsonl`
- `selector_logs.jsonl`
- `summary.json`
- `summary_rows.csv`

Merged run directories also write the same `summary_rows.csv` normalization for downstream tables and budget-quality plots.

For benchmark preparation, raw dataset conversion, and non-2Wiki experiment flows, use [docs/current-implementation.md](docs/current-implementation.md) as the operational reference rather than expanding the README into a command catalog.
