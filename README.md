# webwalker

`webwalker` is a `query-time pre-RAG corpus selector` for naturally linked corpora.

## What It Is

- A research prototype for selecting a compact evidence set before downstream RAG or GraphRAG.
- A selector-first system: the primary output is a selected corpus or evidence set under an explicit token budget.
- A lightweight, training-free, budget-aware experimentation repo.

## What It Is Not

- Not a complete RAG system.
- Not an answerer-first system.
- Not the current round's learned retriever.
- Not a production service.

## Where To Read Next

- [docs/paper-positioning.md](docs/paper-positioning.md): paper-facing problem framing, defensible claims, and novelty map.
- [docs/phase-decisions.md](docs/phase-decisions.md): evidence-backed experiment conclusions and next ablations.
- [docs/current-implementation.md](docs/current-implementation.md): what is implemented now, how it is evaluated, and where the current claim boundary sits.
- [docs/corpus-selection-literature.md](docs/corpus-selection-literature.md): paper-facing related-work guide and baseline priority.
- [docs/literature-map.md](docs/literature-map.md): full research inventory.

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
