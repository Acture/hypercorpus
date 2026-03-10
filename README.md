Corpus is a connected graph instead of single documents.

# webwalker

`webwalker` is a research prototype for pre-RAG corpus selection over naturally linked corpora.
The core claim is:

`query-time link-semantic corpus selection can outperform dense top-k and eager GraphRAG on cost, support recall, and controllability`

The current repo contains a local experimentation pipeline for that claim:

- link-context document graph construction
- anchor selection policies
- selector-first graph search over hyperlink semantics
- lazy subgraph extraction over selected documents
- heuristic answer synthesis as a downstream sanity check
- lightweight evaluation against dense RAG, a local GraphRAG-style baseline, and multiple selector families

This is still an offline experimentation repo, not a production service.

## Research Direction

- High-level idea: [Research Ideas.md](Research%20Ideas.md)
- Current implementation and test design: [docs/current-implementation.md](docs/current-implementation.md)

## Current Modules

- [`src/webwalker/graph.py`](src/webwalker/graph.py): `LinkContextGraph`, document nodes, semantic edge payloads, `2wikimultihop` adapter.
- [`src/webwalker/selector.py`](src/webwalker/selector.py): corpus selector protocols, budgets, pathfinding-family selectors, semantic PPR, hybrid selection.
- [`src/webwalker/eval.py`](src/webwalker/eval.py): selector-first evaluator, support-recall and token-cost metrics, selection report ranking.
- [`src/webwalker/subgraph.py`](src/webwalker/subgraph.py): lazy subgraph extraction over visited nodes only.
- [`src/webwalker/answering.py`](src/webwalker/answering.py): heuristic answer generation with evidence.
- [`src/webwalker/walker.py`](src/webwalker/walker.py): legacy greedy semantic walker retained as a baseline.
- [`src/webwalker/store/kvstore/sqlite.py`](src/webwalker/store/kvstore/sqlite.py): streaming SQLite-backed corpus loading for Hotpot-style archives.

## Run Tests

```bash
uv run pytest -q
```

The test suite is intentionally synthetic and fast. It validates contracts and behavior without requiring the full 7GB+ datasets.
