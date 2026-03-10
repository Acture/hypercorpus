Corpus is a connected graph instead of single documents.

# webwalker

`webwalker` is a research prototype for multi-hop QA over naturally linked corpora.
The current repo now contains a minimal end-to-end local pipeline:

- link-context document graph construction
- anchor selection policies
- query-time semantic walking
- lazy subgraph extraction over visited documents
- heuristic answer synthesis with evidence
- lightweight evaluation against `webwalker`, dense RAG, and a local GraphRAG-style baseline

This is still an offline experimentation repo, not a production service.

## Research Direction

- High-level idea: [Research Ideas.md](Research%20Ideas.md)
- Current implementation and test design: [docs/current-implementation.md](docs/current-implementation.md)

## Current Modules

- [`src/webwalker/graph.py`](src/webwalker/graph.py): `LinkContextGraph`, document nodes, semantic edge payloads, `2wikimultihop` adapter.
- [`src/webwalker/walker.py`](src/webwalker/walker.py): `DynamicWalker`, walk budget, stop reasons, edge scoring.
- [`src/webwalker/subgraph.py`](src/webwalker/subgraph.py): lazy subgraph extraction over visited nodes only.
- [`src/webwalker/answering.py`](src/webwalker/answering.py): heuristic answer generation with evidence.
- [`src/webwalker/eval.py`](src/webwalker/eval.py): local evaluator and three comparable pipelines.
- [`src/webwalker/store/kvstore/sqlite.py`](src/webwalker/store/kvstore/sqlite.py): streaming SQLite-backed corpus loading for Hotpot-style archives.

## Run Tests

```bash
uv run pytest -q
```

The test suite is intentionally synthetic and fast. It validates contracts and behavior without requiring the full 7GB+ datasets.
