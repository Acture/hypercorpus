Corpus is a connected graph instead of single documents.

# webwalker

`webwalker` is a research prototype for pre-RAG corpus selection over naturally linked corpora.
The current research question is:

`can link-semantic corpus selection before GraphRAG be cheaper, more accurate, and more controllable than dense top-k or eager full-corpus extraction?`

The current repo contains a local experimentation pipeline for that claim:

- link-context document graph construction
- anchor selection policies
- query-time corpus selection over hyperlink semantics
- budget-swept corpus reduction before downstream reasoning
- lazy subgraph extraction over selected documents
- heuristic answer synthesis as a secondary sanity check
- lightweight evaluation against `dense_topk`, `expand_topology`, `expand_anchor`, `expand_link_context`, `webwalker_selector`, `oracle_start_webwalker`, and `eager_full_corpus_proxy`

This is still an offline experimentation repo, not a production service.

## Research Direction

- High-level idea: [Research Ideas.md](Research%20Ideas.md)
- Current implementation and test design: [docs/current-implementation.md](docs/current-implementation.md)

## Current Modules

- [`src/webwalker/graph.py`](src/webwalker/graph.py): `LinkContextGraph`, document nodes, semantic edge payloads, `2wikimultihop` adapter.
- [`src/webwalker/selector.py`](src/webwalker/selector.py): exploratory semantic search selectors retained for ablations and future work.
- [`src/webwalker/walker.py`](src/webwalker/walker.py): `DynamicWalker`, walk budget, stop reasons, edge scoring.
- [`src/webwalker/subgraph.py`](src/webwalker/subgraph.py): lazy subgraph extraction over selected nodes only.
- [`src/webwalker/answering.py`](src/webwalker/answering.py): heuristic answer generation with evidence.
- [`src/webwalker/eval.py`](src/webwalker/eval.py): selector-first evaluator, budget sweeps, GraphRAG proxy baseline, and secondary end-to-end summaries.
- [`src/webwalker/datasets/twowiki.py`](src/webwalker/datasets/twowiki.py): `2WikiMultihopQA` graph + question loaders.
- [`src/webwalker/experiments.py`](src/webwalker/experiments.py): batch experiment runner, summary generation, and GraphRAG-compatible CSV export.
- [`src/webwalker/store/kvstore/sqlite.py`](src/webwalker/store/kvstore/sqlite.py): streaming SQLite-backed corpus loading for Hotpot-style archives.

## Run Tests

```bash
uv run pytest -q
```

The test suite is intentionally synthetic and fast. It validates contracts and experiment behavior without requiring the full datasets.

## Run 2Wiki Experiments

```bash
uv run webwalker-cli experiments run-2wiki \
  --questions /path/to/dev.json \
  --graph-records /path/to/para_with_hyperlink.jsonl \
  --output /tmp/webwalker-2wiki \
  --limit 100 \
  --selectors dense_topk,expand_topology,expand_anchor,expand_link_context,webwalker_selector,oracle_start_webwalker,eager_full_corpus_proxy \
  --budget-ratios 0.01,0.02,0.05,0.10,1.00 \
  --seed 7 \
  --max-steps 3 \
  --top-k 2 \
  --with-e2e \
  --export-graphrag-inputs
```

The command writes:

- `results.jsonl`: one record per case, selector, and budget ratio with nested `selection` and optional `end_to_end`
- `summary.json`: aggregated selector metrics grouped by `selector x budget`
- `graphrag_inputs/`: GraphRAG-compatible CSV slices with fixed `id,title,text,url` columns
