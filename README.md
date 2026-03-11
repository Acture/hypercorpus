Corpus is a connected graph instead of single documents.

# webwalker

`webwalker` is a research prototype for pre-RAG corpus selection over naturally linked corpora.
The current research question is:

`can link-semantic corpus selection before GraphRAG be cheaper, more accurate, and more controllable than dense top-k or eager full-corpus extraction?`

The current repo contains a local experimentation pipeline for that claim:

- link-context document graph construction
- anchor selection policies
- query-time corpus selection over hyperlink semantics
- token-budgeted corpus reduction before downstream reasoning
- lazy subgraph extraction over selected documents
- heuristic or fixed-reader answer synthesis as a secondary sanity check
- lightweight evaluation against `seed_rerank`, `seed_plus_topology_neighbors`, `seed_plus_anchor_neighbors`, `seed_plus_link_context_neighbors`, `seed__anchor_overlap__single_path_walk`, `seed__link_context_overlap__single_path_walk`, `seed__anchor_overlap__two_hop_single_path_walk`, `seed__link_context_overlap__two_hop_single_path_walk`, and the `seed__link_context_llm__*` diagnostics

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
- [`src/webwalker/datasets/twowiki_store.py`](src/webwalker/datasets/twowiki_store.py): sharded 2Wiki store prepare flow, lazy graph backend, and object-store-compatible runtime.
- [`src/webwalker/experiments.py`](src/webwalker/experiments.py): batch experiment runner, summary generation, and GraphRAG-compatible CSV export.
- [`src/webwalker/store/kvstore/sqlite.py`](src/webwalker/store/kvstore/sqlite.py): streaming SQLite-backed corpus loading for Hotpot-style archives.

## Run Tests

```bash
uv run pytest -q
```

The test suite is intentionally synthetic and fast. It validates contracts and experiment behavior without requiring the full datasets.

## Fetch 2Wiki Data

Write a zero-download smoke dataset first:

```bash
uv run webwalker-cli datasets write-2wiki-sample \
  --output-dir /tmp/webwalker-2wiki-sample
```

Fetch the official 2Wiki question split only:

```bash
uv run webwalker-cli datasets fetch-2wiki \
  --output-dir /data/2wikimultihop \
  --split dev
```

Fetch the shared hyperlink graph when you are ready for the large artifact:

```bash
uv run webwalker-cli datasets fetch-2wiki \
  --output-dir /data/2wikimultihop \
  --split dev \
  --graph
```

The fetch command writes a stable layout:

- `questions/<split>.json`
- `graph/para_with_hyperlink.jsonl`

This split keeps the large graph download explicit instead of hiding it behind the experiment command.

Inspect local/raw/store state before downloading or preparing:

```bash
uv run webwalker-cli datasets inspect-2wiki-store \
  --cache-dir ~/.cache/webwalker/2wiki
```

Prepare a sharded local store from existing raw files or URLs:

```bash
uv run webwalker-cli datasets prepare-2wiki-store \
  --output-dir /data/2wiki-store \
  --questions-source dataset/2wikimultihop/data_ids_april7 \
  --graph-source dataset/2wikimultihop/para_with_hyperlink.jsonl
```

The prepared store contains:

- `manifest.json`
- `questions/{train,dev,test}.json`
- `index/catalog.sqlite`
- `shards/part-*.jsonl.gz`

## Run 2Wiki Experiments

```bash
uv run webwalker-cli experiments run-2wiki \
  --questions /path/to/dev.json \
  --graph-records /path/to/para_with_hyperlink.jsonl \
  --output /tmp/webwalker-2wiki \
  --limit 100 \
  --selectors seed_rerank,seed_plus_topology_neighbors,seed_plus_anchor_neighbors,seed_plus_link_context_neighbors,seed__link_context_overlap__single_path_walk,oracle_seed__link_context_overlap__single_path_walk,gold_support_context,full_corpus_upper_bound \
  --token-budgets 128,256,512,1024 \
  --seed 7 \
  --max-steps 3 \
  --top-k 2 \
  --no-e2e \
  --export-graphrag-inputs
```

The command writes:

- `results.jsonl`: one record per case, selector, and budget with nested `selection` and optional `end_to_end`
- `selector_logs.jsonl`: one record per scored walk step with candidate scores, rationale, latency, cache, and token usage
- `summary.json`: aggregated selector metrics grouped by `selector x budget x selector_provider x selector_model`
- `graphrag_inputs/`: GraphRAG-compatible CSV slices with fixed `id,title,text,url` columns

To run the fixed reader evaluation with an LLM-scored selector, enable e2e explicitly and pin both selector and reader configuration:

```bash
uv run webwalker-cli experiments run-2wiki \
  --questions /path/to/dev.json \
  --graph-records /path/to/para_with_hyperlink.jsonl \
  --output /tmp/webwalker-2wiki-e2e \
  --limit 100 \
  --selectors seed__link_context_llm__single_path_walk,oracle_seed__link_context_llm__single_path_walk,gold_support_context,full_corpus_upper_bound \
  --token-budgets 128,256,512,1024 \
  --selector-provider openai \
  --selector-model gpt-4.1-mini \
  --selector-cache-path /tmp/webwalker-selector-cache.jsonl \
  --with-e2e \
  --answerer llm_fixed \
  --answer-model gpt-4.1-mini \
  --answer-cache-path /tmp/webwalker-answer-cache.jsonl
```

## Run Store-Backed Chunked Experiments

```bash
uv run webwalker-cli experiments run-2wiki-store \
  --store /data/2wiki-store \
  --exp-name pilot-2wiki \
  --split dev \
  --chunk-size 100 \
  --chunk-index 0 \
  --selectors seed_rerank,seed_plus_topology_neighbors,seed_plus_anchor_neighbors,seed_plus_link_context_neighbors,seed__link_context_overlap__single_path_walk,oracle_seed__link_context_overlap__single_path_walk,gold_support_context,full_corpus_upper_bound \
  --token-budgets 64,128,256,512,1024 \
  --no-e2e \
  --no-export-graphrag-inputs
```

Each chunk writes to `runs/<exp-name>/chunks/<chunk-id>/`.

Merge chunk outputs back into a single summary:

```bash
uv run webwalker-cli experiments merge-2wiki-results \
  --run-dir runs/pilot-2wiki
```
