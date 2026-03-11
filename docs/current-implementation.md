# Current Implementation

## Positioning

`webwalker` is currently a research prototype for validating a specific selection hypothesis:

`query -> lightweight start retrieval -> corpus selection over semantic hyperlinks -> lazy subgraph extraction -> optional answer synthesis -> evaluation`

The implementation is local-first, selector-first, and still centered on "select a smaller corpus before downstream RAG or GraphRAG." The lexical selector family is fully local and testable without network calls. The LLM selector family is optional, provider-backed, and covered with mocked tests.

## What Is Implemented

### 1. Link-context graph

- `LinkContextGraph` stores document nodes plus semantic edge payloads.
- Each edge keeps `anchor_text`, `sentence`, `sent_idx`, and optional reference metadata.
- `LinkContextGraph.from_2wikimultihop_records(...)` adapts `2wikimultihop` paragraph records into a graph while preserving hyperlink context.
- The graph also implements `topk_similar(...)`, so it can serve as both graph storage and a lightweight dense-retrieval stand-in.

### 2. Stable start-policy contract

- `StartPolicy.select_start(...)` returns `list[T]` consistently.
- `MaxPhiOverAnchors` returns a ranked node list instead of a single node.
- `SelectByCosTopK` returns node ids only, not `(node, score)` tuples.
- Helper functions in `webwalker.candidate` expose both the ranked list and the single best start node.

### 3. Corpus-selection baselines

- The default evaluator now uses a fixed reviewer-facing comparison set:
  - `seed_rerank`
  - `seed_plus_topology_neighbors`
  - `seed_plus_anchor_neighbors`
  - `seed_plus_link_context_neighbors`
  - `seed__anchor_overlap__single_path_walk`
  - `seed__link_context_overlap__single_path_walk`
  - `seed__anchor_overlap__two_hop_single_path_walk`
  - `seed__link_context_overlap__two_hop_single_path_walk`
- Diagnostic selectors remain available outside the default study matrix:
  - `seed__title_aware__single_path_walk`
  - `oracle_seed__link_context_overlap__single_path_walk`
  - `seed__link_context_llm__single_path_walk`
  - `seed__link_context_llm__two_hop_single_path_walk`
  - `oracle_seed__link_context_llm__single_path_walk`
  - `random__single_path_walk`
  - `full_corpus_upper_bound`
- The renamed walker family is intentionally explicit about control variables:
  - `seed__...` means dense-seed start policy
  - `..._overlap__...` means lexical overlap scoring
  - `..._llm__...` means provider-backed step scoring with rationale and token accounting
  - `...__single_path_walk` means a single start node and a single greedy walk path
- `full_corpus_upper_bound` is a GraphRAG-compatible full-corpus proxy:
  - it always selects the full graph
  - it reports full-corpus token cost
  - it exports the exact CSV slice that a downstream GraphRAG run would consume

### 4. Experiment budget and controllability

- The experiment layer uses a single budget object with:
  - `max_steps`
  - `top_k`
  - `token_budget_tokens`
  - `token_budget_ratio`
- Exactly one of `token_budget_tokens` or `token_budget_ratio` must be set.
- Absolute token budgets are now the primary controllability axis.
- The default token-budget sweep is:
  - `128`
  - `256`
  - `512`
  - `1024`
- Ratio budgets remain available for legacy comparisons.

### 5. Selector-first evaluation

- `Evaluator` now treats corpus selection as the primary output.
- Primary reported metrics:
  - `support_recall`
  - `support_precision`
  - `support_f1`
  - `path_hit` when available
  - `selected_nodes_count`
  - `selected_token_estimate`
  - `compression_ratio`
  - `selection_runtime_s`
  - `budget_adherence`
- Selector-side LLM accounting is recorded when applicable:
  - `selector_provider`
  - `selector_model`
  - `selector_prompt_tokens`
  - `selector_completion_tokens`
  - `selector_total_tokens`
  - `selector_runtime_s`
  - `selector_llm_calls`
- Secondary reported metrics:
  - `answer_em`
  - `answer_f1`
  - `answer`
  - `confidence`
- Step-level selector traces are written to `selector_logs.jsonl` with:
  - candidate scorecards
  - structured rationale
  - backend/provider/model
  - latency, tokens, and cache hit state
  - fallback reason when an LLM step drops back to overlap scoring
- Experiment summaries are grouped by `selector x budget x selector_provider x selector_model`, not by selector only.

### 6. 2Wiki experiment support

- `load_2wiki_graph(...)` reads hyperlink paragraph JSONL and builds a title-keyed `LinkContextGraph`.
- `load_2wiki_questions(...)` reads the question JSON and maps supporting facts into evaluation metadata.
- `prepare_2wiki_store(...)` converts local raw files or URL sources into:
  - `manifest.json`
  - `questions/{train,dev,test}.json`
  - `index/catalog.sqlite`
  - `shards/part-*.jsonl.gz`
- `ShardedLinkContextStore` keeps the selector runtime off the full raw graph:
  - metadata and retrieval come from `catalog.sqlite`
  - document sentences are loaded lazily from the selected shard
- `run_2wiki_experiment(...)` runs the selector matrix over a batch of cases and writes:
  - `results.jsonl`
  - `selector_logs.jsonl`
  - `summary.json`
  - GraphRAG-compatible CSV slices under `graphrag_inputs/`
- `run_2wiki_store_experiment(...)` runs the same selector matrix against a prepared store and writes chunked outputs under `runs/<exp>/chunks/<chunk-id>/`
- The CLI entrypoint is:
  - `webwalker-cli experiments run-2wiki ...`
  - `webwalker-cli experiments run-2wiki-store ...`
  - `webwalker-cli experiments merge-2wiki-results ...`
  - `webwalker-cli datasets fetch-2wiki ...`
  - `webwalker-cli datasets prepare-2wiki-store ...`
  - `webwalker-cli datasets inspect-2wiki-store ...`
  - `webwalker-cli datasets write-2wiki-sample ...`
- The output schema is selector-first:
  - each record contains nested `selection`
  - each record may optionally contain `end_to_end`
  - each record is keyed by selector and budget
  - selector logs are written separately from selection summaries

### 7. Dataset fetch workflow

- `fetch-2wiki` keeps question files and the large hyperlink graph separate.
- Default fetch behavior is question-only so the large shared graph download stays explicit.
- `inspect-2wiki-store` reports:
  - local raw data availability
  - local store/cache state
  - free disk space
  - remote raw archive sizes when they are discoverable
  - a recommended next action
- Extracted layout is stable:
  - `questions/<split>.json`
  - `graph/para_with_hyperlink.jsonl`
- `prepare-2wiki-store` adds hard space guards and defaults to `--no-keep-raw`.
- `write-2wiki-sample` writes a tiny smoke dataset in the same layout, so the experiment CLI can be tested without network or large downloads.

### 8. Lazy extraction and secondary end-to-end QA

- `SubgraphExtractor.extract(...)` only reads from selected nodes.
- `Answerer.answer(...)` remains heuristic and local.
- `LLMAnswerer` is available as an explicit fixed-reader path for end-to-end answer checks.
- End-to-end answer quality is still produced because reviewers will likely want it, but it is treated as secondary evidence rather than the primary claim.

### 9. Corpus storage

- `SQLiteKVStore` supports `get(...)` for query-time document access.
- `build_from_tar_bz2(...)` remains the streaming ingest path for large Hotpot-style Wikipedia archives.
- `TwoWikiStoreManifest` and `catalog.sqlite` now provide the selector runtime with:
  - node -> shard mapping
  - lightweight full-corpus retrieval
  - link-context lookup without loading the full raw graph
- `LocalDirectoryStore` and `S3CompatibleObjectStore` share the same `ObjectStore` contract so a prepared store can live on local disk or behind an `s3://` URI.

## What Is Still Not Implemented

- multi-path or beam-based walker families
- real eager GraphRAG indexing/query execution
- full GraphRAG integration as the answer backend
- training or RL-based selector policies
- real benchmark runners over HotpotQA or MuSiQue
- open-web crawling or online serving

## Concrete Test Design

The repo uses fast synthetic tests instead of requiring the real corpora for basic validation.

### Contract tests

- `test_candidate_policy.py`
  - verifies ranked start-policy output
  - verifies helper API behavior
  - locks the contract that start selection returns node ids

### Graph adapter tests

- `test_graph.py`
  - verifies `2wikimultihop` records become semantic edges with preserved anchor text and sentence context
  - verifies induced subgraph behavior

### Selector and walker tests

- `test_selector.py`
  - verifies the exploratory selector-family library still behaves correctly
  - verifies selector budgets and search variants
- `test_walker.py`
  - verifies the single-path greedy walk baselines still behave as expected
  - verifies cycle avoidance
  - verifies stop conditions for dead ends and low-confidence edges

### Evaluation tests

- `test_eval.py`
  - verifies the reviewer-facing selector registry runs
  - verifies semantic-link ablations differ for topology, anchor-only, overlap-walk, and link-context selection
  - verifies budget adherence for dense, expansion, single-path walkers, and eager full-corpus proxy selection
  - verifies secondary end-to-end results are still attached when enabled
  - verifies legacy selector names now fail fast

### 2Wiki loader and runner tests

- `test_twowiki.py`
  - verifies title-keyed graph construction from hyperlink paragraph records
  - verifies question loading and supporting-fact mapping
- `test_experiments.py`
  - verifies selector x budget output files
  - verifies `summary.json` grouping by selector, budget, provider, and model
  - verifies GraphRAG-compatible CSV export
  - verifies chunked store-backed runs and merged summaries
- `test_cli_experiments.py`
  - verifies the Typer CLI command works end to end on a tiny sample
  - verifies `--token-budgets`, `--budget-ratios`, `--no-e2e`, and `--no-export-graphrag-inputs`
  - verifies `run-2wiki-store` and `merge-2wiki-results`
- `test_selector_llm.py`
  - verifies multi-backend selector LLM configuration and fail-fast behavior
  - verifies selector cache, rationale logging, token accounting, and two-hop score parsing with mocked providers
- `test_dataset_fetch.py`
  - verifies split-specific 2Wiki extraction, archive retention, and the sample dataset writer
  - verifies the prepare flow rejects unsafe free-space configurations
- `test_cli_datasets.py`
  - verifies dataset fetch and sample writer CLI flows without external network access
  - verifies store prepare and inspect commands
- `test_twowiki_store.py`
  - verifies manifest/catalog/shard generation
  - verifies lazy shard loading and mocked remote-store downloads

### Storage tests

- `test_store.py`
  - verifies streaming import from a synthetic `.tar.bz2`
  - verifies `SQLiteKVStore.get(...)` returns decoded JSON payloads

## Default Assumptions In Code

- The repo is an offline research sandbox.
- Natural hyperlinks are the primary graph structure.
- The primary claim is pre-RAG corpus selection, not end-to-end QA dominance.
- End-to-end QA and answer heuristics are secondary reviewer-facing sanity checks.
- Real GraphRAG execution is not part of this round; the eager comparison is proxied by full-corpus export and cost accounting.
