# Current Implementation

## Positioning

`webwalker` currently implements a selector-first experimentation pipeline:

`query -> start retrieval -> budgeted corpus selection over natural hyperlinks -> lazy extraction -> optional answer synthesis -> evaluation`

The current repo is an offline research sandbox for pre-RAG corpus selection. It is local-first where possible, LLM-assisted only when explicitly configured, and organized around selecting a smaller evidence set before downstream RAG or GraphRAG.

## Current Claim Boundary

- The primary claim is pre-RAG corpus selection, not end-to-end QA dominance.
- The headline metric is `support_f1_zero_on_empty`.
- `answer_em` and `answer_f1` are secondary reviewer-facing sanity checks.
- The full-corpus comparison is a GraphRAG proxy, not a full GraphRAG integration.
- The current version is intentionally lightweight, training-free, and budget-aware.

## What Is Implemented

### Link-Context Graph and Data Loading

- `LinkContextGraph` stores document nodes and semantic hyperlink edges.
- Each edge keeps `anchor_text`, `sentence`, `sent_idx`, and optional reference metadata.
- `LinkContextGraph.from_2wikimultihop_records(...)` adapts `2wikimultihop` paragraph records into a hyperlink graph while preserving link context.
- The graph also exposes `topk_similar(...)`, so it can act as a lightweight dense-retrieval stand-in for seed selection.

### Start Policies and Selector Families

- Start policies return ranked node lists through a stable `StartPolicy.select_start(...)` contract.
- Dense and lexical seed selection are both supported.
- The current experiment runner supports fixed selector matrices plus extra diagnostic selectors outside the default study set.
- Selector-side LLM scoring is optional and records provider, model, token usage, runtime, cache state, and fallback behavior.

### Budgeted Evaluation

- The experiment layer uses an explicit budget object with `max_steps`, `top_k`, `token_budget_tokens`, and `token_budget_ratio`.
- Absolute token budgets are the primary controllability axis for current studies.
- The evaluator treats corpus selection as the primary output.
- Primary metrics include:
  - `support_recall`
  - `support_precision`
  - `support_f1`
  - `support_f1_zero_on_empty`
  - `budget_adherence`
  - `budget_utilization`
  - `empty_selection`
  - `selection_runtime_s`
  - `selected_nodes_count`
  - `selected_token_estimate`
- Secondary metrics include:
  - `answer_em`
  - `answer_f1`
  - `answer`
  - `confidence`
- Summaries are grouped by `selector x budget x selector_provider x selector_model`.

### 2Wiki Runtime and Storage

- `load_2wiki_graph(...)` and `load_2wiki_questions(...)` support direct 2Wiki evaluation.
- `prepare_2wiki_store(...)` converts raw files into a sharded store with:
  - `manifest.json`
  - `questions/{train,dev,test}.json`
  - `index/catalog.sqlite`
  - `shards/part-*.jsonl.gz`
- `ShardedLinkContextStore` keeps query-time selection off the full raw graph by loading document sentences lazily from selected shards.
- `run_2wiki_experiment(...)` and `run_2wiki_store_experiment(...)` write:
  - `results.jsonl`
  - `selector_logs.jsonl`
  - `summary.json`
  - optional GraphRAG-compatible CSV slices

### Lazy Extraction and Secondary Answering

- `SubgraphExtractor.extract(...)` only reads from selected nodes.
- `Answerer.answer(...)` remains heuristic and local.
- `LLMAnswerer` is available as an explicit fixed-reader path for secondary end-to-end checks.

## Selector Family Interpretation

### Dense-Seed Baselines

- `hop_0__dense` variants answer the question: how far can plain lexical or dense seed retrieval go under the same token budget.
- `__budget_fill_relative_drop` variants test whether filling the remaining budget improves all-case evidence recovery.

### Hyperlink-Local Walk

- `single_path_walk` variants start from a seed and expand over natural hyperlinks.
- `link_context_*` scorers use anchor text and surrounding sentence context as the local decision unit.
- These are the current core selector family for the paper-facing story.

### Broad Search Variants

- `beam` and `astar` families explore wider and deeper graph frontiers.
- They are currently best treated as search-space diagnostics rather than the main claimed contribution.

### Diagnostic Upper Bounds

- `gold_support_context` is an oracle-style upper bound for evidence coverage under the same token budget accounting.
- `full_corpus_upper_bound` is a full-corpus proxy for GraphRAG-style eager inclusion.

## Current Evidence Snapshot

The current phase-decision anchor is the completed `phase-decision-30` run on `2Wiki dev`, `30` cases, `token budget 256`, and `no-e2e`. The strongest tested operating point is:

`top_1 + sentence_transformer seed + hop_2 + single_path_walk + link_context_llm + budget_fill_relative_drop`

This run supports three narrow conclusions:

- Sentence-transformer seeds outperform lexical seeds in the tested families.
- `top_1` is a better operating point than `top_3` at this budget.
- Broad `hop_3` search with `beam` or `astar` is not a good direction under a fixed `256`-token budget.

Those conclusions should be read together with [phase-decisions.md](phase-decisions.md), which separates supported findings from claims that still need ablation or direct baseline comparisons.

## What Is Still Not Implemented

- Full GraphRAG indexing and query execution
- A production answer backend
- Training or RL-based selector policies
- Direct MDR, GraphRetriever, or HippoRAG reproductions
- Real benchmark runners over HotpotQA or MuSiQue
- Open-web crawling or online serving

## Concrete Test Design

The repo uses fast synthetic tests instead of requiring the real corpora for basic validation.

### Contract and Graph Tests

- `test_candidate_policy.py` verifies ranked start-policy behavior and helper APIs.
- `test_graph.py` verifies graph construction from `2wikimultihop` records and preserved link-context payloads.

### Selector and Walker Tests

- `test_selector.py` covers exploratory selector-family behavior, search variants, and budget handling.
- `test_walker.py` covers greedy walk behavior, cycle avoidance, and stop conditions.

### Evaluation and Experiment Tests

- `test_eval.py` verifies selector registries, budget adherence, GraphRAG proxy selection, and optional secondary end-to-end attachment.
- `test_experiments.py` verifies selector-by-budget outputs, grouped summaries, CSV export, chunked runs, and merged summaries.
- `test_cli_experiments.py` verifies the Typer CLI on tiny samples.
- `test_selector_llm.py` verifies multi-backend selector configuration, mocked token accounting, parsing, caching, and fallback behavior.

### Dataset and Storage Tests

- `test_twowiki.py`, `test_twowiki_store.py`, `test_dataset_fetch.py`, and `test_cli_datasets.py` cover 2Wiki loading, store preparation, lazy shard access, and sample dataset workflows.
- `test_store.py` verifies streaming import and `SQLiteKVStore.get(...)`.

## Default Assumptions In Code

- Natural hyperlinks are the primary graph structure.
- The primary output is selected evidence under an explicit token budget.
- End-to-end answering remains secondary to selector evaluation in the current repo.
