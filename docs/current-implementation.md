# Current Implementation

Purpose: canonical implementation-status document for `hypercorpus`.
Canonical for: current pipeline shape, supported selector families, evaluation contract, missing pieces, and test coverage.
Not for / See also: paper-facing claims live in `phase-decisions.md`; active experiment sequencing lives in `next-phase-experiments.md`; framing lives in `paper-positioning.md`.

## Positioning

`hypercorpus` currently implements a selector-first experimentation pipeline:

`query -> start retrieval -> budgeted subgraph/corpus discovery over natural hyperlinks -> optional lazy extraction -> optional answer synthesis -> evaluation`

The current repo is an offline research sandbox for selector-first subgraph/corpus discovery. It is local-first where possible, LLM-assisted only when explicitly configured, and organized around selecting a smaller support-bearing subgraph from a naturally linked corpus.

## Current Claim Boundary

- The primary claim is pre-RAG subgraph/corpus discovery, not end-to-end QA dominance.
- The headline metric is `support_f1_zero_on_empty`.
- `answer_em` and `answer_f1` are secondary reviewer-facing sanity checks.
- The full-corpus comparison is a GraphRAG proxy, not a full GraphRAG integration.
- The current method story is `dense-started constrained subgraph selection`, not "walk instead of dense".
- The current version is intentionally lightweight, zero-shot where possible, and budget-aware.
- The current implementation supports an algorithmic discovery paper more directly than an answerer-first NLP paper.
- Downstream local-context packaging is not the core problem definition for the current paper.

## What Is Implemented

### Link-Context Graph and Data Loading

- `LinkContextGraph` stores document nodes and semantic hyperlink edges.
- Each edge keeps `anchor_text`, `sentence`, `sent_idx`, and optional reference metadata.
- `LinkContextGraph.from_2wikimultihop_records(...)` adapts `2wikimultihop` paragraph records into a hyperlink graph while preserving link context.
- The graph also exposes `topk_similar(...)`, so it can act as a lightweight dense-retrieval stand-in for seed selection.

### Start Policies and Selector Families

- Start policies return ranked node lists through a stable `StartPolicy.select_start(...)` contract.
- Dense and lexical seed selection are both supported.
- The current experiment runner supports named selector presets, including `paper_recommended`, `paper_recommended_local`, and `branchy_profiles`, plus extra diagnostic selectors outside the default study set.
- The experiment runner also supports local-only study presets, including `single_path_edge_ablation_local`, `baseline_retest_local`, and `branchy_profiles_384_512`, which bundle selector defaults and token budgets for planned experiments.
- `paper_recommended` is the full paper-facing preset and requires explicit selector LLM configuration. `paper_recommended_local` is the local-only variant that avoids `link_context_llm`.
- Each run now writes `run_manifest.json`, `evaluated_case_ids.txt`, and `study_comparison_rows.csv`, so broader phase samples can be replayed exactly without relying on `--limit`.
- Each run also writes `subset_comparison_rows.csv`, so harder-case and path-aware slices can be compared against the dense control without reprocessing raw logs.
- Selector-side LLM scoring is optional and records provider, model, token usage, runtime, cache state, and fallback behavior.
- Step-scorer composition is now exposed as named profiles for overlap and sentence-transformer scorers. LLM scorer aggregation remains fixed today.

### Budgeted Evaluation

- The experiment layer uses an explicit budget object with `max_steps`, `top_k`, `token_budget_tokens`, and `token_budget_ratio`.
- Both absolute token budgets and corpus-ratio budgets are implemented.
- Fixed token budgets remain useful on fragment-level calibration surfaces such as reduced `2Wiki`.
- Ratio budgets are the cleaner control surface for coarse full-node selector studies such as canonical `IIRC`, where full-document node size makes tiny fixed token caps misleading.
- The evaluator treats subgraph/corpus discovery as the primary output.
- Primary metrics include:
  - `support_recall`
  - `support_precision`
  - `support_f1`
  - `support_f1_zero_on_empty`
  - `support_set_em`
  - `budget_adherence`
  - `budget_utilization`
  - `empty_selection`
  - `selection_runtime_s`
  - `selected_nodes_count`
  - `selected_token_estimate`
- Reviewer-facing diagnostics also include:
  - `avg_path_hit` as the unified path-level exactness metric (`Path Recall` analogue)
  - `bridge` / `comparison` subset slices when `question_type` is available or inferable
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
  - `summary_rows.csv`
  - optional GraphRAG-compatible CSV slices
- merged run directories also export the same normalized `summary_rows.csv`, which is the current thin reporting layer for tables and budget-quality curves

### Additional Benchmark Adapters and Raw Preparation

- `run_iirc`, `run_musique`, and `run_hotpotqa` support direct benchmark evaluation from normalized questions/graph inputs.
- `run_iirc_store`, `run_musique_store`, and `run_hotpotqa_store` run the same benchmarks against prepared stores.
- The dataset CLI supports built-in raw fetch and conversion for `IIRC`, `MuSiQue`, and `HotpotQA`.
- Raw benchmark flows now follow:
  - `fetch-*` -> `<output>/raw/source-manifest.json`
  - `convert-*-raw` -> normalized `questions/`, `graph/normalized.jsonl`, and `conversion-manifest.json`
  - `prepare-*-store-from-raw` -> final prepared store under `store/`
- `HotpotQA distractor` can be converted directly from case-local `context`.
- `HotpotQA fullwiki` still expects a supplied normalized graph bundle; this pass does not ingest a raw Wikipedia dump.

### Lazy Extraction and Secondary Answering

- `SubgraphExtractor.extract(...)` only reads from selected nodes.
- `Answerer.answer(...)` remains heuristic and local.
- `LLMAnswerer` is available as an explicit fixed-reader path for secondary end-to-end checks.

## Selector Family Interpretation

### Dense-Seed Baselines

- `hop_0__dense` variants answer the question: how far can plain lexical or dense seed retrieval go under the same selector budget.
- In the current story, `hop_0__dense` is both a stage-1 seed prior and the main flat control.
- `hop_2__iterative_dense` variants provide an `MDR-style` iterative dense baseline using repeated dense retrieval over accumulated query-plus-context text under the same selector-budget accounting.
- `hop_2__mdr_light` variants provide a repo-native comparison point that expands each frontier node with its own dense query and then merges the hop candidates under the same selector-budget accounting. This is not a trained MDR reproduction.
- `__budget_fill_relative_drop` variants test whether filling the remaining budget improves all-case evidence recovery.

### Hyperlink-Local Walk

- `single_path_walk` variants start from a seed and expand over natural hyperlinks.
- `link_context_*` scorers use anchor text and surrounding sentence context as the local decision unit.
- `link_context_llm_controller` adds a zero-shot semantic controller that chooses actions (`stop`, `choose_one`, `choose_two`) rather than only emitting per-edge scores.
- These are the current core selector family for the paper-facing story.

### Constrained Multi-Path Controller

- `constrained_multipath` is a controller-guided branchy family with at most two live branches and one scout fork.
- It is designed as a dense-anchored selector, not a general beam-search replacement.
- It keeps branch count, backtracking, and budget pacing explicitly bounded so precision collapse is measurable rather than hidden inside a broad search frontier.

### Broad Search Variants

- `beam` and `astar` families explore wider and deeper graph frontiers.
- `ucs` and `beam_ppr` are also available in the canonical selector space.
- In the current paper story they are still search-space diagnostics rather than the main claimed contribution.
- The current `2Wiki` phase result should not be read as a universal negative result for broad search. These variants are still candidates for harder datasets and for scorer-calibration studies.

### Diagnostic Upper Bounds

- `gold_support_context` is an oracle-style upper bound for evidence coverage under the same selector-budget accounting.
- `full_corpus_upper_bound` is a full-corpus proxy for GraphRAG-style eager inclusion.

## Evidence And Planning Pointers

- Use `phase-decisions.md` for completed empirical conclusions and their current claim boundary.
- Use `next-phase-experiments.md` for active sequencing, run recovery status, and near-term execution order.
- Keep this document focused on what the code can do today, not which experiment should run next.

## What Is Still Not Implemented

- Full GraphRAG indexing and query execution
- A production answer backend
- Per-dataset fine-tuned selector policies
- Unified controller distillation and offline bandit calibration
- Direct trained MDR, GraphRetriever, or HippoRAG reproductions
- Automatic raw Wikipedia dump ingestion for fullwiki-style corpora
- Large-scale comparative studies over `IIRC`, `HotpotQA`, or `MuSiQue`
- Open-web crawling or online serving

## Concrete Test Design

The repo uses fast synthetic tests instead of requiring the real corpora for basic validation.

### Contract and Graph Tests

- `test_candidate_policy.py` verifies ranked start-policy behavior and helper APIs.
- `test_graph.py` verifies graph construction from `2wikimultihop` records and preserved link-context payloads.

### Selector and Walker Tests

- `test_selector.py` covers exploratory selector-family behavior, search variants, controller parsing, and budget handling.
- `test_walker.py` covers greedy walk behavior, cycle avoidance, and stop conditions.

### Evaluation and Experiment Tests

- `test_eval.py` verifies selector registries, budget adherence, GraphRAG proxy selection, and optional secondary end-to-end attachment.
- `test_experiments.py` verifies selector-by-budget outputs, grouped summaries, subset-aware CSV export, chunked runs, and merged summaries.
- `test_cli_experiments.py` verifies the Typer CLI on tiny samples.
- `test_selector_llm.py` verifies multi-backend selector configuration, controller actions, mocked token accounting, parsing, caching, and fallback behavior.

### Dataset and Storage Tests

- `test_twowiki.py`, `test_twowiki_store.py`, `test_dataset_fetch.py`, and `test_cli_datasets.py` cover 2Wiki loading, store preparation, lazy shard access, and sample dataset workflows.
- `test_store.py`, `test_raw_converters.py`, `test_cli_raw_datasets.py`, `test_musique.py`, `test_hotpotqa.py`, and `test_iirc.py` cover benchmark raw fetch/conversion, generic store preparation, and adapter behavior.

## Default Assumptions In Code

- Natural hyperlinks are the primary graph structure.
- The primary output is selected evidence under an explicit selector budget.
- End-to-end answering remains secondary to selector evaluation in the current repo.
