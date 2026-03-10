# Current Implementation

## Positioning

`webwalker` is currently a research prototype for validating a specific retrieval hypothesis:

`query -> anchor retrieval -> corpus selection over semantic hyperlinks -> lazy subgraph extraction -> answer synthesis -> evaluation`

The implementation is local, heuristic, and testable without network calls. The current center of gravity is no longer "build a walker," but "optimize the corpus selector that runs before downstream RAG or GraphRAG."

## What Is Implemented

### 1. Link-context graph

- `LinkContextGraph` stores document nodes plus semantic edge payloads.
- Each edge keeps `anchor_text`, `sentence`, `sent_idx`, and optional reference metadata.
- `LinkContextGraph.from_2wikimultihop_records(...)` adapts `2wikimultihop` paragraph records into a graph while preserving hyperlink context.
- The graph also implements `topk_similar(...)`, so it can serve as both graph storage and a lightweight dense-retrieval stand-in.

### 2. Stable start-policy contract

- `StartPolicy.select_start(...)` now returns `list[T]` consistently.
- `MaxPhiOverAnchors` returns a ranked node list instead of a single node.
- `SelectByCosTopK` now returns node ids only, not `(node, score)` tuples.
- Helper functions in `webwalker.candidate` expose both the ranked list and the single best start node.

### 3. Query-time corpus selector

- `selector.py` is now the primary query-time selection layer.
- Implemented selector families:
  - semantic beam search
  - semantic A*-like best-first search
  - semantic greedy best-first search
  - semantic uniform-cost search
  - semantic personalized PageRank
  - hybrid pathfinding + PPR variants
- Selection is budgeted by node count, hop count, and token count.
- The main scoring signal comes from link semantics only:
  - anchor overlap with query
  - surrounding sentence overlap with query
- The output is a weighted, query-specific subgraph candidate set rather than a single path.

### 4. Legacy walker baseline

- `DynamicWalker.walk(...)` remains in the repo as a greedy baseline.
- It is still useful as a cheap reference point, but it is no longer the main abstraction for research progress.

### 5. Lazy subgraph extraction

- `SubgraphExtractor.extract(...)` only reads from visited nodes.
- It emits:
  - scored evidence snippets from visited documents
  - scored relations from visited sources
- Unvisited targets are filtered unless the edge itself is query-relevant through anchor or target-title overlap.
- This keeps the extracted subgraph query-bounded rather than expanding back into eager global indexing.

### 6. Answer synthesis

- `Answerer.answer(...)` produces:
  - `answer`
  - `confidence`
  - evidence items
- The current implementation is heuristic:
  - relation targets are preferred as answer candidates
  - year extraction is used for `when`-style questions
  - capitalized phrase extraction is used for `who` / `where`-style questions
- This is enough to validate pipeline behavior and regression-test the research scaffold.

### 7. Evaluation orchestrator

- `Evaluator` now runs a selector-centric experiment matrix over the same graph and case:
  - `dense_rag`
  - `baseline_graphrag`
  - `webwalker` as a legacy greedy baseline
  - selector-family standalone pipelines
  - selector-family hybrid-with-PPR pipelines
- `baseline_graphrag` is intentionally a local eager stand-in, not Microsoft GraphRAG proper.
- Reported metrics:
  - support recall
  - support precision
  - selected nodes
  - selected links
  - tokens per recalled support document
  - coverage ratio
  - runtime
  - time-to-first-answer
  - token-cost estimate
  - visited steps
  - exact-match correctness when an expected answer is provided
- `CaseEvaluation.selection_report()` sorts results selector-first, prioritizing:
  - higher support recall
  - lower token cost
  - higher coverage ratio

### 8. Corpus storage

- `SQLiteKVStore` now supports `get(...)` for query-time document access.
- `build_from_tar_bz2(...)` remains the streaming ingest path for large Hotpot-style Wikipedia archives.

## What Is Still Not Implemented

- model-backed selector scoring and heuristics
- full GraphRAG integration as the answer backend
- training or RL-based selector policies
- real benchmark runners over HotpotQA / MuSiQue / 2Wiki question sets
- open-web crawling or online serving

## Concrete Test Design

The repo now uses fast synthetic tests instead of requiring the real corpora for basic validation.

### Contract tests

- `test_candidate_policy.py`
  - verifies ranked start-policy output
  - verifies helper API behavior
  - locks the contract that start selection returns node ids

### Graph adapter tests

- `test_graph.py`
  - verifies `2wikimultihop` records become semantic edges with preserved anchor text and sentence context
  - verifies induced subgraph behavior

### Walker tests

- `test_walker.py`
  - verifies the legacy greedy baseline still behaves as expected
  - verifies cycle avoidance
  - verifies stop conditions for dead ends and low-confidence edges

### Selector tests

- `test_selector.py`
  - verifies the weighted-subgraph contract
  - verifies budget presets
  - verifies beam search keeps bridge paths
  - verifies A* / GBFS / UCS produce distinct priority behavior
  - verifies semantic PPR prefers relevant branches over noisy hubs
  - verifies hybrid selection improves support recall in a synthetic case

### Subgraph and answer tests

- `test_subgraph.py`
  - verifies extraction only uses visited nodes
  - verifies answer generation prefers the relevant relation target for a `where` query

### Evaluation smoke test

- `test_eval.py`
  - verifies the full selector matrix runs
  - verifies metrics are present
  - verifies selection reporting is available and ordered around recall and cost

### Storage tests

- `test_store.py`
  - verifies streaming import from a synthetic `.tar.bz2`
  - verifies `SQLiteKVStore.get(...)` returns decoded JSON payloads

## Default Assumptions In Code

- The repo is an offline research sandbox.
- Natural hyperlinks are the primary graph structure.
- The corpus selector is the primary optimization target.
- The current answerer and baselines are heuristic placeholders designed to keep downstream validation cheap.
- The next implementation layer should swap heuristics for model-backed selector components without changing the graph, selector, subgraph, and evaluation interfaces.
