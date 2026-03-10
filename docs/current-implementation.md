# Current Implementation

## Positioning

`webwalker` is currently a research prototype for validating a specific retrieval hypothesis:

`query -> anchor retrieval -> semantic graph walk -> lazy subgraph extraction -> answer synthesis -> evaluation`

The implementation is local, heuristic, and testable without network calls. It is meant to make the research loop concrete before introducing small LLMs or full GraphRAG orchestration.

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

### 3. Query-time walker

- `DynamicWalker.walk(...)` executes a greedy semantic walk over outgoing links.
- `WalkBudget` controls max steps, revisit policy, and minimum edge score.
- `WalkResult` records ordered steps, visited nodes, and an explicit stop reason.
- Current edge scoring is lexical and local:
  - anchor overlap with query
  - surrounding sentence overlap with query
  - target title overlap with query
  - novelty bonus for unseen nodes

### 4. Lazy subgraph extraction

- `SubgraphExtractor.extract(...)` only reads from visited nodes.
- It emits:
  - scored evidence snippets from visited documents
  - scored relations from visited sources
- Unvisited targets are filtered unless the edge itself is query-relevant through anchor or target-title overlap.
- This keeps the extracted subgraph query-bounded rather than expanding back into eager global indexing.

### 5. Answer synthesis

- `Answerer.answer(...)` produces:
  - `answer`
  - `confidence`
  - evidence items
- The current implementation is heuristic:
  - relation targets are preferred as answer candidates
  - year extraction is used for `when`-style questions
  - capitalized phrase extraction is used for `who` / `where`-style questions
- This is enough to validate pipeline behavior and regression-test the research scaffold.

### 6. Evaluation orchestrator

- `Evaluator` runs three local pipelines over the same graph and case:
  - `webwalker`
  - `dense_rag`
  - `baseline_graphrag`
- `baseline_graphrag` is intentionally a local stand-in, not Microsoft GraphRAG proper.
- Reported metrics:
  - runtime
  - time-to-first-answer
  - token-cost estimate
  - visited steps
  - exact-match correctness when an expected answer is provided

### 7. Corpus storage

- `SQLiteKVStore` now supports `get(...)` for query-time document access.
- `build_from_tar_bz2(...)` remains the streaming ingest path for large Hotpot-style Wikipedia archives.

## What Is Still Not Implemented

- small-LLM path-bound graph extraction
- full GraphRAG integration as the answer backend
- training or RL-based walker policies
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
  - verifies next-hop selection on a small semantic graph
  - verifies cycle avoidance
  - verifies stop conditions for dead ends and low-confidence edges

### Subgraph and answer tests

- `test_subgraph.py`
  - verifies extraction only uses visited nodes
  - verifies answer generation prefers the relevant relation target for a `where` query

### Evaluation smoke test

- `test_eval.py`
  - verifies all three pipelines run
  - verifies metrics are present
  - verifies the `webwalker` pipeline returns the expected answer on the synthetic graph

### Storage tests

- `test_store.py`
  - verifies streaming import from a synthetic `.tar.bz2`
  - verifies `SQLiteKVStore.get(...)` returns decoded JSON payloads

## Default Assumptions In Code

- The repo is an offline research sandbox.
- Natural hyperlinks are the primary graph structure.
- The current answerer and baselines are heuristic placeholders designed to preserve the architecture and keep tests cheap.
- The next implementation layer should swap heuristics for model-backed components without changing the graph, walk, subgraph, and evaluation interfaces.
