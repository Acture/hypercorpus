# Hypercorpus Related Work Outline

## 1. Benchmarks And Evaluation Surfaces
- `HotpotQA`
- `IIRC`
- `MuSiQue`
- Positioning point:
  - these datasets motivate multi-hop evidence discovery, but the paper stays selector-first rather than answerer-first

## 2. Reasoning-Path Retrieval And Multi-Hop Evidence Retrieval
- `GraphRetriever`
- `MDR`
- `DecompRC`
- `IRCoT`
- Positioning point:
  - these methods retrieve or assemble multi-hop evidence paths, but they differ in whether they rely on learned iterative retrieval, explicit decomposition, or stronger reasoning-time control

## 3. Graph Retrieval And GraphRAG-Adjacent Systems
- `HippoRAG`
- `RAPTOR`
- other graph or structured retrieval systems cited in the existing literature map
- Positioning point:
  - the paper does not compete as a full memory-RAG replacement
  - the contrast is query-time evidence discovery over naturally linked corpora versus eager global structure construction

## 4. Hyperlink Navigation And Adjacent Web-Style Navigation
- goal-driven web navigation work
- hyperlink-aware local navigation ideas
- Positioning point:
  - Hypercorpus uses naturally linked corpora and local link semantics, but is evaluated as retrieval and evidence assembly rather than as a general navigation agent

## 5. What To Emphasize In The Narrative
- Treat dense retrieval, iterative retrieval, graph retrieval, and LLM-guided retrieval as existing design spaces.
- Emphasize the project’s distinctive combination:
  - dense-started evidence assembly
  - natural hyperlink locality
  - explicit token-budget control
  - selector-first evaluation

## 6. What Not To Overclaim
- Do not write as if the paper already beats all graph-based or learned retrievers.
- Do not describe `mdr_light` as direct `MDR`.
- Do not let related work pull the paper into a full QA-system comparison frame.
