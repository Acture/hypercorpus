# Corpus-Selection Literature Guide

This document is the paper-facing related-work guide for `webwalker`.

Treat the paper as a `linked-corpora discovery` paper first and as a `downstream QA support` paper second.

Use it to answer four writing questions:

- what problem are we actually claiming to solve
- which baselines are mandatory
- which systems are nearest comparators
- which rebuttal points should already be prepared

`docs/literature-map.tsv` is the canonical machine-readable bibliography.  
`docs/corpus-selection-literature.tsv` is a paper-oriented subset or export view and should not be treated as the source of truth.

## Narrative Order

### 1. Define The Problem As Subgraph Or Corpus Discovery, Not Answer Generation

Start from the multi-hop datasets and critiques:

- `HotpotQA`
- `2WikiMultiHopQA`
- `MuSiQue`
- `Compositional Questions Do Not Necessitate Multi-hop Reasoning`
- `IIRC`

The paper should frame the task as budgeted discovery of compact evidence for downstream reasoning rather than proposing a new answer generator.

### 2. Establish The Main Baseline Family

The core baseline story is still retrieval-centric:

- `dense top-k`
- `MDR`
- `Answering Any-hop Open-domain Questions with Iterative Document Reranking`
- `HopRetriever`

These are the first systems the paper needs to beat, match, or clearly complement.

### 3. Position Natural Hyperlink Navigation As The Distinctive Twist

Use adjacent navigation, focused-crawling work, and IR-native complex-answer-retrieval papers to justify why natural hyperlinks matter:

- `End-to-End Goal-Driven Web Navigation`
- `Focused crawling`
- `Topic-Sensitive PageRank`
- `Personalized PageRank`
- `Benchmark for Complex Answer Retrieval`
- `Characterizing Question Facets for Complex Answer Retrieval`
- `Local and Global Query Expansion for Hierarchical Complex Topics`
- `Why Does This Entity Matter? Support Passage Retrieval for Entity Retrieval`

These works support two points at once: query-conditioned movement on a linked graph is a real problem setting, and the underlying retrieval problem is also legible as complex-answer retrieval rather than only as multi-hop QA.

### 4. Use Graph-Retrieval Systems As Comparators, Not The Main Head-To-Head Yet

For the current paper stage, use graph retrieval systems to locate `webwalker` in the literature:

- `GraphRetriever`
- `HippoRAG`
- `KG2RAG`
- `PathRAG`
- `Walk&Retrieve`
- `Exploiting Relevance Feedback in Knowledge Graph Search`
- `Keyword Search over Knowledge Graphs via Static and Dynamic Hub Labelings`

These are important comparators, but the current evidence does not yet support claiming direct superiority over them. For a more algorithmic `KDD/general discovery` story, the last two papers are especially useful because they frame query-time graph discovery as a standalone search problem rather than a downstream QA helper.

### 5. Keep End-To-End GraphRAG Systems Secondary

Use these mostly to define the boundary of what `webwalker` is not:

- `GraphRAG`
- `RAPTOR`
- `LightRAG`

The current paper should not collapse back into an end-to-end GraphRAG comparison story.

## Baseline Priority

This priority order reflects the current `phase-decision-30` evidence.

### Tier 1: Must Compare Directly

- `dense top-k`
- `MDR`

These are mandatory because the current best result is still best understood as a stronger selector operating point, not yet as a graph-retrieval win over the broader literature.

### Tier 2: Near-Neighbour Retrieval Comparators

- `iterative document reranking`
- `HopRetriever`

These strengthen the claim that the contribution is about evidence selection rather than only about graph traversal.

### Tier 3: Narrative Comparators

- `GraphRetriever`
- `HippoRAG`

These are the closest systems conceptually, but not yet the first baselines to implement inside the current framework.

### Tier 4: Secondary System Comparators

- `GraphRAG`
- `LightRAG`
- `RAPTOR`

Use them to define system boundaries and motivate why `webwalker` focuses on pre-RAG corpus selection instead of full-stack retrieval-and-generation.

## Comparator Taxonomy

### Dense And Iterative Retrieval

- `MDR`
- `HopRetriever`
- `iterative reranking`

These systems answer: how far can you go without relying on natural hyperlink structure.

### Learned Path Retrieval

- `GraphRetriever`

This is the closest learned ancestor. It matters for positioning, but it is not a drop-in backend for the current selector architecture.

### Graph Retrieval And Subgraph Assembly

- `HippoRAG`
- `KG2RAG`
- `PathRAG`
- `Walk&Retrieve`

These systems are important when the paper discusses path or subgraph outputs rather than flat document sets.

### System-Level GraphRAG

- `GraphRAG`
- `RAPTOR`
- `LightRAG`

These are useful as downstream context, but they are not the main target of the current phase-decision evidence.

## Rebuttal-Ready Talking Points

### "Why Is This Not Just Another RAG System?"

Because the current output is a selected evidence set under a token budget, not a final answer. The contribution is in pre-RAG corpus selection.

### "Why Is The Main Baseline MDR Instead Of GraphRetriever?"

Because `MDR` is the strongest retrieval-centric baseline family for the claim being made now: better budgeted evidence selection than flat dense retrieval. `GraphRetriever` is a learned path retriever and belongs in the nearest-related-work discussion, but it is not the first direct baseline to wire into the current framework.

### "Why Not Claim LLM-Guided Graph Retrieval As The Novelty?"

Because the current experiments do not isolate that effect cleanly enough. The present evidence supports budget-aware selection and the operating-point story more strongly than a claim about LLM edge scoring itself.

### "Why Talk About Hyperlinks Instead Of Building A Graph First?"

Because the paper's distinctive angle is that natural hyperlinks already provide usable local structure and semantics at query time. That is different from papers whose main contribution is eager graph construction or learned graph memory.

### "What Is The Cleanest New Component?"

`budget_fill_relative_drop` is the strongest current candidate because it is distinctive and already shows a clear benefit in the completed phase-decision results.
