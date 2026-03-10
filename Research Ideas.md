# Core Research Direction: Semantic Link-Guided Corpus Selection Before RAG

## The Problem
There are two major blind spots in current retrieval pipelines over linked corpora:
1. **The GraphRAG Extraction Bottleneck**: Microsoft's GraphRAG requires "eager" information extraction over the entire corpus using large, expensive LLMs to build an entity-relation graph from scratch. This process is excessively slow and costly, especially since it completely ignores the existing, natural graph structure of documents.
2. **The Weakness of Dense Top-k as a Corpus Selector**: Dense retrieval gives a fast ranked list, but it treats documents as isolated items. It usually misses bridge documents and throws away the semantic value embedded in the hyperlinks themselves.
3. **The Neglect of Natural Semantic Links**: Many document corpora (like Wikipedia, internal wikis, or the broader web) are *already naturally connected* by hyperlinks. Traditional graph algorithms (like PageRank or Random Walk) use topology but ignore the rich semantics of links such as anchor text and surrounding sentence context.

## Proposed Solution: Semantic Link-Guided Corpus Selector
We propose using the **naturally existing document hyperlink graph**, enriched with local link semantics, as a query-time corpus selector before downstream RAG or GraphRAG.

1. **Semantic Link-Guided Selection (The Selector)**: Instead of a blind topological walk or a dense top-k over isolated documents, a lightweight selector operates directly on the natural hyperlink graph. At query time it scores links using anchor text and local sentence context, then expands a budgeted subgraph using path search and local diffusion.
2. **Lazy Subgraph Construction (The Filter)**: Only the selected documents are passed downstream. This avoids global eager extraction and yields a smaller, query-specific subgraph for later reasoning.
3. **Targeted RAG / GraphRAG (The Solver)**: A downstream answer stage then consumes only the selected corpus. The selector therefore becomes the true optimization target: how to maximize support recall and bridge recall under a fixed budget.

## Key Research Questions & Contributions

1. **Natural Links vs. Eager GraphRAG**: Can link-semantic corpus selection avoid GraphRAG's full-corpus indexing cost while preserving or improving downstream answer quality?
2. **Natural Links vs. Dense Top-k**: Can a query-time selector recover more support documents and bridge documents than dense top-k at the same or lower token budget?
3. **Controllability**: Does an explicit selector with hop, node, and token budgets provide a better recall-cost tradeoff than dense retrieval or eager global extraction?
4. **Selector Policy Design**: Which selection policy works best over semantic hyperlinks: beam search, A*-like best-first search, greedy best-first, uniform-cost search, PPR, or hybrid path-plus-diffusion variants?

## Proposed Implementation Plan in `webwalker`
- **Link Context Parsing**: Update the data ingestion pipeline (handling `2wikimultihop`/`musique`) to preserve not just the document text, but the outgoing hyperlinks and the semantic context (sentences) surrounding those links.
- **Anchor Selection (`candidate.policy`)**: Select initial starting documents using fast dense retrieval (`SelectByCosTopK`).
- **Corpus Selector Module (`webwalker.selector`)**: Implement selector policies that:
  - Read outgoing semantic links.
  - Expand a budgeted candidate corpus using link semantics rather than document-only similarity.
  - Emit a weighted subgraph for downstream RAG / GraphRAG.
- **Evaluation**: Compare selector policies against dense top-k and a local eager GraphRAG-style baseline using:
  - support document recall
  - bridge document recall
  - token cost
  - recall at fixed budget
  - downstream EM/F1 as a secondary check
