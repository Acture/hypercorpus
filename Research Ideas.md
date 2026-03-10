# Core Research Direction: Semantic Link-Guided Graph Walking & Subgraph Screening

## The Problem
There are two major blind spots in current knowledge-graph retrieval methods:
1. **The GraphRAG Extraction Bottleneck**: Microsoft's GraphRAG requires "eager" information extraction over the entire corpus using large, expensive LLMs to build an entity-relation graph from scratch. This process is excessively slow and costly, especially since it completely ignores the existing, natural graph structure of documents.
2. **The Neglect of Natural Semantic Links**: Many document corpora (like Wikipedia, internal wikis, or the broader web) are *already naturally connected* by hyperlinks. Traditional graph algorithms (like PageRank or Random Walk) utilize the topology of these links but ignore their rich semantics (e.g., anchor text, surrounding context). Conversely, GraphRAG extracts semantics but ignores the existing natural hyperlink topology, preferring to rebuild a graph from scratch.

## Proposed Solution: Semantic Link-Guided Walker + Small LLM Subgraph Screening
We propose leveraging the **naturally existing document hyperlink graph**, enriched with local semantic context, to perform "Lazy" Subgraph Screening. 

1. **Semantic Link-Guided Walking (The Navigator)**: Instead of a blind topological walk or rebuilding the graph, a highly optimized, lightweight mechanism (like a small SLM, RL agent, or context-aware GNN) "walks" the *natural document hyperlink graph*. At each node (document), the walker examines the outgoing hyperlinks and their **semantic context** (the anchor text and surrounding sentence) to decide the most promising path to resolve the query.
2. **Small LLM Subgraph Screening (The Filter)**: As the walker traverses these semantically-rich natural links, it uses a fast, small LLM (e.g., Llama-3-8B, Qwen-2.5-7B) to quickly screen and extract fine-grained entities/relations *only* for the documents visited on the path. This creates a highly relevant, query-specific "subgraph."
3. **Targeted GraphRAG (The Solver)**: A more capable LLM then applies GraphRAG-style reasoning (like community summarization or global path reasoning) strictly on this dynamically extracted, highly condensed subgraph to synthesize the final answer.

## Key Research Questions & Contributions

1. **Unlocking Natural Semantics vs. Eager Extraction**: By utilizing native document hyperlinks and their contextual semantics for navigation, can we bypass the need for GraphRAG's exorbitant full-corpus indexing phase while maintaining >90% of its multi-hop QA accuracy?
2. **Efficiency vs. Accuracy**: Does "Lazy" path-bound extraction by a 7B-class LLM provide a dense enough subgraph for targeted GraphRAG, compared to the global pass done by GPT-4-class models?
3. **Walker Policy Design (The "Web Walker")**: What is the optimal policy for evaluating semantic links? Does reinforcement learning (RL) guided by QA reward out-perform LLM zero-shot routing when deciding which hyperlinks to follow based on anchor text?

## Proposed Implementation Plan in `webwalker`
- **Link Context Parsing**: Update the data ingestion pipeline (handling `2wikimultihop`/`musique`) to preserve not just the document text, but the outgoing hyperlinks and the semantic context (sentences) surrounding those links.
- **Anchor Selection (`candidate.policy`)**: Select initial starting documents using fast dense retrieval (`SelectByCosTopK`).
- **Semantic Walker Module (`webwalker`)**: Implement a new `DynamicWalker` class that:
  - Reads the current document's outgoing semantic links.
  - Scores which web link to walk next based on the user's multi-hop query.
  - Passes the visited documents to a small, local LLM to extract the targeted subgraph.
- **Evaluation**: Compare `webwalker` against standard GraphRAG (which rebuilds the graph ignoring natural links) and traditional RAG in aspects of **Time-to-First-Answer**, **Total Token Cost**, and **Exact Match (EM) Score** on multi-hop benchmarks.
