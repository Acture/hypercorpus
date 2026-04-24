# CSPaper Review — "Follow the Links" v7 (SIGIR, GPT-5)
- Source: cspaper.org
- Venue simulated: SIGIR
- Reviewed: April 24, 2026
- Score: **2 Weak (or Lean) Reject**
- Confidence: 4 High
- Significance: **3 Fair** (stable from v6)
- Does it have a point: **4 Good** (stable)

## Key changes from v6:
- Now sees 100-case data with F1=0.503 and bootstrap CIs
- Acknowledges CIs in Table 2 as a strength
- Acknowledges H/ρ_fill/θ ablations (Tables 8, 9) as strengths
- Still 2/Weak Reject — same blockers: table rendering, trained baselines, leakage ablation
- New: explicitly asks for mask-anchor + context-shuffle + knapsack trimming ablations
- New related work: G-Retriever, Optimal Subgraph for RAG, BubbleRAG, Deng 2022 (hyperlink subgraph retrieval)
- New: "Section 5.3 contains corrupted text" — rendering issue

---

## Desk Rejection Assessment:

### Paper Length
Pass ✅.

### Topic Compatibility
Pass ✅. The paper targets retrieval under token budgets, hyperlink-aware graph traversal, and multi-hop QA evaluation, which are core SIGIR topics (search/retrieval models, RAG, evaluation).

### Minimum Quality
Pass ✅. The paper contains Abstract, Introduction, Method (Sections 2–3), Experiments and Results (Sections 4–6), Related Work (Section 7), Limitations (Section 8), and Conclusion (Section 9). However, there are notable clarity and methodological issues discussed in the review.

### Prompt Injection and Hidden Manipulation Detection
Pass ✅. I did not detect hidden prompts or manipulative instructions in the provided content.

## Expected Review Outcome:

### Paper Summary
The paper formulates budgeted subgraph selection over naturally linked corpora and proposes a zero-shot three-stage selector: dense seed retrieval, hyperlink-local link-context walking, and budget-aware fill with a relative-drop gate. On IIRC (61k full Wikipedia articles) and 2WikiMultiHopQA, the controller-guided walk reportedly improves document-level support F1 at matched token budgets, while exhibiting "self-limiting" behavior under ratio-controlled budgets.

### Relevance to SIGIR Perspectives
4 Good relevance: Clearly relevant to SIGIR, focusing on retrieval and budget-aware evidence selection in linked corpora.

### Appropriateness
4 Good: Fits the technical full-paper expectations for IR; the framing is selector-first and empirical. It is not a perspective piece, but it aligns with SIGIR's core tracks.

### Perceived significance or potential impact of the perspective contribution
3 Fair: The idea of hyperlink-local selection under explicit token budgets is interesting, but the empirical evidence is limited by small samples, missing strong baselines, and several implementation ambiguities.

### Quality of perspective presentation
3 Fair: The paper is generally understandable, but there are clarity/notation issues, table formatting errors, and some inconsistencies that hinder precise interpretation.

### Does it have a point?
4 Good: The central thesis, "exploit hyperlink-local semantics to self-limit evidence selection under token budgets," is clear and argued throughout with controlled components.

### Context
3 Some awareness but misses key references.: The related work cites several adjacent threads, but it omits multiple directly related and recent graph-based subgraph retrieval/RAG papers and hyperlink-based retrieval works.

### Strengths
- Clear and timely problem framing. Casting "budgeted subgraph selection" as a selector-first objective with explicit token accounting is useful for disentangling retrieval quality from downstream generation.
- Methodical three-stage pipeline with a consistent budget accounting mechanism. Figure 1 succinctly lays out the components and budget constraint Στ(v) ≤ ρT; the hard constraint and relative-drop gate are conceptually straightforward.
- Exploiting link-local signals (anchor text and containing sentence) before fetching targets is reasonable and often underused. Algorithm 1 operationalizes a simple walk policy with pluggable link scorers.
- Evidence that a quality-aware controller can resist over-filling under generous ratio budgets, contrasting with flat dense similarity that tends to admit a long tail. Figure 4 shows the controller's F1 staying flat across 256–512 token budgets, matching the "self-limiting" claim.
- Informative ablations on the calibration set: Table 5 demonstrates lookahead matters for edge scoring (LA=2 best), and Table 8 shows H=2 as the sweet spot. Table 9 confirms the backfill gate trades precision vs recall without inverting method ranking.
- Figure 2's precision–recall scatter makes a concrete point that broader graph exploration (beam/A*) increases recall at the expense of severe precision collapse at tight budgets.
- Qualitative walk trace (Figure 3) helps ground the mechanism.

### Weaknesses
1. Limited and fragile empirical basis on IIRC — 100-case canonical sample, only Table 2 has CIs.
2. Missing strong baselines — no trained MDR, GraphRetriever, or recent graph-RAG/subgraph selection.
3. Notation and algorithmic clarity issues — Eq(3) novelty term, Algorithm 1 variable naming.
4. Table and text formatting errors — Table 3 garbled, Section 5.3 corrupted text.
5. F1∅ coupled with budget fill — needs systematic with/without fill for all selectors.
6. Incomplete reproducibility details — tokenizer, embedding truncation, controller prompt/params.
7. Closed LLM reliance without leakage ablation — mask-anchor experiment needed.
8. K=64 cap confounds ratio-budget analysis.
9. Runtime cost not quantified as Pareto frontier.
10. Graph construction details underspecified.
11. Walk structure comparison uses different configs.
12. Trimming heuristic could discard high-value seeds.

### Potentially Missing Related Work
1. Li et al. (2024) G-Retriever: Graph-Based Evidence Retrieval for Multi-Hop QA
2. Chen et al. (2025) Optimal Informative Subgraph Retrieval for Graph-based RAG
3. Zhang et al. (2025) BubbleRAG: Evidence-Driven RAG for Black-Box KGs
4. Yu et al. (2024) Graph-based RAG for Long-Context QA
5. Wang et al. (2023) Efficient K-Hop Subgraph Retrieval for KG QA
6. Xiong et al. (2023) GraphRAG: Leveraging KGs for RAG
7. Deng et al. (2022) Subgraph Retrieval for Multi-Hop QA over Wikipedia Hyperlinks
8. Zhou et al. (2021) Improving Dense Passage Retrieval with Document Graphs

### Your final vote for this paper
2 Weak (or Lean) Reject

### Reviewer's Confidence
4 High: Solid expertise, strong confidence.

### Confidential remarks for the program committee
N/A.
