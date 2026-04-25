# Review 22 — CSPaper (v17, hyper_17.pdf)

- **Date**: 2026-04-25
- **Venue target**: SIGIR
- **Score**: 4/5 (Weak Accept) — UP from 2/5!
- **Confidence**: 4 High
- **Paper version**: hyper_17.pdf

## Score trajectory

| # | Version | Reviewer | Score |
|---|---------|----------|-------|
| 16 | v7b | CSPaper | 2/5 |
| 17 | v11 | CSPaper | 2/5 |
| 20 | v12 | CSPaper | 2/5 |
| 22 | v17 | CSPaper | **4/5** |

## Desk Rejection: PASS ✅

## Sub-scores

| Dimension | Score |
|-----------|-------|
| Relevance to SIGIR | 5 Excellent |
| Appropriateness | 4 Good |
| Significance | 4 Good |
| Presentation quality | 3 Fair |
| Has a point? | 4 Good |
| Context/related work | 3 Some awareness, misses key refs |

## Strengths (8 items)

1. Clear selector-first formulation with F1∅ metric
2. Link-context graph abstraction — practical, no offline graph construction
3. Controller-guided walk with ranked-choice (direct-support, bridge-potential, redundancy-risk)
4. Strong IIRC signal: F1=0.503 vs dense 0.337; controller on higher iso-F1 contour
5. Careful budget analysis: dense precision collapse vs controller self-limiting
6. Walk structure + edge-scorer + depth ablations illuminate design choices
7. Qualitative trace example (Figure 4)
8. Practical discussion: cost/runtime, caching, preliminary downstream QA

## Weaknesses (5 clusters)

### W1: Comparator coverage
- No trained multi-hop retrievers (MDR, GraphRetriever, HippoRAG, QA-GNN)
- No stronger dense baselines (DPR/Contriever + cross-encoder reranking) in main table

### W2: Evaluation scope
- 100-case subsets not full dev sets
- 2Wiki N=88 vs 100 mismatch
- Some ablations on 20-30 cases; CIs only in Table 2

### W3: Inconsistencies/clarity
- Table 2 "Nodes=1.000" vs text "≈1.8 nodes" — MUST reconcile
- "LLM/rok" column header unclear
- "st_future_bassy" typo in selector description
- F10 vs F1θ notation varies

### W4: Controller dependence/reproducibility
- Main wins require closed LLM
- Exact prompts, decision policies, filtering heuristics not specified

### W5: Generalization
- Wikipedia-only evaluation
- No non-Wikipedia test

## 10 Missing Related Work Citations
1. BubbleRAG (Li et al. 2025) — KG subgraph retrieval under constraints
2. BubbleRAG (Zhao et al. 2025) — optimal informative subgraph for QA
3. HyperGraphRAG (Hu et al. 2025) — entity-centric subgraph selection
4. Diffusive Hypergraph Retrieval (An et al. 2024) — budget-aware biased walks
5. DynaGRAG (Thakrar et al. 2024) — dynamic graph retrieval for RAG
6. AtlasKV (Wang et al. 2025) — KG integration with budget-aware pruning
7. BridgeRAG (Chen et al. 2026) — training-free bridge-aware scoring
8. Query-Driven Graph Retrieval (Liu et al. 2026) — adaptive pruning for multi-hop
9. Retrieving Relevant KG Subgraphs (Huang et al. 2025) — subgraph retrieval for dialogue
10. G-Refer (Gupta et al. 2024) — hybrid dense+graph retrieval

## 14 Detailed Comments
1. Add trained MDR or DPR/Contriever + cross-encoder reranker
2. Re-run 2Wiki on same 88-case subset for apples-to-apples
3. k-sweep for seed count sensitivity
4. Dense without fill + dense+fill with K varied in main table
5. ρ_fill sensitivity on IIRC ratio budgets
6. Reconcile Nodes=1.000 vs ≈1.8 discrepancy
7. Full controller prompt + JSON schema + prefilter specs
8. Per-hop error decomposition by model
9. Per-case node counts alongside F1
10. Coverage@tokens metric for >2 gold pages
11. Non-Wikipedia corpus experiment
12. Link filtering impact on non-LLM walk
13. Token budget + seeds/hops in Figure 2 legend
14. Clean typos/notation

## Final Verdict
4 Weak Accept — "Strong idea with clear IR relevance and promising empirical gains under tight token budgets; presentation has inconsistencies, and the comparator set omits trained multi-hop/graph baselines. With clarifications and stronger baselines, this could be a solid accept."

## Key Delta from Previous Reviews
- **Score jumped 2/5 → 4/5** — first time breaking out of the 2/5 band
- Strengths now recognized: 8 explicit strengths vs mostly concerns before
- "Promising empirical gains" language — significant tone shift
- Remaining concerns are concrete and addressable (vs previously structural/fundamental)
- Still wants: MDR, reconcile nodes discrepancy, exact prompts, non-Wikipedia test
