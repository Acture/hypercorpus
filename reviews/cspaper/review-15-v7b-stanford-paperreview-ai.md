# Stanford PaperReview.ai — "Follow the Links" v7b (with K variation)
- Source: https://paperreview.ai
- Venue simulated: ICLR
- Submitted: April 24, 2026
- Score: **5.4/10** (↑ from 5.2)

## Key changes from v7:
- Score UP 5.2 → 5.4
- New strength: "convincingly highlight budget discipline" for ratio analysis
- New strength: "backfill pool size is not the bottleneck" K variation acknowledged implicitly
- Weaknesses stable: still wants trained baselines, E2E QA, open-model controllers
- New suggestion: "token-aware walk" (refuse to traverse if exceeding budget) — interesting idea
- New related work mentions: BAVT, CatRAG, ToPG, REMINDRAG, AdaGReS

---

## Score: 5.4/10 (ICLR calibrated)

## Summary
The paper formulates budgeted subgraph selection on naturally linked corpora and proposes a zero-shot selector that starts from dense seeds, follows hyperlinks using link-local semantics (anchor text and sentence context), and enforces an explicit token budget with a quality-gated backfill step. The most effective variant uses an LLM controller to judge direct support, bridge potential, and redundancy risk at each hop, yielding higher document-level support F1 than flat dense retrieval on IIRC at matched fixed-token budgets and demonstrating "self-limiting" selection under ratio-controlled budgets.

## Strengths
- Introduces a selector-first formulation of budgeted subgraph selection that explicitly treats token budget as a hard constraint rather than a post-hoc truncation.
- Leverages a link-context graph that exploits anchor text and sentence context on existing hyperlinks, avoiding expensive corpus-wide graph construction.
- Designs a practical, modular pipeline: dense seed retrieval, hyperlink-local walking with edge lookahead without fetching targets, and a budget-fill step with a relative-drop quality gate.
- Proposes an LLM-guided controller that decomposes link utility into direct support, bridge potential, and redundancy risk, and decides when to continue, branch, or stop, encouraging compact, high-precision selections.
- Evaluates on two multi-hop QA corpora under both fixed-token and ratio-controlled budgets, which probe complementary aspects (selection quality vs budget discipline).
- Provides a decent set of ablations: edge-scorer profiles and lookahead, walk depth, walk structure (single-path vs beam/A*), and budget-fill threshold sensitivity.
- Reports precision/recall, budget utilization/adherence, and in one table includes bootstrap CIs for F1.
- Careful discussion of why ratio budgets expose different behavior than fixed-token budgets, along with qualitative "self-limiting" interpretation.
- The finding that quality-aware link-following can remain compact under generous budgets while flat dense retrieval "fills mechanically" is operationally meaningful.

## Weaknesses
- Heavy reliance on a capable closed-source LLM for the main reported gains; the non-LLM single-path variant underperforms dense retrieval on full-article IIRC, suggesting the method's advantage is contingent on expensive inference.
- The walk ignores token costs during traversal and enforces the budget with post-hoc trimming, potentially discarding later (useful) selections.
- Results are reported on fixed, small 100-case canonical samples without stratification.
- Missing direct comparisons to state-of-the-art trained multi-hop retrievers (e.g., MDR) or established graph-based systems (e.g., GraphRetriever, HippoRAG/CatRAG).
- End-to-end QA accuracy is not reported.
- Some experiments compare strategies under different seed/hop settings that may confound fairness.
- Tables include extraction artifacts and inconsistent numbers.
- Limited detail on the LLM controller's decision interface.
- Reproducibility details are incomplete: no code/prompt artifacts.
- No comparison to budget-aware post-retrieval selectors (e.g., submodularity-aware selection).

## Questions for Authors
1. How sensitive are results to sampling? Full dev split results?
2. Trained multi-hop baseline comparisons (MDR, GraphRetriever, HippoRAG/CatRAG)?
3. Mask anchor/sentence ablation + per-dimension controller ablation?
4. Token-aware walk (refuse to traverse if exceeding budget)?
5. Corrected Table 3 with exact metrics?
6. Full quantitative results for multiple open-weight controllers?
7. End-to-end QA (EM/F1) with a fixed reader?

## Overall Assessment
Estimated Score: 5.4/10. Promising but not yet ready for a top-tier venue. Main gains hinge on closed-source LLM with substantial runtime cost, evaluation limited to small samples, missing strong trained baselines, and reporting artifacts.
