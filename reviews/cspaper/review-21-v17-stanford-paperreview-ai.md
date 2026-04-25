# Review 21 — Stanford PaperReview.ai (v17, hyper_17.pdf)

- **Date**: 2026-04-25
- **Venue target**: ICLR
- **Score**: 6.1 / 10
- **Paper version**: hyper_17.pdf (4-layer evidence integration)

## Score trajectory

| # | Version | Reviewer | Score |
|---|---------|----------|-------|
| 18 | v11 | Stanford | 5.4 |
| 19 | v12 | Stanford | 5.8 |
| 21 | v17 | Stanford | 6.1 |

## Summary

This paper formulates budgeted subgraph selection over naturally linked corpora and proposes a zero-shot, hyperlink-aware selector that starts from dense seeds and performs a link-context walk guided by either lightweight edge scoring or an LLM controller. The core contribution is a self-limiting controller that reasons over anchor/sentence contexts and explicitly decides to continue, branch, or stop, yielding compact selections that respect hard token budgets and avoid the precision collapse observed in dense backfill. On IIRC with full-article context (61k pages), the controller achieves notable gains in support F1 (0.503 vs. 0.337 for a strengthened dense baseline) while selecting fewer than two documents per query, and exhibits budget independence under ratio-controlled budgets.

## Strengths

### Technical novelty and innovation
- Clear "selector-first" formulation with explicit ratio-based token budget and link-context graph
- Self-limiting, LLM-guided walk policy decomposing link utility into direct-support, bridge-potential, redundancy-risk
- Fill treated as baseline strengthener rather than method crutch — surfaces budget discipline
- Zero-shot traversal over existing hyperlinks without pre-building corpus-wide KGs

### Experimental rigor and validation
- Two multi-hop QA surfaces (2Wiki calibration; IIRC main) with careful budget regimes
- Per-case scatter, bootstrap CIs, ablations on edge scoring, walk depth, fill thresholds, controller model sensitivity, error taxonomy, budget utilization diagnostics
- Defensively strengthened dense baseline with quality-gated backfill

### Clarity of presentation
- Clear problem statement, graph formalization, budget accounting
- "Selector-first" lens and F1∅ metric well-motivated and consistently applied
- Visualizations and tables effectively communicate tradeoffs

### Significance
- Meaningful practical gap in RAG/multi-hop retrieval
- Self-limiting behavior has clear downstream cost/quality implications
- Preliminary E2E results indicating nearly oracle-level reader performance

## Weaknesses

### Technical limitations
1. **Seed dependence**: k=1 seeds throughout; error analysis shows most losses are upstream seed failures; no multi-seed ablation
2. **Proprietary LLM dependence**: GPT-5/GPT-4.1 controller; open-weight alternatives underperform or run slower
3. **Wikipedia-only corpora**: portability claim not empirically validated on other domains

### Experimental gaps
4. **Missing trained baselines**: No MDR, HopRAG, SPRIG, SAGE comparisons
5. **Small N for some analyses**: 20–40 cases; 2Wiki sample size mismatch (N=88 vs 100)
6. **Limited ρ_fill sweep on IIRC**: Only characterized on 2Wiki; broader threshold sweep on IIRC needed
7. **Controller prompt/prefilter details**: Only briefly described; replication difficult

### Presentation
8. **Pilot vs main-sample mixing**: Intermixes pilot and main results across tables/footnotes

### Missing comparisons
9. **No supervised multi-hop retrievers**: MDR, HopRetriever
10. **No graph-centric RAGs**: ToPG, SAGE, A2RAG, HopRAG
11. **Dense baseline landscape**: Contriever, SOTA embeddings under fixed-token with tuned fill thresholds

## Questions for Authors

1. **Seed dependence**: Why fix k=1? Ablations with k∈{2,3,5}?
2. **Prefilter sensitivity**: How sensitive to two-stage prefilter (lexical then ST top-5)?
3. **Stronger baselines**: MDR + at least one graph/RAG baseline under same fixed-token budget?
4. **ρ_fill sweep on IIRC**: Same 2Wiki sweep on IIRC?
5. **Controller reproducibility**: Release prompts, prefilter code, cached traces? Full open-weight results?
6. **Gold mapping**: How many cases dropped due to title mismatches? Mismatch-induced bias?
7. **Generalization**: Small-scale non-Wikipedia experiment?
8. **Multi-seed + deeper horizon**: Can controller aggregate across multiple seeds with H=3?

## Overall Assessment

> "I find the conceptual clarity and empirical signal compelling enough for ICLR, provided the authors address the most pressing experimental comparisons (stronger baselines, multi-seed sensitivity, threshold sweeps on IIRC) and bolster reproducibility details. I recommend a weak accept."

**Verdict**: Weak accept. Key blockers: stronger baselines, multi-seed ablation, IIRC ρ_fill sweep, reproducibility details.

## Delta from v12 → v17

New positives recognized:
- Controller model sensitivity analysis (GPT-4.1 > GPT-5, Llama-70B marginal)
- Mask ablation with real results (−0.027 F1)
- E2E QA downstream results
- K-variation data
- Cross-sample robustness
- Fill strategy comparison (neighbor/diverse/relative-drop)
- Contriever and dense-rerank encoder-agnostic evidence

Persistent concerns:
- Missing trained baselines (MDR, HopRAG) — still #1 blocker
- Seed dependence (k=1) — now more salient with error taxonomy
- Wikipedia-only evaluation
- Small N for some ablations
