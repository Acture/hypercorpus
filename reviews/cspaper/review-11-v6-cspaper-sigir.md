# CSPaper Review — "Follow the Links" v6 (SIGIR, GPT-5)
- Source: cspaper.org
- Venue simulated: SIGIR
- Reviewed: April 23, 2026
- Score: **2 Weak (or Lean) Reject**
- Confidence: 4 High
- Significance: **3 Fair** (upgraded from 2 Poor in earlier rounds)
- Does it have a point: **4 Good** (stable)

## Key changes from previous rounds:
- Significance upgraded 2→3: "selector-first framing and self-limiting observation are interesting"
- New strengths acknowledged: bootstrap CIs in Table 2, sensitivity tables (H, ρ_fill, θ)
- New weaknesses: Table 3 "corrupted/misaligned", Table 6 missing model rows cited in text
- Still requesting: trained MDR baseline, larger eval, end-to-end QA, dense similarity decay plots
- New related work: Xiong/Chen 2018 (progressive multi-hop RL), Dhingra 2020, Min 2019 (doc graphs), Nie 2021, Zhu 2022

## Score: 2 (unchanged) — but significance improved and ablations acknowledged

---

## Desk Rejection Assessment:

### Paper Length
Pass ✅.

### Topic Compatibility
Pass ✅. The paper targets budget-aware retrieval and graph-based evidence selection for multi-hop QA over Wikipedia, which falls squarely within SIGIR's core areas of search/retrieval models, ML for IR, and RAG-style retrieval.

### Minimum Quality
Pass ✅. The submission includes Abstract, Introduction, Problem Formulation/Method (Sections 2–3), Experiments and Results (Sections 4–6), Related Work (Section 7), Limitations (Section 8), and Conclusion (Section 9). However, there are issues with evaluation completeness and clarity that affect the final recommendation.

### Prompt Injection and Hidden Manipulation Detection
Pass ✅. I found no signs of prompt injections, hidden reviewer-targeting instructions, or manipulation.

## Expected Review Outcome:

### Paper Summary
The paper studies "budgeted subgraph selection" for evidence retrieval over naturally linked corpora and proposes a zero-shot, hyperlink-aware selector that starts from dense seeds, walks using link-context scoring, and then backfills under a token budget with a relative-drop quality gate. On IIRC (61K full-context Wikipedia articles), the LLM-controller variant claims higher support-F1 at matched token budgets than flat dense retrieval and shows "self-limiting" behavior under ratio budgets.

### Relevance to SIGIR Perspectives
4 Good relevance: Clearly relevant to SIGIR, though more narrowly scoped or focused on a subarea. Justification: The paper addresses IR/RAG retrieval under budgets over hyperlink graphs, a timely IR topic. The scope is focused on multi-hop QA retrieval over Wikipedia.

### Appropriateness
3 Fair: Partially appropriate; some elements fit but others feel misaligned with the intended goals of the track. Justification: This reads as a methodological empirical paper rather than a perspectives-style synthesis. Still, method and evaluation are aligned with SIGIR's full paper track expectations.

### Perceived significance or potential impact of the perspective contribution
3 Fair: Some value, but the contribution is incremental or the impact is uncertain. Justification: The selector-first framing and the "self-limiting" observation are interesting. Impact is limited by evaluation on small 100-case slices, missing trained multi-hop baselines, and reliance on a closed LLM controller.

### Quality of perspective presentation
3 Fair: Understandable overall but has structural or clarity issues. Justification: The core idea and pipeline are clear, but there are equation/notation mistakes, corrupted tables, and some inconsistencies that impede clarity.

### Does it have a point?
4 Good: The main point is visible, though occasionally diluted or not consistently emphasized. Justification: The central message—hyperlink-local navigation plus quality-aware stopping outperforms flat dense retrieval at matched budgets and avoids precision collapse under ratio budgets—is consistent.

### Context
3 Some awareness but misses key references. Justification: The paper surveys dense and graph-based RAG work, but misses several directly relevant baselines on graph navigation and budgeted/adaptive evidence selection.

### Strengths
- Problem framing: Treating "budgeted subgraph selection" as a selector-first objective under explicit token budgets (Section 2) is a useful lens for comparing retrieval strategies independent of readers.
- Zero-shot over link-context graphs: The method exploits natural hyperlink structure without pre-constructing a KG and uses anchor text and sentence context to guide traversal (Sections 3.2–3.5). This is practical when hyperlinks already exist.
- Explicit budget accounting: Ratio-controlled budgets and a simple but effective quality gate for backfill (Equation 6) make budget use observable and comparable across selectors (Section 3.6).
- Evidence that link-following recovers bridges: On 2WikiMultiHopQA, the single-path non-LLM walk improves support recall over dense by 0.128 at a 256-token budget (Table 4), aligning with the hypothesis that bridges are accessible via hyperlinks but not by query-document similarity alone.
- Budget-discipline analysis: The ratio-budget analysis shows stark differences in budget usage. The controller returns ~1.8 nodes regardless of available headroom while dense fills to K=64 with very low precision (Table 3, Section 5.2).
- Clear pipeline visualization: Figure 1 nicely summarizes stages and budget enforcement in a single diagram.
- Qualitative trace: Figure 3 provides a concrete example where the controller follows anchors/contexts to reach gold support, which helps ground the approach.
- Figure 2: The precision-recall scatter demonstrates why unconstrained beam/A* collapse precision at this budget; it visually reinforces the "constrained quality-aware walk" design choice.
- Table 2: Main IIRC result at 512 tokens, with bootstrap CIs, shows a sizable gap between the controller (.474) and dense (.337), supporting the headline claim under matched budgets.
- Table 5: Edge-scorer ablation demonstrates the value of lookahead and the st_future_heavy profile, supporting the design of the non-LLM walk module.

### Weaknesses
1. Narrow and potentially fragile evaluation scope — Only 100-case "canonical" slices are used on both IIRC and 2Wiki (Section 4.1, Sampling protocol). This is small relative to standard IR evaluation practice, making results sensitive to sample composition. No stratification by hop type/difficulty and no large-scale test are provided.

2. Missing strong trained baselines and limited comparisons — The main external comparators are a flat dense retriever and two "repo-native" iterative dense baselines (Section 3.4, 4.4). Absent are strong, trained multi-hop retrievers like MDR with learned reformulation, PathRetriever variants, or graph-policy learners.

3. Heavy dependence on a closed LLM controller without sufficient ablations — Table 6 only shows GPT-5 vs Dense, omitting the other models cited in text. This inconsistency prevents assessing robustness to controller choice and cost-quality tradeoffs.

4. Clarity and correctness issues in equations, algorithm, and tables — Equation (3) has unclear notation; Algorithm 1 uses inconsistent variable names; Table 3 appears corrupted with misaligned headers.

5. Potential confounds in walk-structure comparison — Figure 2 and Table 7 compare strategies under different seeds/hops configurations.

6. Limited generality beyond Wikipedia — No tests outside Wikipedia or on sparse-link domains.

7. Budget accounting and selection policy interactions — Trimming policy ("last added, first removed") may prune relevant additions.

8. Evaluation metrics and reporting gaps — No end-to-end QA accuracy; CIs only in Table 2; controller runtime/token sensitivity missing.

9. Reproducibility and implementation details — No code or prompts referenced.

10. Minor formatting/typo issues — Stray symbols, inconsistent naming.

### Potentially Missing Related Work
1. Xiong, Chen, Li (2018): Progressive Multi-hop Reasoning over Documents with RL
2. Dhingra, Yang, Cohen (2020): Differentiable Reasoning over a KG for Multi-hop QA
3. Min, Chen, Hajishirzi (2019): Multi-hop QA through Multi-scale Document-Level Graphs
4. Nie, Chen, Bansal (2021): Global Optimization of Retrieval for Long Document QA
5. Zhu, Yu, Jiang (2022): Adaptive Evidence Selection for Open-Domain QA

### Your final vote for this paper
2 Weak (or Lean) Reject

Rationale: The idea of budgeted hyperlink-local selection and the "self-limiting" lens are interesting and relevant, and several analyses are thoughtful. However, the empirical foundation is too narrow (100-case slices), strong trained baselines are missing, critical tables are corrupted or incomplete, the controller ablation is insufficient, and clarity issues in equations/algorithm hinder reproducibility. With a more rigorous, larger-scale evaluation and stronger baselines, this could become competitive.

### Reviewer's Confidence
4 High: Solid expertise, strong confidence.

### Confidential remarks for the program committee
N/A
