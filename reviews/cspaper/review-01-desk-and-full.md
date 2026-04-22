# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper targets retrieval over hyperlinked corpora, budgeted selection, and multi-hop QA evaluation, which fall squarely within SIGIR's core IR, NLP-for-IR, and RAG topics.

## Minimum Quality
Pass ✅. The submission contains Abstract, Introduction (Section 1), Problem Formulation (Section 2), Method (Section 3), Experimental Setup (Section 4), Main Results (Section 5), Analysis (Section 6), Related Work (Section 7), Limitations (Section 8), and Conclusion (Section 9). However, the empirical section is preliminary and limited in scope.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. No signs of hidden prompts, steganography, or manipulative instructions.

# Expected Review Outcome:

## Paper Summary
The paper formulates budgeted subgraph selection over naturally linked corpora, proposing a zero-shot "dense-started" selector that walks Wikipedia hyperlinks using anchor and sentence-level link context, followed by a budget-aware fill mechanism. An LLM-controlled multi-path walking variant is reported to improve support F1 over flat dense retrieval on a small IIRC pilot (N=20) and shows a recall gain on 2WikiMultiHopQA under fixed budgets.

## Relevance to SIGIR Perspectives
4 Good relevance: Clearly relevant to SIGIR, though more narrowly scoped or focused on a subarea.  
The topic addresses retrieval and RAG over hyperlinked corpora with explicit cost constraints, relevant to IR and evaluation.

## Appropriateness
3 Fair: Partially appropriate; some elements fit but others feel misaligned with the intended goals of the track.  
The paper reads as a technical empirical study rather than a perspectives-style piece; nevertheless it is method- and evaluation-focused and within SIGIR's full paper remit.

## Perceived significance or potential impact of the perspective contribution
2 Poor: Limited significance; offers little new beyond existing discussions.  
The formulation and hyperlink-walk idea are interesting, but the empirical evidence is too preliminary (N=20 on IIRC) and missing key baselines to substantiate impact.

## Quality of perspective presentation
3 Fair: Understandable overall but has structural or clarity issues.  
The write-up is largely clear, but several inconsistencies (notation, missing figure, metric naming) and insufficient experimental detail reduce clarity.

## Does it have a point?
3 Fair: The paper has a point, but it is vague, muddled, or obscured by details.  
The central claim is that hyperlink-local walking under a token budget can recover bridge evidence that flat dense misses. The claim is plausible but weakened by small-scale, uneven evaluations and budget-setting asymmetries.

## Context
2 Weak situating.: Little engagement with prior work.  
Several important, closely related works are missing, including strong trained multi-hop retrievers, DPR-style dense baselines, and recent graph-augmented RAG methods.

## Strengths
- Clear problem framing: formulates "budgeted subgraph selection" with an explicit ratio-controlled token budget and a selector-first evaluation using support F1 with zero-on-empty (Section 2.2–2.3).
- Sensible exploitation of hyperlink-local signals: uses anchor text and sentence context to score outgoing edges without fetching targets (Section 3.5), which is pragmatic for linked corpora.
- Method modularity: consistent budget accounting across selectors, with dense seeding and comparable walk strategies, including Algorithm 1 for the single-path walk (Section 3.4).
- Budget fill with relative drop: an explicit mechanism to improve budget utilization while guarding against low-quality filler (Section 3.6). Conceptually useful for selector-first settings.
- Empirical indication of bridge recovery: On 2WikiMultiHopQA (Table 3), the single-path walk achieves higher recall than dense under fixed budgets; on IIRC, the controller variant shows a preliminary F1 gain over matched-node-count dense retrieval (Table 2).
- Explicit cost discussion: reports runtime and node counts and comments on trade-offs, acknowledging LLM-controller overhead (Table 2 and Section 5.1).
- Equations and algorithmic clarity: Edge-scoring formulations (Equations 3–5) and Algorithm 1 specify decision rules, providing a concrete basis for walk behavior.

Specific figure/table references:
- Table 2: shows preliminary IIRC results where the LLM-controller achieves F1θ = 0.471 with 1.75 nodes, contrasted with dense regimes including a ratio-controlled setup with many nodes and very low precision. This table centralizes the paper's claim.
- Table 3: calibration on 2WikiMultiHopQA where "Single-path walk" improves recall to 0.730 vs. dense at 0.640 with F1θ rising from 0.396 to 0.414, supporting the "bridge evidence" narrative.
- Algorithm 1: clearly presents the single-path walk, which helps assess what information influences a hop decision.
- Equations (3)–(5): articulate the scoring logic that leverages anchor and sentence context and optional lookahead; these back the claim that link-local signals can guide traversal.

## Weaknesses
1) Preliminary and underpowered evaluation on IIRC undermines the main claim.  
- Table 2 explicitly marks the IIRC ratio-controlled results as preliminary with N=20. This sample is too small to support strong claims, especially given the high variance of multi-hop retrieval tasks. The core result, a +0.063 F1 gain over "dense at tokens-512" and a dramatic contrast versus ratio-controlled dense, needs confirmation on the full 100-case set the authors state is "forthcoming" (Sections 5.1 and 5.3). Without it, conclusions are tentative.

2) Budget-setting asymmetries obfuscate the comparison.  
- In Table 2, the "Dense" row under ratio budgets selects 64 documents, while the "Controller" row returns 1.75 nodes, essentially comparing a budget-filling dense regime to a self-limiting controller that stops early. The paper then compares the controller to "dense at matched node count" using fixed tokens (512) rather than the same ratio regime, conflating two distinct budget definitions. This creates ambiguity about whether gains are due to better selection or simply stricter stopping. A fair comparison would hold the budget regime constant and align stopping criteria or enforce the same max node/tokens across methods.

3) Missing or weak baselines reduce credibility.  
- The paper compares primarily to flat dense and "repo-native" iterative dense ("mdr_light") without including strong trained multi-hop retrievers (e.g., MDR with trained encoders) or widely used DPR/DrQA-style baselines. Given Section 7 acknowledges these systems, their omission is a serious gap. The claims that natural-hyperlink walks outperform dense-only iterative methods need validation against state-of-the-art trained retrievers and graph-based methods.

4) Inconsistent metrics and notation.  
- The manuscript alternates between F1_{∅} (Section 2.3) and F1θ (Tables 2–3). It is unclear whether θ refers to a threshold distinct from the zero-on-empty variant. This inconsistency makes it difficult to interpret headline numbers and replicate the pipeline.
- Equation (6) appears incorrect as written: admit(v_t) = 1[sim(q, v_t) ≥ ρ_fill · sim(q, v_t)], which is tautologically true for ρ_fill ≤ 1. The surrounding text implies it should be ≥ ρ_fill · sim(q, v_1) (the top backfill candidate). This is a critical gating condition for budget fill; the typo impacts clarity and potentially implementation.
- There are multiple uses of a symbol resembling "not nu" (e.g., w_n · 1[v∉S]) typeset as "\notν[…]" throughout Equations (3)–(6), which is confusing.

5) Missing figure and unclear pipeline visualization.  
- Section 3 mentions "Figure ??" to illustrate the pipeline, but the figure is missing. This hinders understanding of the multi-path controller's branching/backtrack rules and how budget enforcement interleaves with walk steps.

6) Reproducibility concerns and underspecified LLM controller.  
- The LLM controller uses "GPT-5 via Copilot SDK" (Section 4.5), but no prompt templates, decoding parameters, or input formatting are provided, nor are candidate prefilter thresholds fully specified. The walk budget H=2, branching limits, redundancy thresholds, and early stopping are described qualitatively, yet without enough detail to replicate decisions. Reported sensitivity to model capability (Section 6) reinforces the need for precise controller specification.

7) Graph construction and corpus preprocessing are underdescribed.  
- Section 3.2 defines the link-context triple, but critical details are missing: handling of redirects/disambiguations, normalization of anchors, whether templates and infobox links are included, deduplication of multiple anchors to the same target, and the overall graph statistics (nodes, edges, average out-degree). Especially for IIRC "full-context" Wikipedia, these choices can heavily influence traversal quality.

8) Dataset and sampling protocol leave room for bias.  
- Both datasets are evaluated on "fixed 100-case canonical" samples (Section 4.1), with no description of the sampling protocol, random seed, or whether items are stratified by hop type or difficulty. The IIRC headline result is on N=20, which further exacerbates selection bias risk.

9) Ambiguous budget regimes and utilization reporting.  
- Although Section 2.3 defines budget adherence and utilization, Table 2 for IIRC does not report utilization or adherence. The ratio-controlled "dense" line shows "64.0 nodes" regardless of ρ, which is odd if ρ is varied from 0.01 to 1.0; at minimum, an explanation is required. The claim that the controller is "budget independent" (Section 6) should be validated with utilization numbers and evidence that backfill did not trigger.

10) Limited ablations for scorer design and hyperparameters.  
- Edge scoring families (Equations 3–5) introduce several weight profiles and lookahead (L) settings. The paper states the best non-LLM configuration was chosen via ablation on 2Wiki, but corresponding ablation tables or sensitivity plots are not provided. Without these, it is unclear whether improvements are robust or sensitive to tuning.

11) Fairness of the "full-corpus" comparator.  
- The paper repeatedly contrasts to a ρ=1 "full-corpus" inclusion baseline with near-zero precision (e.g., Table 3, utilization 404.7×). While illustrative, this comparator is strawman-like; it is obvious that retaining the entire corpus collapses precision. The more relevant question is how the proposed method compares to realistic, budgeted pipelines that already prune aggressively.

12) Claims about controller necessity not yet substantiated on the main surface.  
- Section 6 argues the controller is "essential" based on 2Wiki runs and earlier IIRC trials, but the key IIRC improvement is only on N=20 and not contrasted against a tuned non-LLM walk on the same surface.

Specific figure/table/equation references:
- Table 2: core comparison suffers from the budget-regime mismatch and constant 64-node behavior under "ρ=0.01–1.0" for dense, which needs justification; it currently weakens the argument.
- Table 3: calibration gains are small (+0.018 F1θ) and need stronger baselines to be persuasive.
- Algorithm 1: single-path walk is clear, but the multi-path controller logic is not formalized; a figure or explicit pseudocode is missing.
- Equation (6): appears incorrect; likely should reference v_1 rather than v_t on the right-hand side. This is non-trivial since it governs the budget-fill quality gate.

## Potentially Missing Related Work
1) Chen, D., et al. Reading Wikipedia to Answer Open-Domain Questions (2017).  
- Directly relevant as a foundational retrieve-then-read baseline over Wikipedia. Should be cited in Related Work and used as a classic pipeline reference in Section 7 and baselines discussion.

2) Karpukhin, V., et al. Dense Passage Retrieval for Open-Domain Question Answering (2020).  
- Highly relevant dense retrieval baseline underlying many recent systems. DPR should be referenced in Sections 3.3 and 7, and ideally included or discussed as a stronger dense baseline beyond "multi-qa-MiniLM-L6-cos-v1".

3) Feldman, Y., and El-Yaniv, R. Multi-Hop Paragraph Retrieval for Open-Domain QA (2019/2021 variants).  
- Iterative retrieval baseline closely aligned with the paper's iterative-dense comparisons; should be cited in Section 7 and compared to repo-native "mdr_light" with some justification.

4) Xiong, W., et al. Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval (MDR, 2021).  
- Although cited as [29], the paper does not include trained MDR comparisons. It should be elevated as a core baseline; Section 8 acknowledges this gap, but Section 5 should either include MDR results or provide a principled rationale.

5) Welbl, J., et al. Constructing Datasets for Multi-hop Reading Comprehension Across Documents (2018).  
- A foundational multi-hop dataset and problem framing; while HotpotQA is cited, this work provides broader context. Add in Section 7 to strengthen historical grounding.

6) Yasunaga, M., et al. QA-GNN (2021).  
- Graph-based QA that integrates textual evidence with KG structure. Relevant to the paper's "graph vs. hyperlink" positioning; include in Section 7 to sharpen contrasts with constructed-graph approaches.

7) Nie, Y., et al. Query-Focused Evidence Extraction for Bridging Multi-Hop Reasoning and Retrieval (2022).  
- Focused evidence selection across multiple documents, conceptually adjacent to budgeted subgraph selection. Add in Related Work to position the paper's selector-first framing.

8) Zhang, S., et al. Graph-Augmented Retrieval for Open-Domain QA (2023).  
- Recent graph-augmented retrieval approach. Should be discussed in Section 7 as a directly comparable alternative to hyperlink-based traversal.

## Detailed comments to authors
- Budget comparability and stopping criteria: Please provide results where all methods operate under the same budget definition and constraints. For example, fix ρ and enforce identical token or node caps and identical budget-fill rules across dense and walk variants. Report budget utilization and adherence (as defined in Section 2.3) for IIRC in Table 2 to validate the "budget independence" claim for the controller and clarify the dense selector's 64-node behavior across all ρ.
- Clarify metric naming and compute: Throughout Sections 2 and 5, standardize on F1_{∅} or clearly define F1θ and its relationship to "zero-on-empty". Reprint the exact metric formula and confirm whether θ plays any role beyond walk thresholds.
- Fix Equation (6): The gating condition should reference the score of the top backfill candidate v_1, not v_t on both sides. Please also restate the exact sim function and any normalization used, and report ablations over ρ_fill.
- Provide the missing figure: Section 3 references "Figure ??". Please include an architecture/pipeline diagram detailing the controller's choose_one/choose_two/stop actions, scout-branch constraints, redundancy-risk thresholds, and the backtrack rule. This would also aid in replicability.
- LLM controller specification: Include the exact prompts, candidate prefilter thresholds, token/window limits, parsing of direct/bridge/redundancy subscores, and stopping policy. If the controller aggregates multi-edge signals, provide the combination formula. Add token accounting for LLM calls in Table 2.
- Stronger baselines: Add trained MDR, DPR, and a classical BM25+re-rank pipeline. If training MDR is out of scope, compare against a published MDR checkpoint on IIRC-style data or provide a rigorous justification. Also consider a graph-augmented retriever (e.g., a recent GAR or QA-GNN variant) as a comparison point.
- Edge scorer ablations: Provide a results table showing overlap vs. sentence-transformer scorers, lookahead depths (L), and the named weight profiles (e.g., st_balanced vs. st_future_heavy). Quantify variance and report statistical tests on the 2Wiki runs; then replicate on IIRC once the full 100-case set is available.
- Graph construction details: Please document how you extract hyperlinks and contexts, how redirects/disambiguations are handled, anchor normalization, multi-edge consolidation, and edge count statistics. Provide corpus graph statistics for IIRC (|V|, |E|, degree distributions).
- Sampling protocol: Describe how the "canonical 100-case" subsets for IIRC and 2Wiki were selected. Include random seeds and stratification criteria if any.
- Table-specific questions:
  - Table 2: Why does "Dense ρ=0.01–1.0" always yield 64.0 nodes? Is this a fixed cap independent of ρ? If so, report how it was chosen and how it relates to ρ. Please add Budget Utilization and Adherence columns.
  - Table 3: Please complete missing precision/recall and utilization cells for the iterative baselines. Also provide node counts to align with Table 2's presentation.
- Algorithm 1: The walk threshold θ is set to 0 in Section 4.5; please include runs with non-zero θ to examine precision–recall trade-offs and early stopping effects. Also provide an ablation on H (e.g., H=1,2,3).
- Sensitivity to model capability: Section 6 mentions large degradation with a weaker controller LLM. Please include a table quantifying performance vs. LLM choice (model size/capability), and if possible, a non-LLM emulation using learned scorers to understand how much of the benefit is model reasoning vs. link context.
- Fairness of the "full corpus" comparator: While instructive, it is a strawman. Consider comparing to budget-aware post-retrieval pruners (e.g., AdaGReS-style) that optimize redundancy–relevance tradeoffs under the same token cap to show end-to-end budget value.

## Your final vote for this paper
2 Weak (or Lean) Reject  
The idea of hyperlink-local, budgeted subgraph selection is promising and relevant, and the paper includes clear algorithmic components and some encouraging signals (notably Table 3 and the controller's pilot in Table 2). However, the empirical foundation is too preliminary, comparisons are not budget-aligned, strong baselines are missing, and several clarity and reproducibility issues persist (equation error, missing figure, underspecified controller, underdescribed graph construction). With a full 100-case IIRC evaluation, stronger baselines, corrected metrics/equations, and tighter budget-aligned comparisons, this work could be much more compelling.

## Reviewer's Confidence
4 High: Solid expertise, strong confidence.  
I am confident in the assessment of methodology, evaluation design, and baselines, though final performance verdicts could change with more complete IIRC results.

## Confidential remarks for the program committee
N/A
