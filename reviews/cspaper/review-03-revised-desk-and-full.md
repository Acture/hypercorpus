# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The work targets retrieval over hyperlinked corpora, multi-hop QA, and budget-aware selection, which are central to SIGIR topics in "Search and Ranking," "Machine Learning for IR," and "NLP for IR (RAG/QA)."

## Minimum Quality
Pass ✅. The paper contains Abstract, Introduction, Problem Formulation, Method, Experiments, Results/Analysis, Related Work, Limitations, and Conclusion. The methodology and empirical sections are substantive, though some baselines and analyses are missing.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. No hidden prompts, steganographic instructions, or manipulative content detected in the provided text or figures.

# Expected Review Outcome:
## Paper Summary
The paper formulates budgeted subgraph selection over naturally linked corpora and proposes a zero-shot, dense-started selector that traverses hyperlink-local context (anchor text and sentence context) before a budget-aware fill stage. On IIRC (61k full articles) and a 2WikiMultiHopQA calibration, the controller-guided variant improves support F1 over flat dense retrieval at matched token budgets.

## Relevance to SIGIR Perspectives
4 Good relevance: Clearly relevant to SIGIR, though more narrowly scoped or focused on a subarea.
Rationale: The paper addresses core IR retrieval under constraints with hyperlinks, highly pertinent to multi-hop QA/RAG.

## Appropriateness
4 Good: Mostly appropriate, though slightly closer to a technical/empirical report than to a perspective or position piece.
Rationale: This reads as a full technical paper with experiments and a concrete method; the framing and style fit the full paper track.

## Perceived significance or potential impact of the perspective contribution
2 Poor: Limited significance; offers little new beyond existing discussions.
Rationale: The selector-first budgeted perspective is interesting, but impact is muted by small-scale evaluations (100-case samples), lack of comparisons to strong supervised or modern open baselines, and model-dependence on a closed controller.

## Quality of perspective presentation
3 Fair: Understandable overall but has structural or clarity issues.
Rationale: Overall clear, but there are notational inconsistencies (e.g., F1θ vs. F1∅), typos in equations and algorithm, and some missing experimental details limit clarity and reproducibility.

## Does it have a point?
3 Fair: The paper has a point, but it is vague, muddled, or obscured by details.
Rationale: The thesis—"exploit hyperlinks with budget discipline beats flat dense under matched tokens"—is clear, but the practical import is diluted by limited baselines and small sample sizes, making it hard to generalize the claimed advantage.

## Context
2 Weak situating.: Little engagement with prior work.
Rationale: The paper covers multi-hop and GraphRAG threads, but omits several closely related retrieval and budgeted-evidence selection works and stronger multi-hop retrieval baselines.

## Strengths
- Clear problem setup. Section 2 frames budgeted subgraph selection over hyperlink graphs with an explicit cost model. Equation (1) formalizes the constrained optimization with F1∅ as the objective, and Section 2.3 defines budget adherence/utilization.
- Sensible pipeline. Figure 1 lays out a simple, auditable architecture: dense seed, hyperlink-local walk with link-context scoring, then budget-fill with a similarity-gated admission (Equation 6).
- Link-context exploitation. The method avoids pre-building synthetic graphs and leverages existing hyperlink semantics and anchor/sentence context, a reasonable and cost-effective design for Wikipedia-like corpora.
- Selector-first evaluation. The paper deliberately decouples selection quality from reader models, focusing on support F1 and budget behavior; this is a useful perspective for IR.
- Evidence that navigation helps on bridge cases. On 2Wiki (Table 4), single-path walk improves recall by +0.128 over dense at 256 tokens while maintaining similar precision; Table 5 shows two-step lookahead helps (0.394 vs. 0.354 F1).
- Budget discipline insights. Table 3 contrasts ratio-controlled behavior: the controller self-limits (~1.8 nodes) while dense fills to 64, illustrating how budget use can distort quality. Figure 4 further shows controller F1 flat across token budgets, highlighting self-limiting behavior.
- Qualitative transparency. Figure 3's walk trace is instructive: it illustrates how anchor sentences align with the query to guide hops that recover gold support.
- Analysis of walk structures. Table 6 and Figure 2 convincingly show that unconstrained beam/A* exploration collapses precision despite higher recall, motivating constrained, quality-aware walking.

## Weaknesses
- Very limited evaluation scale and statistical support. All IIRC experiments use a fixed 100-case "canonical" dev subset (Section 4.1–4.2). No variance estimates, confidence intervals, or statistical tests are provided. This makes conclusions fragile. For instance, Table 2's +0.10 F1 improvement for the controller over dense at 512 tokens could be sample-sensitive without uncertainty quantification.
- Baseline coverage is insufficient for SIGIR standards. There is no comparison to strong supervised multi-hop retrievers (e.g., MDR with trained query encoders) despite extensive prior evidence they help, nor to modern hybrid sparse-dense pipelines or strong rerankers. The "mdr_light" is a repo-native approximation and not representative. The paper positions against GraphRAG broadly but does not compare to representative open implementations with budget controls.
- Absence of budgeted context-selection baselines. Systems like FiD-style selectors under a fixed token budget and submodular/greedy selection under token constraints would be highly relevant comparators to the proposed "budget fill with relative drop." Without them, it is hard to ascribe gains to the link-context walk rather than to the gating/fill policy.
- Controller reliance and reproducibility. The best numbers hinge on a closed LLM ("GPT-5 via Copilot SDK"). Section 6 acknowledges steep degradation with a smaller model (F1∅ down to 0.235), but there is no evaluation with widely-available open models or an ablation reporting quality-cost trade-offs, latency, or token counts. This weakens claims of portability and practical deployability.
- Ambiguities/inconsistencies in metrics and notation. The manuscript alternates between F1∅ and F1θ; Table 2's header uses F1θ but the text discusses F1∅ throughout. Resolve the notation and confirm what is actually computed. Also, Table 3 reports the same precision/recall for dense across ρ values, which is unexpected given changing utilization; provide explanation or correct if mistaken.
- Equations contain notational/typing errors. In Equation (3), the novelty term appears as w_n·\not{v}[v∉S], which looks like a malformed indicator; define it clearly and justify how this term is used. In (3) t_o is undefined (likely t_v). In Algorithm 1 line 11, v_{ℓ+1} appears to be a typo and should be v_{t+1}. Also, "return S ▷ score below threshold" occurs with θ=0 in experiments; clarify when θ>0 is used.
- Budget fill design and side effects underexplored. Equation (6) thresholds candidates relative to the top backfill score with ρ_fill=0.5, but the sensitivity of F1 to this gate is not reported. The gate can easily over-admit near-duplicates or correlated distractors when dense similarity is anisotropic. Provide ablations varying ρ_fill and K (backfill pool size).
- Incomplete reporting of runtime and cost. The paper states it records wall-clock runtime and LLM tokens, but these are not tabulated for the main results. Without cost-performance trade-offs, the reader cannot judge whether the controller's gains are worthwhile in practice.
- Seed and hop sensitivity missing. Results lock in k=1 seed and H=2 hops (Section 4.5), but no sensitivity analysis explores whether larger k or H changes outcomes, particularly on IIRC where the hyperlink graph is "denser and noisier." This undermines generality claims.
- Dataset protocol raises concerns. Both datasets use dev splits and a 100-case canonical sample chosen by "deterministic ordering," with no stratification by hop type/difficulty and no test split reporting. This is not a robust evaluation protocol and risks overfitting to a narrow slice of data characteristics. Additionally, for IIRC, the full-article corpus and link extraction details are given, but there is no check for leakage or consistency with the original benchmark's context assumptions.
- Missing and mismatched details in sections. The paper refers to multi-qa-MiniLM-L6-cos-v1 in Section 3.3, then to multi-qa-MiniLM-16-cpsv1124 in Sections 4.4–4.5; clarify which model is used where. The seed top-k varies in different places (top-1 vs. top-3 for baselines in Table 4), which can confound comparisons.
- Limited external validity. The paper claims portability to technical or biomedical corpora, but all evaluations are on Wikipedia-derived benchmarks. Without at least one non-Wikipedia corpus or even a link-scarce condition, generality remains speculative.

Figure- and table-specific critiques:
- Figure 2: The precision-recall scatter convincingly shows single-path outperforming beam/A* in F1 (dashed iso-F1∅ curves). However, the legend and plotting choices obscure exact operating points for iterative-dense baselines noted in text; consider overlaying them for completeness or clarifying why they are omitted.
- Figure 3: The walk trace example is helpful, but it would be stronger if the authors included the exact anchor/context snippets and their similarity/LLM scores to demonstrate why "London" is preferred over other neighbors. As is, it is qualitative.
- Figure 4: The "budget-independence" of the controller is visually clear, but the y-axis suggests small absolute gains over dense as tokens increase. Adding error bars or per-point CIs is important to interpret stability.
- Table 2: Main IIRC result. Provide standard deviations or bootstrap CIs across the 100 cases. Also, report cost (mean runtime, LLM tokens/case) to ground the practical trade-off.
- Table 3: Budget utilization table. Controller utilization is "-" across rows; if you enforce a hard token budget, please report utilization and adherence explicitly for parity. The dense P/R values remain identical across ρ; explain this or fix.
- Table 4 and Table 5: These are useful ablations on 2Wiki; please add significance testing for F1 deltas and detail how many cases ended empty before/after fill to substantiate the "0.0 empty rate" claim.

## Potentially Missing Related Work
1) Izacard, G., Grave, E. (2020), Leveraging Passage Retrieval with Generative Models for Open-Domain QA (FiD). Directly relevant to budgeted evidence assembly and a strong retrieval-plus-reader baseline under fixed token budgets. Discuss as a flat-dense alternative with budgeted context packing and consider a FiD-style selection baseline in Section 7 and comparisons in Section 5.  
2) Dhingra, B., Qi, P., Liu, Z. (2020), Differentiable Reasoning over a Knowledge Graph for QA. Provides graph-structured multi-hop retrieval/reasoning that implicitly controls exploration under constraints. Cite in Related Work (Graph-based retrieval) and contrast with discrete hyperlink walks.  
3) Zhao, J., Wang, Y., Lin, J. (2023), Lexically-Accelerated Dense Retrieval. Two-stage retrieval followed by graph exploration; relevant as a non-hyperlink graph expansion from dense seeds. Add to Related Work and consider as an additional baseline where a document proximity graph is explored without LLM control.  
4) Lin, Y., Wang, H., Dehghani, M. (2024), End-to-End Beam Retrieval for Multi-Hop QA. Frames retrieval as beam search with budgeted context, a close alternative to your constrained multi-path strategy. Compare conceptually and, if possible, empirically at matched token budgets.  
5) Wang, H., Wu, T., Xiong, C. (2025), Graph Neural Network Enhanced Retrieval for QA. Uses passage graphs to select compact evidence sets; include in Graph-based retrieval and discuss differences vs. natural hyperlinks and zero-shot scoring.  
6) Wei, Q., Ning, H., Han, C. (2025), Query-Aware Multi-Path KG Fusion for RAG. Builds compact, query-conditioned subgraphs with multi-path fusion; contrast with your controller's branching and budget enforcement in Section 7.  
7) Anonymous (2024), Semantic Contribution-Aware Adaptive Retrieval for Black-Box Models. Adaptive selection and pruning under context budgets is directly related to your budget-fill stage; add to Budget-Aware Retrieval.  
8) Anonymous (2025), Fixed-Budget Evidence Assembly in Multi-Hop RAG. Explicitly optimizes evidence selection under strict budgets for multi-hop; cite in Budget-Aware and Adaptive Retrieval and consider as a baseline for fixed-token settings.

## Detailed comments to authors
- Methodology and notation
  1) Please correct Equation (3): define the novelty indicator cleanly (e.g., 1[v∉S]) and fix t_o to t_v if that is intended. Specify tokenization/normalization for φ(·,·).  
  2) Algorithm 1 line 11 likely has a typo: v_{ℓ+1} should be v_{t+1}. Also, "return S ▷ score below threshold" occurs with θ=0 in experiments; clarify when θ>0 is used.  
  3) You use multi-qa-MiniLM-L6-cos-v1 in Section 3.3 but multi-qa-MiniLM-16-cpsv1124 in Sections 4.4–4.5. Which model actually produced the reported results? Please standardize and re-run if needed.  
  4) For the LLM controller, provide the exact prompt template, candidate-construction details, and any temperature/top-p settings. This is essential for reproducibility; if prompts cannot be shared, consider releasing a redacted but structurally equivalent template.
- Evaluation protocol
  5) Expand beyond 100-case "canonical" samples. At minimum, report results on the full dev set or two disjoint 100-case samples with CIs; ideally add a test split for IIRC. Include bootstrap confidence intervals or paired tests for all F1∅ comparisons in Tables 2–6.  
  6) Report runtime and LLM token usage per case for all selectors in Table 2 and Table 3. Without this, it is impossible to judge the value of the controller's gains.  
  7) Add sensitivity analyses: k seeds (e.g., 1/3/5), H hops (e.g., 1/2/3), ρ_fill (0.3–0.8), and backfill pool K (16/32/64/128). This will show whether the gains are robust to budget and hyperparameters.  
  8) Provide a strong supervised multi-hop baseline. MDR with trained query encoders is a natural choice; if training is out-of-scope, include at least a modern hybrid sparse+dense baseline with a reranker (e.g., BM25+DPR+cross-encoder) under identical budgets.  
  9) Clarify why Table 3 shows identical P/R for dense across all ρ values. Were the same 64 documents recovered regardless of ratio due to the K=64 cap? If so, state explicitly and explain how this interacts with utilization; otherwise fix the table. Also add the controller's utilization values rather than "-".  
  10) Unify F1 notation. You switch between F1∅ and F1θ in text and tables. Please settle on one and ensure figures/tables match.  
  11) For Figure 3, include the actual anchor text and containing sentence, and, if possible, the controller's per-candidate reasoning or scores. This would substantiate the qualitative narrative.  
  12) For Figure 2, add iterative-dense/mdr_light points if available, or justify omission. Also add error bars if feasible.  
  13) Address dataset protocol concerns: justify the deterministic 100-case sampling, confirm no data leakage (e.g., redirects/aliases mapping), and clarify how gold support mapping to your full-article store is handled.  
  14) Consider at least one non-Wikipedia corpus with hyperlinks (e.g., a software docs site) to support your portability claim.  
  15) Given controller dependence on a closed model, provide results with a strong open model and discuss quality-cost trade-offs. Even if lower, this will help practitioners assess feasibility.

## Your final vote for this paper
2 Weak (or Lean) Reject
Rationale: The paper proposes a reasonable and clearly motivated selector-first formulation and presents promising early evidence that hyperlink-local navigation plus budget discipline can beat flat dense retrieval at matched tokens. However, the empirical foundation is too thin for a SIGIR full paper: limited to 100-case subsets, lacking strong supervised and hybrid baselines, missing runtime/cost reporting, and containing several notation/implementation inconsistencies. The controller's reliance on a closed LLM further complicates reproducibility, and sensitivity analyses and ablations are insufficient. With stronger, broader experiments, corrected notation, and better baseline coverage, this could become a compelling contribution.

## Reviewer's Confidence
4 High: Solid expertise, strong confidence.

## Confidential remarks for the program committee
N/A.
