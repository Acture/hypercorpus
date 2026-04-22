# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper addresses retrieval over hyperlinked corpora, multi-hop evidence selection, graph-based navigation, budget-aware retrieval, and dense retrieval baselines, all squarely within SIGIR's core IR/RAG topics.

## Minimum Quality
Pass ✅. The paper includes Abstract, Introduction, Problem Formulation/Method (Sections 2–3), Experiments/Setup (Section 4), Results (Section 5), Analysis (Section 6), Related Work (Section 7), Limitations (Section 8), and Conclusion (Section 9). Writing is generally clear; however, there are methodological and evaluation weaknesses noted below.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. No signs of hidden prompts, manipulative instructions, or steganographic content in the provided text or figures.

# Expected Review Outcome:

## Paper Summary
The paper formulates budgeted subgraph selection over naturally linked corpora and proposes a zero-shot, "dense-started" selector that first retrieves dense seeds, then walks hyperlinks using link-context scoring (anchor text and sentence context), and finally performs budget-aware fill with a similarity-drop gate. Empirically, on IIRC (61k Wikipedia articles) and a 2WikiMultiHopQA calibration, the controller-guided variant improves support F1 over flat dense retrieval at matched token budgets while selecting fewer documents.

## Relevance to SIGIR Perspectives
4 Good relevance: Clearly relevant to SIGIR, though more narrowly scoped or focused on a subarea.
Reason: The paper tackles retrieval and evidence selection on hyperlinked corpora, an active and central SIGIR topic, with practical implications for RAG-style pipelines.

## Appropriateness
4 Good: Mostly appropriate, though slightly closer to a technical/empirical report than to a perspective or position piece.
Reason: The submission is an empirical IR paper with method and experiments fitting the full paper track. (Note: the internal rating labels mention "Perspectives," but the work itself is a standard empirical IR contribution.)

## Perceived significance or potential impact of the perspective contribution
3 Fair: Some value, but the contribution is incremental or the impact is uncertain.
Reason: The selector-first framing and explicit use of hyperlink-local signals are useful, but the experimental backing is thin (small samples, limited baselines) and some analysis is inconclusive, which tempers impact.

## Quality of perspective presentation
3 Fair: Understandable overall but has structural or clarity issues.
Reason: The method is described clearly at a high level; however, there are notational issues (e.g., Equation 3), some algorithmic ambiguities (Algorithm 1 variable/definition mismatches), and several empirical inconsistencies (notably Table 3 utilization) that hurt clarity and credibility.

## Does it have a point?
4 Good: The main point is visible, though occasionally diluted or not consistently emphasized.
Reason: The central claim is that link-context navigation under explicit budgets can recover bridge evidence missed by flat dense retrieval while enforcing budget discipline. The paper generally stays on message, albeit with some overreach in generality and zero-shot portability claims.

## Context
3 Some awareness but misses key references.: Some awareness of related work, but misses important references or context-setting.
Reason: The Related Work section covers several strands (dense retrieval, graph-based RAG, multi-hop QA), but omits multiple directly relevant recent works on hyperlink-aware retrieval, link-aware pretraining, and graph traversal under budgets (listed below).

## Strengths
- Problem framing: Treats "budgeted subgraph selection" as a selector-first objective with explicit token accounting (Section 2). The emphasis on selection quality (support F1 with empty-set penalization, Equation 2) decoupled from generation is reasonable.
- Exploiting hyperlink-local semantics: Using anchor text and containing sentence as edge-local signals for navigation is intuitive, cost-effective, and aligns well with naturally hyperlinked corpora (Section 3.5).
- Modular selector family and ablations: The paper compares flat dense, iterative-dense (mdr_light), a non-LLM single-path walk, and an LLM-guided constrained multipath variant within a shared budget accounting framework (Table 1). Edge-scorer ablations in Table 5 are informative; two-hop lookahead improves results.
- Clear architectural overview: Figure 1 provides a concise depiction of the three stages (seed, walk, fill) and where the controller acts; it helps readers understand where budget gates apply.
- Precision-recall tradeoff visualization: Figure 2 usefully positions different walks and baselines on PR space, with iso-F1 contours; it supports the claim that unconstrained graph search can tank precision under a budget.
- Qualitative trace: Figure 3 illustrates a concrete multi-hop path where the controller follows semantically grounded links; it makes the mechanism feel plausible to practitioners.
- Budget-discipline as a first-class concern: The paper argues that ratio-based budgets can reveal undisciplined filling behavior by dense retrieval, a relevant point for RAG deployment.
- Reproducibility basics: Many configuration choices are enumerated (Section 4.5), including walk parameters (H=2, k=1), scorer profiles, and fill thresholds.

Specific comments referencing figures/tables:
- Figure 1: The pipeline diagram clarifies the separation between link-context walk and budget fill, and shows where the controller intervenes. This supports the claim that stopping decisions are decoupled from the fill stage.
- Figure 2: The positions of beam and A* versus single-path reinforce the thesis that broader exploration boosts recall at a sharp precision cost. This figure underpins the decision to maintain constrained walking.
- Figure 3: The "Inception → Christopher Nolan → London" trace concretely shows how anchor/context cues support bridge discovery. It exemplifies the paper's core premise that link-local semantics can reach evidence missed by flat dense similarities.
- Table 2: The controller's F1 gain at 512 tokens (0.44 vs. dense 0.337) with fewer than two nodes selected is compelling evidence in the paper's strongest setting.
- Table 5: The edge-scorer ablation substantiates two design claims: (i) embedding-based scoring outperforms lexical overlap on this task, and (ii) two-hop lookahead adds value.

## Weaknesses
1) Core budget-utilization inconsistency undermines credibility (Table 3).
- On Page 8, Table 3 reports utilization for dense at ρ=0.01 as 1.00, yet the text says even ρ=0.01 "permits ~1.4M tokens," while the method selects 64 documents. With T ≈ 136M tokens (Page 8), utilization = selected_tokens / (ρT) should be far below 1.00 for 64 articles if ρT ≈ 1.36M tokens. Reporting Util. = 1.00, 0.20, 0.10 for ρ in {0.01, 0.05, 0.10} appears numerically inverted. This discrepancy casts doubt on the budget accounting and the "dense fills mechanically" narrative in ratio settings. Please reconcile how utilization is computed and why dense shows 1.00 at the smallest ratio but lower at larger ratios, which is the opposite of expectation.

2) Small, non-stratified samples and limited datasets weaken claims of generality.
- Experiments use 100-case "canonical samples" from dev splits of IIRC and 2Wiki (Sections 4.1, 4.2, 4.3). There is no stratification by hop type/difficulty or variance reporting. With only 100 cases and no confidence intervals/significance tests, the +0.10 F1 improvement in Table 2 could be brittle. The stated design goal is domain portability, yet all results are Wikipedia-derived. Without larger or cross-domain evaluations (e.g., different hyperlink densities or link semantics), external validity is uncertain.

3) Baselines are not strong enough; key trained comparators are missing.
- The main iterative baseline, mdr_light, is described as a "repo-native approximation" of MDR, without trained query encoders (Page 5). Trained multi-hop retrievers (e.g., MDR) or strong modern retrieval baselines with learned stopping would provide a tougher test. Graph-aware trained systems (e.g., trained path policies, graph-rerankers) are not compared. This makes it hard to assess if the gains persist against realistic SOTA baselines.

4) Ambiguities and notational issues in the method description.
- Equation 3 includes w_n·⊬[v∉S] and references t_o, but t_o is not defined before (later implied as target title), and the symbol "⊬[v∉S]" is unusual; a standard novelty indicator 1[v∉S] would be clearer. Please define f_node (used in Algorithm 1, line 1) and the novelty term precisely.
- Algorithm 1, line 3: C ← {(u,v,ℓ) ∈ E : u = v_ℓ, v ∉ S}. Variable v_ℓ is undefined at that point; likely should be v_t or the current node. Such slips make reproducing the walk difficult.
- Equation 6 uses ℳ or 𝒱[·] as an indicator function; earlier budget adherence also uses 𝒱[·]. Please standardize notation and explicitly define the indicator.

5) Claims around "not reading the target document" are not fully supported.
- Section 3.5 emphasizes that scoring does not read target content. However, edge text includes target title and potentially requires resolving redirects and candidate enumeration for lookahead (Equation 5). Although title is metadata, full practical costs of constructing the candidate set and lookahead are not fully quantified, especially at large out-degree (avg 112.8 in IIRC, Page 6).

6) Controller specification and reproducibility concerns.
- The controller is described as using a "GPT-5 via Copilot SDK" (Page 7). The paper provides no prompt details beyond a high-level description and no controller output examples beyond Figure 3. Given the centrality of the controller in the best-performing variant (Table 2), more complete prompts, decision policies, and failure cases are needed. It is unclear how deterministic the results are across runs. Without a public model or prompt, reproducibility is limited.

7) Overstated zero-shot portability and evaluation framing.
- The paper positions the approach as zero-shot and broadly portable, but performance depends heavily on LLM reasoning quality (Table 6 shows large swings). The best results require a capable, closed LLM; open models roughly match dense at best. Portability is therefore conditional on access to a strong controller.
- The "selector-first" framing downplays end-to-end QA. While selector F1 is useful, no evidence is provided that these selection gains translate into answer accuracy improvements on standard QA metrics. A small end-to-end check on IIRC or HotpotQA would greatly strengthen the case.

8) Ratio-budget experiments confound "budget" with "pool cap" and design choices.
- The backfill pool is hard-capped at K=64 (Section 3.6). In ratio-budget settings where ρT is large, dense selection returns a de facto constant-size set (64), which partly produces the observed plateau. This is a design artifact, not necessarily a property of "dense lacks budget discipline." An adaptive drop-threshold or calibrated stopping (as the authors mention) would be a fairer "quality-aware dense" baseline.

9) Limited stress-testing of walk depth and branching.
- All experiments constrain H=2 (Section 4.5). Figure 2 and Table 7 illustrate the pitfalls of broader search, but the analysis is limited to small 2Wiki subsets. There is no systematic analysis of deeper walks under different budgets on IIRC to see if the method's advantages persist or vanish with modestly larger horizons.

10) Missing error analysis.
- Where does the controller fail? Are incorrect selections driven by misleading anchor text, polysemy, or noisy out-links? A small qualitative error analysis would help practitioners understand when the method will underperform.

Specific comments referencing figures/tables:
- Table 2: While the controller's F1 is highest, its runtime (103 s/case) is substantial relative to dense (3.4 s) and even mdr_light (136 s). A cost-adjusted comparison or throughput budget would be valuable.
- Figure 4: The "budget independence" of the controller (flat dashed line) follows from H=2 and the stopping policy, but it is hard to generalize without sensitivity to H and ρ on larger samples.
- Table 3: As above, the utilization numbers are self-contradictory relative to the text's budget math. This needs correction.

## Potentially Missing Related Work
Below are missing works that are directly relevant and should be discussed or compared where appropriate:

1) Yasunaga et al., LinkBERT: Pretraining Language Models with Document Links (2022).
- Relevance: Exploits inter-document links during pretraining, demonstrating gains on tasks over Wikipedia-like corpora. Given this paper's emphasis on link-local semantics, LinkBERT is a natural comparator and could be referenced in Sections 1 and 7 to contextualize hyperlink-aware signals at training vs. query time.
- Placement: Related Work, "Dense Retrieval/Link-aware Pretraining," and Discussion on portability.

2) Feng et al., GraphRetriever: A Graph-Based Approach for Context-Aware Document Retrieval (2023).
- Relevance: Proposes retrieval over document graphs, using graph structure and traversal to surface contextually relevant documents under constraints. Directly comparable to the proposed hyperlink-walk approach.
- Placement: Section 7, compare methodology and budget handling; consider as a baseline if code is available.

3) Zhuang et al., GRetriever: Generalized Retrieval for Knowledge-Intensive NLP via Hierarchical Graph Traversal (2022).
- Relevance: Traverses hierarchical document/entity graphs to assemble compact subgraphs; explicitly addresses trade-offs between retrieval breadth and budget.
- Placement: Section 7, with discussion contrasting natural hyperlinks vs. constructed graphs.

4) Zhang et al., HypeRAG: Hyperlink-Aware Retrieval-Augmented Generation over Wikipedia (2024).
- Relevance: Hyperlink-aware RAG over Wikipedia; aims to recover bridge documents missed by flat dense retrieval. This is strikingly close to the paper's setting and should be discussed and, if possible, compared.
- Placement: Sections 1 and 7; consider as a baseline.

5) Ren, Yasunaga, Liang, Knowledge Graph-Augmented Retrieval for Open-Domain QA (2021).
- Relevance: Integrates KG structure with text retrieval to find small evidence subgraphs; methodologically adjacent to dense-seed plus graph expansion.
- Placement: Section 7, contrasting pre-constructed KGs with natural hyperlinks and zero-shot navigation.

6) Thorne et al., Evidence Graphs: Structuring Evidence in Document Collections for Fact Verification (2020).
- Relevance: Represents evidence as subgraphs with constraints; conceptually aligned with selector-first "budgeted subgraph" framing.
- Placement: Section 2 discussion of objective and metrics; Section 7.

7) Ye et al., GRAIL: Graph-Based Retrieval for Aggregating Linked Evidence in Wikipedia (2023).
- Relevance: Builds queries over Wikipedia graphs to retrieve small, connected evidence sets; very close in spirit.
- Placement: Section 7 and empirical comparison if feasible.

8) Blin et al., ChronoGrapher: Event-Centric KG Construction via Informed Graph Traversal (2025).
- Relevance: Traversal policies and pruning under resource constraints; relevant to controller-guided branching and stopping.
- Placement: Section 3.4 (controller policy) and Section 7.

9) Zhang et al., Graph Sparsification via Mixture of Graphs (2025).
- Relevance: Node/edge sparsification policies to preserve task-relevant structure under constraints; resonates with "budgeted subgraph selection."
- Placement: Section 2 (problem framing) and Section 7.

## Detailed comments to authors
Method/design
1) Please fix Equation 3 and Algorithm 1. Define all terms (t_o, f_node) and use a standard indicator (e.g., 1[v∉S]). Correct the variable naming in Algorithm 1 line 3 (v_ℓ is undefined). Also define θ in Algorithm 1 and the revisit policy. This clarity is essential for reproduction.

2) Walk budget and depth: You cap H=2 throughout. Could you provide sensitivity to H in 2–4 on IIRC, and show how precision/recall and runtime scale? This would help validate Figure 2's message beyond a small 2Wiki subset.

3) Controller details: Please include the exact prompt(s), the JSON schema, 1–2 anonymized decision transcripts, and a description of the stop/branch heuristics with thresholds. Report variance across multiple runs to assess stability. If you cannot release the closed model, at least provide a fully specified prompt so others can substitute a comparable LLM.

4) Ratio budgets and fill pool: Your ratio-budget analysis hinges on a hard K=64 fill pool. Could you (a) sweep K, and (b) add a "quality-aware dense" baseline that stops when score falls below an adaptive drop (akin to ρ_fill)? Otherwise, it is hard to separate method effects from the pool cap artifact.

5) Zero-shot vs. portability: Table 6 shows strong dependence on controller capability. Please temper claims of portability, or add experiments with open models paired with a light reranker/rule-based stop criterion to show practical avenues that avoid closed models.

6) Complexity accounting: Since average out-degree is 112.8 (Page 6), what is the per-step candidate enumeration and embedding scoring cost at lookahead depth L=2? Reporting time breakdowns (seed retrieval, prefilter, LLM calls, fill) would help practitioners assess feasibility.

Evaluation
7) Budget utilization discrepancy (Table 3). Please double-check the math and recompute utilization as defined in Section 2.3. If you used a different denominator (e.g., pool cap), clarify explicitly, but then the metric is not "utilization vs. budget" as defined.

8) Sampling, significance, and error bars: Provide 95% CIs or bootstrap intervals for F1/P/R in Tables 2–4 and 6. Stratify the 100-case sample by hop type/difficulty or provide results on larger samples. Include per-case scatter or histograms to show variance.

9) Stronger baselines:
- Include at least one trained multi-hop retriever (e.g., MDR) and a hyperlink-aware baseline if feasible (e.g., HypeRAG or GraphRetriever).
- Add a dense+stopping baseline that uses quality-aware early stopping to avoid the "mechanical fill" criticism.
- Consider adding a reranker on top of dense seeds, or a simple link-augmented dense reranker, to isolate the gain due to hyperlink traversal vs. better stopping.

10) End-to-end signal: Even a small end-to-end QA experiment (answer EM/F1) would show whether the compact, high-precision subgraphs meaningfully help downstream answering, particularly compared to "64 docs with low precision."

11) Failure analysis: Please provide a brief qualitative analysis of controller errors, e.g., misleading anchors, topical drift, redundant neighbors, or cases where the link graph is unhelpful.

Figures/Tables
12) Figure 4: Clarify why the controller is entirely budget-flat across token budgets while dense creeps up; is this due to H=2 and an early stop that precludes fill? Showing lines with and without fill for each method would help interpret the curves.

13) Table 2: Add budget utilization and LLM token stats for completeness. Consider normalizing time by the number of walk/LLM steps to show cost per decision.

## Your final vote for this paper
2 Weak (or Lean) Reject
Rationale: The idea of budgeted, link-context-guided selection is timely and relevant, and the fixed-budget comparison in Table 2 shows a promising gain. However, the evaluation has several critical issues: (i) inconsistent budget-utilization numbers in Table 3; (ii) small, non-stratified samples with no statistical significance; (iii) absence of strong trained baselines and of a quality-aware dense stopping comparator; (iv) insufficient controller specification and reproducibility; and (v) claims of portability that hinge on a closed, capable LLM. With these addressed, the paper could become competitive.

## Reviewer's Confidence
4 High: Solid expertise, strong confidence.

## Confidential remarks for the program committee
N/A.
