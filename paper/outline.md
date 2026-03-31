# Hypercorpus Paper Outline

## Working Title
Dense-Started Budgeted Subgraph Selection over Naturally Linked Corpora

## Default Package
- Default venue: `SIGIR`
- Paper type: retrieval / evidence-discovery paper
- Core identity: selector-first, not QA-first
- Page limit: 8 pages + references
- Realistic venue window: `CIKM 2026` (abstract 2026-05-18, full paper 2026-05-25) or `SIGIR 2027` (likely ~2027-01)

## One-Sentence Thesis
For naturally linked corpora, a dense-started, budgeted selector can recover stronger support-bearing subgraphs than flat dense selection and repo-native iterative dense baselines, while staying lighter than eager graph construction or dataset-specific training.

## Abstract Skeleton
1. Motivate the problem as budgeted subgraph selection over naturally linked corpora.
2. State the method as dense-started constrained subgraph selection under explicit selector budgets.
3. Emphasize natural hyperlink structure plus local link semantics.
4. Summarize the main comparison surface: dense, repo-native iterative dense baselines, and real `MDR`.
5. Close with the main empirical tradeoff: support quality versus retained corpus mass and runtime.

## Section Skeleton

### 1. Introduction (~1.5 pages)

**Status: partially evidence-ready; contributions paragraph blocked on full-IIRC + real MDR.**

- Why flat dense top-k is weak for bridge-heavy evidence discovery.
  - Evidence: on 2Wiki (100 cases, budget 256), dense control achieves `support_f1 = 0.3964` but single-path walk reaches `0.4139` (+0.0175 absolute); recall gap is larger: `0.64` vs `0.73`.
  - Source: `runs/2wiki-baseline-retest-s100-v1`, `runs/2wiki-single-path-edge-ablation-s100-v1`
- Why eager graph construction is too expensive or mismatched for naturally linked corpora.
  - Evidence: `full_corpus_upper_bound` achieves `support_f1 = 0.0049` (recall 1.0, precision 0.002) -- full-corpus inclusion destroys precision.
  - Source: `runs/2wiki-baseline-retest-s100-v1`
- Why selector-budgeted subgraph selection is the right problem surface.
  - Argument: selector-side corpus mass is a real constraint; the paper should compare selected subgraphs against the full-corpus reference rather than against an answer-packaging stage.
- Contributions paragraph:
  - **[BLOCKED on WS1/ACT-5]**: Do not finalize until full-IIRC + real `MDR` comparison is locked.
  - Safe components to list now: problem formulation, budget-aware selector family, hyperlink-local subgraph selection, `budget_fill_relative_drop`.

### 2. Problem Formulation (~0.75 pages)

**Status: evidence-ready from positioning docs.**

- Input: query plus naturally linked corpus (documents connected by hyperlinks with anchor text and sentence context).
- Output: compact evidence set or induced subgraph under a selector budget.
- Objective: maximize support recovery quality (support F1) under budget and cost constraints.
- Evaluation is selector-first; QA is secondary.
- Formal notation for:
  - `LinkContextGraph G = (V, E)` with edge semantics `(anchor_text, sentence, sent_idx)`.
  - `SelectorBudget B = (\rho, H, k)`.
  - Objective: `argmax_{S subset V, \tau(S) <= \rho T} support_f1(S, gold)`.
- Source: `docs/paper-positioning.md`, `src/hypercorpus/graph.py`, `src/hypercorpus/selector.py`

### 3. Method (~2 pages)

**Status: evidence-ready for method description; ablation clarity depends on full-IIRC runs.**

- **Stage 1: Dense start prior.**
  - Sentence-transformer seed retrieval (`multi-qa-MiniLM-L6-cos-v1`).
  - Demonstrated stronger than lexical seeds: on 2Wiki 30-case, lexical dense F1 = 0.3535 vs sentence-transformer dense F1 = 0.4124.
  - Source: `runs/phase-decision-30`
- **Stage 2: Constrained hyperlink-local subgraph selection.**
  - `single_path_walk`: greedy walk over natural hyperlinks using link-context scoring.
  - Scorers: anchor overlap, sentence-transformer, LLM-based.
  - `link_context_sentence_transformer` with `lookahead_2` and `st_future_heavy` profile is current non-LLM winner.
  - Source: `runs/2wiki-single-path-edge-ablation-s100-v1`
- **Budget accounting and token-aware stopping.**
  - `budget_fill_relative_drop`: fill remaining budget after walk with dense retrieval, dropping lowest-relevance nodes.
  - Effective component: eliminates empty selections entirely (0.0 empty rate across all tested selectors).
  - Source: `runs/phase-decision-30`, `runs/2wiki-single-path-edge-ablation-s100-v1`
- **Family framing** (comparison class, not all claimed as contributions):
  - `dense`: flat dense top-k control and stage-1 seed prior.
  - `iterative_dense`: repo-native iterative dense baseline (MDR-style repeated dense retrieval).
  - `mdr_light`: repo-native comparison point with frontier-node dense expansion.
  - `single_path_walk`: hyperlink-local walk with link-context scoring.
  - `constrained_multipath`: controller-guided branchy selector with bounded branch count.
  - Source: `docs/current-implementation.md`, selector family documentation.

### 4. Experimental Setup (~1 page)

**Status: setup description evidence-ready; 2Wiki calibration results ready; full-IIRC main table blocked on canonical selector-budget reruns + real MDR.**

- **Datasets:**
  - `2WikiMultihopQA` (dev set): calibration only, not the main paper result.
    - 100-case canonical sample. Source: `runs/2wiki-sample-s100-v1/evaluated_case_ids.txt`.
  - `IIRC` (dev set): main harder-dataset judgment surface.
    - Partial store had 5,184 articles; the canonical full-context store now has 61,304 articles under `dataset/iirc/store`.
    - All partial-IIRC results are deprecated.
    - 100-case canonical sample source: `runs/iirc-sample-s100-dense-v1/chunks/chunk-00000/evaluated_case_ids.txt`.
- **Metrics:**
  - Primary: `support_f1_zero_on_empty` (penalizes empty selections).
  - Secondary: `support_precision`, `support_recall`, `budget_utilization`, `budget_adherence`.
  - Diagnostic: `selection_runtime_s`, `selector_total_tokens`, `selected_nodes_count`, `selected_corpus_mass`.
  - Answer-generation metrics remain available in generic summaries but are out of scope for this selector paper.
- **Budgets:**
  - `2Wiki` calibration: fixed budgets such as `128` and `256`.
  - `IIRC` main table: selector-budget ratios such as `0.01`, `0.02`, `0.05`, `0.10`, and `1.0` as the full-corpus reference.
- **Comparison class:**
  - `dense` (top-1 seed, sentence-transformer, budget fill)
  - `mdr_light` (repo-native iterative dense)
  - best non-LLM `single_path_walk` (sentence-transformer scorer, lookahead 2, st_future_heavy)
  - current controller winner (constrained multipath + LLM controller) -- **[BLOCKED on full-IIRC]**
  - real `MDR` -- **[BLOCKED on ACT-5 export/train/index/run]**
  - `gold_support_context` (oracle upper bound)
  - `full_corpus_upper_bound` (GraphRAG proxy)

### 5. Main Results (~1.5 pages)

**Status: 2Wiki calibration results ready; full-IIRC main table blocked on WS1/ACT-5.**

- **Table 1 (main):** Full-IIRC selector comparison under ratio-controlled budgets.
  - **[BLOCKED]**: Requires one canonical full-IIRC shortlist rerun surface + real MDR.
  - Data should come from one rerun on `dataset/iirc/store`, not from historical coarse-node fixed-budget runs.
- **Table 3 / Supplementary:** 2Wiki calibration summary.
  - Evidence ready. Baseline ordering on tokens-256 (100-case sample):
    1. `gold_support_context`: 1.0000
    2. `dense` (top-1, ST): 0.3964
    3. `iterative_dense` (top-3, ST): 0.3852
    4. `mdr_light` (top-3, ST): 0.3766
    5. `dense` (top-3, ST): 0.3759
    6. `iterative_dense` (top-1, ST): 0.3702
    7. `mdr_light` (top-1, ST): 0.3702
    8. `full_corpus_upper_bound`: 0.0049
  - Source: `docs/next-phase-experiments.md`, `runs/2wiki-baseline-retest-s100-v1`
  - Single-path winner vs dense control delta at tokens-256: +0.0175 F1, +0.09 recall.
  - Source: `runs/2wiki-single-path-edge-ablation-s100-v1`
- **Preliminary partial-IIRC signal (deprecated, not for paper):**
  - On partial store (5,184 articles), chunk-00001 only:
    - Best at tokens-256: `single_path_walk` (ST, st_future_heavy) F1 = 0.2200.
    - Best at tokens-384/512: `constrained_multipath` (LLM controller) F1 = 0.2850.
  - Source: `runs/iirc-controller-shortlist-v1/chunks/chunk-00001/summary.json`
  - **These numbers must not appear in the paper.** Full-IIRC will replace them.

### 6. Analysis (~1 page)

**Status: partially writable from 2Wiki; full analysis blocked on full-IIRC.**

- **Precision / recall tradeoff.**
  - 2Wiki evidence: single-path walk gains recall (+0.09 at budget 256) while maintaining precision gain (+0.0472).
  - Source: `runs/2wiki-single-path-edge-ablation-s100-v1`
- **Budget sensitivity.**
  - 2Wiki evidence: dense control precision drops from 0.3247 (128) to 0.2226 (512) as budget grows; walk selectors show similar trend but maintain higher recall.
  - Source: `runs/2wiki-baseline-retest-s100-v1`
  - **[BLOCKED]**: Full budget sensitivity analysis requires the ratio-controlled IIRC main table.
- **When controller behavior helps versus hurts.**
  - Preliminary signal: on partial IIRC, `constrained_multipath + llm_controller` is the best at budgets 384/512 but not at 256.
  - **[BLOCKED]**: Requires full-IIRC to confirm.
- **Why full-IIRC replaces partial-IIRC.**
  - Factual: partial store had 5,184 articles; full store has 61,304 articles (11.8x expansion).
  - Impact: retrieval difficulty increases substantially, making gains more meaningful.
  - Source: conversion fix commit history, `docs/next-phase-experiments.md`

### 7. Related Work (~0.75 pages)

**Status: structure ready from `paper/related-work-outline.md` (owned by WS4/ACT-8).**

- Multi-hop evidence retrieval and reasoning-path retrieval (MDR, GraphRetriever, DecompRC, IRCoT).
- Iterative dense retrieval and decomposition-guided retrieval.
- Graph retrieval / GraphRAG baselines (HippoRAG, RAPTOR).
- Web or hyperlink navigation as adjacent inspiration.
- **Do not edit here -- owned by WS4/ACT-8.**

### 8. Limitations (~0.25 pages)

**Status: writable now.**

- Current story is evidence-discovery-first, not end-to-end QA superiority.
- Real `MDR` is the first external comparator; broader external coverage (GraphRetriever, HippoRAG) may remain future work.
- Results may still be budget- and corpus-dependent.
- The method relies on the existence of natural hyperlink structure in the corpus.
- Current evaluation is on English Wikipedia-derived corpora; generalization to other domains is untested.

### 9. Conclusion (~0.25 pages)

**Status: blocked on full-IIRC + real MDR for final claim language.**

- Restate the problem and operating point.
- Emphasize the selector-first contribution.
- Keep any broad claim bounded by the final IIRC evidence surface.
- **[BLOCKED]**: Do not write conclusion until main claim boundary is frozen (CTRL gate 4).

## Writing Gates
- Do not finalize the introduction contribution list before the full-IIRC + real `MDR` comparison is locked.
- Do not write broad superiority language before the main claim boundary is frozen.
- Do not let the paper body drift into a QA-system framing.
- Do not cite any partial-IIRC numbers (from the 5,184-article store) as paper evidence.
- Do not write "we beat X" before the exact comparison table exists.

## Evidence Inventory

| Evidence Source | Status | Paper Section |
| --- | --- | --- |
| `runs/phase-decision-30` (2Wiki, 30 cases, budget 256) | Complete | Method motivation, early calibration |
| `runs/2wiki-sample-s100-v1` (2Wiki, 100 cases, budget 128) | Complete | Sample definition |
| `runs/2wiki-single-path-edge-ablation-s100-v1` (2Wiki, 100 cases, budgets 128/256) | Complete | Method validation, edge-scorer ablation |
| `runs/2wiki-baseline-retest-s100-v1` (2Wiki, 100 cases, budgets 128-512) | Complete | Calibration table, baseline ordering |
| `runs/iirc-sample-s100-dense-v1` (IIRC partial, 100 cases) | Deprecated | Sample definition only |
| `runs/iirc-controller-shortlist-v1` (IIRC partial, incomplete) | Deprecated | None -- must rerun on full store |
| Full-IIRC canonical selector-table experiments | **Not started** | Table 1 (main), Analysis |
| Real MDR export/train/index/run | **Not started** | Table 1 (main), Contributions |
