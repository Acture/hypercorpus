# Hypercorpus Claim Ledger

## Purpose
Track which paper claims are already supported, which are still conditional, and which experiments or tables are required before they become safe.

## Claim Status Table

| # | Claim | Status | Evidence Source | Required To Lock |
| --- | --- | --- | --- | --- |
| C1 | Budgeted evidence discovery over natural hyperlinks is a coherent problem formulation. | **Supported** | `docs/paper-positioning.md`, `phase-decision-30` | Keep framing consistent in the draft. |
| C2 | Dense-started hyperlink-local selection beats the flat dense control on `2Wiki`. | **Supported (calibration only)** | `runs/2wiki-single-path-edge-ablation-s100-v1`: single-path winner F1 = 0.4139 vs dense F1 = 0.3964 at tokens-256, 100 cases. Delta: +0.0175 F1, +0.09 recall. | Reframe as calibration evidence only. Not the main result. |
| C3 | `2Wiki` should not be the main decision surface. | **Supported** | `docs/next-phase-experiments.md`. Dense control is already strong on 2Wiki; deltas are small. Harder dataset needed to show method value. | Close remaining 2Wiki wrap-up. |
| C4 | Full-IIRC replaces the old partial-IIRC store as the only valid IIRC judgment surface. | **Supported on implementation facts** | Full-context fetch + conversion fix. Partial store: 5,184 articles. Full store: 61,304 articles (11.8x expansion). | Land the canonical full-IIRC artifact set under `dataset/iirc_full/`. **(WS1/ACT-5 dependency)** |
| C5 | The project has a reviewer-acceptable external comparator through real `MDR`. | **In progress** | Official `MDR` integration exists in code (`baselines/mdr` submodule, `baselines/mdr.py`). | Close `export -> train -> index -> run` pipeline, and surface reviewer-friendly diagnostics (`support_set_em`, `avg_path_hit`, `bridge/comparison` splits) in the same comparison surface. **(WS1/ACT-5 dependency)** |
| C6 | Sentence-transformer seeds are stronger than lexical seeds. | **Supported** | `runs/phase-decision-30`: ST dense F1 = 0.4124 vs lexical dense F1 = 0.3535 (30 cases, budget 256). Confirmed on 100-case sample: ST start_hit = 0.97 vs lexical start_hit = 0.63. | Stable. |
| C7 | `budget_fill_relative_drop` is an effective component. | **Supported** | `runs/phase-decision-30`: eliminates empty selections (0.0 empty rate). Improves all-case F1. Without fill, lexical dense has 26.7% empty rate and F1 drops from 0.3535 to 0.3022. | Stable. Can be written up as a contribution component. |
| C8 | Under a fixed 256-token budget, wider/deeper graph search (beam, astar) hurts by collapsing precision. | **Supported on 2Wiki** | `runs/phase-decision-30`: beam overlap F1 = 0.2272, astar overlap F1 = 0.1472 vs dense F1 = 0.4124. Precision collapse is clear. | Needs IIRC confirmation that this pattern transfers. |
| C9 | `lookahead_2 + st_future_heavy` is the best non-LLM single-path scorer profile. | **Supported on 2Wiki** | `runs/2wiki-single-path-edge-ablation-s100-v1`: F1 = 0.3935 (128) and 0.4139 (256) vs all other profiles on 100 cases. Clear winner. | Needs IIRC transfer validation. |
| C10 | The controller-guided selector is the main contribution, not just one diagnostic option. | **Open** | Partial IIRC chunk-00001 hints: `constrained_multipath + llm_controller` F1 = 0.2850 at budgets 384/512 vs dense control values. But partial store is deprecated. | Full-IIRC comparison required. **(WS1/ACT-5 dependency)** |
| C11 | The paper beats trained iterative retrieval on the main harder dataset. | **Open** | None yet. | Real `MDR` on full-IIRC. **(WS1/ACT-5 dependency)** |
| C12 | The paper supports a broader superiority claim against graph-based systems (GraphRetriever, HippoRAG). | **Not supported** | None. | Only consider after real `MDR` is closed and the main story still has room. Phase constraint: no expansion before real MDR is closed. |
| C13 | The method is stronger as evidence discovery, not necessarily as end-to-end QA. | **Supported** | Current docs, experiment setup, all evaluation uses `support_f1_zero_on_empty` as headline metric. QA metrics (`answer_em`, `answer_f1`) are secondary. | Stable. Keep QA as secondary sanity check only. |
| C14 | Dense-started evidence assembly improves on flat dense, not "walk replaces dense". | **Supported** | `docs/next-phase-experiments.md`: "the current method claim is not 'replace dense', but 'improve dense-seeded evidence assembly under the same token budget'". Stage-1 dense is shared. | Stable framing. |
| C15 | The 2Wiki repo-native baseline ordering is: dense > iterative_dense > mdr_light. | **Supported** | `runs/2wiki-baseline-retest-s100-v1` at tokens-256: dense = 0.3964, iterative_dense (top-3) = 0.3852, mdr_light (top-3) = 0.3766. | Stable for calibration table. |

## Current No-Claim Zone
- Do not claim end-to-end QA superiority.
- Do not claim broad wins over all learned or graph-based retrievers.
- Do not claim the controller is the isolated source of gain until the main comparison surface is complete.
- Do not cite any partial-IIRC numbers (5,184-article store) as evidence.
- Do not claim superiority over real MDR until the comparison exists.

## Evidence Dependency Map

```
C1 (problem formulation)         --> Ready
C6 (ST > lexical seeds)          --> Ready
C7 (budget_fill effective)       --> Ready
C13 (evidence-first framing)     --> Ready
C14 (dense-started framing)      --> Ready
C15 (2Wiki baseline ordering)    --> Ready

C2 (walk > dense on 2Wiki)       --> Ready (calibration only)
C8 (broad search hurts)          --> Ready on 2Wiki; needs IIRC confirmation
C9 (st_future_heavy best)        --> Ready on 2Wiki; needs IIRC confirmation
C3 (2Wiki not main surface)      --> Ready

C4 (full-IIRC replaces partial)  --> BLOCKED on WS1/ACT-5 (canonical store landing)
C5 (real MDR comparator)         --> BLOCKED on WS1/ACT-5 (export/train/index/run)
C10 (controller as main claim)   --> BLOCKED on C4, C5
C11 (beats trained retrieval)    --> BLOCKED on C4, C5
C12 (beats graph systems)        --> BLOCKED on C10, C11 + additional baselines
```

## Near-Term Closure Tasks
1. **[WS1/ACT-5]** Land full-IIRC canonical store (61,304 articles).
2. **[WS1/ACT-5]** Close real `MDR` as a runnable comparator (export -> train -> index -> run).
3. **[WS1/ACT-5]** Rerun IIRC controller shortlist on full store at budgets 384 and 512.
4. **[WS1/ACT-5]** Generate the full-IIRC paper-facing table.
5. **[WS1/ACT-5]** Land reviewer-friendly MDR-aligned diagnostics in the unified evaluation (`support_set_em`, `avg_path_hit`, `bridge/comparison` split).
6. **[CTRL/ACT-9]** Freeze the main claim boundary (CTRL gate 4) before drafting introduction and conclusion.
7. **[WS3/ACT-6]** Begin prose drafting only after gate 4 is passed.
