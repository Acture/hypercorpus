# Hypercorpus Claim Ledger

## Purpose
Track which paper claims are already supported, which are still conditional, and which experiments or tables are required before they become safe.

## Claim Status Table

| # | Claim | Status | Evidence Source | Required To Lock |
| --- | --- | --- | --- | --- |
| C1 | Budgeted evidence discovery over natural hyperlinks is a coherent problem formulation. | **Supported** | `docs/paper-positioning.md`, `phase-decision-30` | Keep framing consistent in the draft. |
| C2 | Dense-started hyperlink-local selection beats the flat dense control on `2Wiki`. | **Supported (calibration only)** | `runs/2wiki-single-path-edge-ablation-s100-v1` on `data/2wiki-store-1k`: single-path winner F1 = 0.4139 vs dense F1 = 0.3964 at tokens-256, 100 cases. Delta: +0.0175 F1, +0.09 recall. | Reframe as calibration evidence only. Not the main result. |
| C3 | `2Wiki` should not be the main decision surface. | **Supported** | `docs/next-phase-experiments.md`. Dense control is already strong on 2Wiki; deltas are small. Harder dataset needed to show method value. | Close remaining 2Wiki wrap-up. |
| C4 | Full-IIRC replaces the old partial-IIRC store as the only valid IIRC judgment surface. | **Supported on implementation facts** | Full-context fetch + conversion fix. Partial store: 5,184 articles. Full store: 61,304 articles (11.8x expansion). | Keep the canonical full-IIRC artifact set stable under `dataset/iirc/store`. **(WS1/ACT-5 dependency)** |
| C5 | The project has a reviewer-acceptable external comparator through real `MDR`. | **In progress** | Official `MDR` integration exists in code (`baselines/mdr` submodule, `baselines/mdr.py`). | Close `export -> train -> index -> run` pipeline, and surface reviewer-friendly diagnostics (`support_set_em`, `avg_path_hit`, `bridge/comparison` splits) in the same comparison surface. **(WS1/ACT-5 dependency)** |
| C6 | Sentence-transformer seeds are stronger than lexical seeds. | **Supported** | `runs/phase-decision-30`: ST dense F1 = 0.4124 vs lexical dense F1 = 0.3535 (30 cases, budget 256). Confirmed on 100-case sample: ST start_hit = 0.97 vs lexical start_hit = 0.63. | Stable. |
| C7 | `budget_fill_relative_drop` is an effective component. | **Supported** | `runs/phase-decision-30`: eliminates empty selections (0.0 empty rate). Improves all-case F1. Without fill, lexical dense has 26.7% empty rate and F1 drops from 0.3535 to 0.3022. | Stable. Can be written up as a contribution component. |
| C8 | Under a fixed 256-token budget, wider/deeper graph search (beam, astar) hurts by collapsing precision. | **Supported on 2Wiki** | `runs/phase-decision-30` on `data/2wiki-store-1k`: beam overlap F1 = 0.2272, astar overlap F1 = 0.1472 vs dense F1 = 0.4124. Precision collapse is clear. | Needs IIRC confirmation that this pattern transfers. |
| C9 | `lookahead_2 + st_future_heavy` is the best non-LLM single-path scorer profile. | **Supported on 2Wiki** | `runs/2wiki-single-path-edge-ablation-s100-v1` on `data/2wiki-store-1k`: F1 = 0.3935 (128) and 0.4139 (256) vs all other profiles on 100 cases. Clear winner. | Needs IIRC transfer validation. |
| C10 | The controller-guided selector is the main contribution, not just one diagnostic option. | **Preliminary positive signal (20-case pilot)** | `runs/iirc-controller-pilot-v2` (20 cases, full-IIRC store, constrained_multipath + llm_controller, openai/gpt-5.3-codex, ratio-controlled budgets): F1 = 0.4613 (ratio-0.01), 0.4756 (ratio-0.02) vs dense F1 = 0.4076 (tokens-512) on the same 20 cases (`runs/iirc-dense-full-v1` chunk-00000). Non-controller walk variants still lose to dense, confirming the controller as the sole viable headline method. | Replication on 100-case canonical surface with dense and mdr_light baselines in the same ratio-controlled comparison. **(WS1/ACT-5 dependency)** |
| C11 | The paper beats trained iterative retrieval on the main harder dataset. | **Open** | None yet. | Real `MDR` on full-IIRC. **(WS1/ACT-5 dependency)** |
| C12 | The paper supports a broader superiority claim against graph-based systems (GraphRetriever, HippoRAG). | **Not supported** | None. | Only consider after real `MDR` is closed and the main story still has room. Phase constraint: no expansion before real MDR is closed. |
| C13 | The method is an evidence-discovery / selector paper, not an end-to-end QA paper. | **Supported** | Current docs, experiment setup, and paper-facing exports all center `support_f1_zero_on_empty`, support-set metrics, selector runtime, and selector-side corpus mass. Answer metrics remain available in generic summaries but are out of scope for the current paper. | Stable. Keep answer-side reporting out of the selector-paper main tables. |
| C14 | Dense-started subgraph selection improves on flat dense, not "walk replaces dense". | **Supported** | `docs/next-phase-experiments.md`: "the current method claim is not 'replace dense', but 'improve dense-seeded subgraph selection under the same selector budget'". Stage-1 dense is shared. | Stable framing. |
| C15 | The 2Wiki repo-native baseline ordering is: dense > iterative_dense > mdr_light. | **Supported** | `runs/2wiki-baseline-retest-s100-v1` on `data/2wiki-store-1k` at tokens-256: dense = 0.3964, iterative_dense (top-3) = 0.3852, mdr_light (top-3) = 0.3766. | Stable for calibration table. |

## Current No-Claim Zone
- Do not claim end-to-end QA superiority.
- Do not claim broad wins over all learned or graph-based retrievers.
- Do not claim the controller is the isolated source of gain until the main comparison surface is complete.
- Do not cite any partial-IIRC numbers (5,184-article store) as evidence.
- Do not claim superiority over real MDR until the comparison exists.

## Paper Guidance From Current Data
- Treat `full-IIRC` as the only paper-facing decision surface. Historical `runs/iirc-*` directories are not interchangeable: `iirc-controller-shortlist-v1/chunks/chunk-00000/summary.json` and `iirc-local-full-v1/chunks/chunk-00000/summary.json` both point back to the same 100-case source list and the same 20-case slice definition (`case_start: 0`, `case_limit: 20`), yet they report dense-256 `support_f1_zero_on_empty` values of `0.2200` and `0.3867`. Until reconciled, only a freshly rerun canonical full-IIRC table should drive claims.
- Treat `iirc_selector_main` as the canonical implementation preset for that paper-facing surface. It fixes the shortlist selector set and uses ratio-controlled selector budgets (`0.01`, `0.02`, `0.05`, `0.10`, `1.0`) rather than coarse-node fixed token caps.
- Treat `mdr_light` as an internal baseline and the best non-LLM `single_path_walk` row as an ablation, not as the method headline. The main method candidate remains `constrained_multipath + llm_controller`.
- Treat the latest full-IIRC local runs as a historical coarse-node diagnostic, not a headline result: the complete 100-case non-controller surface in `runs/iirc-local-full-v1` ranks `dense > mdr_light > best single_path_walk` at every fixed-node-text budget (`256`: `0.3187 > 0.3103 > 0.2730`, `384`: `0.3140 > 0.3097 > 0.2709`, `512`: `0.3370 > 0.3242 > 0.2866` on `support_f1_zero_on_empty`). These runs narrow the non-controller family, but they are not the canonical ratio-controlled selector surface for this paper.
- Treat the dense control on the full-IIRC surface as operationally stable. `runs/iirc-local-full-v1` and `runs/iirc-dense-full-v1` now agree exactly on the dense row for all five completed chunks and all three budgets, so the current canonical dense baseline is no longer drifting across those two run directories.
- Treat all current `2Wiki` tables as reduced-store calibration. `phase-decision-30`, `2wiki-baseline-retest-s100-v1`, and `2wiki-single-path-edge-ablation-s100-v1` all read from `data/2wiki-store-1k`, not the full `data/2wiki-store`. That is enough for operating-point and ablation guidance, but not for a broad full-corpus retrieval claim.
- Treat `HotpotQA distractor` as supportive sanity evidence only. The current `s20` slice suggests repo-native `mdr_light` can beat flat dense, but the sample is too small and too task-specific to anchor the paper narrative.
- Treat `MuSiQue` and the `graphrag/input/wiki.csv` import path as implementation assets, not stable paper evidence. They show benchmark/export coverage, not a locked comparison surface.
- Treat `runs/iirc-controller-pilot-v2` as the first real controller signal on full-IIRC: F1 = 0.46 vs dense 0.41 on 20 cases. This is directional only until replicated on the 100-case canonical surface with dense and mdr_light baselines in the same ratio-controlled run.
- Write the contribution as `dense-started constrained subgraph selection under explicit selector budgets`. The 20-case pilot supports a controller-led headline, but do not claim it until the 100-case surface is locked.
- Write selector-side cost in terms of retained corpus mass, selector runtime, and selector LLM usage. Do not let paper-facing tables drift back into downstream answer packaging or reader-context token wording.
- Be precise about dataset unification. The evaluation interface is unified, but the underlying corpora are still adapter-specific (`2Wiki` carries `mentions`; `IIRC` carries `links`). This supports a common evaluation contract claim, not a single native corpus-schema claim.

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
C10 (controller as main claim)   --> Preliminary signal (20-case pilot); needs 100-case confirmation
C11 (beats trained retrieval)    --> BLOCKED on C4, C5
C12 (beats graph systems)        --> BLOCKED on C10, C11 + additional baselines
```

## Near-Term Closure Tasks
1. **[WS1/ACT-5]** Keep the canonical full-IIRC store stable under `dataset/iirc/store` and stop citing any legacy `dataset/iirc_full/*` path.
2. **[WS1/ACT-5]** Close real `MDR` as a runnable comparator (export -> train -> index -> run).
3. **[WS1/ACT-5]** Rerun the IIRC controller shortlist on the full store under selector-budget ratios (`0.01`, `0.02`, `0.05`, `0.10`, `1.0`).
4. **[WS1/ACT-5]** Generate the full-IIRC paper-facing selector table from one canonical rerun surface.
5. **[WS1/ACT-5]** Land reviewer-friendly MDR-aligned diagnostics in the unified evaluation (`support_set_em`, `avg_path_hit`, `bridge/comparison` split).
6. **[CTRL/ACT-9]** Freeze the main claim boundary (CTRL gate 4) before drafting introduction and conclusion.
7. **[WS3/ACT-6]** Begin prose drafting only after gate 4 is passed.
