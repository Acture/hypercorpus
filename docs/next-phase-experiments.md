# Next-Phase Experiments

Purpose: working sequencing note for active experiments.
Canonical for: current sequencing, entry criteria, local recovery commands, and immediate next actions.
Not for / See also: completed claim support lives in `phase-decisions.md`; implementation surface lives in `current-implementation.md`; stable framing lives in `paper-positioning.md`; live execution status and handoffs live in Linear; milestone-level conclusions live in Notion.

Date: `2026-03-23`

This document is the working experiment sequencing note. Keep completed, paper-facing decisions in `phase-decisions.md`. Do not use this file as a multi-worktree sync board. Keep run-level blocker state, owner changes, and handoff notes in Linear, not in repo-local status prose.

## Headline Policy Shift

The main experiment story now shifts from "walk versus dense" to `dense-started constrained subgraph selection`.

- Shared stage-1 prior and control:
  - `dense`
- Main repo-native comparison class:
  - `iterative_dense`
  - `mdr_light`
- Future external comparison class:
  - `MDR`
  - `GraphRetriever`
  - `HippoRAG`

Interpretation rule:

- `dense` remains mandatory because it is both the seed prior and the simplest strong control
- the current method claim is not "replace dense", but "improve dense-seeded evidence assembly under the same token budget"
- until a direct external baseline is wired in, all headline decisions stay on repo-native baselines plus harder-dataset transfer

The fixed judgment rule for the current phase is:

- the method must beat the shared `dense` control and `mdr_light` on at least one harder-dataset operating point
- `MDR` remains a desired future external comparison, but it is not a blocking prerequisite for the current internal phase

## Current 2Wiki Execution State

### Scope

- Dataset: `2Wiki dev`
- Sample size: `100`
- Sample source: `runs/2wiki-sample-s100-v1`
- Main metric: `summary_rows.csv.support_f1_zero_on_empty`
- Constraints:
  - `--no-e2e`
  - `--no-export-graphrag-inputs`
  - local-only, no selector LLM

### Completed Runs

#### Sample-Defining Run

- Run: `runs/2wiki-sample-s100-v1`
- Selector: `top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop`
- Budget: `128`
- Status: completed and merged
- Canonical sample IDs: `runs/2wiki-sample-s100-v1/evaluated_case_ids.txt`

#### Single-Path Edge Ablation

- Run: `runs/2wiki-single-path-edge-ablation-s100-v1`
- Study preset: `single_path_edge_ablation_local`
- Status: completed and merged

Winner on `tokens-128`:

- `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop`
- `support_f1_zero_on_empty = 0.3935`
- `support_precision = 0.3247`
- `support_recall = 0.5575`
- vs control delta:
  - `support_f1 = +0.0415`
  - `support_precision = +0.0352`
  - `support_recall = +0.0575`

Winner on `tokens-256`:

- `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop`
- `support_f1_zero_on_empty = 0.4139`
- `support_precision = 0.3101`
- `support_recall = 0.7300`
- vs control delta:
  - `support_f1 = +0.0575`
  - `support_precision = +0.0472`
  - `support_recall = +0.0900`

Current conclusion:

- `lookahead_2 + sentence_transformer + st_future_heavy` is the clear Phase 1 single-path winner.
- The current overlap control does not remain competitive on the `100`-case sample.

#### Baseline Retest

- Run: `runs/2wiki-baseline-retest-s100-v1`
- Study preset: `baseline_retest_local`
- Status: completed (all 5 chunks finished)

## 2Wiki Wrap-Up Only

`2Wiki` is no longer the main decision surface. It is now only a wrap-up and calibration stage for dense-seeded selector behavior.

Required remaining work:

1. Finish and merge `runs/2wiki-baseline-retest-s100-v1`.
2. Record the current best dense-anchored selector operating point.
3. Record the current repo-native baseline ordering.
4. Record the delta between the current `single_path` winner and the best repo-native baseline.
5. Export subset-aware comparison rows so harder-case signals are visible separately from all-case averages.

Repo-native baseline ordering on `tokens-256` (from `runs/2wiki-baseline-retest-s100-v1`, 100-case sample):

| Rank | Selector | `support_f1_zero_on_empty` |
| --- | --- | --- |
| 1 | `gold_support_context` | 1.0000 |
| 2 | `top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop` | 0.3964 |
| 3 | `top_3_seed__sentence_transformer__hop_2__iterative_dense__budget_fill_relative_drop` | 0.3852 |
| 4 | `top_3_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop` | 0.3766 |
| 5 | `top_3_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop` | 0.3759 |
| 6 | `top_1_seed__sentence_transformer__hop_2__iterative_dense__budget_fill_relative_drop` | 0.3702 |
| 7 | `top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop` | 0.3702 |
| 8 | `full_corpus_upper_bound` | 0.0049 |

Explicitly out of scope on `2Wiki`:

- do not run `2Wiki branchy_profiles_384_512`
- do not expand the `2Wiki` sample to chase small gains
- do not use `2Wiki` as the main evidence for whether broad walk is the paper story

Current dense-anchored control:

- `top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop`

Current best `single_path` winner:

- `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop`

Current semantic-controller lane to develop next:

- `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm_controller__lookahead_2`
- `top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2`

Current `2Wiki` baseline retest status:

- run: `runs/2wiki-baseline-retest-s100-v1`
- status: completed (all 5 chunks finished)

After `2Wiki` wrap-up, the only internal conclusions to carry forward are:

- the current best dense-anchored selector
- the repo-native baseline ordering on the retest sample
- subset-aware deltas against the dense control

Immediate next actions:

1. ~~Resume and complete `baseline_retest_local`.~~ Done.
2. ~~Merge the run and write down the best repo-native baseline ordering plus the delta to the `single_path` winner.~~ Done; see table above.
3. Move directly to `IIRC`; do not reopen extra `2Wiki` confirmation work first.

## IIRC As The Main Next Phase

> **Deprecation notice (2026-03-23):** All runs in this section that reference `--store dataset/iirc/store` used the partial store (5,184 articles). The full-context store (61,304 articles) is now canonical. These results are retained for historical reference only and must not be cited as evidence in the paper.

`IIRC` is the only required harder-dataset mainline for the next phase.

Execution order is fixed.

### 1. IIRC Sample-Defining Run

- sample size: `100`
- selector: `top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop`
- budget: `128`
- purpose: create a canonical `evaluated_case_ids.txt`

### 2. IIRC WebWalker Shortlist

Do not run the full study preset first. Run only this shortlist:

- `top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop`
- `top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop`
- current best `single_path` selector from `2Wiki`
- current best `constrained_multipath` controller selector
- `gold_support_context`
- `full_corpus_upper_bound`

Budgets are fixed to:

- `256`
- `384`
- `512`

### 3. IIRC Judgment Rule

The shortlist must use the exact same `case_ids_file` across all selectors.

The comparison must land on the same evidence-budget frame, using:

- `support_f1_zero_on_empty`
- `support_precision`
- `support_recall`
- `avg_path_hit`

The first required questions on `IIRC` are:

- does the semantic-controller selector beat the dense control
- does it also beat `mdr_light`
- are gains concentrated on harder subsets rather than only easy all-case averages
- does precision collapse while recall rises

## Conditional Branchy Lane On IIRC

Branchy search is no longer a default next step. It is conditional.

Only add a branchy lane on `IIRC` if the controller-based `single_path` lane is still only slightly above the best repo-native baseline.

If triggered, run at most two candidates:

- `constrained_multipath + llm_controller`
- one diagnostic `beam` or `ucs` comparator only if needed

This branchy lane is for method diagnosis, not the default headline path.

## HotpotQA Fullwiki Gate

`HotpotQA fullwiki` is the second harder-dataset phase, not the immediate next step.

It enters the mainline only if both conditions hold:

1. the normalized graph bundle is ready
2. `IIRC` shows at least one positive signal:
   - a semantic-controller selector beats both `dense` and `mdr_light` by absolute `>= 0.02`
   - or `constrained_multipath` beats the best `single_path` controller by absolute `>= 0.02`

If `IIRC` does not show that signal, do not invest in a long `HotpotQA fullwiki` run yet. Return first to the baseline / method story problem.

If `HotpotQA fullwiki` is unlocked, only run:

- hypercorpus shortlist:
  - `single_path winner`
  - at most one branchy winner, and only if `IIRC` showed a real gain
- external baseline shortlist:
  - `MDR`, only if connected by then
  - `HippoRAG` only if already connected and still worth the extra run cost

## Backup Benchmark Candidates

These datasets are recorded as backup or future-track options only; they do not change the current execution order of `IIRC` first and `HotpotQA fullwiki` second.

### Diagnostic Backup Only

- `WikiHop`
  - relevant as an early hyperlink-derived multi-document benchmark, but not a near-term headline dataset because it is not the same as current full-corpus evidence-budget evaluation and should not displace `IIRC` or `HotpotQA fullwiki`

### Future Expansion Track, Not Current Phase

- `WebSRC`
  - attractive for a true web/HTML story, but it requires DOM-aware page handling and a different data interface than the repo's current normalized graph + node-support format
- citation-style scientific QA datasets such as `SciDQA`-like directions
  - conceptually interesting as citation-graph walking, but they require new citation-edge graph construction, PDF/paper parsing, and new gold-evidence alignment

### Do Not Prioritize For The Current Paper Story

- `TimeQuestions` / `TempQuestions`
  - temporal QA or KGQA is too far from the current natural-hyperlink text-navigation claim
- `ASQA` / open-domain `SQuAD` variants
  - useful for answer quality or open-domain retrieval, but not a good fit for the current support-node / support-path evidence metrics

The current adapters and normalized dataset format expect a graph plus per-case `gold_support_nodes`, `gold_start_nodes`, and optional `gold_path_nodes`, so new datasets should only move into the mainline if they can be mapped cleanly into that interface without redefining the evaluation task.

## Documentation Rules

- `phase-decisions.md`: completed, paper-facing phase decisions only
- `next-phase-experiments.md`: active plan, dense-anchored sequencing, local status, and recovery commands
- run directories under `runs/`: artifacts only, not separate human-written status notes

## Minimum Completion Standards

### 2Wiki Wrap-Up

- `runs/2wiki-baseline-retest-s100-v1` is fully merged
- best repo-native baseline is explicitly identified
- dense-control subset deltas are explicitly written down
- this delta is not used as headline acceptance by itself

### IIRC

- a shared `evaluated_case_ids.txt` exists for `IIRC`
- all shortlist selectors use the same sample
- both sides report evidence metrics on the same budget definition
- the run answers:
  - whether a semantic-controller selector beats `dense`
  - whether it also beats `mdr_light`
  - whether the gain is `>= 0.02`
  - whether precision collapses

### HotpotQA Fullwiki

- graph bundle ready
- positive `IIRC` signal exists
- otherwise `HotpotQA fullwiki` stays deferred
