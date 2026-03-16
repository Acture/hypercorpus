# Next-Phase Experiments

Purpose: single working plan and run-status log for active experiments.
Canonical for: current sequencing, entry criteria, local recovery commands, and immediate next actions.
Not for / See also: completed claim support lives in `phase-decisions.md`; implementation surface lives in `current-implementation.md`; stable framing lives in `paper-positioning.md`.

Date: `2026-03-13`

This document is the working experiment plan. Keep completed, paper-facing decisions in `phase-decisions.md`. Do not create sibling status notes under `runs/` for the current phase; put recovery state here.

## Headline Policy Shift

The main experiment story now shifts from repo-native baselines to published methods.

- Headline baselines:
  - `MDR`
  - `HippoRAG`
  - `GraphRetriever` only after the first two
- Internal checks only:
  - `dense`
  - `iterative_dense`
  - `mdr_light`

Interpretation rule:

- winning over repo-native dense is no longer sufficient for the paper story; the next headline comparator is `MDR`
- beating `dense` only shows the method is not obviously broken
- beating `MDR` is the first threshold for a paper-level method claim
- until `MDR` is connected, all current results should be treated as pre-headline evidence

The fixed judgment rule for the current paper direction is:

- if `webwalker` only beats repo-native `dense` but does not beat `MDR`, the current paper story does not hold
- if `webwalker` stably beats `MDR` on a harder dataset, `dense` can move to appendix / sanity-check status

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

### In-Progress Recovery

#### Baseline Retest

- Run: `runs/2wiki-baseline-retest-s100-v1`
- Study preset: `baseline_retest_local`
- Status: interrupted for connectivity reasons
- Do not merge this run yet.

Chunk state:

- complete: `chunk-00000`
- complete: `chunk-00001`
- complete: `chunk-00002`
- partial, must rerun: `chunk-00003`
- not started: `chunk-00004`

Important note:

- `chunk-00003` has partial artifacts and should be treated as invalid until rerun to completion.

Resume commands:

```bash
uv run webwalker-cli experiments run-2wiki-store \
  --store data/2wiki-store-1k \
  --exp-name 2wiki-baseline-retest-s100-v1 \
  --output-root runs \
  --split dev \
  --case-ids-file runs/2wiki-sample-s100-v1/evaluated_case_ids.txt \
  --study-preset baseline_retest_local \
  --chunk-size 20 \
  --chunk-index 3 \
  --no-e2e \
  --no-export-graphrag-inputs

uv run webwalker-cli experiments run-2wiki-store \
  --store data/2wiki-store-1k \
  --exp-name 2wiki-baseline-retest-s100-v1 \
  --output-root runs \
  --split dev \
  --case-ids-file runs/2wiki-sample-s100-v1/evaluated_case_ids.txt \
  --study-preset baseline_retest_local \
  --chunk-size 20 \
  --chunk-index 4 \
  --no-e2e \
  --no-export-graphrag-inputs

uv run webwalker-cli experiments merge-2wiki-results \
  --run-dir runs/2wiki-baseline-retest-s100-v1
```

## 2Wiki Wrap-Up Only

`2Wiki` is no longer the main decision surface. It is now only a wrap-up and calibration stage.

Required remaining work:

1. Finish and merge `runs/2wiki-baseline-retest-s100-v1`.
2. Record the current best `single_path` operating point.
3. Record the current repo-native baseline ordering.
4. Record the delta between the current `single_path` winner and the best repo-native baseline.

Explicitly out of scope on `2Wiki`:

- do not run `2Wiki branchy_profiles_384_512`
- do not expand the `2Wiki` sample to chase small gains
- do not use `2Wiki` as the main evidence for whether broad walk is the paper story

After `2Wiki` wrap-up, the only internal conclusions to carry forward are:

- the current best `single_path` selector
- the repo-native baseline ordering on the retest sample
- the delta between `single_path winner` and best repo-native baseline, but this delta is not a headline acceptance criterion

Immediate next actions:

1. Resume and complete `baseline_retest_local`.
2. Merge the run and write down the best repo-native baseline ordering plus the delta to the `single_path` winner.
3. Move directly to `IIRC`; do not reopen extra `2Wiki` confirmation work first.

## IIRC As The Main Next Phase

`IIRC` is the only required harder-dataset mainline for the next phase.

Execution order is fixed.

### 1. IIRC Sample-Defining Run

- sample size: `100`
- selector: `top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop`
- budget: `128`
- purpose: create a canonical `evaluated_case_ids.txt`

### 2. IIRC WebWalker Shortlist

Do not run the full study preset first. Run only this shortlist:

- `single_path winner` from `2Wiki`
- best repo-native baseline from the completed `2Wiki` baseline retest
- `gold_support_context`
- `full_corpus_upper_bound`

Budgets are fixed to:

- `256`
- `384`
- `512`

### 3. IIRC MDR Run

The `MDR` run must use the exact same `case_ids_file` as the `IIRC` webwalker shortlist.

The comparison must land on the same evidence-budget frame, using:

- `support_f1_zero_on_empty`
- `support_precision`
- `support_recall`

The first required questions on `IIRC` are:

- does `webwalker` beat `MDR`
- is the absolute gain at least `0.03`
- does precision collapse while recall rises

## Conditional Branchy Lane On IIRC

Branchy search is no longer a default next step. It is conditional.

Only add a branchy lane on `IIRC` if the `single_path winner` is still only slightly above the best repo-native baseline.

If triggered, run at most two candidates:

- `beam + ST lookahead_2 + st_future_heavy`
- `ucs + ST lookahead_2 + st_future_heavy`

This branchy lane is for method diagnosis, not the default headline path.

## HotpotQA Fullwiki Gate

`HotpotQA fullwiki` is the second harder-dataset phase, not the immediate next step.

It enters the mainline only if both conditions hold:

1. the normalized graph bundle is ready
2. `IIRC` shows at least one positive signal:
   - `webwalker single_path winner` beats `MDR` by absolute `>= 0.03`
   - or a branchy candidate beats the `single_path winner` by absolute `>= 0.02`

If `IIRC` does not show that signal, do not invest in a long `HotpotQA fullwiki` run yet. Return first to the baseline / method story problem.

If `HotpotQA fullwiki` is unlocked, only run:

- webwalker shortlist:
  - `single_path winner`
  - at most one branchy winner, and only if `IIRC` showed a real gain
- published baseline shortlist:
  - `MDR`
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
- `next-phase-experiments.md`: active plan, sequencing, local status, and recovery commands
- run directories under `runs/`: artifacts only, not separate human-written status notes

## Minimum Completion Standards

### 2Wiki Wrap-Up

- `runs/2wiki-baseline-retest-s100-v1` is fully merged
- best repo-native baseline is explicitly identified
- the delta between `single_path winner` and best repo-native baseline is explicitly written down
- this delta is not used as headline acceptance

### IIRC

- a shared `evaluated_case_ids.txt` exists for `IIRC`
- `webwalker shortlist` and `MDR` use the same sample
- both sides report evidence metrics on the same budget definition
- the run answers:
  - whether `webwalker` beats `MDR`
  - whether the gain is `>= 0.03`
  - whether precision collapses

### HotpotQA Fullwiki

- graph bundle ready
- positive `IIRC` signal exists
- otherwise `HotpotQA fullwiki` stays deferred
