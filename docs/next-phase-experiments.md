# Next-Phase Experiments

Date: `2026-03-13`

This document is a working plan for the next experiment phase. It is not a canonical paper-facing decision log. Completed, paper-facing decisions should remain in `docs/phase-decisions.md`.

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

## 2Wiki Wrap-Up Only

`2Wiki` is no longer the main decision surface. It is now only a wrap-up and calibration stage.

Required remaining work:

1. Finish and merge `runs/2wiki-baseline-retest-s100-v1`.
2. Record the current best `single_path` operating point.
3. Record the current repo-native baseline ordering.

Explicitly out of scope on `2Wiki`:

- do not run `2Wiki branchy_profiles_384_512`
- do not expand the `2Wiki` sample to chase small gains
- do not use `2Wiki` as the main evidence for whether broad walk is the paper story

Current best `single_path` winner:

- `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop`

Current `2Wiki` baseline retest recovery status:

- run: `runs/2wiki-baseline-retest-s100-v1`
- completed chunks: `chunk-00000`, `chunk-00001`, `chunk-00002`
- partial and must rerun: `chunk-00003`
- not started: `chunk-00004`
- do not merge until `chunk-00003` and `chunk-00004` are completed and the run is merged cleanly

After `2Wiki` wrap-up, the only internal conclusions to carry forward are:

- the current best `single_path` selector
- the repo-native baseline ordering on the retest sample
- the delta between `single_path winner` and best repo-native baseline, but this delta is not a headline acceptance criterion

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

## Tracking Docs

Document ownership is split as follows:

- `docs/phase-decisions.md`
  - completed, paper-facing phase decisions only
- `docs/next-phase-experiments.md`
  - this working next-step plan
  - baseline policy switch to published methods first
  - `2Wiki` wrap-up and direct move to `IIRC`
  - `HotpotQA fullwiki` entry conditions
  - current `single_path` winner
  - current `2Wiki` baseline retest recovery status
- `runs/2wiki-phase1-local-s100-status.md`
  - local execution state and recovery commands for `2Wiki`
- `runs/iirc-phase2-local-s100-status.md`
  - future local execution state doc for the `IIRC` phase

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
