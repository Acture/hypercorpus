# Phase Decisions

This file is the canonical experiment-facing record for paper claims. Every phase-decision entry should separate:

- `supported on <run>`
- `needs ablation`
- `not yet compared`

The goal is to keep paper-facing claims tied to specific evidence instead of spreading provisional conclusions across unrelated docs.

## Template

Each new entry should include:

- `run id`
- `dataset`
- `sample size`
- `budget`
- `tested selector family`
- `winners / losers`
- `supported findings`
- `needs ablation`
- `not yet compared`
- `next experiments`

## phase-decision-30

### Run Metadata

- `run id`: `phase-decision-30`
- `dataset`: `2WikiMultiHopQA dev`
- `sample size`: `30` cases
- `budget`: `256` tokens
- `reader`: `no-e2e`
- `summary source`: `runs/phase-decision-30/chunks/range-00000-00029/summary.json`

### Tested Selector Family

- dense seed baselines with and without `budget_fill_relative_drop`
- `top_1` two-hop single-path walk with `link_context_llm`
- `top_3` three-hop `beam` and `astar` variants with overlap or LLM link-context scoring
- `gold_support_context` as an oracle-style upper bound

### Winners And Losers

#### Best Tested Operating Point

`top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm__lookahead_1__budget_fill_relative_drop`

- `support_f1_zero_on_empty`: `0.4724`
- `support_recall`: `0.7500`
- `support_precision`: `0.3783`
- `budget_utilization`: `0.9660`
- `empty_selection_rate`: `0.0000`

#### Strong Dense Baseline

`top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop`

- `support_f1_zero_on_empty`: `0.4124`
- `support_recall`: `0.6167`
- `support_precision`: `0.3337`
- `budget_utilization`: `0.9650`
- `empty_selection_rate`: `0.0000`

#### Clearly Weak Direction At This Budget

`top_3 + hop_3 + beam/astar`

- lexical `beam overlap`: `0.2272`
- lexical `beam llm`: `0.2472`
- lexical `astar overlap`: `0.1472`
- lexical `astar llm`: `0.1747`
- sentence-transformer `beam llm`: `0.3110`
- sentence-transformer `astar llm`: `0.1906`

### Supported Findings

These findings are `supported on phase-decision-30`.

- The best tested operating point in this phase is `top_1 + sentence_transformer seed + hop_2 + single_path_walk + link_context_llm + budget_fill_relative_drop`.
- Sentence-transformer seeds are consistently stronger than lexical seeds in the tested families.
- `top_1` is a better operating point than `top_3` under the current `256`-token budget.
- Broad `hop_3` search with `beam` or `astar` is not a good direction under this budget because precision collapses while utilization is already high.
- `budget_fill_relative_drop` is an effective component for reducing empty selections and improving all-case support F1.
- `gold_support_context` reaches `avg_budget_adherence = 0.6` and `avg_budget_utilization = 1.0324`, which means the current oracle evidence often does not fit cleanly into `256` tokens.

### Needs Ablation

These points are suggested by current results but are not isolated enough to claim causally.

- Whether the hop-2 winner is strong because of walk structure, because of the LLM edge scorer, or because of stronger seeds plus budget fill.
- Whether `link_context_llm` is better than overlap or sentence-transformer edge scoring within the same `top_1 + sentence_transformer + hop_2 + single_path` family.
- Whether the current gains persist at larger token budgets or different sample slices.

### Not Yet Compared

- `MDR`
- `GraphRetriever`
- `HippoRAG`
- direct iterative reranking baselines
- full end-to-end QA systems under the same evidence budget

These systems should not be described as beaten or matched by the current phase.

### Next Experiments

- Add a clean edge-scorer ablation for `top_1 + sentence_transformer + hop_2 + single_path`.
- Add direct `dense top-k` and `MDR` style baselines under the same token-budget accounting.
- Repeat the current selector family on a larger phase sample before elevating claims beyond the present operating-point story.
- Treat further `beam` and `astar` expansion as low priority unless a new budget or pruning idea changes the precision tradeoff.
