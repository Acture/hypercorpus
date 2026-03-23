# Hypercorpus Tables And Figures

## Main Tables

### Table 1. Full-IIRC Main Comparison

**Status: BLOCKED on WS1/ACT-5 (full-IIRC canonical store + real MDR).**

- Dataset: full-IIRC canonical store (61,304 articles)
- Budgets: `384`, `512`
- Rows:
  - `dense` (top-1, sentence-transformer, budget_fill_relative_drop)
  - `mdr_light` (top-1, sentence-transformer, budget_fill_relative_drop)
  - best non-LLM `single_path_walk` (ST, lookahead_2, st_future_heavy, budget_fill_relative_drop)
  - current controller winner (constrained_multipath + llm_controller)
  - real `MDR`
  - `gold_support_context` (oracle)
- Columns:
  - `support_f1_zero_on_empty`
  - `support_precision`
  - `support_recall`
  - `budget_utilization`
  - `selection_runtime_s`
  - `selector_total_tokens` (LLM cost)
- Data source: will come from rerun of `runs/iirc-controller-shortlist-v1` on full store.
- Dependencies: full-IIRC canonical store landed, real MDR pipeline closed.

### Table 2. Full-IIRC Hard-Subset Comparison

**Status: BLOCKED on WS1/ACT-5.**

- Same selector rows as Table 1.
- Only harder slices (cases where dense control fails to recover any support).
- Goal: show whether gains are concentrated on the difficult portion rather than only all-case averages.
- Data source: `subset_comparison_rows.csv` from full-IIRC runs.

### Table 3. 2Wiki Calibration Summary

**Status: READY.**

- Purpose: calibration only, not the main paper result.
- Dataset: 2WikiMultihopQA dev, 100-case canonical sample.
- Budget: `256` tokens.
- Actual data (from `runs/2wiki-baseline-retest-s100-v1` and `runs/2wiki-single-path-edge-ablation-s100-v1`):

| Selector | `support_f1` | `precision` | `recall` | `budget_util` |
| --- | --- | --- | --- | --- |
| `gold_support_context` | 1.0000 | 1.000 | 1.000 | 0.944 |
| `single_path_walk` (ST, st_future_heavy, la2) | 0.4139 | 0.310 | 0.730 | 0.971 |
| `dense` (top-1, ST) | 0.3964 | 0.289 | 0.640 | -- |
| `iterative_dense` (top-3, ST) | 0.3852 | -- | -- | -- |
| `mdr_light` (top-3, ST) | 0.3766 | -- | -- | -- |
| `dense` (top-3, ST) | 0.3759 | -- | -- | -- |
| `full_corpus_upper_bound` | 0.0049 | 0.002 | 1.000 | 404.7 |

- Source run IDs:
  - `runs/2wiki-baseline-retest-s100-v1` (baseline ordering)
  - `runs/2wiki-single-path-edge-ablation-s100-v1` (single-path winner)
  - Sample: `runs/2wiki-sample-s100-v1/evaluated_case_ids.txt`
- Notes:
  - The single-path walk shows +0.0175 F1 over dense, driven by +0.09 recall gain.
  - Dense control is surprisingly strong on 2Wiki; this motivates testing on harder IIRC.

## Analysis Figures

### Figure 1. System Architecture / Pipeline Diagram

**Status: Can be drafted now.**

- Show the three-stage pipeline:
  1. Dense seed retrieval (sentence-transformer top-k).
  2. Constrained hyperlink-local walk with link-context scoring.
  3. Budget-aware fill (budget_fill_relative_drop).
- Show `LinkContextGraph` structure: nodes (documents), edges (hyperlinks with anchor text + sentence context).
- Source: `docs/current-implementation.md`, `src/hypercorpus/graph.py`, `src/hypercorpus/selector.py`

### Figure 2. Cost-Quality Curve

**Status: BLOCKED on WS1/ACT-5 for IIRC data. Can draft 2Wiki version now.**

- X-axis: selector-side cost (runtime or LLM token cost)
- Y-axis: `support_f1_zero_on_empty`
- Use the full-IIRC comparison surface (primary) and 2Wiki (secondary/supplementary).
- Key comparison: non-LLM single-path walk (zero LLM cost) vs controller variants (LLM cost) vs real MDR (training cost).
- 2Wiki data available:
  - Dense: runtime ~0.23s, F1 = 0.3964, 0 LLM tokens.
  - Single-path (ST): runtime ~0.23s, F1 = 0.4139, 0 LLM tokens.
  - Source: `runs/2wiki-single-path-edge-ablation-s100-v1`

### Figure 3. Budget Sensitivity

**Status: BLOCKED on WS1/ACT-5 for IIRC data. Preliminary 2Wiki data available.**

- Compare `384` vs `512` (paper budgets) and optionally `128` vs `256` (diagnostic).
- Show which selector family scales with extra budget and which saturates early.
- 2Wiki preliminary (baseline retest, 20-case chunk, tokens-256 vs tokens-512):
  - Dense (top-1): F1 drops from 0.4199 (256) to 0.3206 (512) -- precision collapse with more budget.
  - This pattern may differ on IIRC.
  - Source: `runs/2wiki-baseline-retest-s100-v1`

### Figure 4. Precision vs Recall Tradeoff

**Status: BLOCKED on WS1/ACT-5 for IIRC data. Can be sketched from 2Wiki.**

- Emphasize whether controller variants buy precision at the cost of recall.
- 2Wiki data: single-path walk gains both recall (+0.09) and precision (+0.047) over dense at tokens-256.
- Source: `runs/2wiki-single-path-edge-ablation-s100-v1`

## Optional Tables

### Table 4. Main Claim Ablation

**Status: BLOCKED on WS1/ACT-5 and claim boundary decision.**

- Only build if the controller remains the paper center after real `MDR`.
- Candidate rows:
  - best non-LLM `single_path_walk`
  - current controller winner
  - one tightly scoped component ablation if needed (e.g., with vs without budget_fill).
- Data source: full-IIRC runs.

### Table 5. External Comparator Expansion

**Status: Not planned for current phase.**

- Only if the project later unlocks `GraphRetriever` or `HippoRAG`.
- Phase constraint: no expansion before real MDR is closed.

### Table 6. Edge-Scorer Ablation (2Wiki)

**Status: READY.**

- Purpose: supplementary / appendix table showing edge-scorer profiles on 2Wiki.
- Key comparison at tokens-128 (100 cases):

| Profile | `support_f1` | `recall` | `precision` |
| --- | --- | --- | --- |
| `st_future_heavy` (la2) | 0.3935 | 0.558 | 0.325 |
| `st_balanced` (la2) | 0.3846 | 0.545 | 0.318 |
| `st_direct_heavy` (la2) | 0.3561 | 0.505 | 0.293 |
| `st_balanced` (la1) | 0.3536 | 0.500 | 0.291 |
| `overlap_balanced` (la1) | 0.3520 | 0.500 | 0.289 |
| `anchor_overlap` (la1) | 0.3510 | 0.500 | 0.288 |

- Source: `runs/2wiki-single-path-edge-ablation-s100-v1`

## Source Tracking
- Every final table should be tied back to:
  - run id
  - dataset store path
  - selector set
  - exact budget set
- Final captions should reference run ids, not informal descriptions.
- Partial-IIRC runs (5,184-article store) must not be used as data sources for any table.
