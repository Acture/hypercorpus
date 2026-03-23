# Hypercorpus Tables And Figures

## Main Tables

### Table 1. Full-IIRC Main Comparison
- Dataset: full-IIRC canonical store
- Budgets: `384`, `512`
- Rows:
  - `dense`
  - `mdr_light`
  - best non-LLM `single_path_walk`
  - current controller winner
  - real `MDR`
- Columns:
  - support F1 on all cases
  - precision
  - recall
  - budget utilization
  - runtime
  - selector-side token cost

### Table 2. Full-IIRC Hard-Subset Comparison
- Same selector rows as Table 1.
- Only harder slices.
- Goal: show whether gains are concentrated on the difficult portion rather than only all-case averages.

### Table 3. 2Wiki Calibration Summary
- Purpose: calibration only, not the main paper result.
- Show:
  - dense control
  - best non-LLM `single_path_walk`
  - repo-native baseline ordering

## Analysis Figures

### Figure 1. Cost-Quality Curve
- X-axis: selector-side cost or runtime
- Y-axis: support F1 on all cases
- Use the full-IIRC comparison surface

### Figure 2. Budget Sensitivity
- Compare `384` vs `512`
- Show which selector family scales with extra budget and which saturates early

### Figure 3. Precision vs Recall Tradeoff
- Emphasize whether controller variants buy precision at the cost of recall
- Use full-IIRC only

## Optional Tables

### Table 4. Main Claim Ablation
- Only build if the controller remains the paper center after real `MDR`
- Candidate rows:
  - best non-LLM `single_path_walk`
  - current controller winner
  - one tightly scoped component ablation if needed

### Table 5. External Comparator Expansion
- Only if the project later unlocks `GraphRetriever` or `HippoRAG`
- Not part of the current closure phase

## Source Tracking
- Every final table should be tied back to:
  - run id
  - dataset store path
  - selector set
  - exact budget set
- Final captions should reference run ids, not informal descriptions.
