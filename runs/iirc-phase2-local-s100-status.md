> **Deprecated (2026-03-23):** These results were produced with the partial IIRC store (5,184 articles). The full-context store (61,304 articles) is now canonical. Do not cite these numbers as paper evidence.

# IIRC Phase 2 Local S100 Status

Last updated: `2026-03-16`

## Current Run Family

- run: `runs/iirc-controller-shortlist-v1`
- shared case ids:
  - `runs/iirc-sample-s100-dense-v1/chunks/chunk-00000/evaluated_case_ids.txt`
- budgets:
  - `256`
  - `384`
  - `512`
- selectors:
  - `top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop`
  - `top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop`
  - `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop`
  - `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm_controller__lookahead_2`
  - `top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2`

## Chunk Status

- completed:
  - `chunk-00001`
- not started:
  - `chunk-00000`
  - `chunk-00002`
  - `chunk-00003`
  - `chunk-00004`

Completed chunk output:

- `runs/iirc-controller-shortlist-v1/chunks/chunk-00001/summary.json`
- `runs/iirc-controller-shortlist-v1/chunks/chunk-00001/summary_rows.csv`
- `runs/iirc-controller-shortlist-v1/chunks/chunk-00001/study_comparison_rows.csv`
- `runs/iirc-controller-shortlist-v1/chunks/chunk-00001/subset_comparison_rows.csv`

## Current Read

- `tokens-256`
  - best selector:
    - `top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop`
  - `support_f1_zero_on_empty = 0.2200`
- `tokens-384`
  - best selector:
    - `top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2`
  - `support_f1_zero_on_empty = 0.2850`
- `tokens-512`
  - best selector:
    - `top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2`
  - `support_f1_zero_on_empty = 0.2850`

Interpretation:

- keep the non-LLM sentence-transformer `single_path_walk` selector as the tight-budget mainline
- treat `constrained_multipath + link_context_llm_controller` as the stronger moderate-budget extension
- de-prioritize `single_path_walk + link_context_llm_controller`

## Next Required Work

1. Run the remaining shortlist chunks on the same shared `100`-case slice.
2. Merge the chunked results into one `IIRC` replay result.
3. Add the controller ablation:
   - constrained multipath controller
   - constrained search without `choose_two` / backtrack
   - best non-LLM sentence-transformer `single_path_walk`
4. Export cost-quality views that include selector token cost and selected evidence length.

## Command Template

```bash
uv run webwalker-cli experiments run-iirc-store \
  --store dataset/iirc/store \
  --output-root /Users/acture/repos/hypercorpus/runs \
  --exp-name iirc-controller-shortlist-v1 \
  --split dev \
  --case-ids-file /Users/acture/repos/hypercorpus/runs/iirc-sample-s100-dense-v1/chunks/chunk-00000/evaluated_case_ids.txt \
  --chunk-size 20 \
  --chunk-index <INDEX> \
  --selectors "top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop,top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop,top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop,top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm_controller__lookahead_2,top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2" \
  --token-budgets 256,384,512 \
  --selector-provider anthropic \
  --selector-model claude-haiku-4-5-20251001 \
  --selector-api-key-env ANTHROPIC_API_KEY \
  --selector-cache-path runs/iirc-controller-shortlist-v1/selector-cache.jsonl \
  --sentence-transformer-model multi-qa-MiniLM-L6-cos-v1 \
  --sentence-transformer-cache-path ~/.cache/webwalker/embeddings.sqlite3 \
  --sentence-transformer-device mps \
  --no-e2e \
  --no-export-graphrag-inputs \
  --restart
```
