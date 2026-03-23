# Hypercorpus Paper Outline

## Working Title
Dense-Started Budgeted Evidence Discovery over Naturally Linked Corpora

## Default Package
- Default venue: `SIGIR`
- Paper type: retrieval / evidence-discovery paper
- Core identity: selector-first, not QA-first

## One-Sentence Thesis
For naturally linked corpora, a dense-started, budgeted selector can assemble stronger compact evidence sets than flat dense assembly and repo-native iterative dense baselines, while staying lighter than eager graph construction or dataset-specific training.

## Abstract Skeleton
1. Motivate the problem as budgeted evidence discovery before downstream RAG or GraphRAG.
2. State the method as dense-started constrained subgraph selection under explicit token budgets.
3. Emphasize natural hyperlink structure plus local link semantics.
4. Summarize the main comparison surface: dense, repo-native iterative dense baselines, and real `MDR`.
5. Close with the main empirical tradeoff: evidence quality under fixed budgets.

## Section Skeleton

### 1. Introduction
- Why flat dense top-k is weak for bridge-heavy evidence discovery.
- Why eager graph construction is too expensive or mismatched for naturally linked corpora.
- Why token-budgeted evidence assembly is the right problem surface.
- Contributions paragraph should stay conservative until the real `MDR` comparison is locked.

### 2. Problem Formulation
- Input: query plus naturally linked corpus.
- Output: compact evidence set or induced subgraph under a token budget.
- Objective: maximize support recovery quality under budget and cost constraints.
- Evaluation is selector-first; QA is secondary.

### 3. Method
- Stage 1: dense start prior.
- Stage 2: constrained hyperlink-local evidence assembly.
- Budget accounting and token-aware stopping.
- Family framing:
  - dense control
  - repo-native iterative dense baselines
  - hyperlink-local walk
  - controller-guided constrained multipath

### 4. Experimental Setup
- Datasets:
  - `2Wiki` for calibration only
  - `IIRC` as the main harder-dataset judgment surface
- Metrics:
  - support F1 on all cases
  - precision / recall
  - budget utilization and adherence
  - selector runtime / token cost
- Comparison class:
  - dense
  - `mdr_light`
  - best non-LLM `single_path_walk`
  - current controller winner
  - real `MDR`

### 5. Main Results
- Main table on full-IIRC, budgets `384` and `512`.
- Subset table for harder slices.
- Cost-quality view.
- Secondary 2Wiki calibration summary only.

### 6. Analysis
- Precision / recall tradeoff.
- Budget sensitivity.
- When controller behavior helps versus hurts.
- Why full-IIRC replaces partial-IIRC as the only valid IIRC judgment surface.

### 7. Related Work
- Multi-hop evidence retrieval and reasoning-path retrieval.
- Iterative dense retrieval and decomposition-guided retrieval.
- Graph retrieval / GraphRAG baselines.
- Web or hyperlink navigation as adjacent inspiration.

### 8. Limitations
- Current story is evidence-discovery-first, not end-to-end QA superiority.
- Real `MDR` is the first external comparator; broader external coverage may remain future work.
- Results may still be budget- and corpus-dependent.

### 9. Conclusion
- Restate the problem and operating point.
- Emphasize the selector-first contribution.
- Keep any broad claim bounded by the final IIRC evidence surface.

## Writing Gates
- Do not finalize the introduction contribution list before the full-IIRC + real `MDR` comparison is locked.
- Do not write broad superiority language before the main claim boundary is frozen.
- Do not let the paper body drift into a QA-system framing.
