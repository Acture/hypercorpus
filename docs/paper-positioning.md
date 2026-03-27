# Paper Positioning

Purpose: canonical paper-facing framing for `hypercorpus`.
Canonical for: problem statement, thesis, stable research questions, safe claims, and novelty boundaries.
Not for / See also: completed empirical support lives in `phase-decisions.md`; active experiment sequencing lives in `next-phase-experiments.md`; code capability lives in `current-implementation.md`; venue-specific packaging lives in `notes/venue-strategy.md`.

## Problem Framing

`hypercorpus` targets a narrow problem:

`query-time budgeted subgraph/corpus discovery for naturally linked corpora before downstream RAG or GraphRAG`

The paper should not be framed as a new full-stack RAG system or primarily as a QA model. The core question is whether a lightweight selector can discover a compact evidence set or induced subgraph under a fixed token budget by exploiting natural hyperlink structure and local link semantics.

## Why This Problem Exists

Three stable pressures motivate the project:

- Full-stack GraphRAG pipelines often rely on eager global structuring, which is expensive and discards the fact that many corpora already contain usable link structure.
- Flat dense top-k is a weak corpus selector for bridge-heavy questions because it treats documents as isolated items and often misses support pages that are only visible after one step of navigation.
- Natural hyperlinks already contain semantics through anchor text and local sentence context, but many retrieval baselines use topology alone or ignore the links entirely.

## One-Sentence Thesis

For naturally linked corpora, the paper studies whether a dense-started semantic controller can assemble better compact evidence sets than flat dense evidence assembly under the same token budget, without requiring eager graph construction or per-dataset fine-tuning.

## Paper Form

Write the project as an algorithmic discovery paper.

- Primary output: a selected corpus or induced subgraph under an explicit token budget.
- Primary objective: evidence discovery quality before downstream reasoning.
- Primary comparison class: dense control plus internal ablation baselines, with real `MDR` as the main external comparator, not full answer-generation systems.
- QA remains an evaluation context, not the core identity of the method.

## Stable Research Questions

These are the durable questions that should survive changes in the current operating point:

1. Can link-semantic corpus selection avoid eager whole-corpus graph construction while staying competitive on evidence recovery?
2. Can a query-time selector recover more support and bridge evidence than flat dense top-k under the same token budget?
3. Does explicit budget control over hops, nodes, and tokens produce a better recall-cost tradeoff than purely dense or iterative retrieval?
4. Which selector policy family best uses natural hyperlink structure: local walk, branchy search, or controlled local propagation?

## Main Claims We Can Defend Now

These claims are currently `supported on phase-decision-30`, a completed `2Wiki dev / 30 cases / token budget 256 / no-e2e` phase-decision run.

- Budgeted subgraph/corpus discovery over natural hyperlinks is a coherent problem formulation for pre-RAG retrieval.
- In the current phase sample, dense-seeded hyperlink-local selection is a coherent operating point and the right level of comparison against flat dense evidence assembly.
- Under a fixed `256`-token budget, wider and deeper graph search can hurt the operating point by collapsing precision.
- Budget-aware fill is an effective component in the current system and should be treated as a real part of the paper contribution.

## Claims We Should Not Make Yet

These claims are not supported by current evidence and should stay out of the main paper story for now.

- We do not yet compare directly against trained `MDR`.
- We do not yet outperform `GraphRetriever`.
- We do not yet outperform `HippoRAG`.
- We cannot yet claim that zero-shot semantic control is the main source of the gain.
- We cannot yet claim end-to-end QA superiority.
- We cannot yet claim the current result generalizes beyond this task, budget, or sample size.

## Closest Related Systems And How They Differ

### MDR

`MDR` remains the main external comparator because it represents trained iterative dense retrieval without relying on natural hyperlink structure. The repo-native `mdr_light` row is only an internal baseline; it is not the paper's method and should not be presented as such.

### GraphRetriever

`GraphRetriever` is the closest learned ancestor in spirit. It is a trained reasoning-path retriever over a paragraph graph. `hypercorpus` differs by targeting a dense-started, query-time selector over natural hyperlinks rather than a task-specific learned graph retriever. It is not a drop-in backend for the current framework.

### HippoRAG

`HippoRAG` is an important graph-retrieval comparator, especially for comparing natural hyperlink walks against graph-based neighborhood retrieval. It is not the main head-to-head for the current phase because the present claim is still selector-first rather than full memory-RAG replacement.

## Novelty Map

### Standard Design Space

- lexical seeds
- dense seeds
- beam or A-star graph search
- PPR-style graph priors
- LLM-guided retrieval as a broad category

These should be written as established ingredients, not as paper novelty.

### Current Effective Core Combination

- dense-started natural hyperlink link-context selection
- fixed-budget corpus assembly
- selector-first evaluation with all-case and subset-aware evidence metrics

This combination should now be read more narrowly: `dense` is the control, the best non-LLM `single_path_walk` row is an ablation, and the actual main-method candidate is the controller-guided multi-path family.

### Candidate New Components

- `budget_fill_relative_drop`

This is the cleanest candidate for a named new component because it is both distinctive in the current design and clearly effective in the current phase-decision results.

### Candidate But Not Yet Isolated

- structured semantic controller over hyperlink decisions

This is promising as a modular design, but current experiments do not isolate it cleanly enough to claim it as the main source of the observed gain.
