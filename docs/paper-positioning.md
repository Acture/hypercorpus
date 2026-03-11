# Paper Positioning

## Problem Framing

`webwalker` targets a narrow problem:

`query-time budgeted subgraph/corpus discovery for naturally linked corpora before downstream RAG or GraphRAG`

The paper should not be framed as a new full-stack RAG system or primarily as a QA model. The core question is whether a lightweight selector can discover a compact evidence set or induced subgraph under a fixed token budget by exploiting natural hyperlink structure and local link semantics.

## One-Sentence Thesis

For naturally linked corpora, a lightweight, training-free, budget-aware discovery algorithm can recover better compact evidence sets than flat dense top-k baselines, without requiring eager graph construction or a learned retrieval policy.

## Paper Form

Write the project as an algorithmic discovery paper.

- Primary output: a selected corpus or induced subgraph under an explicit token budget.
- Primary objective: evidence discovery quality before downstream reasoning.
- Primary comparison class: dense and iterative retrieval baselines, not full answer-generation systems.
- QA remains an evaluation context, not the core identity of the method.

## Main Claims We Can Defend Now

These claims are currently `supported on phase-decision-30`, a completed `2Wiki dev / 30 cases / token budget 256 / no-e2e` phase-decision run.

- Budgeted subgraph/corpus discovery over natural hyperlinks is a coherent problem formulation for pre-RAG retrieval.
- In the current phase sample, a dense-seeded hyperlink-local walk outperforms flat dense top-k selection on all-case support F1.
- Under a fixed `256`-token budget, wider and deeper graph search can hurt the operating point by collapsing precision.
- Budget-aware fill is an effective component in the current system and should be treated as a real part of the paper contribution.

## Claims We Should Not Make Yet

These claims are not supported by current evidence and should stay out of the main paper story for now.

- We do not yet outperform `MDR`.
- We do not yet outperform `GraphRetriever`.
- We do not yet outperform `HippoRAG`.
- We cannot yet claim that LLM edge scoring is the main source of the gain.
- We cannot yet claim end-to-end QA superiority.
- We cannot yet claim the current result generalizes beyond this task, budget, or sample size.

## Closest Related Systems And How They Differ

### MDR

`MDR` is the main strong baseline to compare against next. It represents iterative dense retrieval without relying on natural hyperlink structure. If `webwalker` cannot beat or clearly complement this family, the paper story remains incomplete.

### GraphRetriever

`GraphRetriever` is the closest learned ancestor in spirit. It is a trained reasoning-path retriever over a paragraph graph. `webwalker` differs by targeting a training-free, query-time selector over natural hyperlinks rather than a learned graph retriever. It is not a drop-in backend for the current framework.

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

- natural hyperlink link-context selection
- fixed-budget corpus assembly
- selector-first evaluation with all-case evidence metrics

This combination is the current strongest candidate for the paper's main algorithmic systems contribution.

### Candidate New Components

- `budget_fill_relative_drop`

This is the cleanest candidate for a named new component because it is both distinctive in the current design and clearly effective in the current phase-decision results.

### Candidate But Not Yet Isolated

- structured LLM edge scorer

This is promising as a modular design, but current experiments do not isolate it cleanly enough to claim it as the main source of the observed gain.
