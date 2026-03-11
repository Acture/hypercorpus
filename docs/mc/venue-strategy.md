# Venue Strategy

Last updated: `2026-03-12`

## Canonical Paper Form

Write `webwalker` as:

`query-time budgeted subgraph/corpus discovery over naturally linked corpora`

Do not write it primarily as:

- a full-stack RAG system
- an answer-generation paper
- an agentic web-navigation paper
- an LLM-edge-scoring novelty paper

## A-Class Venue Fit

### SIGIR

Current strongest fit.

- Most natural home if the paper remains retrieval-centric.
- Best if the main story is budgeted evidence discovery, selector policies, and retrieval tradeoffs.
- Least rewriting pressure relative to the current repo.

### KDD

Strong option if the paper is made more algorithmic and more general.

- Best if the contribution is presented as a general discovery problem over linked corpora rather than as a QA-specific retrieval trick.
- Requires stronger emphasis on objective design, algorithmic mechanism, tradeoff analysis, and generalization across corpora or datasets.
- Reviewers are more likely than SIGIR reviewers to ask whether the method is genuinely a new algorithm rather than a retrieval engineering combination.

### WWW

Strong option if the paper foregrounds the natural hyperlink graph and web-style corpora.

- Best if query-conditioned navigation over linked documents is the center of the story.
- Works well if the paper leans into hyperlink semantics, graph traversal, and web-like corpora structure.

### ACL

Possible but not the native home.

- Viable if the paper is packaged around multi-hop QA, grounding, or language-mediated retrieval decisions.
- Less natural if the core contribution is the retrieval algorithm itself rather than a new language model or reasoning method.

## Current Strategic Decision

Keep the canonical project framing venue-agnostic but algorithm-forward:

- problem: budgeted subgraph/corpus discovery
- mechanism: natural-link traversal with local link semantics
- objective: discovery quality under explicit budget
- evaluation: evidence metrics first, answer metrics second

This framing can still be specialized later:

- `SIGIR` version: retrieval-first
- `KDD` version: algorithm-and-objective-first
- `WWW` version: hyperlink-and-web-graph-first
- `ACL` version: QA-and-grounding-first

## What Must Be True To Sell The KDD Story

- The method must read as a distinct algorithm, not only as a benchmark-specific operating point.
- Core components such as `budget_fill_relative_drop` must have clean ablations and a clear role in the objective.
- Results should extend beyond a single small `2Wiki` slice.
- The paper should report cost-quality curves, budget sensitivity, and stronger baseline coverage.
- The paper should emphasize general linked-corpus discovery rather than only Wikipedia multi-hop QA.

## Current Risk Notes

- Current public evidence is still narrow: `phase-decision-30` on `2Wiki dev`, `30` cases, `256` tokens.
- `MDR`, `GraphRetriever`, and `HippoRAG` are still not direct completed comparisons in this repo.
- The present docs support an algorithmic paper direction, but not yet a broad superiority claim.
