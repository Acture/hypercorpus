# Venue Strategy Notes

Purpose: venue-specific packaging note for `webwalker`.
Canonical for: per-venue narrative deltas, fit criteria, and current submission risks.
Not for / See also: shared framing lives in `../paper-positioning.md`; completed evidence lives in `../phase-decisions.md`; active plan lives in `../next-phase-experiments.md`.

Last updated: `2026-03-12`

This note assumes the canonical framing from `paper-positioning.md` and only records how that framing changes by venue.

## Current Read

- `SIGIR` is the default strongest fit while the story remains retrieval-centric.
- `KDD` is attractive if the paper becomes more explicitly algorithmic and general across linked corpora.
- `WWW` is attractive if the hyperlink graph and web-style navigation story move to the foreground.
- `ACL` is possible, but it is not the native home unless the paper is repackaged around QA and grounding.

## Venue Deltas

### SIGIR

- Most natural home if the paper stays focused on budgeted evidence discovery and retrieval tradeoffs.
- Requires the least reframing relative to the current repo and docs.

### KDD

- Best if the contribution is presented as a general discovery problem over linked corpora rather than as a QA-specific retrieval trick.
- Needs more emphasis on objective design, algorithmic mechanism, tradeoff analysis, and cross-corpus generality.
- Reviewers are more likely to ask whether the method is genuinely a distinct algorithm rather than a retrieval engineering combination.

### WWW

- Best if query-conditioned navigation over linked documents is the center of the story.
- Works well if the paper leans into hyperlink semantics, graph traversal, and web-like corpora structure.

### ACL

- Viable if the paper is packaged around multi-hop QA, grounding, or language-mediated retrieval decisions.
- Less natural if the contribution remains primarily about retrieval and discovery control.

## Extra Bar For KDD

- The method must read as a distinct algorithm, not only as a benchmark-specific operating point.
- Core components such as `budget_fill_relative_drop` must have clean ablations and a clear role in the objective.
- Results should extend beyond a single small `2Wiki` slice.
- The paper should report cost-quality curves, budget sensitivity, and stronger baseline coverage.
- The paper should emphasize general linked-corpus discovery rather than only Wikipedia multi-hop QA.

## Current Risk Notes

- Current public evidence is still narrow: `phase-decision-30` on `2Wiki dev`, `30` cases, `256` tokens.
- `MDR`, `GraphRetriever`, and `HippoRAG` are still not direct completed comparisons in this repo.
- The present docs support an algorithmic paper direction, but not yet a broad superiority claim.
