# Docs Guide

This directory is organized around one paper-facing question:

`how should webwalker be argued as query-time budgeted subgraph/corpus discovery over naturally linked corpora`

## Reading Order

1. `paper-positioning.md`
2. `phase-decisions.md`
3. `current-implementation.md`
4. `corpus-selection-literature.md`
5. `literature-map.md`
6. `mc/venue-strategy.md`

## Layering

### Thesis Layer

- `paper-positioning.md`: the canonical narrative for problem framing, novelty boundaries, and claims that are safe to write in a paper.
- `corpus-selection-literature.md`: the paper-facing related-work guide and baseline order.

### Evidence Layer

- `phase-decisions.md`: the canonical experiment-facing record for supported findings, missing ablations, and not-yet-compared systems.

### Implementation Layer

- `current-implementation.md`: what exists in code now, what is still missing, and what the current implementation can honestly support.

### Bibliography Layer

- `literature-map.tsv`: canonical machine-readable bibliography.
- `literature-map.md`: human-readable annotated inventory derived from the bibliography.
- `corpus-selection-literature.tsv`: paper-oriented export view, not source of truth.

### Memory Layer

- `mc/`: short working notes for strategy decisions that should be remembered but should not dominate the paper-facing narrative.

## Source-Of-Truth Rules

- Claim support belongs in `phase-decisions.md`, not scattered across narrative docs.
- Problem framing belongs in `paper-positioning.md`, not in ad hoc notes.
- Bibliography truth lives in `literature-map.tsv`.
- Venue strategy, conference fit, and writing-angle decisions belong in `mc/`.

## Current Canonical Framing

Use this repo as a `selector-first`, `budget-aware`, `query-time subgraph/corpus discovery` project.

- Primary output: selected evidence set or induced subgraph under an explicit token budget.
- Primary evaluation object: discovery quality before downstream reasoning.
- Secondary evaluation object: downstream QA quality as a sanity check, not the main identity of the system.
