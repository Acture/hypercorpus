# Hypercorpus Venue Packaging

## Default Venue
- Default target: `SIGIR`
- Reason:
  - the current story is strongest as retrieval and evidence discovery under budget constraints

## SIGIR Package
- Lead with:
  - evidence discovery over naturally linked corpora
  - token-budgeted selector behavior
  - dense control versus iterative and graph-adjacent retrieval baselines
- Keep QA secondary.
- Keep the paper focused on retrieval tradeoffs, compact evidence quality, and cost-quality behavior.

## KDD Alternative
- Only switch if the final evidence supports a stronger algorithmic-generality story.
- Additional bar:
  - cleaner algorithmic identity
  - stronger ablations
  - broader comparator coverage
  - stronger cross-corpus generality

## WWW Alternative
- Only switch if hyperlink navigation and web-like linked structure become the strongest narrative hook.
- Additional bar:
  - stronger emphasis on link semantics and navigation over naturally linked corpora
  - less emphasis on benchmark-specific retrieval packaging

## Venue Decision Rule
- Stay with `SIGIR` unless the full-IIRC + real `MDR` results clearly justify a stronger algorithmic or hyperlink-navigation framing.
- Do not optimize for multiple venues at once during the current closure phase.
