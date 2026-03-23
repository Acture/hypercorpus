# Hypercorpus Venue Packaging

## Default Venue
- Default target: `SIGIR`
- Reason:
  - the current story is strongest as retrieval and evidence discovery under budget constraints

## Timing Snapshot
- Snapshot date: `2026-03-23`
- `CIKM 2026` is the nearest official full-paper window that is still open:
  - abstract deadline: `2026-05-18`
  - full paper deadline: `2026-05-25`
  - notification: `2026-08-06`
  - main conference: `2026-11-09` to `2026-11-11`
- `KDD 2026` research cycle is already closed:
  - abstract deadline: `2026-02-01`
  - full paper deadline: `2026-02-08`
  - notification: `2026-05-16`
  - conference: `2026-08-09` to `2026-08-13`
- `WWW 2026` research track is already closed:
  - abstract deadline: `2025-09-30`
  - full paper deadline: `2025-10-07`
  - notification: `2026-01-13`
  - conference: `2026-07-01` to `2026-07-03`
- `SIGIR 2026` is already in the closed cycle:
  - conference: `2026-07-20` to `2026-07-24`
  - keep `SIGIR` as the default packaging target, but treat `CIKM 2026` as the nearest realistic live submission window unless `SIGIR 2027` dates are published in time

## CCF And Fit Snapshot
- `SIGIR`: `CCF A`
  - current best fit for the project's retrieval and evidence-discovery story
- `KDD`: `CCF A`
  - viable only if the final paper clears a stronger algorithmic-generalization bar
- `WWW`: `CCF A`
  - strongest only if hyperlink navigation becomes the dominant narrative
- `CIKM`: `CCF B`
  - the most pragmatic near-term submission window if the project is not ready for `SIGIR 2027`

## 2027 Planning Assumption
- Treat these as planning assumptions, not official deadlines.
- If the cycle remains similar:
  - `WWW 2027`: likely around `2026-09/10`
  - `SIGIR 2027`: likely around `2027-01`
  - `KDD 2027`: likely around `2027-02`
  - `CIKM 2027`: likely around `2027-05`
- Practical interpretation:
  - `WWW 2027` is not a fallback after missing `SIGIR 2027`; it likely arrives earlier.
  - `CIKM 2027` is the timeline-friendly fallback if the project is not ready by the likely `SIGIR 2027` window.

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
- If the project needs the nearest realistic 2026 submission window, prefer `CIKM 2026`.
- If the project misses the likely `SIGIR 2027` window, treat `CIKM 2027` as the default fallback rather than `WWW 2027`.
- Do not optimize for multiple venues at once during the current closure phase.
