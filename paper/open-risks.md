# Hypercorpus Open Risks

## Reviewer Risks

### 1. The story is still too narrow
- Risk: the paper looks like a tuned operating point instead of a stable method.
- Current mitigation:
  - full-IIRC becomes the main harder-dataset surface
  - `2Wiki` is reduced to calibration
- Remaining need:
  - lock the full-IIRC paper-facing table

### 2. Real `MDR` may weaken the current story
- Risk: once trained iterative retrieval is included, the controller may no longer be the headline win.
- Current mitigation:
  - do not freeze the main claim boundary yet
- Remaining need:
  - close the real `MDR` comparison before drafting strong contribution language

### 3. The paper may drift into a QA framing
- Risk: reviewers read the work as a QA paper and judge it against answer-generation systems.
- Current mitigation:
  - selector-first framing in `docs/paper-positioning.md`
- Remaining need:
  - keep experiments, abstract, and related work aligned with retrieval / evidence discovery

### 4. External baseline coverage is still limited
- Risk: reviewers ask immediately for `GraphRetriever` or `HippoRAG`
- Current mitigation:
  - make real `MDR` the first external comparator
- Remaining need:
  - decide after full-IIRC whether the story is strong enough to justify more expansion

### 5. Multi-agent work may scatter the project state
- Risk: progress becomes invisible because it lives inside separate worktrees
- Current mitigation:
  - Notion stores milestones and decisions
  - Linear stores execution state and handoffs
- Remaining need:
  - keep repo files free of temporary status noise

## Drafting Risks
- Do not let multiple agents edit the same core paper body file at once.
- Do not let historical partial-IIRC runs remain phrased as current evidence.
- Do not write “we beat X” before the exact comparison table exists.
