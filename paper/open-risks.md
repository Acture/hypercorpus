# Hypercorpus Open Risks

## Critical Path Risks

### 1. Full-IIRC canonical store not yet landed (WS1/ACT-5)
- Risk: the paper has no main result table until the full-IIRC store (61,304 articles) is operational and all shortlist selectors have been rerun on it.
- Impact: **blocks** Table 1 (main comparison), Table 2 (hard-subset), all IIRC analysis figures, and the main claim boundary decision.
- Current state: partial store (5,184 articles) exists but is deprecated. Full-context fetch and conversion code is in place. Canonical store not yet landed under `dataset/iirc_full/`.
- Mitigation: WS1/ACT-5 is actively working on this. CTRL gate 1 tracks completion.

### 2. Real MDR not yet runnable (WS1/ACT-5)
- Risk: without a trained iterative retrieval baseline, the paper lacks a reviewer-acceptable external comparator. Claims C10 and C11 remain open.
- Impact: **blocks** the contributions paragraph, conclusion strength, and Table 1.
- Current state: `baselines/mdr` submodule exists, `baselines/mdr.py` wrapper exists. The full pipeline (export -> train -> index -> run) is not closed.
- Mitigation: CTRL gate 2 tracks completion. Do not freeze claim boundary until this is resolved.

### 3. Timing for submission windows
- Risk: `SIGIR 2026` is already closed. `CIKM 2026` has abstract deadline 2026-05-18 and full paper deadline 2026-05-25. If full-IIRC + real MDR are not closed by ~2026-04-15, CIKM 2026 becomes infeasible, and the project falls to `SIGIR 2027` (likely ~2027-01).
- Impact: 2-month delay reduces to ~3 weeks of writing time for CIKM 2026. Missing CIKM means a ~7-month wait for SIGIR 2027.
- Mitigation: prioritize WS1/ACT-5 closure. Treat CIKM 2026 as the actionable deadline, not SIGIR 2027.
- Source: `paper/venue-packaging.md` timing snapshot.

## Reviewer Risks

### 4. The story is still too narrow
- Risk: the paper looks like a tuned operating point instead of a stable method.
- Current state: 2Wiki calibration shows gains (+0.0175 F1) but the delta is small. The method needs to demonstrate larger gains on the harder IIRC surface to be convincing.
- Mitigation:
  - Full-IIRC becomes the main harder-dataset surface.
  - `2Wiki` is reduced to calibration.
  - Preliminary partial-IIRC signal (deprecated) showed `constrained_multipath` at F1 = 0.2850 at budget 384/512, but this needs reconfirmation on the full store.
- Current full-IIRC local ablations now give a complete 100-case non-controller surface: `dense` remains first at every budget, `mdr_light` is second, and the best `single_path_walk` row is a clear third (`256`: `0.3187 > 0.3103 > 0.2730`, `384`: `0.3140 > 0.3097 > 0.2709`, `512`: `0.3370 > 0.3242 > 0.2866` on `support_f1_zero_on_empty`). Those rows should stay in the paper as internal baselines / ablations rather than as the method headline.
- Remaining need: lock the full-IIRC paper-facing table.

### 5. Real MDR may weaken the current story
- Risk: once trained iterative retrieval is included, the controller may no longer be the headline win. If real MDR at budget 384/512 exceeds the best selector, the paper narrative must pivot.
- Mitigation:
  - Do not freeze the main claim boundary yet (CTRL gate 4).
  - The paper can still hold if the method matches or is close to MDR without task-specific training.
  - Fallback narrative: "comparable to trained MDR while being zero-shot and budget-explicit".
- Remaining need: close the real `MDR` comparison before drafting strong contribution language.

### 6. The paper may drift into a QA framing
- Risk: reviewers read the work as a QA paper and judge it against answer-generation systems.
- Mitigation:
  - Selector-first framing in `docs/paper-positioning.md`.
  - All primary metrics are support F1, not answer F1.
  - QA metrics are secondary sanity checks only.
- Remaining need: keep experiments, abstract, and related work aligned with retrieval / evidence discovery.

### 7. External baseline coverage is still limited
- Risk: reviewers ask immediately for `GraphRetriever` or `HippoRAG`.
- Mitigation:
  - Real `MDR` is the first external comparator -- a trained multi-hop retriever is a strong baseline.
  - Phase constraint: no expansion to GraphRetriever/HippoRAG before real MDR is closed.
  - The paper can acknowledge these as future work in Limitations.
- Remaining need: decide after full-IIRC whether the story is strong enough to justify more expansion.

### 8. Claim strength depends on full-IIRC numbers
- Risk: if gains on full-IIRC are smaller than on 2Wiki (or negative), the core claims weaken substantially.
- Current signal: partial-IIRC chunk-00001 showed `constrained_multipath` beating dense at budgets 384/512. But partial store is deprecated and the sample is small (20 cases). On the canonical 100-case non-controller full-IIRC surface, `dense` still leads `mdr_light` and every `single_path_walk` variant at all budgets.
- Mitigation:
  - The 2Wiki calibration data shows the method is at least coherent.
  - The latest full-IIRC local runs do not yet invalidate the controller hypothesis; they only show that the current non-controller local/path variants are not enough on their own.
  - The dense control is now operationally stable across the two full-IIRC run directories: `runs/iirc-local-full-v1` and `runs/iirc-dense-full-v1` agree exactly on the dense row for chunks `00000` through `00004` at budgets `256/384/512`.
  - The full-IIRC store (61,304 articles) makes the retrieval problem harder, which may actually help differentiate the walk-based selector from flat dense.
  - If gains are marginal, pivot to a "comparable under budget" narrative rather than "superior".

### 9. Historical IIRC run surfaces are not mutually comparable
- Risk: the repo contains multiple `runs/iirc-*` summaries that look paper-facing but disagree even when they point to the same store, the same 100-case source list (`runs/iirc-sample-s100-dense-v1/chunks/chunk-00000/evaluated_case_ids.txt`), and the same `chunk-00000` slice definition (`case_start: 0`, `case_limit: 20`). The mismatch is visible in `runs/iirc-controller-shortlist-v1/chunks/chunk-00000/summary.json` versus `runs/iirc-local-full-v1/chunks/chunk-00000/summary.json`, which report dense-256 `support_f1_zero_on_empty` values of `0.2200` and `0.3867`.
- Impact: cherry-picking across historical run directories would make the main IIRC table indefensible.
- Mitigation:
  - Treat all historical `runs/iirc-*` directories as directional or debugging artifacts unless they are explicitly designated canonical.
  - Build Table 1 only from one freshly rerun full-IIRC comparison surface.
  - Record the exact run id, store path, case-id source, and metric contract when the canonical table is locked.

## Operational Risks

### 10. Multi-agent work may scatter the project state
- Risk: progress becomes invisible because it lives inside separate worktrees.
- Mitigation:
  - Notion stores milestones and decisions.
  - Linear stores execution state and handoffs.
  - Every handoff includes: latest commit/branch, artifact path, current blocker, next step.
- Remaining need: keep repo files free of temporary status noise.

## Drafting Risks
- Do not let multiple agents edit the same core paper body file at once.
- Do not let historical partial-IIRC runs (5,184-article store) remain phrased as current evidence.
- Do not write "we beat X" before the exact comparison table exists.
- Do not begin prose writing before CTRL gate 4 (main claim boundary locked).

## Risk Priority Order
1. **Full-IIRC store landing** (blocks everything downstream) -- WS1/ACT-5
2. **Real MDR pipeline closure** (blocks claim boundary) -- WS1/ACT-5
3. **Timing for CIKM 2026** (hard deadline 2026-05-25) -- project-wide
4. **Historical IIRC run-surface drift** (blocks a defensible main table) -- CTRL/ACT-9 + WS1/ACT-5
5. **Claim strength on full-IIRC** (determines paper narrative) -- depends on 1 + 2
6. **External baseline coverage** (reviewer concern) -- future work if needed
