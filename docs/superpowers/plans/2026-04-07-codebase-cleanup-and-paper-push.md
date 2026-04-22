# Codebase Cleanup + Paper Push Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up accumulated git and codebase debt, then unblock and accelerate the paper closure pipeline toward CIKM 2026 (deadline 2026-05-25, 48 days out).

**Architecture:** Two phases: (1) git hygiene and uncommitted-change triage — fast, mechanical, no risk to experiments; (2) paper pipeline assessment and unblocking — strategic decisions on MDR feasibility, IIRC controller pre-gate, and venue go/no-go.

**Tech Stack:** Python 3.13+, uv, git, pytest, ruff

---

## Phase 1: Codebase Cleanup

### Task 1: Triage and commit uncommitted changes

The working tree has 14 modified files (+947/-246 lines) across core selector, copilot, experiments, answering, controller, and test code. These need to either be committed or stashed with intent.

**Files:**
- Modified: `src/hypercorpus/answering.py`, `src/hypercorpus/controller_exposure.py`, `src/hypercorpus/copilot.py`, `src/hypercorpus/experiments.py`, `src/hypercorpus/selector.py`, `src/hypercorpus/selector_llm.py`
- Modified: `tests/test_cli_experiments.py`, `tests/test_eval.py`, `tests/test_experiments.py`, `tests/test_selector_llm.py`, `tests/test_subgraph.py`
- Modified: `pyproject.toml`, `uv.lock`, `docs/next-phase-experiments.md`

- [ ] **Step 1: Review the diff to understand what the changes do**

```bash
git diff --stat HEAD
git diff src/hypercorpus/copilot.py | head -80
git diff src/hypercorpus/selector_llm.py | head -80
git diff src/hypercorpus/answering.py | head -60
```

Purpose: The recent commits (`b68148d` through `6e91aab`) refactored controller decisions and switched to Copilot SDK. These uncommitted changes likely continue that trajectory. Understand whether they form a coherent commit or need splitting.

- [ ] **Step 2: Run tests to verify the uncommitted changes are clean**

```bash
uv run pytest -q
```

Expected: all tests pass. If not, the changes need fixing before committing.

- [ ] **Step 3: Run lint**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean. Fix any issues.

- [ ] **Step 4: Commit the changes if tests and lint pass**

Group into logical commits based on what the diff review reveals. Likely groupings:
- One commit for copilot/selector_llm/controller changes (Copilot SDK continuation)
- One commit for answering.py changes
- One commit for test updates
- Or a single commit if they're all part of the same refactor

```bash
git add src/hypercorpus/copilot.py src/hypercorpus/selector_llm.py src/hypercorpus/controller_exposure.py src/hypercorpus/selector.py src/hypercorpus/experiments.py src/hypercorpus/answering.py
git add tests/test_cli_experiments.py tests/test_eval.py tests/test_experiments.py tests/test_selector_llm.py tests/test_subgraph.py
git add pyproject.toml uv.lock docs/next-phase-experiments.md
git commit -m "<message based on diff review>"
```

### Task 2: Clean up stale git branches

The repo has 10 orphaned `worktree-agent-*` branches and several `codex/*` branches that appear unused. Also 1 stash and a prunable worktree.

- [ ] **Step 1: Verify worktree-agent branches are fully merged or abandoned**

```bash
for branch in (git branch --list 'worktree-agent-*'); git log --oneline -1 $branch; end
```

Check if any have unmerged work. If all are behind or at `master`, they're safe to delete.

- [ ] **Step 2: Verify codex/* branches**

```bash
for branch in codex/analyze-uv-run-experiment-trend codex/baseline codex/metric codex/pause codex/references; git log --oneline -3 $branch; end
```

Check for unmerged work.

- [ ] **Step 3: Check the stash**

```bash
git stash show stash@{0}
```

Determine if the stash on `codex/baseline` contains anything valuable.

- [ ] **Step 4: Clean up the prunable worktree**

```bash
git worktree remove .claude/worktrees/elastic-bell --force
```

This is from the old "webwalker" era and is already marked prunable.

- [ ] **Step 5: Delete confirmed-safe branches**

```bash
# Only after confirming no unmerged work:
git branch -d worktree-agent-a15961fc worktree-agent-a350f2b1 worktree-agent-a45862ed worktree-agent-a552f3f3 worktree-agent-a5921582 worktree-agent-a8a3c0ff worktree-agent-a9006984 worktree-agent-a92a9663 worktree-agent-ab455d42 worktree-agent-affeaf90
# If unmerged but abandoned, use -D instead of -d after confirming with user
```

- [ ] **Step 6: Delete confirmed-safe codex branches and WS branches already merged**

```bash
# ws2/deprecate-partial-iirc-record-2wiki-baseline — WS2 is Done in Linear
git branch -d ws2/deprecate-partial-iirc-record-2wiki-baseline
git branch -d ws2/final-cleanup
# ws3/paper-skeleton-enhancement — check if merged
git log --oneline -1 ws3/paper-skeleton-enhancement
# codex/* — delete after confirming with user
```

- [ ] **Step 7: Drop stash if confirmed valueless**

```bash
git stash drop stash@{0}
```

### Task 3: Verify no dead code or unused files in src/

- [ ] **Step 1: Check for unused imports and dead code**

```bash
uv run ruff check . --select F401,F841
```

- [ ] **Step 2: Check if `utils/fetch_wiki.py` and `utils/jsonl2csv.py` are still used**

```bash
rg "fetch_wiki|jsonl2csv" src/ tests/
```

If unused, delete them.

- [ ] **Step 3: Check if `src/hypercorpus/type.py` (4 lines) is still needed**

```bash
cat src/hypercorpus/type.py
rg "from hypercorpus.type import\|from hypercorpus import type" src/ tests/
```

---

## Phase 2: Paper Pipeline — Status Assessment and Unblocking

### Task 4: Assess MDR pipeline feasibility for CIKM 2026

The real MDR pipeline (A1.2 → A1.3 → A1.4 → A1.5) is the single biggest blocker. All 4 tasks are in Backlog/Todo. The full pipeline is serial: export → train → index → run. Each step requires GPU compute and debugging time.

**This is a decision task, not an implementation task.**

- [ ] **Step 1: Assess MDR infrastructure readiness**

```bash
ls baselines/mdr/ 2>/dev/null | head -20
cat baselines/mdr.py | head -40
```

Check: is the MDR submodule actually usable? Does the wrapper code exist for export/train/index/run? What's missing?

- [ ] **Step 2: Estimate MDR timeline realistically**

The 4 serial MDR steps were originally due 04-01 through 04-10. It's now 04-07 and none have started. Key questions:
- Do you have GPU access for training?
- How long does MDR training typically take on IIRC-scale data?
- Is there a shortcut (pretrained MDR checkpoint from the original paper)?

- [ ] **Step 3: Make the MDR go/no-go decision**

Three options:
1. **Full MDR**: close the real pipeline. Requires ~2 weeks. Risk: may not finish before CIKM deadline leaves insufficient writing time.
2. **Pretrained MDR proxy**: use a pretrained MDR checkpoint (if available from the original paper's release) and run inference only, skipping train. Faster but less rigorous.
3. **Drop MDR, pivot narrative**: reframe the paper without a trained external comparator. Use `mdr_light` as the iterative retrieval baseline. Weaker but unblocks everything immediately. The paper narrative becomes "zero-shot budget-aware selection vs. repo-native iterative baselines" rather than "vs. trained MDR".

**Recommendation:** Option 2 if pretrained checkpoints exist, otherwise Option 3. The CIKM 2026 deadline is 48 days away. With real MDR still 4 serial steps from complete, the risk of missing CIKM while chasing MDR is high.

### Task 5: Unblock the IIRC controller pre-gate

Per `docs/next-phase-experiments.md`, before the paper-facing mainline begins, a short implementation-validation gate is needed:
1. 1-case keycheck on the two `hop_adaptive` controller selectors
2. 3-case smoke test
3. 20-case pilot comparing `dense`, `mdr_light`, and the two `hop_adaptive` selectors

This is the fastest path to real IIRC signal.

- [ ] **Step 1: Verify the controller selectors work on the full-IIRC store**

Run a 1-case keycheck:

```bash
uv run hypercorpus run-iirc-store \
    --store dataset/iirc/store \
    --selector "top_1_seed__sentence_transformer__hop_adaptive__single_path_walk__link_context_llm_controller__lookahead_2" \
    --budget-tokens 384 \
    --limit 1 \
    --output runs/iirc-controller-pregate-1case-v1
```

- [ ] **Step 2: If keycheck passes, run 3-case smoke**

- [ ] **Step 3: If smoke passes, run 20-case pilot with the shortlist**

This pilot is the fastest way to get a IIRC controller signal and determine whether the controller story has legs on the full store.

### Task 6: Update Linear and paper docs to reflect current reality

Many Linear due dates have passed. The project status needs to honestly reflect where things stand.

- [ ] **Step 1: Update Linear due dates**

Recalculate dates backwards from CIKM 2026 deadline (2026-05-25):
- F4 (final proofread): 05-23 (2 days before deadline)
- F2 (anonymization): 05-18
- C8 (internal review): 05-12
- C3-C7 (remaining sections): 05-05
- B5 (writing authorized): 04-28
- B4 (claim boundary locked): 04-25
- B3 (table locked): 04-22
- A2.1+A2.2 (IIRC reruns): 04-18
- B2 (MDR closed) / MDR decision: 04-14
- A1.2-A1.5 (MDR pipeline) OR MDR pivot decision: 04-10

- [ ] **Step 2: Update `paper/open-risks.md` with timing reality**

Risk 3 (timing) is now critical. SIGIR 2026 is closed. CIKM 2026 is the only near-term option. Update with:
- MDR feasibility assessment result from Task 4
- Revised critical path based on MDR decision
- Honest assessment: is CIKM 2026 still feasible?

- [ ] **Step 3: Update `docs/next-phase-experiments.md`**

The IIRC Implementation Pre-Gate section needs to reflect the current state of the `copilot + ranked-choice` controller after the recent refactors (commits `b68148d` through `6e91aab`).

### Task 7: Identify what can be written now (parallel track)

Several paper sections are writable today, independent of IIRC/MDR blockers.

- [ ] **Step 1: List immediately writable deliverables**

Already done per Linear and outline:
- **Section 8: Limitations** — writable now (C7 partially, limitations part)
- **Section 7: Related Work** — E1 (coverage check) is Done, C5 just needs prose expansion
- **Table 3: 2Wiki Calibration** — D2 is Done, data ready
- **Table 6: Edge-scorer ablation** — D4 is Done, data ready
- **Figure 1: Architecture diagram** — D6 is Done

Still blocked:
- Section 1 (Introduction): blocked on claim boundary (B4)
- Sections 4-5 (Experiments+Results): blocked on IIRC table (B3)
- Section 6 (Analysis): blocked on IIRC table (B3)
- Section 9 (Conclusion): blocked on claim boundary (B4)

- [ ] **Step 2: Start Related Work prose expansion (C5/ACT-61)**

This has no blockers (E1 is done). Can be written in parallel with experiments.

---

## Phase 3: Execution Strategy

### Recommended Timeline (CIKM 2026 target)

```
Week 1 (Apr 7-13):
├── [DONE] Codebase cleanup (Tasks 1-3)
├── MDR go/no-go decision (Task 4)
├── IIRC controller pre-gate start (Task 5)
└── Related Work prose draft (Task 7.2)

Week 2 (Apr 14-20):
├── IIRC 20-case pilot complete
├── MDR closed OR pivot narrative locked
├── A2.1 baseline rerun start (if pre-gate passes)
└── B2 (Gate 2) pass

Week 3 (Apr 21-27):
├── A2.1 + A2.2 IIRC reruns complete
├── B3 (Gate 3): lock IIRC table
├── B4 (Gate 4): lock claim boundary
├── Table 1 + Table 2 generated
└── B5 (Gate 5): authorize writing

Weeks 4-5 (Apr 28 - May 11):
├── Draft all remaining sections (C3, C4, C6, C7, C9)
├── Figures 2-4 generated
├── C8 internal review pass
└── Abstract written

Weeks 6-7 (May 12-25):
├── F1 venue format applied
├── F2 anonymization pass
├── F3 supplementary materials
└── F4 final proofread + submit (May 23)
```

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
| --- | --- | --- | --- |
| Real MDR doesn't close in time | High (no GPU work started) | Blocks B2, B3, B4 | Drop MDR, pivot narrative |
| Controller doesn't beat dense on full-IIRC | Medium | Weakens paper story | Pivot to "efficient budget-aware selection" narrative per Risk 8 |
| CIKM 2026 infeasible | Medium-High | 7-month delay to SIGIR 2027 | Aggressive parallelization; MDR decision by Apr 10 |
| Writing takes longer than 2 weeks | Medium | Misses deadline | Pre-draft section skeletons now; parallelize section drafting |

### Decision Required: MDR Strategy

**This is the single most important decision right now.** Everything downstream depends on it. The recommendation is:

1. By **April 10**, decide whether to pursue real MDR or pivot.
2. If pivoting, rewrite claims C5, C10, C11 and update the paper narrative.
3. If pursuing, provide a concrete timeline with GPU access confirmed.

Without this decision, the project remains stuck in the same state it's been in for 2 weeks.
