# Working Memo: Multihop Scaling Gap

Date: 2026-03-25
Status: Exploratory framing memo; pending canonical full-IIRC controller rerun
Linear: ACT-116

This file is a working research memo, not a canonical paper-claims document.

Use it for:
- hypothesis tracking
- fallback paper-framing options
- literature-gap scouting

Do not use it as the source of record for paper-safe claims. Canonical paper-facing state lives in:
- `docs/paper-positioning.md`
- `paper/claim-ledger.md`
- `paper/open-risks.md`
- `paper/tables-and-figures.md`

## Working Hypothesis

No existing paper clearly reports that multihop retrieval methods fail to outperform single-hop dense retrieval when scaled to full-size corpora. Most prior evaluations use distractor settings or otherwise constrained candidate pools rather than full Wikipedia-scale corpora.

This memo tracks whether `hypercorpus` may have a publishable gap around that scaling transition, and whether the eventual paper should frame it as a negative result, a solution paper, or only a motivating observation.

## Internal Signal (not yet canonical evidence)

### Latest local numbers at the time of writing (27 cases, 2026-03-26)

| Selector | B=256 F1 | B=384 F1 | B=512 F1 | Runtime |
|---|---|---|---|---|
| dense(hop0) | **0.286** | **0.360** | **0.350** | 3.1s |
| mdr_light(hop2) | 0.262 | 0.336 | 0.333 | 172s |
| walk:st_balanced | 0.226 | 0.277 | 0.283 | 2.3s |
| walk:st_future(la2) | 0.242 | 0.288 | 0.269 | 5.9s |
| walk:overlap_balanced | 0.220 | 0.277 | 0.277 | 2.5s |
| walk:overlap_title | 0.220 | 0.277 | 0.277 | 2.5s |

These numbers were useful for early framing, but they are not the current canonical paper surface. Later docs and runs narrowed the safe interpretation further.

### Early observations from that slice

1. Dense led across the tested budgets on that slice.
2. `mdr_light` was close to dense but not clearly better.
3. The non-controller walk variants were weaker than dense on that slice.
4. This suggested that local path heuristics alone might be insufficient at larger corpus scale.

### Current safe interpretation

The stronger, paper-safe version of the above is narrower:

1. Current full-IIRC local ablations can be read as a narrowing result, not a main result.
2. They suggest that the current non-controller path heuristics are not enough on their own.
3. They do not yet invalidate the controller-guided multi-path hypothesis, because the canonical full-IIRC controller rerun is still pending.

## Why local walk may underperform dense

Dense budget-fill picks from embedding-ranked candidates. Local walk picks from hyperlink neighbors of the seed document. At larger corpus scale, hyperlink neighbors may behave more like broad encyclopedic cross-references than query-specific evidence pointers.

This remains a useful hypothesis, but it is still a hypothesis. It should not be written as a locked paper claim until the controller rerun and main full-IIRC table are complete.

## Implication for paper narrative

This memo supports a possible three-act story:

1. `dense` baseline can recover useful first-hop evidence but may stall on harder evidence assembly.
2. Structure-guided local walk provides traversal but may still lose to dense if link-local signals are too noisy.
3. `LLM`-guided multi-path control is the actual main-method candidate; the hypothesis is that semantic guidance is needed to overcome the noise floor at scale.

This story is only a working framing option until the full-IIRC controller row is rerun on the canonical surface.

## Related Work

### Closest papers (none clearly establish the same scaling claim)

1. **MultiHop-RAG** (Tang & Yang, 2024) — Finds RAG "unsatisfactory" on multi-hop queries, but does not isolate corpus-size effects or compare against single-hop dense across scale.
2. **FAIR-RAG** (Aghajani Asl et al., 2025) — Notes iterative retrieval noise propagation and proposes a gap-analysis solution, but does not isolate the scaling problem directly.
3. **SentGraph** (Liang et al., 2026) — Argues chunk retrieval often yields irrelevant or incoherent context; focuses on sentence-level graph construction rather than the full-corpus scaling transition.
4. **MR.COD** (Lu et al., 2022) — Studies cross-document multi-hop retrieval in open settings, but does not report that multihop advantage disappears at scale.
5. **PRISM** (Nahid & Rafiei, 2025) — Uses LLM-style agentic retrieval and emphasizes filtering distractions, which aligns with the controller-guided motivation.

### What may still be a gap

- Systematic full-corpus comparison of multihop vs single-hop under explicit evidence budgets
- A clear report that non-controller multihop advantage weakens or disappears at corpus scale
- Evaluation of hyperlink traversal strategies at full-corpus scale rather than distractor pools

These are still candidate gap statements, not paper-safe claims.

## Framing options

### Option A: Negative-result / findings paper

**Title sketch**: "Do Multi-hop Retrievers Scale? A Budget-Constrained Evaluation on Full Corpora"

- Contribution: evaluation framework plus a negative scaling result
- Viable even if LLM-guided walk does not beat dense
- Would treat `MDR` and other multihop baselines mainly as evidence for the difficulty of scaling

### Option B: Full paper with a solution

**Title sketch**: "Beyond Link Traversal: Why Multihop Retrieval Needs Semantic Guidance at Scale"

- Negative ablation result becomes motivation
- `LLM`-guided walk / constrained multipath becomes the answer
- Requires the controller row to show a meaningful gain over dense on the canonical surface

## Decision point

Pending canonical `constrained_multipath + llm_controller` results on full-IIRC.

If the controller row clearly beats dense on the canonical surface, Option B remains alive.
If it does not, Option A or a more conservative evidence-assembly framing becomes more likely.

## Next steps

- [x] Run 100-case dense-only baseline (`iirc-dense-full-v1`)
- [ ] Complete the current full-IIRC local ablation run
- [ ] Run canonical full-IIRC controller shortlist
- [ ] Reconcile this memo against `paper/claim-ledger.md` after the controller rerun
- [ ] Decide whether this memo still supports a paper framing, or should remain only internal notes
