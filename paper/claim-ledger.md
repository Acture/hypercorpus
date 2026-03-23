# Hypercorpus Claim Ledger

## Purpose
Track which paper claims are already supported, which are still conditional, and which experiments or tables are required before they become safe.

| Claim | Status | Current Evidence | Required To Lock |
| --- | --- | --- | --- |
| Budgeted evidence discovery over natural hyperlinks is a coherent problem formulation. | Supported | `docs/paper-positioning.md`, `phase-decision-30` | Keep framing consistent in the draft. |
| Dense-started hyperlink-local selection beats the flat dense control on the current small `2Wiki` phase sample. | Supported but narrow | `phase-decision-30` and `2Wiki` single-path ablation docs | Reframe as calibration evidence only. |
| `2Wiki` should not be the main decision surface. | Supported | `docs/next-phase-experiments.md` and current project plan | Close the remaining baseline retest run. |
| Full-IIRC replaces the old partial-IIRC store as the only valid IIRC judgment surface. | Supported on implementation facts | full-context fetch + conversion fix; live graph count rises from `5184` to `61304` | Land the canonical full-IIRC artifact set under `dataset/iirc_full/`. |
| The project has a reviewer-acceptable external comparator through real `MDR`. | In progress | official `MDR` integration exists in code | Close `export -> train -> index -> run` and generate a paper-facing table. |
| The controller-guided selector is the main contribution, not just one diagnostic option. | Open | current shortlist hints at precision gains | Compare against real `MDR` on full-IIRC and decide whether this remains the paper center. |
| The paper beats trained iterative retrieval on the main harder dataset. | Open | none yet | Real `MDR` on full-IIRC. |
| The paper supports a broader superiority claim against graph-based systems such as `GraphRetriever` or `HippoRAG`. | Not supported | none | Only consider after real `MDR` is closed and the main story still has room. |
| The method is stronger as evidence discovery, not necessarily as end-to-end QA. | Supported | current docs and experiment setup | Keep QA as a secondary sanity check only. |

## Current No-Claim Zone
- Do not claim end-to-end QA superiority.
- Do not claim broad wins over all learned or graph-based retrievers.
- Do not claim the controller is the isolated source of gain until the main comparison surface is complete.

## Near-Term Closure Tasks
1. Close full-IIRC canonical artifacts.
2. Close real `MDR` as a runnable comparator.
3. Generate the full-IIRC paper-facing table.
4. Freeze the main claim boundary before drafting the introduction and conclusion in full prose.
