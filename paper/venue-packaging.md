# Hypercorpus Venue Packaging

Purpose: venue-specific formatting notes and submission strategy for the hypercorpus paper.
SIGIR is the default venue. KDD and WWW are packaging alternatives only.

---

## Default Venue: SIGIR

### Format

- Page limit: 8 pages + unlimited references, double-column ACM sigconf format.
- Submission tracks: Full paper (research contribution) is the target. Short paper (4 pages) is a fallback if the evidence surface remains narrow.
- Template: ACM `acmart` class with `sigconf` option. Use `\setcopyright{acmlicensed}`.
- Anonymization: double-blind review.

### Framing emphasis

- Lead with **retrieval innovation**: budgeted hyperlink-graph evidence selection as a pre-RAG retrieval contribution.
- Frame the problem as "evidence discovery under token budget constraints" rather than "multi-hop QA."
- Primary metric story: support F1 under fixed token budgets, not answer EM/F1.
- Emphasize the selector-first evaluation paradigm: the contribution is the selected evidence set, not the final answer.
- Dense top-k and MDR are the main baselines. GraphRetriever and HippoRAG are narrative comparators.

### Key reviewer expectations

- **Strong baselines**: SIGIR reviewers expect comparison against state-of-the-art retrieval methods. MDR is mandatory. At least one graph-based comparator (GraphRetriever or HippoRAG) strengthens the paper significantly.
- **Retrieval metrics**: precision, recall, F1 on evidence recovery. Budget adherence and utilization as secondary metrics. Answer quality as a sanity check only.
- **Reproducibility**: code release expectation is high. The existing test suite and CLI are assets. Dataset preparation scripts should be documented.
- **Efficiency analysis**: SIGIR reviewers value cost-quality tradeoffs. Report selector runtime, token cost, and budget utilization alongside quality metrics.
- **Statistical rigor**: report results over enough cases to support significance claims. The current 30-case calibration sample is not sufficient for the main table; full-IIRC is required.

### Sections to emphasize

- Problem formulation (Section 2): the budget-constrained evidence discovery formulation is novel and must be clearly defined.
- Method (Section 3): the hyperlink-local walk, budget-aware stopping, and `budget_fill_relative_drop` component.
- Main results (Section 5): full-IIRC table at budgets 384 and 512 is the headline.
- Analysis (Section 6): precision-recall tradeoffs, budget sensitivity, when walks help versus hurt.

### Sections to keep brief

- Related work: ~1-1.5 pages. Enough to position, not a survey.
- Benchmarks section of experimental setup: readers know HotpotQA and MuSiQue. IIRC needs more explanation since it is less standard.

---

## Alternative: KDD

### Format

- Page limit: 9 pages + references.
- Template: ACM `acmart` class, double-column.
- Tracks: Research track (Applied Data Science track is also possible but less natural for this work).
- Anonymization: double-blind.

### Framing emphasis

- Lead with **graph mining + applied ML**: query-time subgraph discovery as an algorithmic contribution over linked-document graphs.
- Emphasize the algorithmic identity of the selection problem: budget-constrained subgraph discovery is a combinatorial optimization problem with graph structure.
- Emphasize cross-corpus generality more strongly: results on multiple datasets (IIRC, 2Wiki, potentially MuSiQue/HotpotQA) to show the approach transfers.
- Reference the KG search lineage: Su et al. (2015, KDD) on graph relevance feedback and Shi et al. (2020, WWW) on keyword search over KGs position the problem in the graph-mining tradition.

### What would change from SIGIR default

- More emphasis on the graph algorithm: describe the walk strategies, beam search, A* variants, and budget-constrained stopping as graph algorithms, not just retrieval heuristics.
- Stronger ablation requirements: KDD reviewers expect cleaner isolation of each algorithmic component's contribution.
- Scalability analysis: include graph size, walk length distributions, and runtime scaling. The current implementation may need efficiency benchmarking.
- Broader comparator coverage: would benefit from including GraphRetriever and/or HippoRAG as direct comparators, not just narrative references.
- Less emphasis on IR-specific framing (support F1, retrieval metrics terminology).

### Additional bar to clear

- Cleaner algorithmic identity: the method must read as a principled graph algorithm, not just "we follow links."
- Stronger cross-corpus generality: results on at least 3 datasets.
- More formal analysis of budget-quality tradeoffs.

---

## Alternative: WWW

### Format

- Page limit: 12 pages (including references in some years; check the specific CFP).
- Template: ACM `acmart` class, double-column.
- Anonymization: double-blind.

### Framing emphasis

- Lead with **web graph structure and hyperlink semantics**: the paper's distinctive angle is that natural hyperlinks in web-derived corpora already provide usable structure.
- Emphasize anchor text semantics, sentence context around links, and the difference between natural hyperlink graphs and constructed knowledge graphs.
- Frame the contribution as demonstrating that web-native link structure is an underutilized signal for evidence discovery.
- Reference the focused crawling and web navigation lineage more prominently: Chakrabarti et al. (1999), Nogueira and Cho (2016), Topic-Sensitive PageRank.

### What would change from SIGIR default

- More emphasis on hyperlink semantics: dedicate space to analyzing what anchor text and sentence context contribute, how link density affects selection quality, and when natural links fail.
- More emphasis on web-native evaluation: IIRC is the strongest dataset here because its links are natural Wikipedia hyperlinks, not synthetic.
- More space available (12 pages): can include a longer related work section, more detailed analysis, and potentially a case study showing hyperlink-following behavior on specific examples.
- Less emphasis on retrieval-system-vs-retrieval-system metrics comparisons.
- The web-navigation section of related work (Theme 6) would be promoted from a brief mention to a more substantial discussion.

### Additional bar to clear

- Must convincingly argue that the hyperlink structure (not just the dense seed + budget fill) is the source of the gain.
- Would benefit from analysis of link density, anchor text informativeness, and failure modes.
- Stronger connection to the web mining and web IR communities.

---

## Timing and Deadlines

### Current status (snapshot: 2026-04-07)

| Venue | Cycle | Abstract | Full Paper | Notification | Conference |
| --- | --- | --- | --- | --- | --- |
| SIGIR 2026 | Closed | -- | -- | -- | 2026-07-20 to 2026-07-24 |
| WWW 2026 | Closed | 2025-09-30 | 2025-10-07 | 2026-01-13 | 2026-07-01 to 2026-07-03 |
| KDD 2026 | Closed | 2026-02-01 | 2026-02-08 | 2026-05-16 | 2026-08-09 to 2026-08-13 |
| CIKM 2026 | Open | 2026-05-18 | 2026-05-25 | 2026-08-06 | 2026-11-09 to 2026-11-11 |

### Actionable windows

- **CIKM 2026** (CCF B) is the nearest realistic full-paper window. Abstract deadline: 2026-05-18 (41 days). Full paper: 2026-05-25 (48 days). Viable if the 100-case canonical IIRC surface is closed by ~April 22. A 20-case controller pilot has landed with positive signal (F1 = 0.46 vs dense 0.41). MDR go/no-go decision due ~April 10.
- **SIGIR 2027** (CCF A) is the primary target if the timeline allows. Based on past cycles, likely abstract deadline around 2027-01. This gives approximately 9-10 months.

### 2027 planning assumptions (not official deadlines)

| Venue | Likely Abstract | Likely Full Paper |
| --- | --- | --- |
| WWW 2027 | ~2026-09/10 | ~2026-10 |
| SIGIR 2027 | ~2027-01 | ~2027-01 |
| KDD 2027 | ~2027-02 | ~2027-02 |
| CIKM 2027 | ~2027-05 | ~2027-05 |

### Practical interpretation

- WWW 2027 arrives before SIGIR 2027, so it is not a fallback after missing SIGIR.
- If the project is not ready for SIGIR 2027, CIKM 2027 is the timeline-friendly fallback.
- Do not optimize for multiple venues simultaneously during the current closure phase.

---

## CCF Rankings and Fit Summary

| Venue | CCF | Best fit when... |
| --- | --- | --- |
| SIGIR | A | The story is strongest as retrieval innovation with budget-aware evidence selection. **Current default.** |
| KDD | A | The final paper clears a stronger algorithmic-generalization bar with cross-corpus results and formal analysis. |
| WWW | A | Hyperlink navigation and web-native link semantics become the dominant narrative hook. |
| CIKM | B | Pragmatic near-term option if the project is not ready for CCF-A venues. |

---

## Venue Decision Rule

1. Stay with SIGIR unless the full-IIRC + real MDR results clearly justify a stronger algorithmic or hyperlink-navigation framing.
2. If the project needs the nearest realistic 2026 submission window, prefer CIKM 2026.
3. If the project misses the likely SIGIR 2027 window, treat CIKM 2027 as the default fallback.
4. Do not optimize for multiple venues at once during the current closure phase.
5. Reassess after Gate 3 (paper-facing IIRC table locked) and Gate 4 (main claim boundary locked) from the CTRL workstream.
