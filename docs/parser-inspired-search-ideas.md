# Parser-Inspired Search Ideas

Date: `2026-03-13`

This document is a non-canonical research memo. It records algorithm-design hypotheses for future selector work. It is not an implementation spec, not a completed phase decision, and not a paper-facing claim document.

## Current Grounding

The current `single_path_walk` story is strong because it is narrow, budget-aware, and easy to reason about, but the implementation is still a hard-commit walk rather than a richer control policy.

- `single_path_walk` currently uses `DynamicWalker` with fixed `max_steps = hop_budget + 1`, `min_score = 0.05`, and no explicit early-stop, backtrack, or temporary fork state.
- Current walker stop reasons are only `dead_end`, `budget_exhausted`, and `score_below_threshold`.
- Current lookahead already exists inside the step scorers as immediate plus future scoring, especially `lookahead_2` in the overlap and sentence-transformer scorers.
- Current branchy families already exist as `beam`, `astar`, `ucs`, and `beam_ppr`, and the present phase record suggests precision-collapse risk under the current operating point.

This makes parser-inspired control ideas attractive only if they improve commitment, stopping, or recovery without reopening full branchy search as the default story.

Likely future implementation touchpoints, if any of these ideas are built later, would be `WalkBudget`, `StopReason`, `DynamicWalker`, and step-scorer metadata. This memo does not commit to any interface change.

## Shift-Reduce / Early Reduce

### Parser Analogy

In shift-reduce parsing, the parser does not keep shifting forever. Once the current span already forms a coherent constituent, it can reduce and move on. The key idea is not only how to continue, but when to stop reading and summarize what has already been assembled.

### What It Would Mean In WebWalker

This would add a new stop action for `single_path_walk`: halt the walk when the accumulated path already looks like a high-confidence evidence chunk for the query, instead of blindly consuming the remaining step budget. In practice, the walk would be allowed to say "the path so far is sufficient" and convert the current path into the selected subgraph.

### Why It Fits Or Conflicts With The Current Stack

This fits the current stack very well. The existing walk already has a strict budget, a ranked trace, and a narrow commitment story, but it only stops for failure or exhaustion. Early reduce would preserve the current single-path identity while adding a precision-oriented positive stop condition. It does not require widening the frontier or introducing a general multi-state controller.

### Expected Benefit

Better precision under tight budgets, especially when the correct support set is effectively completed after one or two strong hops and extra expansion mostly adds distractors.

### Main Risk

If the reduce rule fires too early, recall will fall because the system will stop before it collects the last necessary bridge or support page.

### Recommended Status

`P0 absorbable`

### What Evidence Would Justify Building It

It becomes worth building if current `single_path_walk` traces on `IIRC` or later harder datasets show repeated cases where the highest-value support is already assembled before the hop budget is exhausted and later steps mostly lower precision.

## Bounded Backtracking / Error Recovery

### Parser Analogy

Recursive or predictive parsers sometimes walk into the wrong branch and need a cheap recovery rule. They do not reopen the whole search space; they retreat to the latest viable state and try a small number of alternatives.

### What It Would Mean In WebWalker

This would add a constrained recovery rule for `single_path_walk`: if confidence drops sharply across recent hops, or the walker enters a semantic dead zone, return to the latest high-confidence hub and try one alternative outgoing edge. The goal is not general exploration, but cheap recovery from a local wrong turn.

### Why It Fits Or Conflicts With The Current Stack

This mostly fits the current stack because it keeps the core single-path identity and uses recovery only when the active path looks clearly degraded. It conflicts only in the sense that the current walker has no explicit notion of hub quality, rollback state, or "wrong-turn" detection, so the control logic would be more involved than simple early stopping.

### Expected Benefit

Better robustness against brittle one-shot commitment, especially on bridge-style cases where the first locally attractive link is not the right continuation.

### Main Risk

Backtracking can quietly turn into a disguised branchy search if the rollback trigger or retry allowance is too loose.

### Recommended Status

`P1 absorbable`

### What Evidence Would Justify Building It

It becomes worth building if trace inspection shows a common failure pattern where the first hop is good, the second hop sharply degrades, and the correct support node was reachable from a recent high-confidence hub through a different local choice.

## Deferred Commitment / Scout Fork

### Parser Analogy

Generalized parsing keeps multiple locally plausible interpretations alive only when the input is genuinely ambiguous, then prunes them once later evidence resolves the ambiguity.

### What It Would Mean In WebWalker

This would create a very small temporary fork only when the top two candidate links are near-tied. Each branch would receive a tiny scout budget for one probe step, after which one branch would be pruned immediately and the main budget would continue on the stronger continuation. The point is ambiguity resolution, not broad search.

### Why It Fits Or Conflicts With The Current Stack

This only partially fits the current stack. It is attractive because it could sit between rigid `single_path_walk` and the current branchy families, but it also risks recreating the same precision-collapse dynamics that made `beam` and `astar` unattractive in the current phase. The idea is only defensible if the fork is very small, very rare, and very short-lived.

### Expected Benefit

Better handling of locally ambiguous links without committing the full selector to an always-branchy search regime.

### Main Risk

The implementation and evaluation story can easily drift into "small beam search by another name," which would weaken both the algorithmic clarity and the budget-aware narrative.

### Recommended Status

`P2 conditional`

### What Evidence Would Justify Building It

It becomes worth building only if failure analysis shows frequent near-tie first-hop decisions where one probe step is enough to separate the good continuation from the bad one, and where full branchy search still looks too expensive or too imprecise.

## Low-Cost Semantic Lookahead Token

### Parser Analogy

A parser's lookahead token is a tiny, cheap preview signal that helps choose the next action without reading an entire future span in full detail.

### What It Would Mean In WebWalker

This would simplify lookahead by replacing expensive or rich edge-context inspection with tiny previews such as title, lead sentence, page type, or other cheap structural sniffing. The aim is to make lookahead more explicitly "preview-like" rather than treating it as a second round of full context scoring.

### Why It Fits Or Conflicts With The Current Stack

This fits the current stack as a scorer simplification rather than a new search structure. Current `lookahead_2` already partially captures the parser idea because it scores immediate plus future potential, but it does not yet present itself as a deliberately low-cost structural preview mechanism. This idea can strengthen the interpretability of lookahead without changing the broader selector family.

### Expected Benefit

Cheaper and more interpretable future-aware scoring, especially if later experiments show that full edge-context lookahead is overkill relative to smaller structural previews.

### Main Risk

The cheap preview may be too lossy and suppress the very contextual signals that made `lookahead_2` helpful in the first place.

### Recommended Status

`P1 absorbable as scorer simplification`

### What Evidence Would Justify Building It

It becomes worth building if ablations show that current `lookahead_2` gains are real but most of the gain appears recoverable from tiny previews such as title-plus-lead cues rather than richer two-hop text inspection.

## Additional Adjacent Algorithm Ideas

These ideas are not parser-derived, but they are worth tracking because they target the same control problem: how to spend limited query-time budget without collapsing precision. Some of them already have partial analogs in the current stack, so they should be treated as design refinements or coordination ideas rather than immediate claims of novelty.

## Semantic Pheromone / Global Blackboard

### Algorithm Analogy

Ant-colony style search allows weak global coordination across concurrent explorers. One branch can leave a signal that later branches use, so promising areas are not rediscovered from scratch.

### What It Would Mean In WebWalker

This would add a lightweight shared blackboard for concurrent or scout-style walks. If a branch encounters a node or local region with strong semantic promise but cannot fully pursue it because of budget or branch pruning, it can leave behind a compact signal such as the node ID plus a distilled cue. Later branches can then bias their local decisions toward that region.

### Why It Fits Or Conflicts With The Current Stack

This only partially fits the current stack. It is not very relevant to pure `single_path_walk`, but it could help any future scout-fork or limited branchy mode avoid redundant exploration. It also overlaps conceptually with existing `beam_ppr` and more broadly with PPR-style graph priors, which are already part of the standard design space rather than clean novelty candidates.

### Expected Benefit

Lower duplicate exploration cost across short-lived parallel probes, especially when multiple branches independently discover nearby high-value hubs.

### Main Risk

The shared state can make the selector harder to reason about and easier to over-engineer, while also blurring whether gains come from better local decisions or from hidden branch communication.

### Recommended Status

`P2 conditional as branch-coordination layer`

### What Evidence Would Justify Building It

It becomes worth building only if future scout-fork or restricted branchy experiments on harder datasets show repeated duplicate exploration of the same promising regions and clear wasted budget from branch non-communication.

## UCB-Style Exploration / Exploitation Control

### Algorithm Analogy

MCTS and UCB-style rules balance immediate payoff against uncertainty. A move is judged not only by current promise, but also by whether it has been under-explored relative to its possible upside.

### What It Would Mean In WebWalker

This would add a small exploration term to link selection so that the system can occasionally allocate a tiny probe budget to a lower-scoring but less-certain edge. The goal would be to avoid over-committing to the highest local match when a small amount of exploratory evidence could reveal a better downstream path.

### Why It Fits Or Conflicts With The Current Stack

This partially fits the current stack. The current selectors already contain weak analogs such as novelty bonuses, coverage-aware reward, and heuristic shaping, but they do not explicitly represent uncertainty or controlled rollouts. A stronger UCB-like controller would help ambiguity resolution, but it also pushes the system closer to rollout-based search and away from the current clean single-path story.

### Expected Benefit

Better escape from locally attractive but globally bad continuations, especially on deep fullwiki cases where the right bridge is initially less obvious.

### Main Risk

The added exploration pressure can quickly reopen the same cost and precision problems that hurt the broader branchy families, while still remaining a standard search ingredient rather than a distinctive contribution by itself.

### Recommended Status

`P2 conditional as exploration-control scoring`

### What Evidence Would Justify Building It

It becomes worth building only if later trace analysis shows that single-path and bounded-backtracking variants still fail mainly because of overconfident local exploitation, and tiny controlled probes consistently recover hidden support more cheaply than wider beam search.

## Budget Pacing / Dynamic Spend Control

### Algorithm Analogy

Budget pacing in serving or auctions tries to prevent spending too much too early and too little too late. The controller changes how aggressively it spends as confidence and opportunity change over time.

### What It Would Mean In WebWalker

This would turn the token budget into a dynamic control signal instead of a mostly static cap plus later backfill. The walk or search would stay frugal when it is still probing uncertain territory, then spend more aggressively once it appears to be inside a high-value answer region or support neighborhood.

### Why It Fits Or Conflicts With The Current Stack

This fits the current stack well. The repo already treats budget as a first-class object, and `budget_fill_relative_drop` already shows that budget behavior matters, but current budget logic is still mostly post-hoc selection and backfill rather than active pacing during discovery. This is a natural extension of the existing budget-aware story.

### Expected Benefit

Better budget efficiency, especially if the selector needs cheap exploratory movement early and denser local extraction only after it detects that it has reached the right evidence zone.

### Main Risk

If the pacing signal is noisy, the system may stay too conservative for too long or overspend around a false positive hub.

### Recommended Status

`P1 absorbable`

### What Evidence Would Justify Building It

It becomes worth building if traces show a repeated mismatch between where budget is spent and where support is actually recovered, such as early over-expansion in branchy runs or late under-collection in single-path runs despite strong local evidence.

## Reward Shaping / Dead-End Penalty / Hub Bonus

### Algorithm Analogy

Reward shaping modifies intermediate incentives so the search policy reaches useful states faster. Instead of evaluating only the final outcome, it adds small local bonuses and penalties that make desirable trajectories easier to follow.

### What It Would Mean In WebWalker

This would enrich link or state scoring with explicit hub bonuses, dead-end penalties, and semantic trap penalties. A list page, disambiguation page, or other strong bridge node could receive a controlled bonus when it looks like a useful jump point. Conversely, semantic loops across highly similar pages could receive an escalating penalty even when there is no exact revisit.

### Why It Fits Or Conflicts With The Current Stack

This fits the current stack well as a refinement rather than a new search family. The current code already contains partial shaping ingredients such as novelty bonuses, coverage-aware reward, bridge-oriented heuristic bias, and hard no-revisit logic. The missing piece is more explicit shaping for hub usefulness and semantic trap avoidance.

### Expected Benefit

Better local navigation quality without broadening the frontier, especially on cases where the correct route passes through a bridge-like hub or where the search drifts across same-topic near-duplicates.

### Main Risk

Hand-shaped bonuses can become dataset-specific heuristics that look clever on one corpus but fail to generalize on another.

### Recommended Status

`P1 absorbable as heuristic refinement`

### What Evidence Would Justify Building It

It becomes worth building if failure analysis on `IIRC` or later fullwiki runs shows repeated misses at useful bridge pages or repeated semantic wandering among homogeneous pages that are not literal graph cycles.

## Global Semantic PPR

### Algorithm Analogy

This is the classic graph-prior reading of semantic PPR: use a query-conditioned personalization vector, score or weight transitions broadly across the graph, run propagation, and then rank nodes by resulting stationary mass.

### What It Would Mean In WebWalker

This would make semantic PPR a first-class graph-prior baseline or alternate retrieval family. The method would behave less like a local committed walk and more like a global query-conditioned importance computation over the hyperlink graph, followed by budgeted evidence selection from the resulting node ranking.

### Why It Fits Or Conflicts With The Current Stack

It only partially fits the current stack. The repo already has a partial scaffold for this idea through `SemanticPPRSelector` and `beam_ppr`, so the design is not foreign. But as a paper story it conflicts with the current local-walk framing because the search behavior is no longer "follow one promising path" but "compute a soft relevance field over a broader graph." It also weakens the interpretation of token budget if the propagation stage itself is broad.

### Expected Benefit

A strong query-conditioned graph-prior comparator that can test whether smooth propagation over natural links is enough to recover support without path commitment.

### Main Risk

If the propagation is effectively broad or graph-wide, token budget becomes only an output budget and stops reflecting the real search-time cost of the method. That makes comparisons against local walk methods easier to challenge.

### Recommended Status

`P2 comparator, not current core story`

### What Evidence Would Justify Building It

It becomes worth building if the goal is to establish a serious PPR-based comparator against `single_path_walk`, especially for harder settings where the question is whether local commitment beats graph-prior smoothing under the same evidence budget.

## Local Semantic PPR

### Algorithm Analogy

This is a seeded, radius-limited, or frontier-limited semantic PPR. Instead of diffusing across a large graph indiscriminately, propagation stays near query-selected starts or within a local candidate-induced subgraph.

### What It Would Mean In WebWalker

This would treat semantic PPR as a local propagation controller rather than a global graph prior. Seeds would come from the same query-time start policies as the current selectors, propagation would stay inside a bounded neighborhood, and the final node ranking would still be trimmed by the same evidence token budget used elsewhere.

### Why It Fits Or Conflicts With The Current Stack

This fits the current stack much better than the global version. It preserves the local-search flavor, keeps token budget meaningful as an evidence budget, and can coexist with the current selector-first evaluation without forcing a total story change. It still shifts the narrative away from strict path commitment, but not so far that the method becomes a completely different retrieval paradigm.

### Expected Benefit

A cleaner graph-propagation alternative that remains local, budget-aware, and plausibly comparable to `single_path_walk` on the same tasks.

### Main Risk

If the local frontier is too small, the method collapses into a noisy reranking trick; if it is too large, it starts inheriting the same cost and fairness problems as the global version.

### Recommended Status

`P1 conditional if the story pivots toward local propagation`

### What Evidence Would Justify Building It

It becomes worth building if the project wants a more serious semantic-PPR direction without giving up local-budget discipline, or if `single_path_walk` and bounded recovery variants still underperform while a local diffusion prior looks likely to improve recall without destroying precision.

## Prioritization

- `P0`: early reduce
- `P1`: budget pacing
- `P1`: bounded backtracking
- `P1`: low-cost lookahead token
- `P1`: reward shaping
- `P1`: local semantic PPR, if the method story pivots from strict walk to local propagation
- `P2`: scout fork
- `P2`: global semantic PPR as a comparator
- `P2`: semantic pheromone
- `P2`: UCB-style exploration control

The current recommended storyline remains `single_path_walk` first. The most absorbable additions are the ones that improve stopping, pacing, recovery, and local shaping without reopening full branchy search as the default path.

## Not A Canonical Claim Yet

These are algorithm-design hypotheses for future selector work. None of them should be written up as validated contributions until isolated ablations show clear gains on harder datasets such as `IIRC`.
