from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
import heapq
import itertools
from typing import Protocol, Sequence

from webwalker.graph import LinkContext, LinkContextGraph
from webwalker.text import approx_token_count, content_tokens, normalized_token_overlap


class SelectionMode(StrEnum):
	STANDALONE = "standalone"
	HYBRID_WITH_PPR = "hybrid_with_ppr"


@dataclass(slots=True)
class SelectionBudget:
	max_nodes: int | None = 12
	max_hops: int | None = 3
	max_tokens: int | None = 256


SELECTION_BUDGET_PRESETS: dict[str, SelectionBudget] = {
	"node_only": SelectionBudget(max_nodes=12, max_hops=None, max_tokens=None),
	"hop_only": SelectionBudget(max_nodes=None, max_hops=3, max_tokens=None),
	"token_only": SelectionBudget(max_nodes=None, max_hops=None, max_tokens=256),
	"combined_default": SelectionBudget(max_nodes=12, max_hops=3, max_tokens=256),
}


@dataclass(slots=True)
class ScoredNode:
	node_id: str
	score: float
	source_strategy: str
	selected_reason: str


@dataclass(slots=True)
class ScoredLink:
	source: str
	target: str
	anchor_text: str
	sentence: str
	sent_idx: int
	ref_id: str | None
	metadata: dict[str, object]
	score: float
	source_strategy: str
	selected_reason: str

	@classmethod
	def from_link(
		cls,
		link: LinkContext,
		*,
		score: float,
		source_strategy: str,
		selected_reason: str,
	) -> "ScoredLink":
		return cls(
			source=link.source,
			target=link.target,
			anchor_text=link.anchor_text,
			sentence=link.sentence,
			sent_idx=link.sent_idx,
			ref_id=link.ref_id,
			metadata=dict(link.metadata),
			score=score,
			source_strategy=source_strategy,
			selected_reason=selected_reason,
		)


@dataclass(slots=True)
class PathState:
	current_node: str
	path_links: tuple[ScoredLink, ...]
	visited_nodes: tuple[str, ...]
	g_score: float
	h_score: float
	f_score: float
	covered_query_tokens: tuple[str, ...]
	token_cost_estimate: int
	reward_score: float = 0.0
	goal_reached: bool = False

	@property
	def depth(self) -> int:
		return len(self.path_links)

	def coverage_ratio(self, query_token_count: int) -> float:
		if query_token_count <= 0:
			return 0.0
		return len(self.covered_query_tokens) / query_token_count


@dataclass(slots=True)
class CorpusSelectionResult:
	query: str
	ranked_nodes: list[ScoredNode]
	ranked_links: list[ScoredLink]
	selected_node_ids: list[str]
	selected_links: list[ScoredLink]
	token_cost_estimate: int
	strategy: str
	mode: SelectionMode
	debug_trace: list[str]
	coverage_ratio: float


class SemanticEdgeScorer(Protocol):
	def score(self, query: str, graph: LinkContextGraph, link: LinkContext) -> float:
		...


class SemanticHeuristic(Protocol):
	def estimate(
		self,
		query_tokens: frozenset[str],
		graph: LinkContextGraph,
		state: PathState,
		scorer: SemanticEdgeScorer,
	) -> float:
		...


class ModelEdgeScorer(SemanticEdgeScorer, Protocol):
	...


class ModelHeuristic(SemanticHeuristic, Protocol):
	...


class CorpusSelector(Protocol):
	strategy_name: str

	def select(
		self,
		graph: LinkContextGraph,
		query: str,
		start_nodes: list[str],
		budget: SelectionBudget | None = None,
		*,
		scorer: SemanticEdgeScorer | None = None,
		heuristic: SemanticHeuristic | None = None,
	) -> CorpusSelectionResult:
		...


class DefaultSemanticEdgeScorer:
	def __init__(self, *, anchor_weight: float = 0.7, sentence_weight: float = 0.3):
		self.anchor_weight = anchor_weight
		self.sentence_weight = sentence_weight

	def score(self, query: str, graph: LinkContextGraph, link: LinkContext) -> float:
		return (
			normalized_token_overlap(query, link.anchor_text) * self.anchor_weight
			+ normalized_token_overlap(query, link.sentence) * self.sentence_weight
		)


class CoverageSemanticHeuristic:
	def __init__(self, *, bridge_bonus: float = 0.02):
		self.bridge_bonus = bridge_bonus

	def estimate(
		self,
		query_tokens: frozenset[str],
		graph: LinkContextGraph,
		state: PathState,
		scorer: SemanticEdgeScorer,
	) -> float:
		coverage = _coverage_ratio(query_tokens, set(state.covered_query_tokens))
		if self.bridge_bonus <= 0:
			return max(0.0, 1.0 - coverage)

		positive_edges = 0
		for link in graph.links_from(state.current_node):
			if scorer.score(" ".join(sorted(query_tokens)), graph, link) > 0:
				positive_edges += 1
		bridge_factor = min(positive_edges, 3) / 3 if positive_edges else 0.0
		return max(0.0, 1.0 - coverage - self.bridge_bonus * bridge_factor)


class SemanticPPRSelector:
	strategy_name = "semantic_ppr"

	def __init__(
		self,
		*,
		alpha: float = 0.15,
		max_iter: int = 20,
		tol: float = 1e-6,
	):
		self.alpha = alpha
		self.max_iter = max_iter
		self.tol = tol

	def select(
		self,
		graph: LinkContextGraph,
		query: str,
		start_nodes: list[str],
		budget: SelectionBudget | None = None,
		*,
		scorer: SemanticEdgeScorer | None = None,
		heuristic: SemanticHeuristic | None = None,
	) -> CorpusSelectionResult:
		del heuristic
		budget = budget or SelectionBudget()
		scorer = scorer or DefaultSemanticEdgeScorer()
		node_scores, link_scores, debug_trace = self._run_ppr(
			graph,
			query,
			start_nodes,
			scorer,
		)
		return _build_selection_result(
			graph=graph,
			query=query,
			node_scores=node_scores,
			link_scores=link_scores,
			budget=budget,
			strategy=self.strategy_name,
			mode=SelectionMode.STANDALONE,
			debug_trace=debug_trace,
		)

	def _run_ppr(
		self,
		graph: LinkContextGraph,
		query: str,
		start_nodes: list[str],
		scorer: SemanticEdgeScorer,
		*,
		seed_weights: dict[str, float] | None = None,
	) -> tuple[dict[str, float], dict[tuple[str, str, str, str, int, str | None], ScoredLink], list[str]]:
		if not start_nodes:
			raise ValueError("semantic_ppr requires at least one start node.")

		personalization = _normalize_map(seed_weights or _seed_weights_from_graph(graph, query, start_nodes))
		if not personalization:
			personalization = {node_id: 1 / len(start_nodes) for node_id in start_nodes}

		transitions: dict[str, dict[str, float]] = {}
		representative_links: dict[tuple[str, str], LinkContext] = {}
		debug_trace = [f"seed:{node_id}:{weight:.4f}" for node_id, weight in personalization.items()]

		for source in graph.nodes:
			target_weights: dict[str, float] = defaultdict(float)
			target_best_link: dict[str, tuple[float, LinkContext]] = {}
			for link in graph.links_from(source):
				edge_score = scorer.score(query, graph, link)
				if edge_score <= 0:
					continue
				target_weights[link.target] += edge_score
				best_score, _best_link = target_best_link.get(link.target, (float("-inf"), link))
				if edge_score > best_score:
					target_best_link[link.target] = (edge_score, link)

			if target_weights:
				transitions[source] = dict(target_weights)
				for target, (_score, link) in target_best_link.items():
					representative_links[(source, target)] = link

		probabilities = dict(personalization)
		for iteration in range(self.max_iter):
			updated = {node_id: self.alpha * personalization.get(node_id, 0.0) for node_id in graph.nodes}
			dangling_mass = 0.0
			for source in graph.nodes:
				source_prob = probabilities.get(source, 0.0)
				outgoing = transitions.get(source)
				if source_prob <= 0 or not outgoing:
					dangling_mass += source_prob
					continue

				total_weight = sum(outgoing.values())
				if total_weight <= 0:
					dangling_mass += source_prob
					continue

				for target, weight in outgoing.items():
					updated[target] = updated.get(target, 0.0) + (1 - self.alpha) * source_prob * (weight / total_weight)

			if dangling_mass > 0:
				for target, weight in personalization.items():
					updated[target] = updated.get(target, 0.0) + (1 - self.alpha) * dangling_mass * weight

			delta = sum(abs(updated.get(node_id, 0.0) - probabilities.get(node_id, 0.0)) for node_id in graph.nodes)
			debug_trace.append(f"iter:{iteration}:delta={delta:.6f}")
			probabilities = updated
			if delta < self.tol:
				break

		link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = {}
		for (source, target), outgoing_link in representative_links.items():
			source_prob = probabilities.get(source, 0.0)
			outgoing = transitions.get(source, {})
			total_weight = sum(outgoing.values())
			if total_weight <= 0:
				continue
			edge_score = source_prob * (outgoing[target] / total_weight)
			scored_link = ScoredLink.from_link(
				outgoing_link,
				score=edge_score,
				source_strategy=self.strategy_name,
				selected_reason="ppr",
			)
			link_scores[_link_key(scored_link)] = scored_link

		return probabilities, link_scores, debug_trace


class _PathfindingSelector:
	strategy_name = "semantic_pathfinding"

	def __init__(
		self,
		*,
		mode: SelectionMode = SelectionMode.STANDALONE,
		allow_revisit: bool = False,
		max_expansions: int | None = None,
	):
		self.mode = mode
		self.allow_revisit = allow_revisit
		self.max_expansions = max_expansions
		self._ppr = SemanticPPRSelector()

	def select(
		self,
		graph: LinkContextGraph,
		query: str,
		start_nodes: list[str],
		budget: SelectionBudget | None = None,
		*,
		scorer: SemanticEdgeScorer | None = None,
		heuristic: SemanticHeuristic | None = None,
	) -> CorpusSelectionResult:
		if not start_nodes:
			raise ValueError(f"{self.strategy_name} requires at least one start node.")

		budget = budget or SelectionBudget()
		scorer = scorer or DefaultSemanticEdgeScorer()
		heuristic = heuristic or CoverageSemanticHeuristic()
		query_tokens = _query_tokens(query)
		states, debug_trace = self._run_path_search(
			graph=graph,
			query=query,
			query_tokens=query_tokens,
			start_nodes=start_nodes,
			budget=budget,
			scorer=scorer,
			heuristic=heuristic,
		)
		path_result = _build_selection_from_states(
			graph=graph,
			query=query,
			query_tokens=query_tokens,
			states=states,
			budget=budget,
			strategy=self.strategy_name,
			mode=SelectionMode.STANDALONE,
			debug_trace=debug_trace,
		)
		if self.mode == SelectionMode.STANDALONE:
			return path_result

		frontier_scores = {node.node_id: node.score for node in path_result.ranked_nodes}
		seed_weights = _seed_weights_from_graph(graph, query, start_nodes)
		for node_id, score in frontier_scores.items():
			seed_weights[node_id] = seed_weights.get(node_id, 0.0) + score

		ppr_node_scores, ppr_link_scores, ppr_trace = self._ppr._run_ppr(
			graph,
			query,
			list(dict.fromkeys([*start_nodes, *path_result.selected_node_ids])),
			scorer,
			seed_weights=seed_weights,
		)
		return _build_hybrid_result(
			graph=graph,
			query=query,
			query_tokens=query_tokens,
			path_result=path_result,
			ppr_node_scores=ppr_node_scores,
			ppr_link_scores=ppr_link_scores,
			budget=budget,
			strategy=f"{self.strategy_name}_ppr",
			debug_trace=[*path_result.debug_trace, *ppr_trace],
		)

	def _run_path_search(
		self,
		*,
		graph: LinkContextGraph,
		query: str,
		query_tokens: frozenset[str],
		start_nodes: list[str],
		budget: SelectionBudget,
		scorer: SemanticEdgeScorer,
		heuristic: SemanticHeuristic,
	) -> tuple[list[PathState], list[str]]:
		raise NotImplementedError

	def _seed_states(
		self,
		graph: LinkContextGraph,
		query_tokens: frozenset[str],
		start_nodes: list[str],
		heuristic: SemanticHeuristic,
		scorer: SemanticEdgeScorer,
	) -> list[PathState]:
		seeds: list[PathState] = []
		for node_id in start_nodes:
			state = PathState(
				current_node=node_id,
				path_links=(),
				visited_nodes=(node_id,),
				g_score=0.0,
				h_score=0.0,
				f_score=0.0,
				covered_query_tokens=(),
				token_cost_estimate=_node_token_cost(graph, node_id),
			)
			state.h_score = heuristic.estimate(query_tokens, graph, state, scorer)
			state.f_score = state.g_score + state.h_score
			seeds.append(state)
		return seeds

	def _expand_state(
		self,
		*,
		graph: LinkContextGraph,
		query: str,
		query_tokens: frozenset[str],
		state: PathState,
		budget: SelectionBudget,
		scorer: SemanticEdgeScorer,
		heuristic: SemanticHeuristic,
	) -> list[PathState]:
		if budget.max_hops is not None and state.depth >= budget.max_hops:
			return []

		visited_set = set(state.visited_nodes)
		covered_set = set(state.covered_query_tokens)
		query_token_count = max(len(query_tokens), 1)
		next_states: list[PathState] = []
		for link in graph.links_from(state.current_node):
			if not self.allow_revisit and link.target in visited_set:
				continue

			edge_score = scorer.score(query, graph, link)
			if edge_score <= 0:
				continue

			link_tokens = _link_query_tokens(query_tokens, link)
			new_covered_tokens = tuple(sorted(covered_set | link_tokens))
			coverage_gain = (len(new_covered_tokens) - len(covered_set)) / query_token_count
			reward = edge_score + 0.2 * coverage_gain
			edge_cost = 1.0 - min(reward, 1.0)
			scored_link = ScoredLink.from_link(
				link,
				score=edge_score,
				source_strategy=self.strategy_name,
				selected_reason="path",
			)
			token_cost_estimate = state.token_cost_estimate
			if link.target not in visited_set:
				token_cost_estimate += _node_token_cost(graph, link.target)

			next_state = PathState(
				current_node=link.target,
				path_links=(*state.path_links, scored_link),
				visited_nodes=(*state.visited_nodes, link.target),
				g_score=state.g_score + edge_cost,
				h_score=0.0,
				f_score=0.0,
				covered_query_tokens=new_covered_tokens,
				token_cost_estimate=token_cost_estimate,
				reward_score=state.reward_score + reward,
			)
			if not _within_path_budget(next_state, budget):
				continue
			next_state.h_score = heuristic.estimate(query_tokens, graph, next_state, scorer)
			next_state.f_score = next_state.g_score + next_state.h_score
			next_state.goal_reached = next_state.coverage_ratio(len(query_tokens)) >= 1.0
			next_states.append(next_state)
		return next_states

	def _expansion_limit(self, graph: LinkContextGraph, budget: SelectionBudget) -> int:
		if self.max_expansions is not None:
			return self.max_expansions
		node_bound = budget.max_nodes or len(graph.nodes) or 1
		return max(32, node_bound * 8)


class SemanticBeamSelector(_PathfindingSelector):
	strategy_name = "semantic_beam"

	def __init__(
		self,
		*,
		mode: SelectionMode = SelectionMode.STANDALONE,
		beam_width: int = 4,
		allow_revisit: bool = False,
	):
		super().__init__(mode=mode, allow_revisit=allow_revisit)
		self.beam_width = beam_width

	def _run_path_search(
		self,
		*,
		graph: LinkContextGraph,
		query: str,
		query_tokens: frozenset[str],
		start_nodes: list[str],
		budget: SelectionBudget,
		scorer: SemanticEdgeScorer,
		heuristic: SemanticHeuristic,
	) -> tuple[list[PathState], list[str]]:
		debug_trace: list[str] = []
		seeds = self._seed_states(graph, query_tokens, start_nodes, heuristic, scorer)
		for seed in seeds:
			debug_trace.append(f"seed:{seed.current_node}:h={seed.h_score:.4f}")

		all_states = list(seeds)
		frontier = list(seeds)
		while frontier:
			next_frontier: list[PathState] = []
			for state in frontier:
				debug_trace.append(
					f"expand:{state.current_node}:reward={state.reward_score:.4f}:depth={state.depth}"
				)
				next_frontier.extend(
					self._expand_state(
						graph=graph,
						query=query,
						query_tokens=query_tokens,
						state=state,
						budget=budget,
						scorer=scorer,
						heuristic=heuristic,
					)
				)
			if not next_frontier:
				break

			next_frontier.sort(
				key=lambda item: (item.reward_score, item.coverage_ratio(len(query_tokens)), -item.g_score),
				reverse=True,
			)
			frontier = next_frontier[: self.beam_width]
			all_states.extend(frontier)
			debug_trace.append(
				"beam:"
				+ ",".join(
					f"{state.current_node}:{state.reward_score:.4f}"
					for state in frontier
				)
			)
		return all_states, debug_trace


class _PriorityPathSelector(_PathfindingSelector):
	priority_label = "f"

	def _priority(self, state: PathState) -> float:
		raise NotImplementedError

	def _run_path_search(
		self,
		*,
		graph: LinkContextGraph,
		query: str,
		query_tokens: frozenset[str],
		start_nodes: list[str],
		budget: SelectionBudget,
		scorer: SemanticEdgeScorer,
		heuristic: SemanticHeuristic,
	) -> tuple[list[PathState], list[str]]:
		counter = itertools.count()
		debug_trace: list[str] = []
		seeds = self._seed_states(graph, query_tokens, start_nodes, heuristic, scorer)
		all_states = list(seeds)
		heap: list[tuple[float, int, PathState]] = []
		for seed in seeds:
			priority = self._priority(seed)
			heapq.heappush(heap, (priority, next(counter), seed))
			debug_trace.append(
				f"seed:{seed.current_node}:g={seed.g_score:.4f}:h={seed.h_score:.4f}:f={seed.f_score:.4f}"
			)

		expansions = 0
		while heap and expansions < self._expansion_limit(graph, budget):
			priority, _order, state = heapq.heappop(heap)
			debug_trace.append(
				f"pop:{state.current_node}:{self.priority_label}={priority:.4f}:g={state.g_score:.4f}:h={state.h_score:.4f}:f={state.f_score:.4f}"
			)
			expansions += 1
			for child in self._expand_state(
				graph=graph,
				query=query,
				query_tokens=query_tokens,
				state=state,
				budget=budget,
				scorer=scorer,
				heuristic=heuristic,
			):
				all_states.append(child)
				child_priority = self._priority(child)
				heapq.heappush(heap, (child_priority, next(counter), child))
				debug_trace.append(
					f"push:{state.current_node}->{child.current_node}:{self.priority_label}={child_priority:.4f}:g={child.g_score:.4f}:h={child.h_score:.4f}:f={child.f_score:.4f}"
				)
		return all_states, debug_trace


class SemanticAStarSelector(_PriorityPathSelector):
	strategy_name = "semantic_astar"
	priority_label = "f"

	def _priority(self, state: PathState) -> float:
		return state.f_score


class SemanticGBFSSelector(_PriorityPathSelector):
	strategy_name = "semantic_gbfs"
	priority_label = "h"

	def _priority(self, state: PathState) -> float:
		return state.h_score


class SemanticUCSSelector(_PriorityPathSelector):
	strategy_name = "semantic_ucs"
	priority_label = "g"

	def _priority(self, state: PathState) -> float:
		return state.g_score


def _query_tokens(query: str) -> frozenset[str]:
	return frozenset(content_tokens(query))


def _link_query_tokens(query_tokens: frozenset[str], link: LinkContext | ScoredLink) -> set[str]:
	link_tokens = set(content_tokens(f"{link.anchor_text} {link.sentence}"))
	return set(query_tokens & link_tokens)


def _coverage_ratio(query_tokens: frozenset[str], covered_tokens: set[str]) -> float:
	if not query_tokens:
		return 0.0
	return len(query_tokens & covered_tokens) / len(query_tokens)


def _node_token_cost(graph: LinkContextGraph, node_id: str) -> int:
	document = graph.get_document(node_id)
	if document is None:
		return 0
	return approx_token_count(document.text)


def _within_path_budget(state: PathState, budget: SelectionBudget) -> bool:
	if budget.max_nodes is not None and len(set(state.visited_nodes)) > budget.max_nodes:
		return False
	if budget.max_hops is not None and state.depth > budget.max_hops:
		return False
	if budget.max_tokens is not None and state.token_cost_estimate > budget.max_tokens:
		return False
	return True


def _seed_weights_from_graph(
	graph: LinkContextGraph,
	query: str,
	start_nodes: Sequence[str],
) -> dict[str, float]:
	scored = graph.topk_similar(query, list(start_nodes), k=len(start_nodes))
	weights = {node_id: max(score, 0.0) for node_id, score in scored}
	if not any(weights.values()):
		return {node_id: 1.0 for node_id in start_nodes}
	return weights


def _normalize_map(scores: dict[str, float]) -> dict[str, float]:
	positive = {key: max(value, 0.0) for key, value in scores.items() if value > 0}
	if not positive:
		return {}
	total = sum(positive.values())
	if total <= 0:
		return {}
	return {key: value / total for key, value in positive.items()}


def _link_key(link: ScoredLink) -> tuple[str, str, str, str, int, str | None]:
	return (
		link.source,
		link.target,
		link.anchor_text,
		link.sentence,
		link.sent_idx,
		link.ref_id,
	)


def _build_selection_from_states(
	*,
	graph: LinkContextGraph,
	query: str,
	query_tokens: frozenset[str],
	states: list[PathState],
	budget: SelectionBudget,
	strategy: str,
	mode: SelectionMode,
	debug_trace: list[str],
) -> CorpusSelectionResult:
	node_scores: dict[str, float] = defaultdict(float)
	link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = {}
	ordered_states = sorted(
		states,
		key=lambda item: (item.reward_score, item.coverage_ratio(len(query_tokens)), -item.g_score),
		reverse=True,
	)
	for state in ordered_states:
		for node_id in state.visited_nodes:
			node_scores[node_id] = max(node_scores[node_id], state.reward_score)
		for link in state.path_links:
			key = _link_key(link)
			existing = link_scores.get(key)
			if existing is None or link.score > existing.score:
				link_scores[key] = link

	return _build_selection_result(
		graph=graph,
		query=query,
		node_scores=node_scores,
		link_scores=link_scores,
		budget=budget,
		strategy=strategy,
		mode=mode,
		debug_trace=debug_trace,
		query_tokens=query_tokens,
	)


def _build_hybrid_result(
	*,
	graph: LinkContextGraph,
	query: str,
	query_tokens: frozenset[str],
	path_result: CorpusSelectionResult,
	ppr_node_scores: dict[str, float],
	ppr_link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink],
	budget: SelectionBudget,
	strategy: str,
	debug_trace: list[str],
) -> CorpusSelectionResult:
	path_scores = {node.node_id: node.score for node in path_result.ranked_nodes}
	path_norm = _normalize_scores(path_scores)
	ppr_norm = _normalize_scores(ppr_node_scores)

	combined_node_scores: dict[str, float] = {}
	for node_id in set(path_scores) | set(ppr_node_scores):
		combined_node_scores[node_id] = 0.5 * path_norm.get(node_id, 0.0) + 0.5 * ppr_norm.get(node_id, 0.0)

	ranked_nodes = _rank_nodes(combined_node_scores, strategy, "hybrid")
	selected_node_ids, token_cost_estimate = _apply_budget_to_ranked_nodes(graph, ranked_nodes, budget)
	selected_node_set = set(selected_node_ids)

	combined_link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = {
		_link_key(link): link for link in path_result.ranked_links
	}
	for key, link in ppr_link_scores.items():
		if link.source not in selected_node_set or link.target not in selected_node_set or link.score <= 0:
			continue
		existing = combined_link_scores.get(key)
		if existing is None or link.score > existing.score:
			combined_link_scores[key] = link

	ranked_links = sorted(combined_link_scores.values(), key=lambda item: item.score, reverse=True)
	selected_links = [
		link
		for link in ranked_links
		if link.source in selected_node_set and link.target in selected_node_set
	]
	return CorpusSelectionResult(
		query=query,
		ranked_nodes=ranked_nodes,
		ranked_links=ranked_links,
		selected_node_ids=selected_node_ids,
		selected_links=selected_links,
		token_cost_estimate=token_cost_estimate,
		strategy=strategy,
		mode=SelectionMode.HYBRID_WITH_PPR,
		debug_trace=debug_trace,
		coverage_ratio=_selection_coverage_ratio(query_tokens, selected_links),
	)


def _build_selection_result(
	*,
	graph: LinkContextGraph,
	query: str,
	node_scores: dict[str, float],
	link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink],
	budget: SelectionBudget,
	strategy: str,
	mode: SelectionMode,
	debug_trace: list[str],
	query_tokens: frozenset[str] | None = None,
) -> CorpusSelectionResult:
	query_tokens = query_tokens or _query_tokens(query)
	ranked_nodes = _rank_nodes(node_scores, strategy, "selection")
	selected_node_ids, token_cost_estimate = _apply_budget_to_ranked_nodes(graph, ranked_nodes, budget)
	selected_node_set = set(selected_node_ids)
	ranked_links = sorted(link_scores.values(), key=lambda item: item.score, reverse=True)
	selected_links = [
		link
		for link in ranked_links
		if link.source in selected_node_set and link.target in selected_node_set
	]
	return CorpusSelectionResult(
		query=query,
		ranked_nodes=ranked_nodes,
		ranked_links=ranked_links,
		selected_node_ids=selected_node_ids,
		selected_links=selected_links,
		token_cost_estimate=token_cost_estimate,
		strategy=strategy,
		mode=mode,
		debug_trace=debug_trace,
		coverage_ratio=_selection_coverage_ratio(query_tokens, selected_links),
	)


def _rank_nodes(node_scores: dict[str, float], strategy: str, selected_reason: str) -> list[ScoredNode]:
	return sorted(
		[
			ScoredNode(
				node_id=node_id,
				score=score,
				source_strategy=strategy,
				selected_reason=selected_reason,
			)
			for node_id, score in node_scores.items()
		],
		key=lambda item: item.score,
		reverse=True,
	)


def _apply_budget_to_ranked_nodes(
	graph: LinkContextGraph,
	ranked_nodes: list[ScoredNode],
	budget: SelectionBudget,
) -> tuple[list[str], int]:
	selected: list[str] = []
	selected_set: set[str] = set()
	token_cost_estimate = 0
	for node in ranked_nodes:
		if node.node_id in selected_set:
			continue
		if budget.max_nodes is not None and len(selected) >= budget.max_nodes:
			break
		node_tokens = _node_token_cost(graph, node.node_id)
		if budget.max_tokens is not None and token_cost_estimate + node_tokens > budget.max_tokens:
			continue
		selected.append(node.node_id)
		selected_set.add(node.node_id)
		token_cost_estimate += node_tokens
	return selected, token_cost_estimate


def _selection_coverage_ratio(
	query_tokens: frozenset[str],
	selected_links: Sequence[ScoredLink],
) -> float:
	covered_tokens: set[str] = set()
	for link in selected_links:
		covered_tokens |= _link_query_tokens(query_tokens, link)
	return _coverage_ratio(query_tokens, covered_tokens)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
	if not scores:
		return {}
	min_score = min(scores.values())
	max_score = max(scores.values())
	if max_score == min_score:
		return {key: 1.0 for key in scores}
	return {
		key: (value - min_score) / (max_score - min_score)
		for key, value in scores.items()
	}
