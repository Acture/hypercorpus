from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Protocol, Sequence

from hypercorpus.graph import LinkContext, LinkContextGraph
from hypercorpus.text import normalized_token_overlap


class StopReason(StrEnum):
	DEAD_END = "dead_end"
	BUDGET_EXHAUSTED = "budget_exhausted"
	BUDGET_PACING_STOP = "budget_pacing_stop"
	SCORE_BELOW_THRESHOLD = "score_below_threshold"
	CONTROLLER_STOP = "controller_stop"


@dataclass(slots=True)
class WalkBudget:
	max_steps: int = 3
	min_score: float = 0.0
	allow_revisit: bool = False


@dataclass(slots=True)
class StepScorerMetadata:
	scorer_kind: str
	backend: str
	profile_name: str | None = None
	provider: str | None = None
	model: str | None = None
	prompt_version: str | None = None
	candidate_prefilter_top_n: int | None = None
	two_hop_prefilter_top_n: int | None = None
	controller_prompt_version: str | None = None
	controller_prefilter_top_n: int | None = None
	controller_future_top_n: int | None = None


@dataclass(slots=True)
class StepScoreCard:
	edge_id: str
	total_score: float
	subscores: dict[str, float] = field(default_factory=dict)
	rationale: str | None = None
	text: str | None = None
	backend: str = "overlap"
	provider: str | None = None
	model: str | None = None
	latency_s: float = 0.0
	prompt_tokens: int | None = None
	completion_tokens: int | None = None
	total_tokens: int | None = None
	cache_hit: bool | None = None
	fallback_reason: str | None = None
	llm_calls: int | None = None
	best_next_edge_id: str | None = None
	raw_response: str | None = None
	decision_action: str | None = None
	raw_decision_action: str | None = None
	primary_edge_id: str | None = None
	secondary_edge_id: str | None = None
	backup_edge_id: str | None = None
	stop_score: float | None = None
	evidence_cluster_confidence: float | None = None
	prefiltered_candidate_count: int | None = None


@dataclass(slots=True)
class StepCandidateTrace:
	edge_id: str
	source_node_id: str
	target_node_id: str
	anchor_text: str
	sentence: str
	total_score: float
	subscores: dict[str, float] = field(default_factory=dict)
	rationale: str | None = None
	best_next_edge_id: str | None = None
	fallback_reason: str | None = None


@dataclass(slots=True)
class WalkStepLog:
	step_index: int
	current_node_id: str
	path_node_ids: list[str]
	chosen_edge_id: str | None
	chosen_target_node_id: str | None
	backend: str
	provider: str | None
	model: str | None
	latency_s: float
	prompt_tokens: int | None
	completion_tokens: int | None
	total_tokens: int | None
	cache_hit: bool | None
	fallback_reason: str | None
	text: str | None
	raw_response: str | None
	llm_calls: int | None = None
	decision_action: str | None = None
	raw_decision_action: str | None = None
	secondary_edge_id: str | None = None
	backup_edge_id: str | None = None
	stop_score: float | None = None
	evidence_cluster_confidence: float | None = None
	prefiltered_candidate_count: int | None = None
	stop_reason: str | None = None
	candidates: list[StepCandidateTrace] = field(default_factory=list)


@dataclass(slots=True)
class WalkStep:
	index: int
	node_id: str
	score: float
	source_node_id: str | None = None
	anchor_text: str | None = None
	sentence: str | None = None
	edge_id: str | None = None


@dataclass(slots=True)
class WalkResult:
	query: str
	steps: list[WalkStep]
	visited_nodes: list[str]
	stop_reason: StopReason
	scorer_metadata: StepScorerMetadata
	selector_logs: list[WalkStepLog] = field(default_factory=list)


def walk_budget_to_dict(budget: WalkBudget) -> dict[str, Any]:
	return {
		"max_steps": budget.max_steps,
		"min_score": budget.min_score,
		"allow_revisit": budget.allow_revisit,
	}


def walk_budget_from_dict(payload: dict[str, Any]) -> WalkBudget:
	return WalkBudget(
		max_steps=int(payload.get("max_steps", 3)),
		min_score=float(payload.get("min_score", 0.0)),
		allow_revisit=bool(payload.get("allow_revisit", False)),
	)


def walk_step_to_dict(step: WalkStep) -> dict[str, Any]:
	return {
		"index": step.index,
		"node_id": step.node_id,
		"score": step.score,
		"source_node_id": step.source_node_id,
		"anchor_text": step.anchor_text,
		"sentence": step.sentence,
		"edge_id": step.edge_id,
	}


def walk_step_from_dict(payload: dict[str, Any]) -> WalkStep:
	return WalkStep(
		index=int(payload["index"]),
		node_id=str(payload["node_id"]),
		score=float(payload["score"]),
		source_node_id=None
		if payload.get("source_node_id") is None
		else str(payload["source_node_id"]),
		anchor_text=None
		if payload.get("anchor_text") is None
		else str(payload["anchor_text"]),
		sentence=None if payload.get("sentence") is None else str(payload["sentence"]),
		edge_id=None if payload.get("edge_id") is None else str(payload["edge_id"]),
	)


def step_candidate_trace_to_dict(trace: StepCandidateTrace) -> dict[str, Any]:
	return {
		"edge_id": trace.edge_id,
		"source_node_id": trace.source_node_id,
		"target_node_id": trace.target_node_id,
		"anchor_text": trace.anchor_text,
		"sentence": trace.sentence,
		"total_score": trace.total_score,
		"subscores": dict(trace.subscores),
		"rationale": trace.rationale,
		"best_next_edge_id": trace.best_next_edge_id,
		"fallback_reason": trace.fallback_reason,
	}


def step_candidate_trace_from_dict(payload: dict[str, Any]) -> StepCandidateTrace:
	return StepCandidateTrace(
		edge_id=str(payload["edge_id"]),
		source_node_id=str(payload["source_node_id"]),
		target_node_id=str(payload["target_node_id"]),
		anchor_text=str(payload["anchor_text"]),
		sentence=str(payload["sentence"]),
		total_score=float(payload["total_score"]),
		subscores={
			str(key): float(value)
			for key, value in dict(payload.get("subscores", {})).items()
		},
		rationale=None
		if payload.get("rationale") is None
		else str(payload["rationale"]),
		best_next_edge_id=None
		if payload.get("best_next_edge_id") is None
		else str(payload["best_next_edge_id"]),
		fallback_reason=None
		if payload.get("fallback_reason") is None
		else str(payload["fallback_reason"]),
	)


def walk_step_log_to_dict(log: WalkStepLog) -> dict[str, Any]:
	return {
		"step_index": log.step_index,
		"current_node_id": log.current_node_id,
		"path_node_ids": list(log.path_node_ids),
		"chosen_edge_id": log.chosen_edge_id,
		"chosen_target_node_id": log.chosen_target_node_id,
		"backend": log.backend,
		"provider": log.provider,
		"model": log.model,
		"latency_s": log.latency_s,
		"prompt_tokens": log.prompt_tokens,
		"completion_tokens": log.completion_tokens,
		"total_tokens": log.total_tokens,
		"cache_hit": log.cache_hit,
		"fallback_reason": log.fallback_reason,
		"llm_calls": log.llm_calls,
		"text": log.text,
		"raw_response": log.raw_response,
		"decision_action": log.decision_action,
		"raw_decision_action": log.raw_decision_action,
		"secondary_edge_id": log.secondary_edge_id,
		"backup_edge_id": log.backup_edge_id,
		"stop_score": log.stop_score,
		"evidence_cluster_confidence": log.evidence_cluster_confidence,
		"prefiltered_candidate_count": log.prefiltered_candidate_count,
		"stop_reason": log.stop_reason,
		"candidates": [
			step_candidate_trace_to_dict(candidate) for candidate in log.candidates
		],
	}


def walk_step_log_from_dict(payload: dict[str, Any]) -> WalkStepLog:
	return WalkStepLog(
		step_index=int(payload["step_index"]),
		current_node_id=str(payload["current_node_id"]),
		path_node_ids=[str(node_id) for node_id in payload.get("path_node_ids", [])],
		chosen_edge_id=None
		if payload.get("chosen_edge_id") is None
		else str(payload["chosen_edge_id"]),
		chosen_target_node_id=(
			None
			if payload.get("chosen_target_node_id") is None
			else str(payload["chosen_target_node_id"])
		),
		backend=str(payload["backend"]),
		provider=None if payload.get("provider") is None else str(payload["provider"]),
		model=None if payload.get("model") is None else str(payload["model"]),
		latency_s=float(payload.get("latency_s", 0.0)),
		prompt_tokens=None
		if payload.get("prompt_tokens") is None
		else int(payload["prompt_tokens"]),
		completion_tokens=None
		if payload.get("completion_tokens") is None
		else int(payload["completion_tokens"]),
		total_tokens=None
		if payload.get("total_tokens") is None
		else int(payload["total_tokens"]),
		cache_hit=payload.get("cache_hit"),
		fallback_reason=None
		if payload.get("fallback_reason") is None
		else str(payload["fallback_reason"]),
		llm_calls=None if payload.get("llm_calls") is None else int(payload["llm_calls"]),
		text=None if payload.get("text") is None else str(payload["text"]),
		raw_response=None
		if payload.get("raw_response") is None
		else str(payload["raw_response"]),
		decision_action=None
		if payload.get("decision_action") is None
		else str(payload["decision_action"]),
		raw_decision_action=None
		if payload.get("raw_decision_action") is None
		else str(payload["raw_decision_action"]),
		secondary_edge_id=None
		if payload.get("secondary_edge_id") is None
		else str(payload["secondary_edge_id"]),
		backup_edge_id=None
		if payload.get("backup_edge_id") is None
		else str(payload["backup_edge_id"]),
		stop_score=None
		if payload.get("stop_score") is None
		else float(payload["stop_score"]),
		evidence_cluster_confidence=(
			None
			if payload.get("evidence_cluster_confidence") is None
			else float(payload["evidence_cluster_confidence"])
		),
		prefiltered_candidate_count=(
			None
			if payload.get("prefiltered_candidate_count") is None
			else int(payload["prefiltered_candidate_count"])
		),
		stop_reason=None
		if payload.get("stop_reason") is None
		else str(payload["stop_reason"]),
		candidates=[
			step_candidate_trace_from_dict(candidate)
			for candidate in payload.get("candidates", [])
		],
	)


class StepLinkScorer(Protocol):
	metadata: StepScorerMetadata

	def score_candidates(
		self,
		*,
		query: str,
		graph: LinkContextGraph,
		current_node_id: str,
		candidate_links: Sequence[LinkContext],
		visited_nodes: set[str],
		path_node_ids: Sequence[str],
		remaining_steps: int,
	) -> list[StepScoreCard]: ...


class _BaseOverlapStepScorer:
	scorer_kind = "overlap"

	def __init__(
		self,
		*,
		anchor_weight: float,
		sentence_weight: float,
		target_weight: float = 0.0,
		novelty_bonus: float = 0.05,
		lookahead_steps: int = 1,
		lookahead_gamma: float = 0.6,
		profile_name: str | None = None,
	):
		if lookahead_steps <= 0:
			raise ValueError("lookahead_steps must be positive.")
		self.anchor_weight = anchor_weight
		self.sentence_weight = sentence_weight
		self.target_weight = target_weight
		self.novelty_bonus = novelty_bonus
		self.lookahead_steps = lookahead_steps
		self.lookahead_gamma = lookahead_gamma
		self.metadata = StepScorerMetadata(
			scorer_kind=self.scorer_kind,
			backend="overlap",
			profile_name=profile_name,
			provider=None,
			model=None,
			prompt_version=None,
			candidate_prefilter_top_n=None,
			two_hop_prefilter_top_n=None,
		)

	def score_candidates(
		self,
		*,
		query: str,
		graph: LinkContextGraph,
		current_node_id: str,
		candidate_links: Sequence[LinkContext],
		visited_nodes: set[str],
		path_node_ids: Sequence[str],
		remaining_steps: int,
	) -> list[StepScoreCard]:
		del current_node_id, path_node_ids
		cards: list[StepScoreCard] = []
		for index, link in enumerate(candidate_links):
			edge_id = str(index)
			immediate_subscores = self._immediate_subscores(
				query, graph, link, visited_nodes
			)
			immediate_score = self._aggregate_immediate_score(immediate_subscores)
			if self.lookahead_steps > 1 and remaining_steps > 1:
				future_score = self._best_future_score(
					query=query,
					graph=graph,
					link=link,
					visited_nodes=visited_nodes,
					remaining_steps=remaining_steps,
					depth=self.lookahead_steps,
				)
				total_score = _clamp_score(
					immediate_score + self.lookahead_gamma * future_score
				)
				subscores = {
					"immediate_score": immediate_score,
					"future_score": future_score,
				}
				best_next_edge_id = None
			else:
				total_score = immediate_score
				subscores = immediate_subscores
				best_next_edge_id = None
			cards.append(
				StepScoreCard(
					edge_id=edge_id,
					total_score=total_score,
					subscores={
						key: _clamp_score(value) for key, value in subscores.items()
					},
					rationale=None,
					text=None,
					backend="overlap",
					provider=None,
					model=None,
					latency_s=0.0,
					prompt_tokens=None,
					completion_tokens=None,
					total_tokens=None,
					cache_hit=None,
					fallback_reason=None,
					best_next_edge_id=best_next_edge_id,
					raw_response=None,
				)
			)
		return cards

	def _immediate_subscores(
		self,
		query: str,
		graph: LinkContextGraph,
		link: LinkContext,
		visited_nodes: set[str],
	) -> dict[str, float]:
		target_title = _target_title(graph, link.target)
		novelty = 1.0 if link.target not in visited_nodes else 0.0
		return {
			"anchor_overlap": normalized_token_overlap(query, link.anchor_text),
			"sentence_overlap": normalized_token_overlap(query, link.sentence),
			"target_overlap": normalized_token_overlap(query, target_title),
			"novelty": novelty,
		}

	def _aggregate_immediate_score(self, subscores: dict[str, float]) -> float:
		return _clamp_score(
			subscores["anchor_overlap"] * self.anchor_weight
			+ subscores["sentence_overlap"] * self.sentence_weight
			+ subscores["target_overlap"] * self.target_weight
			+ subscores["novelty"] * self.novelty_bonus
		)

	def _best_future_score(
		self,
		*,
		query: str,
		graph: LinkContextGraph,
		link: LinkContext,
		visited_nodes: set[str],
		remaining_steps: int,
		depth: int,
	) -> float:
		if depth <= 1 or remaining_steps <= 1:
			return 0.0

		next_visited = set(visited_nodes)
		next_visited.add(link.target)
		best_future = 0.0
		for neighbor in graph.neighbors(link.target):
			if neighbor in next_visited:
				continue
			for next_link in graph.links_between(link.target, neighbor):
				next_immediate = self._aggregate_immediate_score(
					self._immediate_subscores(query, graph, next_link, next_visited)
				)
				if depth > 2 and remaining_steps > 2:
					next_immediate = _clamp_score(
						next_immediate
						+ self.lookahead_gamma
						* self._best_future_score(
							query=query,
							graph=graph,
							link=next_link,
							visited_nodes=next_visited,
							remaining_steps=remaining_steps - 1,
							depth=depth - 1,
						)
					)
				best_future = max(best_future, next_immediate)
		return _clamp_score(best_future)


class AnchorOverlapStepScorer(_BaseOverlapStepScorer):
	def __init__(
		self,
		*,
		novelty_bonus: float = 0.05,
		lookahead_steps: int = 1,
		lookahead_gamma: float = 0.6,
	):
		super().__init__(
			anchor_weight=1.0,
			sentence_weight=0.0,
			target_weight=0.0,
			novelty_bonus=novelty_bonus,
			lookahead_steps=lookahead_steps,
			lookahead_gamma=lookahead_gamma,
		)


class LinkContextOverlapStepScorer(_BaseOverlapStepScorer):
	def __init__(
		self,
		*,
		anchor_weight: float = 0.6,
		sentence_weight: float = 0.4,
		target_weight: float = 0.0,
		novelty_bonus: float = 0.05,
		lookahead_steps: int = 1,
		lookahead_gamma: float = 0.6,
		profile_name: str | None = None,
	):
		super().__init__(
			anchor_weight=anchor_weight,
			sentence_weight=sentence_weight,
			target_weight=target_weight,
			novelty_bonus=novelty_bonus,
			lookahead_steps=lookahead_steps,
			lookahead_gamma=lookahead_gamma,
			profile_name=profile_name,
		)


class TitleAwareOverlapStepScorer(_BaseOverlapStepScorer):
	def __init__(
		self,
		*,
		anchor_weight: float = 0.45,
		sentence_weight: float = 0.35,
		target_weight: float = 0.20,
		novelty_bonus: float = 0.05,
	):
		super().__init__(
			anchor_weight=anchor_weight,
			sentence_weight=sentence_weight,
			target_weight=target_weight,
			novelty_bonus=novelty_bonus,
			lookahead_steps=1,
			lookahead_gamma=0.6,
			profile_name="overlap_title_aware",
		)


class DynamicWalker:
	def __init__(
		self,
		graph: LinkContextGraph,
		scorer: StepLinkScorer | None = None,
	):
		self.graph = graph
		self.scorer = scorer or TitleAwareOverlapStepScorer()

	def walk(
		self,
		query: str,
		start_nodes: list[str],
		budget: WalkBudget | None = None,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> WalkResult:
		budget = budget or WalkBudget()
		if budget.max_steps <= 0:
			raise ValueError("Walk budget must allow at least one step.")
		if not start_nodes:
			raise ValueError("walk requires at least one start node.")

		if resume_state is not None:
			current = str(resume_state["current_node"])
			visited_nodes = [
				str(node_id) for node_id in resume_state.get("visited_nodes", [])
			]
			if not visited_nodes:
				raise ValueError("DynamicWalker resume_state is missing visited_nodes.")
			visited_set = set(visited_nodes)
			steps = [
				walk_step_from_dict(step) for step in resume_state.get("steps", [])
			]
			if not steps:
				raise ValueError("DynamicWalker resume_state is missing steps.")
			selector_logs = [
				walk_step_log_from_dict(log)
				for log in resume_state.get("selector_logs", [])
			]
			backtracks_used = int(resume_state.get("backtracks_used", 0))
		else:
			current = self._choose_start_node(query, start_nodes)
			visited_nodes = [current]
			visited_set = {current}
			steps = [
				WalkStep(
					index=0,
					node_id=current,
					score=self._node_score(query, current),
				)
			]
			selector_logs = []
			backtracks_used = 0

		stop_reason = StopReason.BUDGET_EXHAUSTED
		while len(steps) < budget.max_steps:
			remaining_steps = budget.max_steps - len(steps)
			candidate_links: list[LinkContext] = []
			for neighbor in self.graph.neighbors(current):
				if not budget.allow_revisit and neighbor in visited_set:
					continue
				candidate_links.extend(self.graph.links_between(current, neighbor))

			if not candidate_links:
				if _apply_single_backup(
					current_node_id=current,
					visited_nodes=visited_nodes,
					visited_set=visited_set,
					steps=steps,
					selector_logs=selector_logs,
					backtracks_used=backtracks_used,
				):
					backtracks_used += 1
					current = visited_nodes[-1]
					continue
				stop_reason = StopReason.DEAD_END
				break

			score_cards = self.scorer.score_candidates(
				query=query,
				graph=self.graph,
				current_node_id=current,
				candidate_links=candidate_links,
				visited_nodes=visited_set,
				path_node_ids=visited_nodes,
				remaining_steps=remaining_steps,
			)
			if not score_cards:
				if _apply_single_backup(
					current_node_id=current,
					visited_nodes=visited_nodes,
					visited_set=visited_set,
					steps=steps,
					selector_logs=selector_logs,
					backtracks_used=backtracks_used,
				):
					backtracks_used += 1
					current = visited_nodes[-1]
					continue
				stop_reason = StopReason.DEAD_END
				break

			best_index, best_card, best_link = _choose_walk_edge(
				candidate_links=candidate_links,
				score_cards=score_cards,
			)

			controller_stop = (
				best_card.decision_action == "stop"
				and len(steps) >= 3
				and (best_card.stop_score or 0.0) >= 0.65
			)

			selector_logs.append(
				WalkStepLog(
					step_index=len(steps) - 1,
					current_node_id=current,
					path_node_ids=list(visited_nodes),
					chosen_edge_id=best_card.edge_id,
					chosen_target_node_id=best_link.target,
					backend=best_card.backend,
					provider=best_card.provider,
					model=best_card.model,
					latency_s=best_card.latency_s,
					prompt_tokens=best_card.prompt_tokens,
					completion_tokens=best_card.completion_tokens,
					total_tokens=best_card.total_tokens,
					cache_hit=best_card.cache_hit,
					fallback_reason=best_card.fallback_reason,
					llm_calls=best_card.llm_calls,
					text=best_card.text,
					raw_response=best_card.raw_response,
					decision_action=best_card.decision_action,
					raw_decision_action=best_card.raw_decision_action,
					secondary_edge_id=best_card.secondary_edge_id,
					backup_edge_id=best_card.backup_edge_id,
					stop_score=best_card.stop_score,
					evidence_cluster_confidence=best_card.evidence_cluster_confidence,
					prefiltered_candidate_count=best_card.prefiltered_candidate_count,
					candidates=[
						StepCandidateTrace(
							edge_id=card.edge_id,
							source_node_id=link.source,
							target_node_id=link.target,
							anchor_text=link.anchor_text,
							sentence=link.sentence,
							total_score=card.total_score,
							subscores=dict(card.subscores),
							rationale=card.rationale,
							best_next_edge_id=card.best_next_edge_id,
							fallback_reason=card.fallback_reason,
						)
						for link, card in zip(
							candidate_links, score_cards, strict=False
						)
					],
				)
			)

			if controller_stop:
				stop_reason = StopReason.CONTROLLER_STOP
				break

			if best_card.total_score < budget.min_score:
				if _apply_single_backup(
					current_node_id=current,
					visited_nodes=visited_nodes,
					visited_set=visited_set,
					steps=steps,
					selector_logs=selector_logs,
					backtracks_used=backtracks_used,
				):
					backtracks_used += 1
					current = visited_nodes[-1]
					continue
				stop_reason = StopReason.SCORE_BELOW_THRESHOLD
				break

			current = best_link.target
			visited_nodes.append(current)
			visited_set.add(current)
			steps.append(
				WalkStep(
					index=len(steps),
					node_id=current,
					score=best_card.total_score,
					source_node_id=best_link.source,
					anchor_text=best_link.anchor_text,
					sentence=best_link.sentence,
					edge_id=str(best_index),
				)
			)
			if checkpoint_callback is not None:
				checkpoint_callback(
					{
						"query": query,
						"start_nodes": list(start_nodes),
						"chosen_start_node": steps[0].node_id,
						"current_node": current,
						"visited_nodes": list(visited_nodes),
						"steps": [walk_step_to_dict(step) for step in steps],
						"selector_logs": [
							walk_step_log_to_dict(log) for log in selector_logs
						],
						"remaining_steps": budget.max_steps - len(steps),
						"budget": walk_budget_to_dict(budget),
						"backtracks_used": backtracks_used,
					}
				)
			if stop_callback is not None:
				stop_callback()
		else:
			stop_reason = StopReason.BUDGET_EXHAUSTED

		return WalkResult(
			query=query,
			steps=steps,
			visited_nodes=visited_nodes,
			stop_reason=stop_reason,
			scorer_metadata=self.scorer.metadata,
			selector_logs=selector_logs,
		)

	def _choose_start_node(self, query: str, start_nodes: list[str]) -> str:
		scored = sorted(
			((node_id, self._node_score(query, node_id)) for node_id in start_nodes),
			key=lambda item: item[1],
			reverse=True,
		)
		return scored[0][0]

	def _node_score(self, query: str, node_id: str) -> float:
		attr = self.graph.node_attr.get(node_id, {})
		text = f"{attr.get('title', '')} {attr.get('text', '')}".strip()
		return normalized_token_overlap(query, text)


def _choose_walk_edge(
	*,
	candidate_links: Sequence[LinkContext],
	score_cards: Sequence[StepScoreCard],
) -> tuple[int, StepScoreCard, LinkContext]:
	indexed = [
		(index, score_cards[index], candidate_links[index])
		for index in range(min(len(candidate_links), len(score_cards)))
	]
	if not indexed:
		raise ValueError("walk edge selection requires non-empty candidates.")
	primary_edge_id = next(
		(
			card.primary_edge_id
			for _index, card, _link in indexed
			if card.decision_action in {"choose_one", "choose_two"}
			and card.primary_edge_id is not None
		),
		None,
	)
	if primary_edge_id is not None:
		for index, card, link in indexed:
			if card.edge_id == primary_edge_id:
				return index, card, link
	return max(
		indexed,
		key=lambda item: (
			item[1].total_score,
			item[1].subscores.get("future_score", 0.0),
			-item[0],
		),
	)


def _apply_single_backup(
	*,
	current_node_id: str,
	visited_nodes: list[str],
	visited_set: set[str],
	steps: list[WalkStep],
	selector_logs: list[WalkStepLog],
	backtracks_used: int,
) -> bool:
	if backtracks_used >= 1 or len(steps) <= 1 or not selector_logs:
		return False
	previous_log = selector_logs[-1]
	backup_edge_id = previous_log.backup_edge_id
	if backup_edge_id is None:
		return False
	if current_node_id != steps[-1].node_id:
		return False
	backup_candidate = next(
		(
			candidate
			for candidate in previous_log.candidates
			if candidate.edge_id == backup_edge_id
		),
		None,
	)
	if backup_candidate is None or backup_candidate.target_node_id in visited_set:
		return False
	removed = steps.pop()
	if removed.node_id in visited_set:
		visited_set.remove(removed.node_id)
	if visited_nodes and visited_nodes[-1] == removed.node_id:
		visited_nodes.pop()
	parent_node_id = steps[-1].node_id
	selector_logs.append(
		WalkStepLog(
			step_index=len(steps) - 1,
			current_node_id=parent_node_id,
			path_node_ids=list(visited_nodes),
			chosen_edge_id=backup_candidate.edge_id,
			chosen_target_node_id=backup_candidate.target_node_id,
			backend=previous_log.backend,
			provider=previous_log.provider,
			model=previous_log.model,
			latency_s=0.0,
			prompt_tokens=None,
			completion_tokens=None,
			total_tokens=None,
			cache_hit=None,
			fallback_reason="controller_backtrack",
			llm_calls=0,
			text=previous_log.text,
			raw_response=previous_log.raw_response,
			decision_action="backtrack",
			raw_decision_action="backtrack",
			secondary_edge_id=previous_log.secondary_edge_id,
			backup_edge_id=None,
			stop_score=previous_log.stop_score,
			evidence_cluster_confidence=previous_log.evidence_cluster_confidence,
			prefiltered_candidate_count=previous_log.prefiltered_candidate_count,
			candidates=list(previous_log.candidates),
		)
	)
	visited_nodes.append(backup_candidate.target_node_id)
	visited_set.add(backup_candidate.target_node_id)
	steps.append(
		WalkStep(
			index=len(steps),
			node_id=backup_candidate.target_node_id,
			score=backup_candidate.total_score,
			source_node_id=backup_candidate.source_node_id,
			anchor_text=backup_candidate.anchor_text,
			sentence=backup_candidate.sentence,
			edge_id=backup_candidate.edge_id,
		)
	)
	return True


def _target_title(graph: LinkContextGraph, node_id: str) -> str:
	return str(graph.node_attr.get(node_id, {}).get("title", node_id))


def _clamp_score(value: Any) -> float:
	try:
		numeric = float(value)
	except (TypeError, ValueError):
		return 0.0
	return max(0.0, min(1.0, numeric))
