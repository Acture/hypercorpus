from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Protocol, Sequence

from hypercorpus.controller_runtime import (
	ControllerCandidateTrace,
	ControllerExecutionResult,
	ControllerRuntimeScorer,
	ControllerStepTrace,
	resolve_controller_backtrack,
)
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
	controller: ControllerStepTrace | None = None
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


def controller_candidate_trace_to_dict(trace: ControllerCandidateTrace) -> dict[str, Any]:
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
		"exposure_status": trace.exposure_status,
		"utility": trace.utility,
		"answer_bearing_link_bonus": trace.answer_bearing_link_bonus,
		"direct_support": trace.direct_support,
		"bridge_potential": trace.bridge_potential,
		"future_potential": trace.future_potential,
		"redundancy_risk": trace.redundancy_risk,
		"generic_concept_like": trace.generic_concept_like,
		"generic_concept_penalty": trace.generic_concept_penalty,
	}


def controller_candidate_trace_from_dict(payload: dict[str, Any]) -> ControllerCandidateTrace:
	return ControllerCandidateTrace(
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
		exposure_status=str(payload.get("exposure_status", "filtered_prefilter")),
		utility=None if payload.get("utility") is None else float(payload["utility"]),
		answer_bearing_link_bonus=None
		if payload.get("answer_bearing_link_bonus") is None
		else float(payload["answer_bearing_link_bonus"]),
		direct_support=None
		if payload.get("direct_support") is None
		else float(payload["direct_support"]),
		bridge_potential=None
		if payload.get("bridge_potential") is None
		else float(payload["bridge_potential"]),
		future_potential=None
		if payload.get("future_potential") is None
		else float(payload["future_potential"]),
		redundancy_risk=None
		if payload.get("redundancy_risk") is None
		else float(payload["redundancy_risk"]),
		generic_concept_like=None
		if payload.get("generic_concept_like") is None
		else bool(payload["generic_concept_like"]),
		generic_concept_penalty=None
		if payload.get("generic_concept_penalty") is None
		else float(payload["generic_concept_penalty"]),
	)


def controller_step_trace_to_dict(trace: ControllerStepTrace) -> dict[str, Any]:
	return {
		"kind": trace.kind,
		"raw_action": trace.raw_action,
		"effective_action": trace.effective_action,
		"primary_edge_id": trace.primary_edge_id,
		"secondary_edge_id": trace.secondary_edge_id,
		"backup_edge_id": trace.backup_edge_id,
		"primary_node_role": trace.primary_node_role,
		"primary_node_role_confidence": trace.primary_node_role_confidence,
		"primary_node_role_rationale": trace.primary_node_role_rationale,
		"secondary_node_role": trace.secondary_node_role,
		"secondary_node_role_confidence": trace.secondary_node_role_confidence,
		"secondary_node_role_rationale": trace.secondary_node_role_rationale,
		"stop_score": trace.stop_score,
		"evidence_cluster_confidence": trace.evidence_cluster_confidence,
		"llm_calls": trace.llm_calls,
		"raw_candidate_count": trace.raw_candidate_count,
		"valid_candidate_count": trace.valid_candidate_count,
		"small_page_bypass": trace.small_page_bypass,
		"dangling_edge_ids": list(trace.dangling_edge_ids),
		"lexical_prefilter_edge_ids": list(trace.lexical_prefilter_edge_ids),
		"semantic_prefilter_edge_ids": list(trace.semantic_prefilter_edge_ids),
		"bonus_rescued_edge_ids": list(trace.bonus_rescued_edge_ids),
		"visible_edge_ids": list(trace.visible_edge_ids),
		"candidates": [
			controller_candidate_trace_to_dict(candidate)
			for candidate in trace.candidates
		],
	}


def controller_step_trace_from_dict(payload: dict[str, Any]) -> ControllerStepTrace:
	return ControllerStepTrace(
		kind=str(payload["kind"]),
		raw_action=None
		if payload.get("raw_action") is None
		else str(payload["raw_action"]),
		effective_action=str(payload["effective_action"]),
		primary_edge_id=None
		if payload.get("primary_edge_id") is None
		else str(payload["primary_edge_id"]),
		secondary_edge_id=None
		if payload.get("secondary_edge_id") is None
		else str(payload["secondary_edge_id"]),
		backup_edge_id=None
		if payload.get("backup_edge_id") is None
		else str(payload["backup_edge_id"]),
		primary_node_role=None
		if payload.get("primary_node_role") is None
		else str(payload["primary_node_role"]),
		primary_node_role_confidence=None
		if payload.get("primary_node_role_confidence") is None
		else float(payload["primary_node_role_confidence"]),
		primary_node_role_rationale=None
		if payload.get("primary_node_role_rationale") is None
		else str(payload["primary_node_role_rationale"]),
		secondary_node_role=None
		if payload.get("secondary_node_role") is None
		else str(payload["secondary_node_role"]),
		secondary_node_role_confidence=None
		if payload.get("secondary_node_role_confidence") is None
		else float(payload["secondary_node_role_confidence"]),
		secondary_node_role_rationale=None
		if payload.get("secondary_node_role_rationale") is None
		else str(payload["secondary_node_role_rationale"]),
		stop_score=None
		if payload.get("stop_score") is None
		else float(payload["stop_score"]),
		evidence_cluster_confidence=None
		if payload.get("evidence_cluster_confidence") is None
		else float(payload["evidence_cluster_confidence"]),
		llm_calls=None if payload.get("llm_calls") is None else int(payload["llm_calls"]),
		raw_candidate_count=int(payload.get("raw_candidate_count", 0)),
		valid_candidate_count=int(payload.get("valid_candidate_count", 0)),
		small_page_bypass=bool(payload.get("small_page_bypass", False)),
		dangling_edge_ids=[str(edge_id) for edge_id in payload.get("dangling_edge_ids", [])],
		lexical_prefilter_edge_ids=[
			str(edge_id) for edge_id in payload.get("lexical_prefilter_edge_ids", [])
		],
		semantic_prefilter_edge_ids=[
			str(edge_id) for edge_id in payload.get("semantic_prefilter_edge_ids", [])
		],
		bonus_rescued_edge_ids=[
			str(edge_id) for edge_id in payload.get("bonus_rescued_edge_ids", [])
		],
		visible_edge_ids=[str(edge_id) for edge_id in payload.get("visible_edge_ids", [])],
		candidates=[
			controller_candidate_trace_from_dict(candidate)
			for candidate in payload.get("candidates", [])
		],
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
		"controller": (
			controller_step_trace_to_dict(log.controller)
			if log.controller is not None
			else None
		),
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
		controller=None
		if payload.get("controller") is None
		else controller_step_trace_from_dict(dict(payload["controller"])),
		stop_reason=None
		if payload.get("stop_reason") is None
		else str(payload["stop_reason"]),
		candidates=[
			step_candidate_trace_from_dict(candidate)
			for candidate in payload.get("candidates", [])
		],
	)


def build_step_candidate_traces(
	*,
	candidate_links: Sequence[LinkContext],
	score_cards: Sequence[StepScoreCard],
) -> list[StepCandidateTrace]:
	return [
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
		for link, card in zip(candidate_links, score_cards, strict=False)
	]


def build_walk_step_log(
	*,
	step_index: int,
	current_node_id: str,
	path_node_ids: Sequence[str],
	chosen_edge_id: str | None,
	chosen_target_node_id: str | None,
	backend: str,
	provider: str | None,
	model: str | None,
	latency_s: float,
	prompt_tokens: int | None,
	completion_tokens: int | None,
	total_tokens: int | None,
	cache_hit: bool | None,
	fallback_reason: str | None,
	llm_calls: int | None,
	text: str | None,
	raw_response: str | None,
	stop_reason: str | None,
	candidates: Sequence[StepCandidateTrace] = (),
	controller: ControllerStepTrace | None = None,
) -> WalkStepLog:
	return WalkStepLog(
		step_index=step_index,
		current_node_id=current_node_id,
		path_node_ids=list(path_node_ids),
		chosen_edge_id=chosen_edge_id,
		chosen_target_node_id=chosen_target_node_id,
		backend=backend,
		provider=provider,
		model=model,
		latency_s=latency_s,
		prompt_tokens=prompt_tokens,
		completion_tokens=completion_tokens,
		total_tokens=total_tokens,
		cache_hit=cache_hit,
		fallback_reason=fallback_reason,
		llm_calls=llm_calls,
		text=text,
		raw_response=raw_response,
		controller=controller,
		stop_reason=stop_reason,
		candidates=list(candidates),
	)


def build_controller_walk_step_log(
	*,
	step_index: int,
	current_node_id: str,
	path_node_ids: Sequence[str],
	execution: ControllerExecutionResult,
) -> WalkStepLog:
	primary = execution.primary
	return build_walk_step_log(
		step_index=step_index,
		current_node_id=current_node_id,
		path_node_ids=path_node_ids,
		chosen_edge_id=primary.edge_id if primary is not None else None,
		chosen_target_node_id=primary.link.target if primary is not None else None,
		backend=execution.backend,
		provider=execution.provider,
		model=execution.model,
		latency_s=execution.latency_s,
		prompt_tokens=execution.prompt_tokens,
		completion_tokens=execution.completion_tokens,
		total_tokens=execution.total_tokens,
		cache_hit=execution.cache_hit,
		fallback_reason=execution.fallback_reason,
		llm_calls=execution.trace.llm_calls,
		text=execution.text,
		raw_response=execution.raw_response,
		controller=execution.trace,
		stop_reason=execution.stop_reason,
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
				if apply_controller_backtrack(
					current_node_id=current,
					visited_nodes=visited_nodes,
					visited_set=visited_set,
					steps=steps,
					selector_logs=selector_logs,
					backtracks_used=backtracks_used,
					max_backtracks=1,
				):
					backtracks_used += 1
					current = visited_nodes[-1]
					continue
				stop_reason = StopReason.DEAD_END
				break

			if isinstance(self.scorer, ControllerRuntimeScorer):
				execution = self.scorer.evaluate_controller_step(
					query=query,
					graph=self.graph,
					current_node_id=current,
					candidate_links=candidate_links,
					visited_nodes=visited_set,
					path_node_ids=visited_nodes,
					remaining_steps=remaining_steps,
					current_depth=max(len(steps) - 1, 0),
					forks_used=0,
					backtracks_used=backtracks_used,
				)
				selector_logs.append(
					build_controller_walk_step_log(
						step_index=len(steps) - 1,
						current_node_id=current,
						path_node_ids=visited_nodes,
						execution=execution,
					)
				)
				if execution.effective_action == "stop":
					if execution.stop_reason == "dead_end" and apply_controller_backtrack(
						current_node_id=current,
						visited_nodes=visited_nodes,
						visited_set=visited_set,
						steps=steps,
						selector_logs=selector_logs,
						backtracks_used=backtracks_used,
						max_backtracks=1,
					):
						backtracks_used += 1
						current = visited_nodes[-1]
						continue
					stop_reason = StopReason(execution.stop_reason or StopReason.DEAD_END)
					break
				best_edge = execution.primary
				assert best_edge is not None
				best_index = best_edge.index
				best_card = best_edge.score_card
				best_link = best_edge.link
			else:
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
					if apply_controller_backtrack(
						current_node_id=current,
						visited_nodes=visited_nodes,
						visited_set=visited_set,
						steps=steps,
						selector_logs=selector_logs,
						backtracks_used=backtracks_used,
						max_backtracks=1,
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
				selector_logs.append(
					build_walk_step_log(
						step_index=len(steps) - 1,
						current_node_id=current,
						path_node_ids=visited_nodes,
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
						stop_reason=None,
						candidates=build_step_candidate_traces(
							candidate_links=candidate_links,
							score_cards=score_cards,
						),
					)
				)

			if best_card.total_score < budget.min_score:
				if apply_controller_backtrack(
					current_node_id=current,
					visited_nodes=visited_nodes,
					visited_set=visited_set,
					steps=steps,
					selector_logs=selector_logs,
					backtracks_used=backtracks_used,
					max_backtracks=1,
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
	return max(
		indexed,
		key=lambda item: (
			item[1].total_score,
			item[1].subscores.get("future_score", 0.0),
			-item[0],
		),
	)


def apply_controller_backtrack(
	*,
	current_node_id: str,
	visited_nodes: list[str],
	visited_set: set[str],
	steps: list[WalkStep],
	selector_logs: list[WalkStepLog],
	backtracks_used: int,
	max_backtracks: int,
) -> bool:
	if not selector_logs:
		return False
	resolution = resolve_controller_backtrack(
		current_node_id=current_node_id,
		steps=steps,
		visited_set=visited_set,
		previous_log=selector_logs[-1],
		backtracks_used=backtracks_used,
		max_backtracks=max_backtracks,
	)
	if resolution is None:
		return False
	removed = steps.pop()
	if removed.node_id in visited_set:
		visited_set.remove(removed.node_id)
	if visited_nodes and visited_nodes[-1] == resolution.removed_node_id:
		visited_nodes.pop()
	selector_logs.append(
		build_walk_step_log(
			step_index=len(steps) - 1,
			current_node_id=resolution.parent_node_id,
			path_node_ids=visited_nodes,
			chosen_edge_id=resolution.edge.edge_id,
			chosen_target_node_id=resolution.edge.link.target,
			backend=resolution.backend,
			provider=resolution.provider,
			model=resolution.model,
			latency_s=0.0,
			prompt_tokens=None,
			completion_tokens=None,
			total_tokens=None,
			cache_hit=None,
			fallback_reason=resolution.fallback_reason,
			llm_calls=0,
			text=resolution.text,
			raw_response=resolution.raw_response,
			controller=resolution.trace,
			stop_reason=None,
		)
	)
	visited_nodes.append(resolution.edge.link.target)
	visited_set.add(resolution.edge.link.target)
	steps.append(
		WalkStep(
			index=len(steps),
			node_id=resolution.edge.link.target,
			score=resolution.edge.score_card.total_score,
			source_node_id=resolution.edge.link.source,
			anchor_text=resolution.edge.link.anchor_text,
			sentence=resolution.edge.link.sentence,
			edge_id=resolution.edge.edge_id,
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
