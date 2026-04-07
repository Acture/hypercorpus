"""Shared controller action interpretation for single-path and multipath search.

This module is the single source of truth for converting a parsed controller
decision into executable traversal semantics:
- resolving primary/secondary/backup edges against visible candidates,
- normalizing the effective action under traversal policy,
- producing canonical per-step traces,
- applying shared backtrack semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from hypercorpus.graph import LinkContext


class ControllerDecisionCandidate(Protocol):
	edge_id: str
	utility: float
	answer_bearing_link_bonus: float
	direct_support: float
	bridge_potential: float
	future_potential: float | None
	redundancy_risk: float
	rationale: str | None
	generic_concept_like: bool
	generic_concept_penalty: float


class ControllerDecisionPayload(Protocol):
	decision: str
	runner_up: str | None
	state: str
	reason: str | None
	effective_action: str | None
	primary_edge_id: str | None
	runner_up_edge_id: str | None
	backup_edge_id: str | None
	backend: str
	provider: str | None
	model: str | None
	candidates: Sequence[ControllerDecisionCandidate]
	text: str | None
	raw_response: str | None
	prompt_tokens: int | None
	completion_tokens: int | None
	total_tokens: int | None
	latency_s: float
	cache_hit: bool | None
	fallback_reason: str | None
	llm_attempts: int
	raw_candidate_count: int
	valid_candidate_count: int
	small_page_bypass: bool
	dangling_edge_ids: Sequence[str]
	lexical_prefilter_edge_ids: Sequence[str]
	semantic_prefilter_edge_ids: Sequence[str]
	bonus_rescued_edge_ids: Sequence[str]
	visible_edge_ids: Sequence[str]


class ControllerScoreCard(Protocol):
	edge_id: str
	total_score: float
	subscores: dict[str, float]
	rationale: str | None
	backend: str
	provider: str | None
	model: str | None
	latency_s: float
	prompt_tokens: int | None
	completion_tokens: int | None
	total_tokens: int | None
	cache_hit: bool | None
	fallback_reason: str | None
	llm_calls: int | None
	best_next_edge_id: str | None
	text: str | None
	raw_response: str | None


class ControllerPathStep(Protocol):
	node_id: str


class ControllerBacktrackLog(Protocol):
	current_node_id: str
	backend: str
	provider: str | None
	model: str | None
	text: str | None
	raw_response: str | None
	controller: ControllerStepTrace | None


@runtime_checkable
class ControllerRuntimeScorer(Protocol):
	def evaluate_controller_step(
		self,
		*,
		query: str,
		graph: object,
		current_node_id: str,
		candidate_links: Sequence[LinkContext],
		visited_nodes: set[str],
		path_node_ids: Sequence[str],
		remaining_steps: int,
		current_depth: int,
		forks_used: int = 0,
		backtracks_used: int = 0,
	) -> "ControllerExecutionResult": ...


@dataclass(slots=True)
class ControllerExecutionPolicy:
	allow_stop: bool = True
	allow_choose_two: bool = False
	allow_backtrack: bool = True


@dataclass(slots=True)
class ControllerCandidateTrace:
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
	exposure_status: str = "filtered_prefilter"
	utility: float | None = None
	answer_bearing_link_bonus: float | None = None
	direct_support: float | None = None
	bridge_potential: float | None = None
	future_potential: float | None = None
	redundancy_risk: float | None = None
	generic_concept_like: bool | None = None
	generic_concept_penalty: float | None = None


@dataclass(slots=True)
class ControllerStepTrace:
	kind: str
	decision: str
	runner_up: str | None
	state: str
	reason: str | None
	effective_action: str
	primary_edge_id: str | None
	secondary_edge_id: str | None
	backup_edge_id: str | None
	llm_calls: int | None
	raw_candidate_count: int
	valid_candidate_count: int
	small_page_bypass: bool
	dangling_edge_ids: list[str] = field(default_factory=list)
	lexical_prefilter_edge_ids: list[str] = field(default_factory=list)
	semantic_prefilter_edge_ids: list[str] = field(default_factory=list)
	bonus_rescued_edge_ids: list[str] = field(default_factory=list)
	visible_edge_ids: list[str] = field(default_factory=list)
	candidates: list[ControllerCandidateTrace] = field(default_factory=list)


@dataclass(slots=True)
class ControllerResolvedEdge:
	index: int
	link: LinkContext
	score_card: ControllerScoreCard

	@property
	def edge_id(self) -> str:
		return self.score_card.edge_id


@dataclass(slots=True)
class ControllerExecutionResult:
	trace: ControllerStepTrace
	primary: ControllerResolvedEdge | None
	secondary: ControllerResolvedEdge | None
	backup: ControllerResolvedEdge | None
	stop_reason: str | None
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

	@property
	def effective_action(self) -> str:
		return self.trace.effective_action


@dataclass(slots=True)
class ControllerBacktrackResolution:
	parent_node_id: str
	removed_node_id: str
	edge: ControllerResolvedEdge
	trace: ControllerStepTrace
	backend: str
	provider: str | None
	model: str | None
	text: str | None
	raw_response: str | None
	fallback_reason: str = "controller_backtrack"


def build_controller_execution_result(
	*,
	decision: ControllerDecisionPayload,
	candidate_links: Sequence[LinkContext],
	score_cards: Sequence[ControllerScoreCard],
	policy: ControllerExecutionPolicy,
) -> ControllerExecutionResult:
	"""Resolve one parsed controller decision against visible scored edges."""

	card_entries = [
		(index, score_cards[index], candidate_links[index])
		for index in range(min(len(score_cards), len(candidate_links)))
	]
	cards_by_edge_id = {
		card.edge_id: (index, card, link) for index, card, link in card_entries
	}
	decision_candidates = {
		candidate.edge_id: candidate for candidate in decision.candidates
	}
	visible_ids = set(decision.visible_edge_ids)
	dangling_ids = set(decision.dangling_edge_ids)

	candidate_traces = [
		_build_candidate_trace(
			edge_id=card.edge_id,
			link=link,
			score_card=card,
			decision_candidate=decision_candidates.get(card.edge_id),
			visible_ids=visible_ids,
			dangling_ids=dangling_ids,
		)
		for _index, card, link in card_entries
	]
	candidate_traces.sort(
		key=lambda trace: (trace.total_score, trace.edge_id), reverse=True
	)

	trace = ControllerStepTrace(
		kind="decision",
		decision=decision.decision,
		runner_up=decision.runner_up,
		state=decision.state,
		reason=decision.reason,
		effective_action=decision.effective_action
		or ("stop" if decision.decision == "stop" else "choose_one"),
		primary_edge_id=decision.primary_edge_id,
		secondary_edge_id=decision.runner_up_edge_id,
		backup_edge_id=decision.backup_edge_id,
		llm_calls=decision.llm_attempts,
		raw_candidate_count=decision.raw_candidate_count,
		valid_candidate_count=decision.valid_candidate_count,
		small_page_bypass=decision.small_page_bypass,
		dangling_edge_ids=list(decision.dangling_edge_ids),
		lexical_prefilter_edge_ids=list(decision.lexical_prefilter_edge_ids),
		semantic_prefilter_edge_ids=list(decision.semantic_prefilter_edge_ids),
		bonus_rescued_edge_ids=list(decision.bonus_rescued_edge_ids),
		visible_edge_ids=list(decision.visible_edge_ids),
		candidates=candidate_traces,
	)
	if not visible_ids:
		trace.effective_action = "stop"
		trace.primary_edge_id = None
		trace.secondary_edge_id = None
		trace.backup_edge_id = None
		return ControllerExecutionResult(
			trace=trace,
			primary=None,
			secondary=None,
			backup=None,
			stop_reason="dead_end",
			backend=decision.backend,
			provider=decision.provider,
			model=decision.model,
			latency_s=decision.latency_s,
			prompt_tokens=decision.prompt_tokens,
			completion_tokens=decision.completion_tokens,
			total_tokens=decision.total_tokens,
			cache_hit=decision.cache_hit,
			fallback_reason=decision.fallback_reason,
			text=decision.text,
			raw_response=decision.raw_response,
		)

	primary = _resolve_edge(
		preferred_edge_id=trace.primary_edge_id,
		cards_by_edge_id=cards_by_edge_id,
		visible_ids=visible_ids,
	)
	runner_up = _resolve_edge(
		preferred_edge_id=decision.runner_up_edge_id,
		cards_by_edge_id=cards_by_edge_id,
		visible_ids=visible_ids,
		exclude={primary.edge_id} if primary is not None else set(),
	)

	effective_action = _normalize_effective_action(
		decision=decision.decision,
		primary=primary,
		runner_up=runner_up,
		policy=policy,
	)
	trace.effective_action = effective_action
	backup_exclude = {primary.edge_id} if primary is not None else set()
	secondary = runner_up if effective_action == "choose_two" else None
	if secondary is not None:
		backup_exclude.add(secondary.edge_id)
	backup = _resolve_edge(
		preferred_edge_id=trace.backup_edge_id,
		cards_by_edge_id=cards_by_edge_id,
		visible_ids=visible_ids,
		exclude=backup_exclude,
	)
	if backup is None and policy.allow_backtrack:
		backup = _resolve_edge(
			preferred_edge_id=None,
			cards_by_edge_id=cards_by_edge_id,
			visible_ids=visible_ids,
			exclude=backup_exclude,
		)
	if not policy.allow_backtrack:
		backup = None
	trace.primary_edge_id = primary.edge_id if primary is not None else None
	trace.secondary_edge_id = secondary.edge_id if secondary is not None else None
	trace.backup_edge_id = backup.edge_id if backup is not None else None
	if effective_action == "stop":
		trace.primary_edge_id = None
		trace.secondary_edge_id = None
		return ControllerExecutionResult(
			trace=trace,
			primary=None,
			secondary=None,
			backup=backup,
			stop_reason=(
				"controller_stop"
				if decision.decision == "stop" and policy.allow_stop
				else "dead_end"
			),
			backend=decision.backend,
			provider=decision.provider,
			model=decision.model,
			latency_s=decision.latency_s,
			prompt_tokens=decision.prompt_tokens,
			completion_tokens=decision.completion_tokens,
			total_tokens=decision.total_tokens,
			cache_hit=decision.cache_hit,
			fallback_reason=decision.fallback_reason,
			text=decision.text,
			raw_response=decision.raw_response,
		)
	return ControllerExecutionResult(
		trace=trace,
		primary=primary,
		secondary=secondary,
		backup=backup,
		stop_reason=None,
		backend=decision.backend,
		provider=decision.provider,
		model=decision.model,
		latency_s=decision.latency_s,
		prompt_tokens=decision.prompt_tokens,
		completion_tokens=decision.completion_tokens,
		total_tokens=decision.total_tokens,
		cache_hit=decision.cache_hit,
		fallback_reason=decision.fallback_reason,
		text=decision.text,
		raw_response=decision.raw_response,
	)


def make_backtrack_trace(
	*,
	previous_trace: ControllerStepTrace,
	backup: ControllerResolvedEdge,
) -> ControllerStepTrace:
	"""Create the canonical synthetic trace for a controller backtrack step."""

	return ControllerStepTrace(
		kind="backtrack",
		decision=previous_trace.decision,
		runner_up=previous_trace.runner_up,
		state=previous_trace.state,
		reason=previous_trace.reason,
		effective_action="backtrack",
		primary_edge_id=backup.edge_id,
		secondary_edge_id=previous_trace.secondary_edge_id,
		backup_edge_id=None,
		llm_calls=0,
		raw_candidate_count=previous_trace.raw_candidate_count,
		valid_candidate_count=previous_trace.valid_candidate_count,
		small_page_bypass=previous_trace.small_page_bypass,
		dangling_edge_ids=list(previous_trace.dangling_edge_ids),
		lexical_prefilter_edge_ids=list(previous_trace.lexical_prefilter_edge_ids),
		semantic_prefilter_edge_ids=list(previous_trace.semantic_prefilter_edge_ids),
		bonus_rescued_edge_ids=list(previous_trace.bonus_rescued_edge_ids),
		visible_edge_ids=list(previous_trace.visible_edge_ids),
		candidates=list(previous_trace.candidates),
	)


def resolve_controller_backtrack(
	*,
	current_node_id: str,
	steps: Sequence[ControllerPathStep],
	visited_set: set[str],
	previous_log: ControllerBacktrackLog,
	backtracks_used: int,
	max_backtracks: int,
) -> ControllerBacktrackResolution | None:
	"""Resolve a reusable backup edge from the previous controller step log."""

	if (
		backtracks_used >= max_backtracks
		or len(steps) <= 1
		or current_node_id != steps[-1].node_id
	):
		return None
	controller = previous_log.controller
	if controller is None or controller.backup_edge_id is None:
		return None
	backup_candidate = next(
		(
			candidate
			for candidate in controller.candidates
			if candidate.edge_id == controller.backup_edge_id
		),
		None,
	)
	if backup_candidate is None or backup_candidate.target_node_id in visited_set:
		return None
	backup_link = LinkContext(
		source=backup_candidate.source_node_id,
		target=backup_candidate.target_node_id,
		anchor_text=backup_candidate.anchor_text,
		sentence=backup_candidate.sentence,
		sent_idx=0,
	)
	backup_card = _score_card_from_candidate_trace(
		candidate=backup_candidate,
		backend=previous_log.backend,
		provider=previous_log.provider,
		model=previous_log.model,
	)
	backup_edge = ControllerResolvedEdge(
		index=int(backup_candidate.edge_id),
		link=backup_link,
		score_card=backup_card,
	)
	return ControllerBacktrackResolution(
		parent_node_id=steps[-2].node_id,
		removed_node_id=steps[-1].node_id,
		edge=backup_edge,
		trace=make_backtrack_trace(previous_trace=controller, backup=backup_edge),
		backend=previous_log.backend,
		provider=previous_log.provider,
		model=previous_log.model,
		text=previous_log.text,
		raw_response=previous_log.raw_response,
	)


def _build_candidate_trace(
	*,
	edge_id: str,
	link: LinkContext,
	score_card: ControllerScoreCard,
	decision_candidate: ControllerDecisionCandidate | None,
	visible_ids: set[str],
	dangling_ids: set[str],
) -> ControllerCandidateTrace:
	if edge_id in dangling_ids:
		exposure_status = "filtered_dangling_target"
	elif edge_id in visible_ids:
		exposure_status = "visible"
	else:
		exposure_status = "filtered_prefilter"
	return ControllerCandidateTrace(
		edge_id=edge_id,
		source_node_id=link.source,
		target_node_id=link.target,
		anchor_text=link.anchor_text,
		sentence=link.sentence,
		total_score=score_card.total_score,
		subscores=dict(score_card.subscores),
		rationale=score_card.rationale,
		best_next_edge_id=score_card.best_next_edge_id,
		fallback_reason=score_card.fallback_reason,
		exposure_status=exposure_status,
		utility=decision_candidate.utility if decision_candidate is not None else None,
		answer_bearing_link_bonus=(
			decision_candidate.answer_bearing_link_bonus
			if decision_candidate is not None
			else None
		),
		direct_support=(
			decision_candidate.direct_support
			if decision_candidate is not None
			else None
		),
		bridge_potential=(
			decision_candidate.bridge_potential
			if decision_candidate is not None
			else None
		),
		future_potential=(
			decision_candidate.future_potential
			if decision_candidate is not None
			else None
		),
		redundancy_risk=(
			decision_candidate.redundancy_risk
			if decision_candidate is not None
			else None
		),
		generic_concept_like=(
			decision_candidate.generic_concept_like
			if decision_candidate is not None
			else None
		),
		generic_concept_penalty=(
			decision_candidate.generic_concept_penalty
			if decision_candidate is not None
			else None
		),
	)


def _score_card_from_candidate_trace(
	*,
	candidate: ControllerCandidateTrace,
	backend: str,
	provider: str | None,
	model: str | None,
) -> ControllerScoreCard:
	@dataclass(slots=True)
	class _BacktrackScoreCard:
		edge_id: str
		total_score: float
		subscores: dict[str, float]
		rationale: str | None
		backend: str
		provider: str | None
		model: str | None
		latency_s: float = 0.0
		prompt_tokens: int | None = None
		completion_tokens: int | None = None
		total_tokens: int | None = None
		cache_hit: bool | None = None
		fallback_reason: str | None = None
		llm_calls: int | None = None
		best_next_edge_id: str | None = None
		text: str | None = None
		raw_response: str | None = None

	return _BacktrackScoreCard(
		edge_id=candidate.edge_id,
		total_score=candidate.total_score,
		subscores=dict(candidate.subscores),
		rationale=candidate.rationale,
		backend=backend,
		provider=provider,
		model=model,
		fallback_reason=candidate.fallback_reason,
		best_next_edge_id=candidate.best_next_edge_id,
	)


def _resolve_edge(
	*,
	preferred_edge_id: str | None,
	cards_by_edge_id: dict[str, tuple[int, ControllerScoreCard, LinkContext]],
	visible_ids: set[str],
	exclude: set[str] | None = None,
) -> ControllerResolvedEdge | None:
	excluded = exclude or set()
	if preferred_edge_id is not None and preferred_edge_id in visible_ids:
		entry = cards_by_edge_id.get(preferred_edge_id)
		if entry is not None and preferred_edge_id not in excluded:
			index, score_card, link = entry
			return ControllerResolvedEdge(index=index, link=link, score_card=score_card)
	visible_entries = [
		(index, score_card, link)
		for edge_id, (index, score_card, link) in cards_by_edge_id.items()
		if edge_id in visible_ids and edge_id not in excluded
	]
	if not visible_entries:
		return None
	index, score_card, link = max(
		visible_entries,
		key=lambda item: (
			item[1].total_score,
			item[1].subscores.get("future_score", 0.0),
			-item[0],
		),
	)
	return ControllerResolvedEdge(index=index, link=link, score_card=score_card)


def _normalize_effective_action(
	*,
	decision: str,
	primary: ControllerResolvedEdge | None,
	runner_up: ControllerResolvedEdge | None,
	policy: ControllerExecutionPolicy,
) -> str:
	if decision == "stop":
		if policy.allow_stop or primary is None:
			return "stop"
		return "choose_one"
	if primary is None:
		return "stop"
	if policy.allow_choose_two and runner_up is not None:
		return "choose_two"
	return "choose_one"
