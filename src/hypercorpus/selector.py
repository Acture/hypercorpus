from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import StrEnum
import heapq
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Literal, Protocol, Sequence, cast

from hypercorpus.embeddings import (
	DEFAULT_CROSS_ENCODER_MODEL,
	DEFAULT_SENTENCE_TRANSFORMER_MODEL,
	CrossEncoderReranker,
	CrossEncoderRerankerConfig,
	SentenceTransformerEmbedder,
	SentenceTransformerEmbedderConfig,
	TextEmbedder,
)
from hypercorpus.graph import LinkContext, LinkContextGraph
from hypercorpus.selector_llm import (
	LLMController,
	LLMControllerStepScorer,
	LLMStepLinkScorer,
	OpenAIApiMode,
	SelectorLLMConfig,
	SelectorProvider,
	TitleAwareOverlapStepScorer,
)
from hypercorpus.text import (
	approx_token_count,
	content_tokens,
	normalized_token_overlap,
)
from hypercorpus.controller_runtime import (
	ControllerExecutionResult,
	ControllerRuntimeScorer,
)
from hypercorpus.walker import (
	AnchorOverlapStepScorer,
	DynamicWalker,
	LinkContextOverlapStepScorer,
	StepLinkScorer,
	StepScoreCard,
	StepScorerMetadata,
	WalkBudget,
	WalkResult,
	WalkStep,
	WalkStepLog,
	_clamp_score,
	apply_controller_backtrack,
	build_controller_walk_step_log,
	build_step_candidate_traces,
	build_walk_step_log,
	walk_step_from_dict,
	walk_step_log_from_dict,
	walk_step_log_to_dict,
	walk_step_to_dict,
)

logger = logging.getLogger(__name__)

SeedStrategyName = Literal["sentence_transformer", "lexical_overlap"]
BudgetFillMode = Literal["score_floor", "always", "relative_drop", "neighbor", "diverse"]
BaselineName = Literal[
	"dense",
	"dense_rerank",
	"iterative_dense",
	"mdr_light",
	"topology_neighbors",
	"anchor_neighbors",
	"link_context_neighbors",
]
SearchStructure = Literal[
	"single_path_walk", "constrained_multipath", "beam", "astar", "ucs", "beam_ppr"
]
EdgeScorerName = Literal[
	"link_context_overlap",
	"link_context_llm",
	"link_context_llm_controller",
	"anchor_overlap",
	"link_context_sentence_transformer",
]
LookaheadName = Literal["lookahead_1", "lookahead_2"]
SelectorFamily = Literal["baseline", "path_search", "diagnostic"]
SelectorPresetName = Literal[
	"full", "paper_recommended", "paper_recommended_local", "branchy_profiles"
]

_SEED_STRATEGIES: set[str] = {"sentence_transformer", "lexical_overlap"}
_BASELINES: set[str] = {
	"dense",
	"dense_rerank",
	"iterative_dense",
	"mdr_light",
	"topology_neighbors",
	"anchor_neighbors",
	"link_context_neighbors",
}
_SEARCH_STRUCTURES: set[str] = {
	"single_path_walk",
	"constrained_multipath",
	"beam",
	"astar",
	"ucs",
	"beam_ppr",
}
_EDGE_SCORERS: set[str] = {
	"link_context_overlap",
	"link_context_llm",
	"link_context_llm_controller",
	"anchor_overlap",
	"link_context_sentence_transformer",
}
_LOOKAHEADS: set[str] = {"lookahead_1", "lookahead_2"}
_BUDGET_FILL_SUFFIXES: dict[str, BudgetFillMode] = {
	"budget_fill_score_floor": "score_floor",
	"budget_fill_always": "always",
	"budget_fill_relative_drop": "relative_drop",
	"budget_fill_neighbor": "neighbor",
	"budget_fill_diverse": "diverse",
}
_DEFAULT_BUDGET_FILL_MMR_LAMBDA = 0.7
_SELECTOR_PATTERN = re.compile(
	r"^top_(?P<seed_top_k>\d+)_seed__(?P<seed_strategy>sentence_transformer|lexical_overlap)__hop_(?P<hop_budget>\d+|adaptive)__(?P<rest>.+)$"
)
_DIAGNOSTIC_SELECTORS = ("gold_support_context", "full_corpus_upper_bound")
_OVERLAP_PROFILE_DEFAULT = "overlap_balanced"
_SENTENCE_TRANSFORMER_PROFILE_DEFAULT = "st_balanced"
_CONTROLLER_ADAPTIVE_HOP_TOKEN = "adaptive"
_CONTROLLER_INTERNAL_SAFETY_HOPS = 10
_OVERLAP_PROFILES: dict[str, dict[str, float]] = {
	"overlap_balanced": {
		"anchor": 0.60,
		"sentence": 0.40,
		"title": 0.00,
		"novelty": 0.05,
	},
	"overlap_anchor_heavy": {
		"anchor": 0.80,
		"sentence": 0.20,
		"title": 0.00,
		"novelty": 0.05,
	},
	"overlap_title_aware": {
		"anchor": 0.45,
		"sentence": 0.35,
		"title": 0.20,
		"novelty": 0.05,
	},
}
_SENTENCE_TRANSFORMER_PROFILES: dict[str, dict[str, float]] = {
	"st_balanced": {"direct": 0.55, "future": 0.35, "novelty": 0.10},
	"st_direct_heavy": {"direct": 0.80, "future": 0.10, "novelty": 0.10},
	"st_future_heavy": {"direct": 0.45, "future": 0.45, "novelty": 0.10},
}
_PAPER_RECOMMENDED_SELECTORS: tuple[str, ...] = (
	"top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_title_aware__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_1__profile_st_balanced__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm__lookahead_1__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop",
	"gold_support_context",
	"full_corpus_upper_bound",
)
_PAPER_RECOMMENDED_LOCAL_SELECTORS: tuple[str, ...] = (
	"top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_title_aware__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_1__profile_st_balanced__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop",
	"gold_support_context",
	"full_corpus_upper_bound",
)
_BRANCHY_PROFILE_SELECTORS: tuple[str, ...] = (
	"top_3_seed__sentence_transformer__hop_3__beam__link_context_overlap__lookahead_1__profile_overlap_balanced",
	"top_3_seed__sentence_transformer__hop_3__beam__link_context_overlap__lookahead_1__profile_overlap_title_aware",
	"top_3_seed__sentence_transformer__hop_3__beam__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
	"top_3_seed__sentence_transformer__hop_3__beam__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
	"top_3_seed__sentence_transformer__hop_3__astar__link_context_overlap__lookahead_1__profile_overlap_balanced",
	"top_3_seed__sentence_transformer__hop_3__astar__link_context_overlap__lookahead_1__profile_overlap_title_aware",
	"top_3_seed__sentence_transformer__hop_3__astar__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
	"top_3_seed__sentence_transformer__hop_3__astar__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
	"top_3_seed__sentence_transformer__hop_3__ucs__link_context_overlap__lookahead_1__profile_overlap_balanced",
	"top_3_seed__sentence_transformer__hop_3__ucs__link_context_overlap__lookahead_1__profile_overlap_title_aware",
	"top_3_seed__sentence_transformer__hop_3__ucs__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
	"top_3_seed__sentence_transformer__hop_3__ucs__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
	"top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_overlap__lookahead_1__profile_overlap_balanced",
	"top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_overlap__lookahead_1__profile_overlap_title_aware",
	"top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
	"top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
	"top_1_seed__sentence_transformer__hop_2__mdr_light",
	"top_1_seed__sentence_transformer__hop_0__dense",
	"gold_support_context",
	"full_corpus_upper_bound",
)


class SelectionMode(StrEnum):
	STANDALONE = "standalone"
	HYBRID_WITH_PPR = "hybrid_with_ppr"


@dataclass(slots=True)
class SelectorBudget:
	max_nodes: int | None = None
	max_hops: int | None = 3
	max_tokens: int | None = 256


def selector_budget_to_dict(budget: SelectorBudget) -> dict[str, Any]:
	return {
		"max_nodes": budget.max_nodes,
		"max_hops": budget.max_hops,
		"max_tokens": budget.max_tokens,
	}


def selector_budget_from_dict(payload: dict[str, Any]) -> SelectorBudget:
	return SelectorBudget(
		max_nodes=None
		if payload.get("max_nodes") is None
		else int(payload["max_nodes"]),
		max_hops=None if payload.get("max_hops") is None else int(payload["max_hops"]),
		max_tokens=None
		if payload.get("max_tokens") is None
		else int(payload["max_tokens"]),
	)


@dataclass(slots=True)
class SelectionTraceStep:
	index: int
	node_id: str
	score: float
	source_node_id: str | None = None
	anchor_text: str | None = None
	sentence: str | None = None


@dataclass(slots=True)
class SelectorMetadata:
	scorer_kind: str
	backend: str
	profile_name: str | None = None
	provider: str | None = None
	model: str | None = None
	seed_strategy: str | None = None
	seed_backend: str | None = None
	seed_model: str | None = None
	prompt_version: str | None = None
	candidate_prefilter_top_n: int | None = None
	two_hop_prefilter_top_n: int | None = None
	seed_top_k: int | None = None
	hop_budget: int | None = None
	search_structure: str | None = None
	edge_scorer: str | None = None
	lookahead_depth: int | None = None
	budget_fill_mode: str | None = None
	budget_fill_pool_k: int | None = None
	budget_fill_score_floor: float | None = None
	budget_fill_relative_drop_ratio: float | None = None
	walk_score_threshold: float | None = None
	rerank_model: str | None = None
	link_context_mask_mode: str | None = None


@dataclass(slots=True)
class SelectorUsage:
	runtime_s: float = 0.0
	llm_calls: int = 0
	prompt_tokens: int = 0
	completion_tokens: int = 0
	total_tokens: int = 0
	cache_hits: int = 0
	step_count: int = 0
	fallback_steps: int = 0
	parse_failure_steps: int = 0
	controller_calls: int = 0
	controller_stop_actions: int = 0
	controller_explicit_stop_actions: int = 0
	controller_budget_pacing_stop_actions: int = 0
	controller_fork_actions: int = 0
	controller_backtrack_actions: int = 0
	controller_prefiltered_candidates: int = 0


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


def scored_node_to_dict(node: ScoredNode) -> dict[str, Any]:
	return {
		"node_id": node.node_id,
		"score": node.score,
		"source_strategy": node.source_strategy,
		"selected_reason": node.selected_reason,
	}


def scored_node_from_dict(payload: dict[str, Any]) -> ScoredNode:
	return ScoredNode(
		node_id=str(payload["node_id"]),
		score=float(payload["score"]),
		source_strategy=str(payload["source_strategy"]),
		selected_reason=str(payload["selected_reason"]),
	)


def scored_link_to_dict(link: ScoredLink) -> dict[str, Any]:
	return {
		"source": link.source,
		"target": link.target,
		"anchor_text": link.anchor_text,
		"sentence": link.sentence,
		"sent_idx": link.sent_idx,
		"ref_id": link.ref_id,
		"metadata": dict(link.metadata),
		"score": link.score,
		"source_strategy": link.source_strategy,
		"selected_reason": link.selected_reason,
	}


def scored_link_from_dict(payload: dict[str, Any]) -> ScoredLink:
	return ScoredLink(
		source=str(payload["source"]),
		target=str(payload["target"]),
		anchor_text=str(payload["anchor_text"]),
		sentence=str(payload["sentence"]),
		sent_idx=int(payload["sent_idx"]),
		ref_id=None if payload.get("ref_id") is None else str(payload["ref_id"]),
		metadata=dict(payload.get("metadata", {})),
		score=float(payload["score"]),
		source_strategy=str(payload["source_strategy"]),
		selected_reason=str(payload["selected_reason"]),
	)


def selection_trace_step_to_dict(step: SelectionTraceStep) -> dict[str, Any]:
	return {
		"index": step.index,
		"node_id": step.node_id,
		"score": step.score,
		"source_node_id": step.source_node_id,
		"anchor_text": step.anchor_text,
		"sentence": step.sentence,
	}


def selection_trace_step_from_dict(payload: dict[str, Any]) -> SelectionTraceStep:
	return SelectionTraceStep(
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
	)


def selector_metadata_to_dict(
	metadata: SelectorMetadata | None,
) -> dict[str, Any] | None:
	if metadata is None:
		return None
	return {
		"scorer_kind": metadata.scorer_kind,
		"backend": metadata.backend,
		"profile_name": metadata.profile_name,
		"provider": metadata.provider,
		"model": metadata.model,
		"seed_strategy": metadata.seed_strategy,
		"seed_backend": metadata.seed_backend,
		"seed_model": metadata.seed_model,
		"prompt_version": metadata.prompt_version,
		"candidate_prefilter_top_n": metadata.candidate_prefilter_top_n,
		"two_hop_prefilter_top_n": metadata.two_hop_prefilter_top_n,
		"seed_top_k": metadata.seed_top_k,
		"hop_budget": metadata.hop_budget,
		"search_structure": metadata.search_structure,
		"edge_scorer": metadata.edge_scorer,
		"lookahead_depth": metadata.lookahead_depth,
		"budget_fill_mode": metadata.budget_fill_mode,
		"budget_fill_pool_k": metadata.budget_fill_pool_k,
		"budget_fill_score_floor": metadata.budget_fill_score_floor,
		"budget_fill_relative_drop_ratio": metadata.budget_fill_relative_drop_ratio,
		"walk_score_threshold": metadata.walk_score_threshold,
		"link_context_mask_mode": metadata.link_context_mask_mode,
	}


def selector_metadata_from_dict(
	payload: dict[str, Any] | None,
) -> SelectorMetadata | None:
	if payload is None:
		return None
	return SelectorMetadata(
		scorer_kind=str(payload["scorer_kind"]),
		backend=str(payload["backend"]),
		profile_name=None
		if payload.get("profile_name") is None
		else str(payload["profile_name"]),
		provider=None if payload.get("provider") is None else str(payload["provider"]),
		model=None if payload.get("model") is None else str(payload["model"]),
		seed_strategy=None
		if payload.get("seed_strategy") is None
		else str(payload["seed_strategy"]),
		seed_backend=None
		if payload.get("seed_backend") is None
		else str(payload["seed_backend"]),
		seed_model=None
		if payload.get("seed_model") is None
		else str(payload["seed_model"]),
		prompt_version=None
		if payload.get("prompt_version") is None
		else str(payload["prompt_version"]),
		candidate_prefilter_top_n=(
			None
			if payload.get("candidate_prefilter_top_n") is None
			else int(payload["candidate_prefilter_top_n"])
		),
		two_hop_prefilter_top_n=(
			None
			if payload.get("two_hop_prefilter_top_n") is None
			else int(payload["two_hop_prefilter_top_n"])
		),
		seed_top_k=None
		if payload.get("seed_top_k") is None
		else int(payload["seed_top_k"]),
		hop_budget=None
		if payload.get("hop_budget") is None
		else int(payload["hop_budget"]),
		search_structure=None
		if payload.get("search_structure") is None
		else str(payload["search_structure"]),
		edge_scorer=None
		if payload.get("edge_scorer") is None
		else str(payload["edge_scorer"]),
		lookahead_depth=None
		if payload.get("lookahead_depth") is None
		else int(payload["lookahead_depth"]),
		budget_fill_mode=None
		if payload.get("budget_fill_mode") is None
		else str(payload["budget_fill_mode"]),
		budget_fill_pool_k=None
		if payload.get("budget_fill_pool_k") is None
		else int(payload["budget_fill_pool_k"]),
		budget_fill_score_floor=(
			None
			if payload.get("budget_fill_score_floor") is None
			else float(payload["budget_fill_score_floor"])
		),
		budget_fill_relative_drop_ratio=(
			None
			if payload.get("budget_fill_relative_drop_ratio") is None
			else float(payload["budget_fill_relative_drop_ratio"])
		),
		walk_score_threshold=(
			None
			if payload.get("walk_score_threshold") is None
			else float(payload["walk_score_threshold"])
		),
		link_context_mask_mode=(
			None
			if payload.get("link_context_mask_mode") is None
			else str(payload["link_context_mask_mode"])
		),
	)


def selector_usage_to_dict(usage: SelectorUsage | None) -> dict[str, Any] | None:
	if usage is None:
		return None
	return {
		"runtime_s": usage.runtime_s,
		"llm_calls": usage.llm_calls,
		"prompt_tokens": usage.prompt_tokens,
		"completion_tokens": usage.completion_tokens,
		"total_tokens": usage.total_tokens,
		"cache_hits": usage.cache_hits,
		"step_count": usage.step_count,
		"fallback_steps": usage.fallback_steps,
		"parse_failure_steps": usage.parse_failure_steps,
		"controller_calls": usage.controller_calls,
		"controller_stop_actions": usage.controller_stop_actions,
		"controller_explicit_stop_actions": usage.controller_explicit_stop_actions,
		"controller_budget_pacing_stop_actions": usage.controller_budget_pacing_stop_actions,
		"controller_fork_actions": usage.controller_fork_actions,
		"controller_backtrack_actions": usage.controller_backtrack_actions,
		"controller_prefiltered_candidates": usage.controller_prefiltered_candidates,
	}


def selector_usage_from_dict(payload: dict[str, Any] | None) -> SelectorUsage | None:
	if payload is None:
		return None
	return SelectorUsage(
		runtime_s=float(payload.get("runtime_s", 0.0)),
		llm_calls=int(payload.get("llm_calls", 0)),
		prompt_tokens=int(payload.get("prompt_tokens", 0)),
		completion_tokens=int(payload.get("completion_tokens", 0)),
		total_tokens=int(payload.get("total_tokens", 0)),
		cache_hits=int(payload.get("cache_hits", 0)),
		step_count=int(payload.get("step_count", 0)),
		fallback_steps=int(payload.get("fallback_steps", 0)),
		parse_failure_steps=int(payload.get("parse_failure_steps", 0)),
		controller_calls=int(payload.get("controller_calls", 0)),
		controller_stop_actions=int(payload.get("controller_stop_actions", 0)),
		controller_explicit_stop_actions=int(
			payload.get("controller_explicit_stop_actions", 0)
		),
		controller_budget_pacing_stop_actions=int(
			payload.get("controller_budget_pacing_stop_actions", 0)
		),
		controller_fork_actions=int(payload.get("controller_fork_actions", 0)),
		controller_backtrack_actions=int(
			payload.get("controller_backtrack_actions", 0)
		),
		controller_prefiltered_candidates=int(
			payload.get("controller_prefiltered_candidates", 0)
		),
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


def path_state_to_dict(state: PathState) -> dict[str, Any]:
	return {
		"current_node": state.current_node,
		"path_links": [scored_link_to_dict(link) for link in state.path_links],
		"visited_nodes": list(state.visited_nodes),
		"g_score": state.g_score,
		"h_score": state.h_score,
		"f_score": state.f_score,
		"covered_query_tokens": list(state.covered_query_tokens),
		"token_cost_estimate": state.token_cost_estimate,
		"reward_score": state.reward_score,
		"goal_reached": state.goal_reached,
	}


def path_state_from_dict(payload: dict[str, Any]) -> PathState:
	return PathState(
		current_node=str(payload["current_node"]),
		path_links=tuple(
			scored_link_from_dict(link) for link in payload.get("path_links", [])
		),
		visited_nodes=tuple(
			str(node_id) for node_id in payload.get("visited_nodes", [])
		),
		g_score=float(payload.get("g_score", 0.0)),
		h_score=float(payload.get("h_score", 0.0)),
		f_score=float(payload.get("f_score", 0.0)),
		covered_query_tokens=tuple(
			str(token) for token in payload.get("covered_query_tokens", [])
		),
		token_cost_estimate=int(payload.get("token_cost_estimate", 0)),
		reward_score=float(payload.get("reward_score", 0.0)),
		goal_reached=bool(payload.get("goal_reached", False)),
	)


def corpus_selection_result_to_dict(result: CorpusSelectionResult) -> dict[str, Any]:
	return {
		"selector_name": result.selector_name,
		"query": result.query,
		"ranked_nodes": [scored_node_to_dict(node) for node in result.ranked_nodes],
		"ranked_links": [scored_link_to_dict(link) for link in result.ranked_links],
		"selected_node_ids": list(result.selected_node_ids),
		"selected_links": [scored_link_to_dict(link) for link in result.selected_links],
		"token_cost_estimate": result.token_cost_estimate,
		"strategy": result.strategy,
		"mode": result.mode.value,
		"debug_trace": list(result.debug_trace),
		"coverage_ratio": result.coverage_ratio,
		"root_node_ids": list(result.root_node_ids),
		"trace": [selection_trace_step_to_dict(step) for step in result.trace],
		"stop_reason": result.stop_reason,
		"selector_metadata": selector_metadata_to_dict(result.selector_metadata),
		"selector_usage": selector_usage_to_dict(result.selector_usage),
		"selector_logs": [walk_step_log_to_dict(log) for log in result.selector_logs],
	}


def corpus_selection_result_from_dict(payload: dict[str, Any]) -> CorpusSelectionResult:
	return CorpusSelectionResult(
		selector_name=str(payload["selector_name"]),
		query=str(payload["query"]),
		ranked_nodes=[
			scored_node_from_dict(node) for node in payload.get("ranked_nodes", [])
		],
		ranked_links=[
			scored_link_from_dict(link) for link in payload.get("ranked_links", [])
		],
		selected_node_ids=[
			str(node_id) for node_id in payload.get("selected_node_ids", [])
		],
		selected_links=[
			scored_link_from_dict(link) for link in payload.get("selected_links", [])
		],
		token_cost_estimate=int(payload.get("token_cost_estimate", 0)),
		strategy=str(payload["strategy"]),
		mode=SelectionMode(str(payload["mode"])),
		debug_trace=[str(item) for item in payload.get("debug_trace", [])],
		coverage_ratio=float(payload.get("coverage_ratio", 0.0)),
		root_node_ids=[str(node_id) for node_id in payload.get("root_node_ids", [])],
		trace=[
			selection_trace_step_from_dict(step) for step in payload.get("trace", [])
		],
		stop_reason=None
		if payload.get("stop_reason") is None
		else str(payload["stop_reason"]),
		selector_metadata=selector_metadata_from_dict(payload.get("selector_metadata")),
		selector_usage=selector_usage_from_dict(payload.get("selector_usage")),
		selector_logs=[
			walk_step_log_from_dict(log) for log in payload.get("selector_logs", [])
		],
	)


@dataclass(slots=True)
class CorpusSelectionResult:
	selector_name: str
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
	root_node_ids: list[str] = field(default_factory=list)
	trace: list[SelectionTraceStep] = field(default_factory=list)
	stop_reason: str | None = None
	selector_metadata: SelectorMetadata | None = None
	selector_usage: SelectorUsage | None = None
	selector_logs: list[WalkStepLog] = field(default_factory=list)


@dataclass(slots=True)
class SelectorSpec:
	canonical_name: str
	family: SelectorFamily
	base_canonical_name: str | None = None
	profile_name: str | None = None
	seed_strategy: SeedStrategyName | None = None
	seed_top_k: int | None = None
	hop_budget: int | None = None
	baseline: BaselineName | None = None
	search_structure: SearchStructure | None = None
	edge_scorer: EdgeScorerName | None = None
	lookahead_depth: int | None = None
	budget_fill_mode: BudgetFillMode | None = None
	budget_fill_pool_k: int | None = None
	budget_fill_score_floor: float | None = None
	budget_fill_relative_drop_ratio: float | None = None
	walk_score_threshold: float | None = None
	rerank_pool_k: int | None = None
	link_context_mask_mode: str | None = None


@dataclass(slots=True)
class SentenceTransformerSelectorConfig:
	model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL
	cache_path: Path | None = None
	device: str | None = None

	def __post_init__(self) -> None:
		if isinstance(self.cache_path, str):
			self.cache_path = Path(self.cache_path)


class SelectionCase(Protocol):
	case_id: str
	query: str
	gold_support_nodes: list[str]
	gold_start_nodes: list[str]


class RuntimeBudget(Protocol):
	token_budget_tokens: int | None
	token_budget_ratio: float | None
	budget_mode: str
	budget_value: int | float
	budget_label: str


class CorpusSelector(Protocol):
	name: str
	spec: SelectorSpec

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
	) -> CorpusSelectionResult: ...


class _SentenceTransformerSupport:
	def __init__(
		self,
		*,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		self.embedder_config = embedder_config or SentenceTransformerSelectorConfig()
		self.embedder_factory = embedder_factory
		self._embedder: TextEmbedder | None = None

	def _get_embedder(self) -> TextEmbedder:
		if self._embedder is None:
			config = SentenceTransformerEmbedderConfig(
				model_name=self.embedder_config.model_name,
				cache_path=self.embedder_config.cache_path,
				device=self.embedder_config.device,
			)
			if self.embedder_factory is not None:
				self._embedder = self.embedder_factory(config)
			else:
				self._embedder = SentenceTransformerEmbedder(config)
		return self._embedder


class CoverageSemanticHeuristic:
	def estimate(
		self,
		query_tokens: frozenset[str],
		graph: LinkContextGraph,
		state: PathState,
	) -> float:
		coverage = _coverage_ratio(query_tokens, set(state.covered_query_tokens))
		if not query_tokens:
			return 0.0
		bridge_edges = 0
		query = " ".join(sorted(query_tokens))
		for link in graph.links_from(state.current_node):
			if (
				normalized_token_overlap(query, link.anchor_text) > 0
				or normalized_token_overlap(query, link.sentence) > 0
			):
				bridge_edges += 1
		bridge_bonus = min(bridge_edges, 3) * 0.03
		return max(0.0, 1.0 - coverage - bridge_bonus)


class SentenceTransformerStepScorer:
	scorer_kind = "sentence_transformer"

	def __init__(
		self,
		*,
		embedder: TextEmbedder,
		lookahead_steps: int = 1,
		direct_weight: float = 0.90,
		future_weight: float = 0.35,
		novelty_weight: float = 0.10,
		profile_name: str | None = None,
	):
		if lookahead_steps <= 0:
			raise ValueError("lookahead_steps must be positive.")
		self.embedder = embedder
		self.lookahead_steps = lookahead_steps
		self.direct_weight = direct_weight
		self.future_weight = future_weight
		self.novelty_weight = novelty_weight
		self.metadata = StepScorerMetadata(
			scorer_kind=self.scorer_kind,
			backend=getattr(embedder, "backend_name", "sentence_transformer"),
			profile_name=profile_name,
			provider=None,
			model=getattr(embedder, "model_name", None),
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
		if not candidate_links:
			return []
		query_embedding = self.embedder.encode([query])[0]
		edge_texts = [_edge_text(graph, link) for link in candidate_links]
		edge_embeddings = self.embedder.encode(edge_texts)
		cards: list[StepScoreCard] = []
		for index, (link, edge_embedding) in enumerate(
			zip(candidate_links, edge_embeddings)
		):
			direct_similarity = _clamp_score(
				_dot_similarity(query_embedding, edge_embedding)
			)
			novelty = 1.0 if link.target not in visited_nodes else 0.0
			future_similarity, best_next_edge_id = self._best_future_score(
				query=query,
				query_embedding=query_embedding,
				graph=graph,
				link=link,
				visited_nodes=visited_nodes,
				remaining_steps=remaining_steps,
				edge_id=str(index),
			)
			if self.lookahead_steps > 1 and remaining_steps > 1:
				total_score = _clamp_score(
					self.direct_weight * direct_similarity
					+ self.future_weight * future_similarity
					+ self.novelty_weight * novelty
				)
				subscores = {
					"direct_support": direct_similarity,
					"future_potential": future_similarity,
					"novelty": novelty,
				}
			else:
				total_score = _clamp_score(
					self.direct_weight * direct_similarity
					+ self.novelty_weight * novelty
				)
				subscores = {
					"direct_support": direct_similarity,
					"novelty": novelty,
				}
				best_next_edge_id = None
			cards.append(
				StepScoreCard(
					edge_id=str(index),
					total_score=total_score,
					subscores=subscores,
					rationale=None,
					text=None,
					backend=self.metadata.backend,
					provider=None,
					model=self.metadata.model,
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

	def _best_future_score(
		self,
		*,
		query: str,
		query_embedding: Sequence[float],
		graph: LinkContextGraph,
		link: LinkContext,
		visited_nodes: set[str],
		remaining_steps: int,
		edge_id: str,
	) -> tuple[float, str | None]:
		if self.lookahead_steps <= 1 or remaining_steps <= 1:
			return 0.0, None
		next_visited = set(visited_nodes)
		next_visited.add(link.target)
		next_links: list[LinkContext] = []
		next_edge_ids: list[str] = []
		for next_index, neighbor in enumerate(graph.neighbors(link.target)):
			if neighbor in next_visited:
				continue
			for inner_index, next_link in enumerate(
				graph.links_between(link.target, neighbor)
			):
				next_edge_offset = len(next_links)
				next_links.append(next_link)
				next_edge_ids.append(f"{edge_id}-{next_edge_offset}")
		if not next_links:
			return 0.0, None
		next_embeddings = self.embedder.encode(
			[_edge_text(graph, next_link) for next_link in next_links]
		)
		scored = [
			(
				_clamp_score(_dot_similarity(query_embedding, embedding)),
				next_edge_ids[index],
			)
			for index, embedding in enumerate(next_embeddings)
		]
		best_score, best_edge_id = max(scored, key=lambda item: (item[0], item[1]))
		return best_score, best_edge_id


class SemanticPPRSelector:
	strategy_name = "semantic_ppr"

	def __init__(self, *, alpha: float = 0.15, max_iter: int = 20, tol: float = 1e-6):
		self.alpha = alpha
		self.max_iter = max_iter
		self.tol = tol

	def select(
		self,
		graph: LinkContextGraph,
		query: str,
		start_nodes: list[str],
		budget: SelectorBudget | None = None,
		*,
		scorer: StepLinkScorer | None = None,
	) -> CorpusSelectionResult:
		budget = budget or SelectorBudget()
		scorer = scorer or LinkContextOverlapStepScorer()
		node_scores, link_scores, debug_trace = self._run_ppr(
			graph, query, start_nodes, scorer
		)
		return _build_corpus_selection_result(
			selector_name=self.strategy_name,
			graph=graph,
			query=query,
			node_scores=node_scores,
			link_scores=link_scores,
			budget=budget,
			strategy=self.strategy_name,
			mode=SelectionMode.STANDALONE,
			debug_trace=debug_trace,
			root_node_ids=start_nodes,
			trace=_trace_from_ranked_nodes(node_scores),
		)

	def _run_ppr(
		self,
		graph: LinkContextGraph,
		query: str,
		start_nodes: list[str],
		scorer: StepLinkScorer,
		*,
		seed_weights: dict[str, float] | None = None,
	) -> tuple[
		dict[str, float],
		dict[tuple[str, str, str, str, int, str | None], ScoredLink],
		list[str],
	]:
		if not start_nodes:
			raise ValueError("semantic_ppr requires at least one start node.")
		personalization = _normalize_map(
			seed_weights or _seed_weights_from_graph(graph, query, start_nodes)
		)
		if not personalization:
			personalization = {node_id: 1 / len(start_nodes) for node_id in start_nodes}

		transitions: dict[str, dict[str, float]] = {}
		representative_links: dict[tuple[str, str], ScoredLink] = {}
		debug_trace = [
			f"seed:{node_id}:{weight:.4f}"
			for node_id, weight in personalization.items()
		]

		for source in graph.nodes:
			candidate_links = list(graph.links_from(source))
			if not candidate_links:
				continue
			cards = scorer.score_candidates(
				query=query,
				graph=graph,
				current_node_id=source,
				candidate_links=candidate_links,
				visited_nodes={source},
				path_node_ids=[source],
				remaining_steps=1,
			)
			target_weights: dict[str, float] = defaultdict(float)
			target_best_link: dict[str, tuple[float, ScoredLink]] = {}
			for link, card in zip(candidate_links, cards):
				if card.total_score <= 0:
					continue
				scored = ScoredLink.from_link(
					link,
					score=card.total_score,
					source_strategy=self.strategy_name,
					selected_reason="ppr",
				)
				target_weights[link.target] += card.total_score
				best_score, _best_link = target_best_link.get(
					link.target, (float("-inf"), scored)
				)
				if card.total_score > best_score:
					target_best_link[link.target] = (card.total_score, scored)
			if target_weights:
				transitions[source] = dict(target_weights)
				for target, (_score, link) in target_best_link.items():
					representative_links[(source, target)] = link

		probabilities = dict(personalization)
		for iteration in range(self.max_iter):
			updated = {
				node_id: self.alpha * personalization.get(node_id, 0.0)
				for node_id in graph.nodes
			}
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
					updated[target] = updated.get(target, 0.0) + (
						1 - self.alpha
					) * source_prob * (weight / total_weight)
			if dangling_mass > 0:
				for target, weight in personalization.items():
					updated[target] = (
						updated.get(target, 0.0)
						+ (1 - self.alpha) * dangling_mass * weight
					)
			delta = sum(
				abs(updated.get(node_id, 0.0) - probabilities.get(node_id, 0.0))
				for node_id in graph.nodes
			)
			debug_trace.append(f"iter:{iteration}:delta={delta:.6f}")
			probabilities = updated
			if delta < self.tol:
				break

		link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = {}
		for (source, target), scored_link in representative_links.items():
			source_prob = probabilities.get(source, 0.0)
			outgoing = transitions.get(source, {})
			total_weight = sum(outgoing.values())
			if total_weight <= 0:
				continue
			edge_score = source_prob * (outgoing[target] / total_weight)
			updated_link = replace(scored_link, score=edge_score)
			link_scores[_link_key(updated_link)] = updated_link
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
		budget: SelectorBudget | None = None,
		*,
		scorer: StepLinkScorer | None = None,
		heuristic: CoverageSemanticHeuristic | None = None,
		selector_name: str | None = None,
		seed_weights: dict[str, float] | None = None,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		if not start_nodes:
			raise ValueError(f"{self.strategy_name} requires at least one start node.")
		budget = budget or SelectorBudget()
		scorer = scorer or LinkContextOverlapStepScorer()
		heuristic = heuristic or CoverageSemanticHeuristic()
		query_tokens = _query_tokens(query)
		if resume_state is not None and resume_state.get("substage") == "ppr":
			path_result = corpus_selection_result_from_dict(
				dict(resume_state["path_result"])
			)
		else:
			states, debug_trace, selector_logs = self._run_path_search(
				graph=graph,
				query=query,
				query_tokens=query_tokens,
				start_nodes=start_nodes,
				budget=budget,
				scorer=scorer,
				heuristic=heuristic,
				resume_state=resume_state,
				checkpoint_callback=checkpoint_callback,
				stop_callback=stop_callback,
			)
			path_result = _build_selection_from_states(
				selector_name=selector_name or self.strategy_name,
				graph=graph,
				query=query,
				query_tokens=query_tokens,
				states=states,
				budget=budget,
				strategy=self.strategy_name,
				mode=SelectionMode.STANDALONE,
				debug_trace=debug_trace,
				root_node_ids=start_nodes,
				selector_metadata=_selector_metadata_from_step_scorer(
					scorer,
					seed_top_k=len(start_nodes),
					hop_budget=budget.max_hops,
					search_structure=self._search_structure_name(),
				),
				selector_usage=_selector_usage_from_logs(selector_logs),
				selector_logs=selector_logs,
				stop_reason="path_search",
			)
		if self.mode == SelectionMode.STANDALONE:
			return path_result

		frontier_scores = {
			node.node_id: node.score for node in path_result.ranked_nodes
		}
		personalization = dict(
			seed_weights or _seed_weights_from_graph(graph, query, start_nodes)
		)
		for node_id, score in frontier_scores.items():
			personalization[node_id] = personalization.get(node_id, 0.0) + score
		if checkpoint_callback is not None:
			checkpoint_callback(
				{
					"substage": "ppr",
					"query": query,
					"start_nodes": list(start_nodes),
					"budget": selector_budget_to_dict(budget),
					"path_result": corpus_selection_result_to_dict(path_result),
					"seed_weights": dict(seed_weights or {}),
				}
			)
		if stop_callback is not None:
			stop_callback()
		ppr_node_scores, ppr_link_scores, ppr_trace = self._ppr._run_ppr(
			graph,
			query,
			list(dict.fromkeys([*start_nodes, *path_result.selected_node_ids])),
			scorer,
			seed_weights=personalization,
		)
		return _build_hybrid_result(
			selector_name=selector_name or self.strategy_name,
			graph=graph,
			query=query,
			query_tokens=query_tokens,
			path_result=path_result,
			ppr_node_scores=ppr_node_scores,
			ppr_link_scores=ppr_link_scores,
			budget=budget,
			strategy=f"{self.strategy_name}_ppr",
			debug_trace=[*path_result.debug_trace, *ppr_trace],
			selector_metadata=path_result.selector_metadata,
			selector_usage=path_result.selector_usage,
			selector_logs=path_result.selector_logs,
		)

	def _search_structure_name(self) -> str:
		return self.strategy_name.removeprefix("semantic_")

	def _run_path_search(
		self,
		*,
		graph: LinkContextGraph,
		query: str,
		query_tokens: frozenset[str],
		start_nodes: list[str],
		budget: SelectorBudget,
		scorer: StepLinkScorer,
		heuristic: CoverageSemanticHeuristic,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> tuple[list[PathState], list[str], list[WalkStepLog]]:
		raise NotImplementedError

	def _seed_states(
		self,
		graph: LinkContextGraph,
		query_tokens: frozenset[str],
		start_nodes: list[str],
		heuristic: CoverageSemanticHeuristic,
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
			state.h_score = heuristic.estimate(query_tokens, graph, state)
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
		budget: SelectorBudget,
		scorer: StepLinkScorer,
		heuristic: CoverageSemanticHeuristic,
	) -> tuple[list[PathState], WalkStepLog | None]:
		if budget.max_hops is not None and state.depth >= budget.max_hops:
			return [], None
		visited_set = set(state.visited_nodes)
		covered_set = set(state.covered_query_tokens)
		query_token_count = max(len(query_tokens), 1)
		candidate_links = [
			link
			for link in graph.links_from(state.current_node)
			if self.allow_revisit or link.target not in visited_set
		]
		if not candidate_links:
			return [], None
		score_cards = scorer.score_candidates(
			query=query,
			graph=graph,
			current_node_id=state.current_node,
			candidate_links=candidate_links,
			visited_nodes=visited_set,
			path_node_ids=list(state.visited_nodes),
			remaining_steps=max((budget.max_hops or 1) - state.depth, 1),
		)
		if not score_cards:
			return [], None

		best_index, best_card, best_link = max(
			(
				(index, score_cards[index], candidate_links[index])
				for index in range(min(len(candidate_links), len(score_cards)))
			),
			key=lambda item: (
				item[1].total_score,
				item[1].subscores.get("future_potential", 0.0),
				-item[0],
			),
		)
		next_states: list[PathState] = []
		for link, card in zip(candidate_links, score_cards):
			if card.total_score <= 0:
				continue
			link_tokens = _link_query_tokens(query_tokens, link)
			new_covered_tokens = tuple(sorted(covered_set | link_tokens))
			coverage_gain = (
				len(new_covered_tokens) - len(covered_set)
			) / query_token_count
			reward = card.total_score + 0.20 * coverage_gain
			edge_cost = 1.0 - min(reward, 1.0)
			scored_link = ScoredLink.from_link(
				link,
				score=card.total_score,
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
			next_state.h_score = heuristic.estimate(query_tokens, graph, next_state)
			next_state.f_score = next_state.g_score + next_state.h_score
			next_state.goal_reached = (
				next_state.coverage_ratio(len(query_tokens)) >= 1.0
			)
			next_states.append(next_state)

		log = build_walk_step_log(
			step_index=state.depth,
			current_node_id=state.current_node,
			path_node_ids=state.visited_nodes,
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
		return next_states, log

	def _expansion_limit(self, graph: LinkContextGraph, budget: SelectorBudget) -> int:
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
		budget: SelectorBudget,
		scorer: StepLinkScorer,
		heuristic: CoverageSemanticHeuristic,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> tuple[list[PathState], list[str], list[WalkStepLog]]:
		if resume_state is not None:
			debug_trace = [str(item) for item in resume_state.get("debug_trace", [])]
			selector_logs = [
				walk_step_log_from_dict(log)
				for log in resume_state.get("selector_logs", [])
			]
			all_states = [
				path_state_from_dict(state)
				for state in resume_state.get("all_states", [])
			]
			frontier = [
				path_state_from_dict(state)
				for state in resume_state.get("frontier", [])
			]
			frontier_index = int(resume_state.get("frontier_index", 0))
			next_frontier = [
				path_state_from_dict(state)
				for state in resume_state.get("next_frontier", [])
			]
		else:
			debug_trace = []
			selector_logs = []
			seeds = self._seed_states(graph, query_tokens, start_nodes, heuristic)
			for seed in seeds:
				debug_trace.append(f"seed:{seed.current_node}:h={seed.h_score:.4f}")
			all_states = list(seeds)
			frontier = list(seeds)
			frontier_index = 0
			next_frontier = []
		while frontier:
			for state in frontier[frontier_index:]:
				debug_trace.append(
					f"expand:{state.current_node}:reward={state.reward_score:.4f}:depth={state.depth}"
				)
				expanded, log = self._expand_state(
					graph=graph,
					query=query,
					query_tokens=query_tokens,
					state=state,
					budget=budget,
					scorer=scorer,
					heuristic=heuristic,
				)
				if log is not None:
					selector_logs.append(log)
				next_frontier.extend(expanded)
				frontier_index += 1
				if checkpoint_callback is not None:
					checkpoint_callback(
						{
							"substage": "search",
							"query": query,
							"start_nodes": list(start_nodes),
							"budget": selector_budget_to_dict(budget),
							"all_states": [
								path_state_to_dict(item) for item in all_states
							],
							"frontier": [path_state_to_dict(item) for item in frontier],
							"frontier_index": frontier_index,
							"next_frontier": [
								path_state_to_dict(item) for item in next_frontier
							],
							"debug_trace": list(debug_trace),
							"selector_logs": [
								walk_step_log_to_dict(item) for item in selector_logs
							],
						}
					)
				if stop_callback is not None:
					stop_callback()
			if not next_frontier:
				break
			next_frontier.sort(
				key=lambda item: (
					item.reward_score,
					item.coverage_ratio(len(query_tokens)),
					-item.g_score,
				),
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
			frontier_index = 0
			next_frontier = []
		return all_states, debug_trace, selector_logs


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
		budget: SelectorBudget,
		scorer: StepLinkScorer,
		heuristic: CoverageSemanticHeuristic,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> tuple[list[PathState], list[str], list[WalkStepLog]]:
		if resume_state is not None:
			debug_trace = [str(item) for item in resume_state.get("debug_trace", [])]
			selector_logs = [
				walk_step_log_from_dict(log)
				for log in resume_state.get("selector_logs", [])
			]
			all_states = [
				path_state_from_dict(state)
				for state in resume_state.get("all_states", [])
			]
			heap = [
				(
					float(entry["priority"]),
					int(entry["order"]),
					path_state_from_dict(dict(entry["state"])),
				)
				for entry in resume_state.get("heap_entries", [])
			]
			next_counter_value = int(resume_state.get("next_counter", len(heap)))
			expansions = int(resume_state.get("expansions", 0))
		else:
			next_counter_value = 0
			debug_trace = []
			selector_logs = []
			seeds = self._seed_states(graph, query_tokens, start_nodes, heuristic)
			all_states = list(seeds)
			heap = []
			for seed in seeds:
				priority = self._priority(seed)
				heapq.heappush(heap, (priority, next_counter_value, seed))
				next_counter_value += 1
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
			children, log = self._expand_state(
				graph=graph,
				query=query,
				query_tokens=query_tokens,
				state=state,
				budget=budget,
				scorer=scorer,
				heuristic=heuristic,
			)
			if log is not None:
				selector_logs.append(log)
			for child in children:
				all_states.append(child)
				child_priority = self._priority(child)
				heapq.heappush(heap, (child_priority, next_counter_value, child))
				next_counter_value += 1
				debug_trace.append(
					f"push:{state.current_node}->{child.current_node}:{self.priority_label}={child_priority:.4f}:g={child.g_score:.4f}:h={child.h_score:.4f}:f={child.f_score:.4f}"
				)
			if checkpoint_callback is not None:
				checkpoint_callback(
					{
						"substage": "search",
						"query": query,
						"start_nodes": list(start_nodes),
						"budget": selector_budget_to_dict(budget),
						"all_states": [path_state_to_dict(item) for item in all_states],
						"heap_entries": [
							{
								"priority": item[0],
								"order": item[1],
								"state": path_state_to_dict(item[2]),
							}
							for item in heap
						],
						"next_counter": next_counter_value,
						"expansions": expansions,
						"debug_trace": list(debug_trace),
						"selector_logs": [
							walk_step_log_to_dict(item) for item in selector_logs
						],
					}
				)
			if stop_callback is not None:
				stop_callback()
		return all_states, debug_trace, selector_logs


class SemanticAStarSelector(_PriorityPathSelector):
	strategy_name = "semantic_astar"
	priority_label = "f"

	def _priority(self, state: PathState) -> float:
		return state.f_score


class SemanticUCSSelector(_PriorityPathSelector):
	strategy_name = "semantic_ucs"
	priority_label = "g"

	def _priority(self, state: PathState) -> float:
		return state.g_score


class CanonicalDenseSelector(_SentenceTransformerSupport):
	def __init__(
		self,
		spec: SelectorSpec,
		*,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.spec = spec
		self.name = spec.canonical_name

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		started_at = time.perf_counter()
		seed_candidates = _select_seed_candidates(
			graph,
			case.query,
			self.spec.seed_top_k or 1,
			seed_strategy=self.spec.seed_strategy or "lexical_overlap",
			embedder=_seed_embedder(self.spec, self._get_embedder),
		)
		root_candidates = [node_id for node_id, _score in seed_candidates]
		node_scores = dict(seed_candidates)
		runtime_s = time.perf_counter() - started_at
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		return _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			node_scores=node_scores,
			link_scores={},
			budget=SelectorBudget(
				max_nodes=None,
				max_hops=0,
				max_tokens=_runtime_budget_token_limit(graph, budget),
			),
			strategy="dense",
			mode=SelectionMode.STANDALONE,
			debug_trace=[
				f"seed:{node_id}:{score:.4f}" for node_id, score in seed_candidates
			],
			root_node_ids=root_candidates,
			trace=_trace_from_seed_candidates(seed_candidates),
			selector_metadata=_baseline_selector_metadata(
				self.spec, seed_backend=seed_backend, seed_model=seed_model
			),
			selector_usage=SelectorUsage(runtime_s=runtime_s),
			stop_reason="top_k_retrieval",
		)


_DENSE_RERANK_DEFAULT_POOL_K = 64


class CanonicalDenseRerankSelector(_SentenceTransformerSupport):
	"""Dense retrieval + cross-encoder reranking baseline."""

	def __init__(
		self,
		spec: SelectorSpec,
		*,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
		cross_encoder_config: CrossEncoderRerankerConfig | None = None,
		cross_encoder_factory: Callable[
			[CrossEncoderRerankerConfig], CrossEncoderReranker
		]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.spec = spec
		self.name = spec.canonical_name
		self._cross_encoder_config = (
			cross_encoder_config or CrossEncoderRerankerConfig()
		)
		self._cross_encoder_factory = cross_encoder_factory
		self._reranker: CrossEncoderReranker | None = None

	def _get_reranker(self) -> CrossEncoderReranker:
		if self._reranker is None:
			if self._cross_encoder_factory is not None:
				self._reranker = self._cross_encoder_factory(self._cross_encoder_config)
			else:
				self._reranker = CrossEncoderReranker(self._cross_encoder_config)
		return self._reranker

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		started_at = time.perf_counter()
		pool_k = self.spec.rerank_pool_k or _DENSE_RERANK_DEFAULT_POOL_K
		seed_candidates = _select_seed_candidates(
			graph,
			case.query,
			pool_k,
			seed_strategy=self.spec.seed_strategy or "lexical_overlap",
			embedder=_seed_embedder(self.spec, self._get_embedder),
		)
		if not seed_candidates:
			runtime_s = time.perf_counter() - started_at
			seed_backend, seed_model = _seed_backend_metadata(
				self.spec, self._get_embedder
			)
			return _build_corpus_selection_result(
				selector_name=self.name,
				graph=graph,
				query=case.query,
				node_scores={},
				link_scores={},
				budget=SelectorBudget(
					max_nodes=None,
					max_hops=0,
					max_tokens=_runtime_budget_token_limit(graph, budget),
				),
				strategy="dense_rerank",
				mode=SelectionMode.STANDALONE,
				debug_trace=["rerank:no_candidates"],
				root_node_ids=[],
				trace=[],
				selector_metadata=_dense_rerank_metadata(
					self.spec,
					seed_backend=seed_backend,
					seed_model=seed_model,
					rerank_model=self._cross_encoder_config.model_name,
				),
				selector_usage=SelectorUsage(runtime_s=runtime_s),
				stop_reason="dense_rerank_no_candidates",
			)

		reranker = self._get_reranker()
		passages = [_node_text(graph, node_id) for node_id, _score in seed_candidates]
		rerank_scores = reranker.score_pairs(case.query, passages)

		reranked = sorted(
			zip(seed_candidates, rerank_scores),
			key=lambda item: item[1],
			reverse=True,
		)

		node_scores = {
			node_id: rerank_score for (node_id, _orig_score), rerank_score in reranked
		}
		root_candidates = [
			node_id
			for (node_id, _orig_score), _rerank_score in reranked[
				: (self.spec.seed_top_k or 1)
			]
		]
		trace = [
			SelectionTraceStep(
				index=index,
				node_id=node_id,
				score=rerank_score,
			)
			for index, ((node_id, _orig_score), rerank_score) in enumerate(reranked)
		]
		debug_trace = [
			f"rerank:{node_id}:{orig_score:.4f}->{rerank_score:.4f}"
			for (node_id, orig_score), rerank_score in reranked
		]

		runtime_s = time.perf_counter() - started_at
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		return _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			node_scores=node_scores,
			link_scores={},
			budget=SelectorBudget(
				max_nodes=None,
				max_hops=0,
				max_tokens=_runtime_budget_token_limit(graph, budget),
			),
			strategy="dense_rerank",
			mode=SelectionMode.STANDALONE,
			debug_trace=debug_trace,
			root_node_ids=root_candidates,
			trace=trace,
			selector_metadata=_dense_rerank_metadata(
				self.spec,
				seed_backend=seed_backend,
				seed_model=seed_model,
				rerank_model=self._cross_encoder_config.model_name,
			),
			selector_usage=SelectorUsage(runtime_s=runtime_s),
			stop_reason="dense_rerank",
		)


class CanonicalIterativeDenseSelector(_SentenceTransformerSupport):
	def __init__(
		self,
		spec: SelectorSpec,
		*,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.spec = spec
		self.name = spec.canonical_name

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		started_at = time.perf_counter()
		embedder = _seed_embedder(self.spec, self._get_embedder)
		hop_budget = self.spec.hop_budget or 1
		per_hop_top_k = self.spec.seed_top_k or 1
		token_budget_limit = _runtime_budget_token_limit(graph, budget)

		if resume_state is not None:
			selected_order = [
				str(node_id) for node_id in resume_state.get("selected_order", [])
			]
			selected_set = {
				str(node_id) for node_id in resume_state.get("selected_set", [])
			}
			node_scores = {
				str(node_id): float(score)
				for node_id, score in resume_state.get("node_scores", {}).items()
			}
			trace = [
				selection_trace_step_from_dict(step)
				for step in resume_state.get("trace", [])
			]
			debug_trace = [str(item) for item in resume_state.get("debug_trace", [])]
			frontier = [str(node_id) for node_id in resume_state.get("frontier", [])]
			root_candidates = [
				str(node_id) for node_id in resume_state.get("root_candidates", [])
			]
			next_hop_index = int(resume_state.get("hop_index", 1))
		else:
			selected_order = []
			selected_set = set()
			node_scores = {}
			trace = []
			debug_trace = []
			seed_candidates = _select_seed_candidates(
				graph,
				case.query,
				per_hop_top_k,
				seed_strategy=self.spec.seed_strategy or "lexical_overlap",
				embedder=embedder,
			)
			root_candidates = [node_id for node_id, _score in seed_candidates]
			frontier = self._accept_candidates(
				seed_candidates,
				selected_order=selected_order,
				selected_set=selected_set,
				node_scores=node_scores,
				trace=trace,
				debug_trace=debug_trace,
				hop_index=0,
				score_reason="seed",
			)
			next_hop_index = 1
			if checkpoint_callback is not None:
				checkpoint_callback(
					self._checkpoint_payload(
						graph=graph,
						case=case,
						hop_index=next_hop_index,
						selected_order=selected_order,
						selected_set=selected_set,
						node_scores=node_scores,
						trace=trace,
						debug_trace=debug_trace,
						frontier=frontier,
						root_candidates=root_candidates,
						token_budget_limit=token_budget_limit,
					)
				)
			if stop_callback is not None:
				stop_callback()

		for hop_index in range(next_hop_index, hop_budget + 1):
			remaining_node_ids = [
				node_id for node_id in graph.nodes if node_id not in selected_set
			]
			if not remaining_node_ids or not frontier:
				break
			hop_candidates = _select_seed_candidates(
				graph,
				self._expansion_query(graph, case.query, frontier),
				per_hop_top_k,
				seed_strategy=self.spec.seed_strategy or "lexical_overlap",
				embedder=embedder,
				candidate_ids=remaining_node_ids,
			)
			if not hop_candidates:
				debug_trace.append(f"hop:{hop_index}:no_candidates")
				break
			frontier = self._accept_candidates(
				hop_candidates,
				selected_order=selected_order,
				selected_set=selected_set,
				node_scores=node_scores,
				trace=trace,
				debug_trace=debug_trace,
				hop_index=hop_index,
				score_reason="iterative_dense",
			)
			if not frontier:
				debug_trace.append(f"hop:{hop_index}:duplicates_only")
				break
			if checkpoint_callback is not None:
				checkpoint_callback(
					self._checkpoint_payload(
						graph=graph,
						case=case,
						hop_index=hop_index + 1,
						selected_order=selected_order,
						selected_set=selected_set,
						node_scores=node_scores,
						trace=trace,
						debug_trace=debug_trace,
						frontier=frontier,
						root_candidates=root_candidates,
						token_budget_limit=token_budget_limit,
					)
				)
			if stop_callback is not None:
				stop_callback()

		runtime_s = time.perf_counter() - started_at
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		result = _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			node_scores=node_scores,
			link_scores={},
			budget=SelectorBudget(
				max_nodes=None, max_hops=hop_budget, max_tokens=token_budget_limit
			),
			strategy="iterative_dense",
			mode=SelectionMode.STANDALONE,
			debug_trace=debug_trace,
			root_node_ids=root_candidates,
			trace=trace,
			selector_metadata=_baseline_selector_metadata(
				self.spec, seed_backend=seed_backend, seed_model=seed_model
			),
			selector_usage=SelectorUsage(runtime_s=runtime_s),
			stop_reason="iterative_dense_retrieval",
		)
		selected_node_ids = self._trim_to_budget(
			graph, selected_order, token_budget_limit
		)
		result.selected_node_ids = selected_node_ids
		result.token_cost_estimate = sum(
			_node_token_cost(graph, node_id) for node_id in selected_node_ids
		)
		selected_node_set = set(selected_node_ids)
		result.trace = [step for step in trace if step.node_id in selected_node_set]
		return result

	def _checkpoint_payload(
		self,
		*,
		graph: LinkContextGraph,
		case: SelectionCase,
		hop_index: int,
		selected_order: Sequence[str],
		selected_set: set[str],
		node_scores: dict[str, float],
		trace: Sequence[SelectionTraceStep],
		debug_trace: Sequence[str],
		frontier: Sequence[str],
		root_candidates: Sequence[str],
		token_budget_limit: int,
	) -> dict[str, Any]:
		remaining_node_ids = [
			node_id for node_id in graph.nodes if node_id not in selected_set
		]
		return {
			"query": case.query,
			"hop_index": hop_index,
			"selected_order": list(selected_order),
			"selected_set": sorted(selected_set),
			"node_scores": dict(node_scores),
			"trace": [selection_trace_step_to_dict(step) for step in trace],
			"debug_trace": list(debug_trace),
			"frontier": list(frontier),
			"root_candidates": list(root_candidates),
			"token_budget_limit": token_budget_limit,
			"remaining_node_ids": remaining_node_ids,
		}

	def _accept_candidates(
		self,
		candidates: Sequence[tuple[str, float]],
		*,
		selected_order: list[str],
		selected_set: set[str],
		node_scores: dict[str, float],
		trace: list[SelectionTraceStep],
		debug_trace: list[str],
		hop_index: int,
		score_reason: str,
	) -> list[str]:
		accepted: list[str] = []
		depth_bonus = max((self.spec.hop_budget or 1) - hop_index + 1, 1)
		for node_id, score in candidates:
			debug_trace.append(f"hop:{hop_index}:{score_reason}:{node_id}:{score:.4f}")
			if node_id in selected_set:
				continue
			selected_order.append(node_id)
			selected_set.add(node_id)
			node_scores[node_id] = depth_bonus + score
			trace.append(
				SelectionTraceStep(
					index=len(trace),
					node_id=node_id,
					score=node_scores[node_id],
				)
			)
			accepted.append(node_id)
		return accepted

	def _expansion_query(
		self, graph: LinkContextGraph, base_query: str, frontier_node_ids: Sequence[str]
	) -> str:
		context = " ".join(
			self._frontier_context(graph, node_id) for node_id in frontier_node_ids
		)
		return f"{base_query} {context}".strip()

	def _frontier_context(self, graph: LinkContextGraph, node_id: str) -> str:
		document = graph.get_document(node_id)
		if document is None:
			return node_id
		lead_sentence = document.sentences[0] if document.sentences else ""
		return f"{document.title} {lead_sentence}".strip()

	def _trim_to_budget(
		self,
		graph: LinkContextGraph,
		selected_order: Sequence[str],
		token_budget_limit: int,
	) -> list[str]:
		if token_budget_limit <= 0:
			return list(selected_order)
		selected: list[str] = []
		used_tokens = 0
		for node_id in selected_order:
			node_tokens = _node_token_cost(graph, node_id)
			if used_tokens + node_tokens > token_budget_limit:
				continue
			selected.append(node_id)
			used_tokens += node_tokens
		return selected


class CanonicalMDRLightSelector(CanonicalIterativeDenseSelector):
	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		started_at = time.perf_counter()
		embedder = _seed_embedder(self.spec, self._get_embedder)
		if embedder is None:
			raise ValueError("mdr_light requires sentence-transformer seed retrieval.")
		hop_budget = self.spec.hop_budget or 1
		per_hop_top_k = self.spec.seed_top_k or 1
		token_budget_limit = _runtime_budget_token_limit(graph, budget)

		if resume_state is not None:
			selected_order = [
				str(node_id) for node_id in resume_state.get("selected_order", [])
			]
			selected_set = {
				str(node_id) for node_id in resume_state.get("selected_set", [])
			}
			node_scores = {
				str(node_id): float(score)
				for node_id, score in resume_state.get("node_scores", {}).items()
			}
			trace = [
				selection_trace_step_from_dict(step)
				for step in resume_state.get("trace", [])
			]
			debug_trace = [str(item) for item in resume_state.get("debug_trace", [])]
			frontier = [str(node_id) for node_id in resume_state.get("frontier", [])]
			root_candidates = [
				str(node_id) for node_id in resume_state.get("root_candidates", [])
			]
			next_hop_index = int(resume_state.get("hop_index", 1))
		else:
			selected_order = []
			selected_set = set()
			node_scores = {}
			trace = []
			debug_trace = []
			seed_candidates = _select_seed_candidates(
				graph,
				case.query,
				per_hop_top_k,
				seed_strategy=self.spec.seed_strategy or "lexical_overlap",
				embedder=embedder,
			)
			root_candidates = [node_id for node_id, _score in seed_candidates]
			frontier = self._accept_candidates(
				seed_candidates,
				selected_order=selected_order,
				selected_set=selected_set,
				node_scores=node_scores,
				trace=trace,
				debug_trace=debug_trace,
				hop_index=0,
				score_reason="seed",
			)
			next_hop_index = 1
			if checkpoint_callback is not None:
				checkpoint_callback(
					self._checkpoint_payload(
						graph=graph,
						case=case,
						hop_index=next_hop_index,
						selected_order=selected_order,
						selected_set=selected_set,
						node_scores=node_scores,
						trace=trace,
						debug_trace=debug_trace,
						frontier=frontier,
						root_candidates=root_candidates,
						token_budget_limit=token_budget_limit,
					)
				)
			if stop_callback is not None:
				stop_callback()

		for hop_index in range(next_hop_index, hop_budget + 1):
			remaining_node_ids = [
				node_id for node_id in graph.nodes if node_id not in selected_set
			]
			if not remaining_node_ids or not frontier:
				break
			merged_candidates: dict[str, float] = {}
			for frontier_node_id in frontier:
				hop_candidates = _select_seed_candidates(
					graph,
					self._expansion_query(graph, case.query, [frontier_node_id]),
					per_hop_top_k,
					seed_strategy=self.spec.seed_strategy or "lexical_overlap",
					embedder=embedder,
					candidate_ids=remaining_node_ids,
				)
				if not hop_candidates:
					debug_trace.append(
						f"hop:{hop_index}:frontier:{frontier_node_id}:no_candidates"
					)
					continue
				for node_id, score in hop_candidates:
					debug_trace.append(
						f"hop:{hop_index}:frontier:{frontier_node_id}:mdr_light:{node_id}:{score:.4f}"
					)
					previous = merged_candidates.get(node_id)
					if previous is None or score > previous:
						merged_candidates[node_id] = score
			if not merged_candidates:
				debug_trace.append(f"hop:{hop_index}:no_candidates")
				break
			ranked_candidates = sorted(
				merged_candidates.items(),
				key=lambda item: (item[1], item[0]),
				reverse=True,
			)[:per_hop_top_k]
			frontier = self._accept_candidates(
				ranked_candidates,
				selected_order=selected_order,
				selected_set=selected_set,
				node_scores=node_scores,
				trace=trace,
				debug_trace=debug_trace,
				hop_index=hop_index,
				score_reason="mdr_light",
			)
			if not frontier:
				debug_trace.append(f"hop:{hop_index}:duplicates_only")
				break
			if checkpoint_callback is not None:
				checkpoint_callback(
					self._checkpoint_payload(
						graph=graph,
						case=case,
						hop_index=hop_index + 1,
						selected_order=selected_order,
						selected_set=selected_set,
						node_scores=node_scores,
						trace=trace,
						debug_trace=debug_trace,
						frontier=frontier,
						root_candidates=root_candidates,
						token_budget_limit=token_budget_limit,
					)
				)
			if stop_callback is not None:
				stop_callback()

		runtime_s = time.perf_counter() - started_at
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		result = _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			node_scores=node_scores,
			link_scores={},
			budget=SelectorBudget(
				max_nodes=None, max_hops=hop_budget, max_tokens=token_budget_limit
			),
			strategy="mdr_light",
			mode=SelectionMode.STANDALONE,
			debug_trace=debug_trace,
			root_node_ids=root_candidates,
			trace=trace,
			selector_metadata=_baseline_selector_metadata(
				self.spec, seed_backend=seed_backend, seed_model=seed_model
			),
			selector_usage=SelectorUsage(runtime_s=runtime_s),
			stop_reason="mdr_light_retrieval",
		)
		selected_node_ids = self._trim_to_budget(
			graph, selected_order, token_budget_limit
		)
		result.selected_node_ids = selected_node_ids
		result.token_cost_estimate = sum(
			_node_token_cost(graph, node_id) for node_id in selected_node_ids
		)
		selected_node_set = set(selected_node_ids)
		result.trace = [step for step in trace if step.node_id in selected_node_set]
		return result


class CanonicalNeighborSelector(_SentenceTransformerSupport):
	def __init__(
		self,
		spec: SelectorSpec,
		*,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.spec = spec
		self.name = spec.canonical_name

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		started_at = time.perf_counter()
		seed_candidates = _select_seed_candidates(
			graph,
			case.query,
			self.spec.seed_top_k or 1,
			seed_strategy=self.spec.seed_strategy or "lexical_overlap",
			embedder=_seed_embedder(self.spec, self._get_embedder),
		)
		root_candidates = [node_id for node_id, _score in seed_candidates]
		ordered_node_ids = list(dict.fromkeys(root_candidates))
		node_scores = dict(seed_candidates)
		link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = {}
		trace = _trace_from_seed_candidates(seed_candidates)
		for root_node_id in root_candidates:
			candidates: list[tuple[float, LinkContext]] = []
			for neighbor in graph.neighbors(root_node_id):
				links = list(graph.links_between(root_node_id, neighbor))
				if not links:
					continue
				best_score, best_link = max(
					(
						(self._score_link(graph, case.query, link), link)
						for link in links
					),
					key=lambda item: (item[0], item[1].target),
				)
				candidates.append((best_score, best_link))
			if not candidates:
				continue
			score, link = max(candidates, key=lambda item: (item[0], item[1].target))
			if link.target in node_scores:
				continue
			ordered_node_ids.append(link.target)
			node_scores[link.target] = max(
				score, _node_score(graph, case.query, link.target)
			)
			scored_link = ScoredLink.from_link(
				link,
				score=score,
				source_strategy=self.name,
				selected_reason="neighbor",
			)
			link_scores[_link_key(scored_link)] = scored_link
			trace.append(
				SelectionTraceStep(
					index=len(trace),
					node_id=link.target,
					score=score,
					source_node_id=link.source,
					anchor_text=link.anchor_text,
					sentence=link.sentence,
				)
			)
		runtime_s = time.perf_counter() - started_at
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		return _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			node_scores=node_scores,
			link_scores=link_scores,
			budget=SelectorBudget(
				max_nodes=None,
				max_hops=1,
				max_tokens=_runtime_budget_token_limit(graph, budget),
			),
			strategy=str(self.spec.baseline),
			mode=SelectionMode.STANDALONE,
			debug_trace=[
				f"seed:{node_id}:{score:.4f}" for node_id, score in seed_candidates
			],
			root_node_ids=root_candidates,
			trace=trace,
			selector_metadata=_baseline_selector_metadata(
				self.spec, seed_backend=seed_backend, seed_model=seed_model
			),
			selector_usage=SelectorUsage(runtime_s=runtime_s),
			stop_reason="neighbor_expansion",
		)

	def _score_link(
		self, graph: LinkContextGraph, query: str, link: LinkContext
	) -> float:
		assert self.spec.baseline is not None
		if self.spec.baseline == "topology_neighbors":
			return (
				normalized_token_overlap(query, _target_title(graph, link.target))
				+ len(graph.neighbors(link.target)) * 0.01
			)
		if self.spec.baseline == "anchor_neighbors":
			return normalized_token_overlap(query, link.anchor_text)
		if self.spec.baseline == "link_context_neighbors":
			return (
				normalized_token_overlap(query, link.anchor_text) * 0.6
				+ normalized_token_overlap(query, link.sentence) * 0.4
			)
		raise ValueError(f"Unsupported neighbor baseline: {self.spec.baseline}")


class CanonicalSinglePathSelector(_SentenceTransformerSupport):
	def __init__(
		self,
		spec: SelectorSpec,
		*,
		llm_config: SelectorLLMConfig | None = None,
		backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.spec = spec
		self.name = spec.canonical_name
		self.llm_config = llm_config or SelectorLLMConfig()
		self.backend_factory = backend_factory

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		started_at = time.perf_counter()
		seed_candidates = _select_seed_candidates(
			graph,
			case.query,
			self.spec.seed_top_k or 1,
			seed_strategy=self.spec.seed_strategy or "lexical_overlap",
			embedder=_seed_embedder(self.spec, self._get_embedder),
		)
		start_nodes = [node_id for node_id, _score in seed_candidates]
		scorer = _build_step_scorer_from_spec(
			self.spec,
			self.llm_config,
			self.backend_factory,
			embedder=_scorer_embedder(self.spec, self._get_embedder)
			or _seed_embedder(self.spec, self._get_embedder),
		)
		walk = DynamicWalker(graph, scorer=scorer).walk(
			case.query,
			start_nodes,
			WalkBudget(
				max_steps=_controller_runtime_max_steps(self.spec),
				min_score=self.spec.walk_score_threshold
				if self.spec.walk_score_threshold is not None
				else 0.05,
				allow_revisit=False,
			),
			resume_state=resume_state,
			checkpoint_callback=checkpoint_callback,
			stop_callback=stop_callback,
		)
		runtime_s = time.perf_counter() - started_at
		result = _corpus_selection_from_walk(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			walk=walk,
			root_node_ids=start_nodes,
			token_budget_limit=_runtime_budget_token_limit(graph, budget),
			runtime_s=runtime_s,
			spec=self.spec,
		)
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		return _apply_selector_metadata(
			result, self.spec, seed_backend=seed_backend, seed_model=seed_model
		)


class CanonicalConstrainedMultipathSelector(_SentenceTransformerSupport):
	def __init__(
		self,
		spec: SelectorSpec,
		*,
		llm_config: SelectorLLMConfig | None = None,
		backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.spec = spec
		self.name = spec.canonical_name
		self.llm_config = llm_config or SelectorLLMConfig()
		self.backend_factory = backend_factory

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		started_at = time.perf_counter()
		seed_candidates = _select_seed_candidates(
			graph,
			case.query,
			self.spec.seed_top_k or 1,
			seed_strategy=self.spec.seed_strategy or "lexical_overlap",
			embedder=_seed_embedder(self.spec, self._get_embedder),
		)
		start_nodes = [node_id for node_id, _score in seed_candidates]
		scorer = _build_step_scorer_from_spec(
			self.spec,
			self.llm_config,
			self.backend_factory,
			embedder=_scorer_embedder(self.spec, self._get_embedder)
			or _seed_embedder(self.spec, self._get_embedder),
		)
		if not isinstance(scorer, LLMControllerStepScorer):
			raise ValueError(
				"constrained_multipath requires link_context_llm_controller."
			)
		max_steps = _controller_runtime_max_steps(self.spec)
		token_budget_limit = _runtime_budget_token_limit(graph, budget)
		if resume_state is not None:
			current = str(resume_state["current_node"])
			visited_nodes = [
				str(node_id) for node_id in resume_state.get("visited_nodes", [])
			]
			visited_set = set(visited_nodes)
			steps = [
				walk_step_from_dict(step) for step in resume_state.get("steps", [])
			]
			if not steps:
				raise ValueError("constrained_multipath resume_state is missing steps.")
			selector_logs = [
				walk_step_log_from_dict(log)
				for log in resume_state.get("selector_logs", [])
			]
			debug_trace = [str(item) for item in resume_state.get("debug_trace", [])]
			forks_used = int(resume_state.get("forks_used", 0))
			backtracks_used = int(resume_state.get("backtracks_used", 0))
			merged_steps = [
				walk_step_from_dict(step)
				for step in resume_state.get("merged_steps", [])
			]
			merged_links = {
				_link_key(link): link
				for link in (
					scored_link_from_dict(link)
					for link in resume_state.get("merged_links", [])
				)
			}
		else:
			current = start_nodes[0]
			visited_nodes = [current]
			visited_set = {current}
			steps = [
				WalkStep(
					index=0,
					node_id=current,
					score=_node_score(graph, case.query, current),
				)
			]
			selector_logs = []
			debug_trace = [
				f"seed:{node_id}:{score:.4f}" for node_id, score in seed_candidates
			]
			forks_used = 0
			backtracks_used = 0
			merged_steps = []
			merged_links = {}

		stop_reason = "budget_exhausted"
		while len(steps) < max_steps:
			current_token_cost = sum(
				_node_token_cost(graph, node_id) for node_id in visited_nodes
			)
			candidate_links = [
				link
				for link in graph.links_from(current)
				if link.target not in visited_set
			]
			if not candidate_links:
				if apply_controller_backtrack(
					current_node_id=current,
					steps=steps,
					visited_nodes=visited_nodes,
					visited_set=visited_set,
					selector_logs=selector_logs,
					backtracks_used=backtracks_used,
					max_backtracks=self.llm_config.max_backtracks_per_case,
				):
					backtracks_used += 1
					current = visited_nodes[-1]
					continue
				stop_reason = "dead_end"
				break
			controller_execution: ControllerExecutionResult | None = None
			if isinstance(scorer, ControllerRuntimeScorer):
				controller_execution = scorer.evaluate_controller_step(
					query=case.query,
					graph=graph,
					current_node_id=current,
					candidate_links=candidate_links,
					visited_nodes=visited_set,
					path_node_ids=visited_nodes,
					remaining_steps=max_steps - len(steps),
					current_depth=max(len(steps) - 1, 0),
					forks_used=forks_used,
					backtracks_used=backtracks_used,
				)
				selector_logs.append(
					build_controller_walk_step_log(
						step_index=len(steps) - 1,
						current_node_id=current,
						path_node_ids=visited_nodes,
						execution=controller_execution,
					)
				)
				decision_state = controller_execution.trace.state
				decision_runner_up = controller_execution.trace.runner_up
				decision_action = controller_execution.effective_action
				if controller_execution.effective_action == "stop":
					stop_reason = controller_execution.stop_reason or "dead_end"
					selector_logs[-1].stop_reason = stop_reason
					debug_trace.append(
						f"{stop_reason}:{current}:{decision_state}:{decision_runner_up or 'none'}"
					)
					if stop_reason == "dead_end" and apply_controller_backtrack(
						current_node_id=current,
						steps=steps,
						visited_nodes=visited_nodes,
						visited_set=visited_set,
						selector_logs=selector_logs,
						backtracks_used=backtracks_used,
						max_backtracks=self.llm_config.max_backtracks_per_case,
					):
						backtracks_used += 1
						current = visited_nodes[-1]
						continue
					break
				primary = controller_execution.primary
				assert primary is not None
				primary_index, primary_card, primary_link = (
					primary.index,
					primary.score_card,
					primary.link,
				)
				secondary = controller_execution.secondary
				secondary_index, secondary_card, secondary_link = (
					(secondary.index, secondary.score_card, secondary.link)
					if secondary is not None
					else (None, None, None)
				)
			else:
				score_cards = scorer.score_candidates(
					query=case.query,
					graph=graph,
					current_node_id=current,
					candidate_links=candidate_links,
					visited_nodes=visited_set,
					path_node_ids=visited_nodes,
					remaining_steps=max_steps - len(steps),
				)
				if not score_cards:
					stop_reason = "dead_end"
					break
				primary_index, primary_card, primary_link = self._choose_edge(
					candidate_links=candidate_links,
					score_cards=score_cards,
					preferred_edge_id=None,
				)
				assert primary_card is not None
				assert primary_link is not None
				secondary_index, secondary_card, secondary_link = (None, None, None)
				decision_state = "need_bridge"
				decision_runner_up = None
				decision_action = "choose_one"
				selector_logs.append(
					build_walk_step_log(
						step_index=len(steps) - 1,
						current_node_id=current,
						path_node_ids=visited_nodes,
						chosen_edge_id=primary_card.edge_id,
						chosen_target_node_id=primary_link.target,
						backend=primary_card.backend,
						provider=primary_card.provider,
						model=primary_card.model,
						latency_s=primary_card.latency_s,
						prompt_tokens=primary_card.prompt_tokens,
						completion_tokens=primary_card.completion_tokens,
						total_tokens=primary_card.total_tokens,
						cache_hit=primary_card.cache_hit,
						fallback_reason=primary_card.fallback_reason,
						llm_calls=primary_card.llm_calls,
						text=primary_card.text,
						raw_response=primary_card.raw_response,
						stop_reason=None,
						candidates=build_step_candidate_traces(
							candidate_links=candidate_links,
							score_cards=score_cards,
						),
					)
				)
			if primary_card.total_score < (
				self.spec.walk_score_threshold
				if self.spec.walk_score_threshold is not None
				else 0.05
			):
				if apply_controller_backtrack(
					current_node_id=current,
					steps=steps,
					visited_nodes=visited_nodes,
					visited_set=visited_set,
					selector_logs=selector_logs,
					backtracks_used=backtracks_used,
					max_backtracks=self.llm_config.max_backtracks_per_case,
				):
					backtracks_used += 1
					current = visited_nodes[-1]
					continue
				stop_reason = "score_below_threshold"
				break

			chosen_card = primary_card
			chosen_link = primary_link
			chosen_index = primary_index
			if (
				decision_action == "choose_two"
				and secondary_card is not None
				and secondary_link is not None
				and forks_used < 1
			):
				winner_card, winner_link, winner_index = (
					primary_card,
					primary_link,
					primary_index,
				)
				loser_card, loser_link = secondary_card, secondary_link
				if self._branch_score(secondary_card) > self._branch_score(
					primary_card
				):
					winner_card, winner_link, winner_index = (
						secondary_card,
						secondary_link,
						secondary_index,
					)
					loser_card, loser_link = primary_card, primary_link
				chosen_card, chosen_link, chosen_index = (
					winner_card,
					winner_link,
					winner_index,
				)
				forks_used += 1
				debug_trace.append(
					f"fork:{current}:keep={winner_link.target}:drop={loser_link.target}:state={decision_state}"
				)
				loser_tokens = _node_token_cost(graph, loser_link.target)
				if (
					loser_card.subscores.get("redundancy_risk", 1.0) < 0.60
					and token_budget_limit > 0
					and loser_tokens <= max(1, math.floor(token_budget_limit * 0.10))
					and loser_link.target not in visited_set
				):
					merged_steps.append(
						WalkStep(
							index=max_steps + len(merged_steps),
							node_id=loser_link.target,
							score=loser_card.total_score,
							source_node_id=loser_link.source,
							anchor_text=loser_link.anchor_text,
							sentence=loser_link.sentence,
							edge_id=loser_card.edge_id,
						)
					)
					merged_link = ScoredLink.from_link(
						loser_link,
						score=loser_card.total_score,
						source_strategy=self.name,
						selected_reason="scout_merge",
					)
					merged_links[_link_key(merged_link)] = merged_link
					debug_trace.append(
						f"merge:{loser_link.target}:{loser_card.total_score:.4f}"
					)

			visited_nodes.append(chosen_link.target)
			visited_set.add(chosen_link.target)
			current = chosen_link.target
			steps.append(
				WalkStep(
					index=len(steps),
					node_id=current,
					score=chosen_card.total_score,
					source_node_id=chosen_link.source,
					anchor_text=chosen_link.anchor_text,
					sentence=chosen_link.sentence,
					edge_id=str(chosen_index),
				)
			)
			debug_trace.append(
				f"walk:{chosen_link.source}->{chosen_link.target}:{chosen_card.total_score:.4f}"
			)
			if (
				budget.budget_mode == "tokens"
				and token_budget_limit > 0
				and current_token_cost >= math.floor(token_budget_limit * 0.35)
				and (decision_runner_up == "stop" or decision_state == "drift_recovery")
				and len(steps) > 1
			):
				stop_reason = "budget_pacing_stop"
				selector_logs[-1].stop_reason = stop_reason
				debug_trace.append(f"budget_pacing_stop:{current}:{current_token_cost}")
				break
			if checkpoint_callback is not None:
				checkpoint_callback(
					self._checkpoint_payload(
						query=case.query,
						start_nodes=start_nodes,
						current_node=current,
						visited_nodes=visited_nodes,
						steps=steps,
						selector_logs=selector_logs,
						remaining_steps=max_steps - len(steps),
						debug_trace=debug_trace,
						forks_used=forks_used,
						backtracks_used=backtracks_used,
						merged_steps=merged_steps,
						merged_links=merged_links,
					)
				)
			if stop_callback is not None:
				stop_callback()
		runtime_s = time.perf_counter() - started_at
		result = self._result_from_steps(
			graph=graph,
			query=case.query,
			start_nodes=start_nodes,
			steps=steps,
			merged_steps=merged_steps,
			merged_links=merged_links,
			token_budget_limit=token_budget_limit,
			selector_logs=selector_logs,
			runtime_s=runtime_s,
			scorer=scorer,
			stop_reason=stop_reason,
			debug_trace=debug_trace,
		)
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		return _apply_selector_metadata(
			result, self.spec, seed_backend=seed_backend, seed_model=seed_model
		)

	def _checkpoint_payload(
		self,
		*,
		query: str,
		start_nodes: Sequence[str],
		current_node: str,
		visited_nodes: Sequence[str],
		steps: Sequence[WalkStep],
		selector_logs: Sequence[WalkStepLog],
		remaining_steps: int,
		debug_trace: Sequence[str],
		forks_used: int,
		backtracks_used: int,
		merged_steps: Sequence[WalkStep],
		merged_links: dict[tuple[str, str, str, str, int, str | None], ScoredLink],
	) -> dict[str, Any]:
		return {
			"query": query,
			"start_nodes": list(start_nodes),
			"current_node": current_node,
			"visited_nodes": list(visited_nodes),
			"steps": [walk_step_to_dict(step) for step in steps],
			"selector_logs": [walk_step_log_to_dict(log) for log in selector_logs],
			"remaining_steps": remaining_steps,
			"debug_trace": list(debug_trace),
			"forks_used": forks_used,
			"backtracks_used": backtracks_used,
			"merged_steps": [walk_step_to_dict(step) for step in merged_steps],
			"merged_links": [
				scored_link_to_dict(link) for link in merged_links.values()
			],
		}

	def _result_from_steps(
		self,
		*,
		graph: LinkContextGraph,
		query: str,
		start_nodes: Sequence[str],
		steps: Sequence[WalkStep],
		merged_steps: Sequence[WalkStep],
		merged_links: dict[tuple[str, str, str, str, int, str | None], ScoredLink],
		token_budget_limit: int,
		selector_logs: Sequence[WalkStepLog],
		runtime_s: float,
		scorer: StepLinkScorer,
		stop_reason: str,
		debug_trace: Sequence[str],
	) -> CorpusSelectionResult:
		node_scores: dict[str, float] = {}
		link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = (
			dict(merged_links)
		)
		trace: list[SelectionTraceStep] = []
		for step in [*steps, *merged_steps]:
			node_scores[step.node_id] = max(
				node_scores.get(step.node_id, 0.0), step.score
			)
			trace.append(
				SelectionTraceStep(
					index=len(trace),
					node_id=step.node_id,
					score=step.score,
					source_node_id=step.source_node_id,
					anchor_text=step.anchor_text,
					sentence=step.sentence,
				)
			)
			if (
				step.source_node_id is not None
				and step.anchor_text is not None
				and step.sentence is not None
			):
				for link in graph.links_between(step.source_node_id, step.node_id):
					if (
						link.anchor_text == step.anchor_text
						and link.sentence == step.sentence
					):
						scored_link = ScoredLink.from_link(
							link,
							score=step.score,
							source_strategy=self.name,
							selected_reason="constrained_multipath",
						)
						link_scores[_link_key(scored_link)] = scored_link
						break
		return _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=query,
			node_scores=node_scores,
			link_scores=link_scores,
			budget=SelectorBudget(
				max_nodes=None,
				max_hops=self.spec.hop_budget,
				max_tokens=token_budget_limit,
			),
			strategy="constrained_multipath",
			mode=SelectionMode.STANDALONE,
			debug_trace=list(debug_trace),
			root_node_ids=start_nodes,
			trace=trace,
			selector_metadata=_selector_metadata_from_step_scorer(
				scorer,
				seed_top_k=self.spec.seed_top_k,
				hop_budget=self.spec.hop_budget,
				search_structure="constrained_multipath",
				edge_scorer=self.spec.edge_scorer,
				lookahead_depth=self.spec.lookahead_depth,
				profile_name=_resolved_profile_name(self.spec),
			),
			selector_usage=_selector_usage_from_logs(
				selector_logs, runtime_override=runtime_s
			),
			selector_logs=selector_logs,
			stop_reason=stop_reason,
		)

	def _choose_edge(
		self,
		*,
		candidate_links: Sequence[LinkContext],
		score_cards: Sequence[StepScoreCard],
		preferred_edge_id: str | None,
	) -> tuple[int, StepScoreCard, LinkContext] | tuple[None, None, None]:
		if preferred_edge_id is not None:
			for index, (link, card) in enumerate(
				zip(candidate_links, score_cards, strict=False)
			):
				if card.edge_id == preferred_edge_id:
					return index, card, link
		if not candidate_links or not score_cards:
			return None, None, None
		return max(
			(
				(index, score_cards[index], candidate_links[index])
				for index in range(min(len(candidate_links), len(score_cards)))
			),
			key=lambda item: (
				item[1].total_score,
				item[1].subscores.get("future_potential", 0.0),
				-item[0],
			),
		)

	def _branch_score(self, card: StepScoreCard) -> float:
		return card.total_score


class CanonicalSearchSelector(_SentenceTransformerSupport):
	def __init__(
		self,
		spec: SelectorSpec,
		*,
		llm_config: SelectorLLMConfig | None = None,
		backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.spec = spec
		self.name = spec.canonical_name
		self.llm_config = llm_config or SelectorLLMConfig()
		self.backend_factory = backend_factory

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		seed_candidates = _select_seed_candidates(
			graph,
			case.query,
			self.spec.seed_top_k or 1,
			seed_strategy=self.spec.seed_strategy or "lexical_overlap",
			embedder=_seed_embedder(self.spec, self._get_embedder),
		)
		start_nodes = [node_id for node_id, _score in seed_candidates]
		path_budget = SelectorBudget(
			max_nodes=None,
			max_hops=self.spec.hop_budget,
			max_tokens=_runtime_budget_token_limit(graph, budget),
		)
		scorer = _build_step_scorer_from_spec(
			self.spec,
			self.llm_config,
			self.backend_factory,
			embedder=_scorer_embedder(self.spec, self._get_embedder)
			or _seed_embedder(self.spec, self._get_embedder),
		)
		heuristic = CoverageSemanticHeuristic()
		assert self.spec.search_structure is not None  # set by SelectorSpec validation
		selector = _build_algorithm(self.spec.search_structure)
		result = selector.select(
			graph,
			case.query,
			start_nodes,
			budget=path_budget,
			scorer=scorer,
			heuristic=heuristic,
			selector_name=self.name,
			seed_weights={node_id: score for node_id, score in seed_candidates},
			resume_state=resume_state,
			checkpoint_callback=checkpoint_callback,
			stop_callback=stop_callback,
		)
		seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
		return _apply_selector_metadata(
			result, self.spec, seed_backend=seed_backend, seed_model=seed_model
		)


class BudgetFillSelector(_SentenceTransformerSupport):
	def __init__(
		self,
		base_selector: CorpusSelector,
		spec: SelectorSpec,
		*,
		embedder_config: SentenceTransformerSelectorConfig | None = None,
		embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder]
		| None = None,
	):
		super().__init__(
			embedder_config=embedder_config, embedder_factory=embedder_factory
		)
		self.base_selector = base_selector
		self.spec = spec
		self.name = spec.canonical_name
		self._seed_candidates_cache: dict[tuple[str, str], list[tuple[str, float]]] = {}

	def select(
		self,
		graph: LinkContextGraph,
		case: SelectionCase,
		budget: RuntimeBudget,
		*,
		base_result: CorpusSelectionResult | None = None,
		resume_state: dict[str, Any] | None = None,
		checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
		stop_callback: Callable[[], None] | None = None,
	) -> CorpusSelectionResult:
		if resume_state is not None:
			result = corpus_selection_result_from_dict(
				dict(resume_state["base_result"])
			)
		elif base_result is not None:
			result = base_result
		else:
			result = self.base_selector.select(graph, case, budget)
		budget_token_limit = _runtime_budget_token_limit(graph, budget)
		if resume_state is not None:
			seed_candidates = [
				(str(item["node_id"]), float(item["score"]))
				for item in resume_state.get("seed_candidates", [])
			]
			candidate_index = int(resume_state.get("candidate_index", 0))
			selected_node_ids = [
				str(node_id) for node_id in resume_state.get("selected_node_ids", [])
			]
			selected_node_set = {
				str(node_id) for node_id in resume_state.get("selected_node_set", [])
			}
			ranked_nodes = [
				scored_node_from_dict(node)
				for node in resume_state.get("ranked_nodes", [])
			]
			trace = [
				selection_trace_step_from_dict(step)
				for step in resume_state.get("trace", [])
			]
			debug_trace = [str(item) for item in resume_state.get("debug_trace", [])]
			token_cost_estimate = int(
				resume_state.get("token_cost_estimate", result.token_cost_estimate)
			)
			first_backfill_score = (
				None
				if resume_state.get("first_backfill_score") is None
				else float(resume_state["first_backfill_score"])
			)
		else:
			cache_key = (case.case_id, case.query)
			if self.spec.budget_fill_mode == "neighbor":
				seed_candidates = _neighbor_fill_candidates(
					graph,
					case.query,
					result.selected_node_ids,
					self.spec.budget_fill_pool_k or 64,
					seed_strategy=self.spec.seed_strategy or "lexical_overlap",
					embedder=_seed_embedder(self.spec, self._get_embedder),
				)
			elif cache_key in self._seed_candidates_cache:
				seed_candidates = self._seed_candidates_cache[cache_key]
			else:
				seed_candidates = _select_seed_candidates(
					graph,
					case.query,
					self.spec.budget_fill_pool_k or 64,
					seed_strategy=self.spec.seed_strategy or "lexical_overlap",
					embedder=_seed_embedder(self.spec, self._get_embedder),
				)
				self._seed_candidates_cache[cache_key] = seed_candidates
			if self.spec.budget_fill_mode == "diverse":
				seed_candidates = _mmr_rerank_candidates(
					[
						(nid, sc)
						for nid, sc in seed_candidates
						if nid not in set(result.selected_node_ids)
					],
					result.selected_node_ids,
					graph,
					self._get_embedder(),
				)
			candidate_index = 0
			selected_node_ids = list(result.selected_node_ids)
			selected_node_set = set(selected_node_ids)
			ranked_nodes = list(result.ranked_nodes)
			trace = list(result.trace)
			debug_trace = list(result.debug_trace)
			token_cost_estimate = result.token_cost_estimate
			first_backfill_score = None
			if checkpoint_callback is not None:
				checkpoint_callback(
					self._checkpoint_payload(
						case=case,
						result=result,
						seed_candidates=seed_candidates,
						candidate_index=candidate_index,
						selected_node_ids=selected_node_ids,
						selected_node_set=selected_node_set,
						ranked_nodes=ranked_nodes,
						trace=trace,
						debug_trace=debug_trace,
						token_cost_estimate=token_cost_estimate,
						first_backfill_score=first_backfill_score,
						budget_token_limit=budget_token_limit,
					)
				)
			if stop_callback is not None:
				stop_callback()

		for offset, (node_id, score) in enumerate(
			seed_candidates[candidate_index:], start=candidate_index
		):
			if node_id in selected_node_set:
				continue
			if first_backfill_score is None:
				first_backfill_score = score
			if not _should_continue_budget_fill(
				self.spec,
				score=score,
				first_backfill_candidate_score=first_backfill_score,
			):
				debug_trace.append(
					f"fill_stop:{self.spec.budget_fill_mode}:{node_id}:{score:.4f}"
				)
				break
			node_tokens = _node_token_cost(graph, node_id)
			if (
				budget_token_limit > 0
				and token_cost_estimate + node_tokens > budget_token_limit
			):
				debug_trace.append(
					f"fill_skip_budget:{self.spec.budget_fill_mode}:{node_id}:{score:.4f}"
				)
				continue
			ranked_nodes.append(
				ScoredNode(
					node_id=node_id,
					score=score,
					source_strategy=self.name,
					selected_reason=f"budget_fill_{self.spec.budget_fill_mode}",
				)
			)
			selected_node_ids.append(node_id)
			selected_node_set.add(node_id)
			token_cost_estimate += node_tokens
			trace.append(
				SelectionTraceStep(
					index=len(trace),
					node_id=node_id,
					score=score,
				)
			)
			debug_trace.append(
				f"fill_add:{self.spec.budget_fill_mode}:{node_id}:{score:.4f}"
			)
			if checkpoint_callback is not None:
				checkpoint_callback(
					self._checkpoint_payload(
						case=case,
						result=result,
						seed_candidates=seed_candidates,
						candidate_index=offset + 1,
						selected_node_ids=selected_node_ids,
						selected_node_set=selected_node_set,
						ranked_nodes=ranked_nodes,
						trace=trace,
						debug_trace=debug_trace,
						token_cost_estimate=token_cost_estimate,
						first_backfill_score=first_backfill_score,
						budget_token_limit=budget_token_limit,
					)
				)
			if stop_callback is not None:
				stop_callback()

		selector_metadata = _apply_budget_fill_metadata(
			result.selector_metadata, self.spec
		)
		selected_links = [
			link
			for link in result.ranked_links
			if link.source in selected_node_set and link.target in selected_node_set
		]
		return replace(
			result,
			selector_name=self.name,
			ranked_nodes=ranked_nodes,
			selected_node_ids=selected_node_ids,
			selected_links=selected_links,
			token_cost_estimate=token_cost_estimate,
			debug_trace=debug_trace,
			coverage_ratio=_selection_coverage_ratio(
				_query_tokens(case.query), selected_links
			),
			trace=trace,
			selector_metadata=selector_metadata,
		)

	def _checkpoint_payload(
		self,
		*,
		case: SelectionCase,
		result: CorpusSelectionResult,
		seed_candidates: Sequence[tuple[str, float]],
		candidate_index: int,
		selected_node_ids: Sequence[str],
		selected_node_set: set[str],
		ranked_nodes: Sequence[ScoredNode],
		trace: Sequence[SelectionTraceStep],
		debug_trace: Sequence[str],
		token_cost_estimate: int,
		first_backfill_score: float | None,
		budget_token_limit: int,
	) -> dict[str, Any]:
		return {
			"query": case.query,
			"base_result": corpus_selection_result_to_dict(result),
			"seed_candidates": [
				{"node_id": node_id, "score": score}
				for node_id, score in seed_candidates
			],
			"candidate_index": candidate_index,
			"selected_node_ids": list(selected_node_ids),
			"selected_node_set": sorted(selected_node_set),
			"ranked_nodes": [scored_node_to_dict(node) for node in ranked_nodes],
			"trace": [selection_trace_step_to_dict(step) for step in trace],
			"debug_trace": list(debug_trace),
			"token_cost_estimate": token_cost_estimate,
			"first_backfill_score": first_backfill_score,
			"budget_token_limit": budget_token_limit,
		}


class GoldSupportContextSelector:
	name = "gold_support_context"
	spec = SelectorSpec(canonical_name="gold_support_context", family="diagnostic")

	def select(
		self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget
	) -> CorpusSelectionResult:
		del budget
		ordered_nodes = list(dict.fromkeys(case.gold_support_nodes))
		node_scores = {node_id: 1.0 for node_id in ordered_nodes}
		return _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			node_scores=node_scores,
			link_scores={},
			budget=SelectorBudget(max_nodes=None, max_hops=None, max_tokens=None),
			strategy="gold_support_context",
			mode=SelectionMode.STANDALONE,
			debug_trace=["gold_support_context"],
			root_node_ids=list(
				dict.fromkeys(case.gold_start_nodes or case.gold_support_nodes)
			),
			trace=_trace_from_ordered_nodes(
				graph, case.query, ordered_nodes, fixed_score=1.0
			),
			stop_reason="gold_support_context",
			ignore_budget=True,
		)


class FullCorpusUpperBoundSelector:
	name = "full_corpus_upper_bound"
	spec = SelectorSpec(canonical_name="full_corpus_upper_bound", family="diagnostic")

	def select(
		self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget
	) -> CorpusSelectionResult:
		del budget
		ordered_nodes = list(graph.nodes)
		node_scores = {
			node_id: _node_score(graph, case.query, node_id)
			for node_id in ordered_nodes
		}
		return _build_corpus_selection_result(
			selector_name=self.name,
			graph=graph,
			query=case.query,
			node_scores=node_scores,
			link_scores={},
			budget=SelectorBudget(max_nodes=None, max_hops=None, max_tokens=None),
			strategy="full_corpus_upper_bound",
			mode=SelectionMode.STANDALONE,
			debug_trace=["full_corpus_proxy"],
			root_node_ids=_dense_seed_nodes(graph, case.query, 3),
			trace=_trace_from_ordered_nodes(graph, case.query, ordered_nodes),
			stop_reason="full_corpus_proxy",
			force_full_corpus=True,
		)


def _split_budget_fill_suffix(name: str) -> tuple[str, BudgetFillMode | None]:
	for suffix, mode in _BUDGET_FILL_SUFFIXES.items():
		marker = f"__{suffix}"
		if name.endswith(marker):
			return name.removesuffix(marker), mode
	return name, None


def _split_profile_suffix(name: str) -> tuple[str, str | None]:
	marker = "__profile_"
	if marker not in name:
		return name, None
	base_name, profile_name = name.rsplit(marker, 1)
	if not base_name or not profile_name:
		return name, None
	return base_name, profile_name


def _resolved_profile_name(spec: SelectorSpec) -> str | None:
	if spec.profile_name is not None:
		return spec.profile_name
	if spec.edge_scorer == "link_context_overlap":
		return _OVERLAP_PROFILE_DEFAULT
	if spec.edge_scorer == "link_context_sentence_transformer":
		return _SENTENCE_TRANSFORMER_PROFILE_DEFAULT
	return None


def _validate_profile_for_spec(spec: SelectorSpec, *, raw_name: str) -> None:
	if spec.profile_name is None:
		return
	if spec.family != "path_search":
		raise ValueError(f"Unknown selector: {raw_name}")
	if spec.edge_scorer == "link_context_overlap":
		if spec.profile_name not in _OVERLAP_PROFILES:
			raise ValueError(f"Unknown selector: {raw_name}")
		return
	if spec.edge_scorer == "link_context_sentence_transformer":
		if spec.profile_name not in _SENTENCE_TRANSFORMER_PROFILES:
			raise ValueError(f"Unknown selector: {raw_name}")
		return
	raise ValueError(f"Unknown selector: {raw_name}")


def parse_selector_spec(name: str) -> SelectorSpec:
	base_name, budget_fill_mode = _split_budget_fill_suffix(name)
	spec_name, profile_name = _split_profile_suffix(base_name)
	if spec_name == "gold_support_context":
		if budget_fill_mode is not None or profile_name is not None:
			raise ValueError(f"Unknown selector: {name}")
		return GoldSupportContextSelector.spec
	if spec_name == "full_corpus_upper_bound":
		if budget_fill_mode is not None or profile_name is not None:
			raise ValueError(f"Unknown selector: {name}")
		return FullCorpusUpperBoundSelector.spec
	match = _SELECTOR_PATTERN.fullmatch(spec_name)
	if match is None:
		raise ValueError(f"Unknown selector: {name}")
	seed_top_k = int(match.group("seed_top_k"))
	seed_strategy = match.group("seed_strategy")
	hop_token = match.group("hop_budget")
	hop_budget = None if hop_token == _CONTROLLER_ADAPTIVE_HOP_TOKEN else int(hop_token)
	rest = match.group("rest")
	parts = rest.split("__")
	if len(parts) == 1:
		if hop_budget is None:
			raise ValueError(f"Unknown selector: {name}")
		baseline = parts[0]
		if baseline not in _BASELINES:
			raise ValueError(f"Unknown selector: {name}")
		if baseline == "dense" and hop_budget != 0:
			raise ValueError(f"Unknown selector: {name}")
		if baseline == "dense_rerank" and hop_budget != 0:
			raise ValueError(f"Unknown selector: {name}")
		if baseline == "dense_rerank" and seed_strategy != "sentence_transformer":
			raise ValueError(f"Unknown selector: {name}")
		if baseline == "iterative_dense" and hop_budget < 1:
			raise ValueError(f"Unknown selector: {name}")
		if baseline == "mdr_light":
			if seed_strategy != "sentence_transformer" or hop_budget < 1:
				raise ValueError(f"Unknown selector: {name}")
		if (
			baseline not in {"dense", "dense_rerank", "iterative_dense", "mdr_light"}
			and hop_budget != 1
		):
			raise ValueError(f"Unknown selector: {name}")
		spec = SelectorSpec(
			canonical_name=name,
			base_canonical_name=spec_name,
			family="baseline",
			profile_name=profile_name,
			seed_strategy=seed_strategy,  # type: ignore[arg-type]
			seed_top_k=seed_top_k,
			hop_budget=hop_budget,
			baseline=baseline,  # type: ignore[arg-type]
			budget_fill_mode=budget_fill_mode,
			budget_fill_pool_k=64 if budget_fill_mode is not None else None,
			budget_fill_score_floor=0.05 if budget_fill_mode == "score_floor" else None,
			budget_fill_relative_drop_ratio=0.5
			if budget_fill_mode in ("relative_drop", "neighbor", "diverse")
			else None,
			rerank_pool_k=64 if baseline == "dense_rerank" else None,
		)
		_validate_profile_for_spec(spec, raw_name=name)
		return spec
	if len(parts) != 3:
		raise ValueError(f"Unknown selector: {name}")
	search_structure, edge_scorer, lookahead = parts
	if (
		search_structure not in _SEARCH_STRUCTURES
		or edge_scorer not in _EDGE_SCORERS
		or lookahead not in _LOOKAHEADS
	):
		raise ValueError(f"Unknown selector: {name}")
	if edge_scorer == "link_context_llm_controller":
		if hop_budget is not None:
			raise ValueError(f"Unknown selector: {name}")
	else:
		if hop_budget is None:
			raise ValueError(f"Unknown selector: {name}")
	if edge_scorer == "link_context_llm_controller" and search_structure not in {
		"single_path_walk",
		"constrained_multipath",
	}:
		raise ValueError(f"Unknown selector: {name}")
	if search_structure == "constrained_multipath" and (
		edge_scorer != "link_context_llm_controller" or lookahead != "lookahead_2"
	):
		raise ValueError(f"Unknown selector: {name}")
	spec = SelectorSpec(
		canonical_name=name,
		base_canonical_name=spec_name,
		family="path_search",
		profile_name=profile_name,
		seed_strategy=seed_strategy,  # type: ignore[arg-type]
		seed_top_k=seed_top_k,
		hop_budget=hop_budget,
		search_structure=search_structure,  # type: ignore[arg-type]
		edge_scorer=edge_scorer,  # type: ignore[arg-type]
		lookahead_depth=int(lookahead.removeprefix("lookahead_")),
		budget_fill_mode=budget_fill_mode,
		budget_fill_pool_k=64 if budget_fill_mode is not None else None,
		budget_fill_score_floor=0.05 if budget_fill_mode == "score_floor" else None,
		budget_fill_relative_drop_ratio=0.5
		if budget_fill_mode in ("relative_drop", "neighbor", "diverse")
		else None,
	)
	_validate_profile_for_spec(spec, raw_name=name)
	return spec


def available_selector_names(*, include_diagnostics: bool = True) -> list[str]:
	from hypercorpus.baselines.mdr import EXTERNAL_MDR_SELECTOR_NAME

	names: list[str] = []
	for seed_strategy in ("sentence_transformer", "lexical_overlap"):
		names.extend(
			[
				f"top_1_seed__{seed_strategy}__hop_0__dense",
				f"top_1_seed__{seed_strategy}__hop_1__topology_neighbors",
				f"top_1_seed__{seed_strategy}__hop_1__anchor_neighbors",
				f"top_1_seed__{seed_strategy}__hop_1__link_context_neighbors",
				f"top_1_seed__{seed_strategy}__hop_2__single_path_walk__anchor_overlap__lookahead_1",
				f"top_1_seed__{seed_strategy}__hop_2__single_path_walk__link_context_overlap__lookahead_1",
				f"top_1_seed__{seed_strategy}__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_1",
				f"top_1_seed__{seed_strategy}__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2",
				f"top_1_seed__{seed_strategy}__hop_2__single_path_walk__link_context_llm__lookahead_1",
				f"top_1_seed__{seed_strategy}__hop_2__single_path_walk__link_context_llm__lookahead_2",
				f"top_1_seed__{seed_strategy}__hop_adaptive__single_path_walk__link_context_llm_controller__lookahead_2",
				f"top_1_seed__{seed_strategy}__hop_adaptive__constrained_multipath__link_context_llm_controller__lookahead_2",
				f"top_3_seed__{seed_strategy}__hop_0__dense",
				f"top_3_seed__{seed_strategy}__hop_1__topology_neighbors",
				f"top_3_seed__{seed_strategy}__hop_1__anchor_neighbors",
				f"top_3_seed__{seed_strategy}__hop_1__link_context_neighbors",
			]
		)
		if seed_strategy == "sentence_transformer":
			names.extend(
				[
					f"top_1_seed__{seed_strategy}__hop_0__dense_rerank",
					f"top_3_seed__{seed_strategy}__hop_0__dense_rerank",
					f"top_1_seed__{seed_strategy}__hop_2__iterative_dense",
					f"top_3_seed__{seed_strategy}__hop_2__iterative_dense",
					f"top_1_seed__{seed_strategy}__hop_2__mdr_light",
					f"top_3_seed__{seed_strategy}__hop_2__mdr_light",
				]
			)
		for search_structure in ("beam", "astar", "ucs", "beam_ppr"):
			names.extend(
				[
					f"top_3_seed__{seed_strategy}__hop_3__{search_structure}__link_context_overlap__lookahead_1",
					f"top_3_seed__{seed_strategy}__hop_3__{search_structure}__link_context_sentence_transformer__lookahead_1",
					f"top_3_seed__{seed_strategy}__hop_3__{search_structure}__link_context_sentence_transformer__lookahead_2",
					f"top_3_seed__{seed_strategy}__hop_3__{search_structure}__link_context_llm__lookahead_1",
					f"top_3_seed__{seed_strategy}__hop_3__{search_structure}__link_context_llm__lookahead_2",
				]
			)
	if include_diagnostics:
		names.extend(_DIAGNOSTIC_SELECTORS)
	names.append(EXTERNAL_MDR_SELECTOR_NAME)
	return names


def available_selector_presets() -> list[str]:
	return ["full", "paper_recommended", "paper_recommended_local", "branchy_profiles"]


def selector_names_for_preset(
	preset: SelectorPresetName | str,
	*,
	include_diagnostics: bool = True,
) -> list[str]:
	if preset == "full":
		return available_selector_names(include_diagnostics=include_diagnostics)
	if preset == "paper_recommended":
		names = list(_PAPER_RECOMMENDED_SELECTORS)
		if include_diagnostics:
			return names
		return [name for name in names if name not in _DIAGNOSTIC_SELECTORS]
	if preset == "paper_recommended_local":
		names = list(_PAPER_RECOMMENDED_LOCAL_SELECTORS)
		if include_diagnostics:
			return names
		return [name for name in names if name not in _DIAGNOSTIC_SELECTORS]
	if preset == "branchy_profiles":
		names = list(_BRANCHY_PROFILE_SELECTORS)
		if include_diagnostics:
			return names
		return [name for name in names if name not in _DIAGNOSTIC_SELECTORS]
	raise ValueError(f"Unknown selector preset: {preset}")


def build_selector(
	name: str,
	*,
	selector_provider: Literal["copilot", "openai", "anthropic", "gemini"]
	| str = "copilot",
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: OpenAIApiMode | str | None = None,
	selector_cache_path: str | None = None,
	selector_backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	sentence_transformer_embedder_factory: Callable[
		[SentenceTransformerEmbedderConfig], TextEmbedder
	]
	| None = None,
	cross_encoder_model: str | None = None,
	cross_encoder_device: str | None = None,
	cross_encoder_factory: Callable[[CrossEncoderRerankerConfig], CrossEncoderReranker]
	| None = None,
	mdr_home: str | Path | None = None,
	mdr_artifact_manifest: str | Path | None = None,
	budget_fill_ratio: float | None = None,
	budget_fill_pool_k: int | None = None,
	walk_score_threshold: float | None = None,
	seed_top_k: int | None = None,
) -> CorpusSelector:
	from hypercorpus.baselines.mdr import (
		EXTERNAL_MDR_SELECTOR_NAME,
		ExternalMDRSelector,
	)

	if name == EXTERNAL_MDR_SELECTOR_NAME:
		if mdr_artifact_manifest is None:
			raise ValueError(
				f"{EXTERNAL_MDR_SELECTOR_NAME} requires mdr_artifact_manifest."
			)
		return ExternalMDRSelector(
			name=name,
			artifact_manifest_path=mdr_artifact_manifest,
			mdr_home=mdr_home,
		)
	spec = parse_selector_spec(name)
	if budget_fill_ratio is not None and spec.budget_fill_mode in (
		"relative_drop",
		"neighbor",
		"diverse",
	):
		spec = replace(spec, budget_fill_relative_drop_ratio=budget_fill_ratio)
	if budget_fill_pool_k is not None and spec.budget_fill_pool_k is not None:
		spec = replace(spec, budget_fill_pool_k=budget_fill_pool_k)
	if walk_score_threshold is not None:
		spec = replace(spec, walk_score_threshold=walk_score_threshold)
	if seed_top_k is not None:
		spec = replace(spec, seed_top_k=seed_top_k)
	llm_config = SelectorLLMConfig(
		provider=cast(SelectorProvider, selector_provider),
		model=selector_model,
		api_key_env=selector_api_key_env,
		base_url=selector_base_url,
		openai_api_mode=cast(OpenAIApiMode | None, selector_openai_api_mode),
		cache_path=None if selector_cache_path is None else Path(selector_cache_path),
	)
	sentence_transformer_config = SentenceTransformerSelectorConfig(
		model_name=sentence_transformer_model or DEFAULT_SENTENCE_TRANSFORMER_MODEL,
		cache_path=None
		if sentence_transformer_cache_path is None
		else Path(sentence_transformer_cache_path),
		device=sentence_transformer_device,
	)
	cross_encoder_config = CrossEncoderRerankerConfig(
		model_name=cross_encoder_model or DEFAULT_CROSS_ENCODER_MODEL,
		device=cross_encoder_device or sentence_transformer_device,
	)
	if spec.family == "diagnostic":
		if name == "gold_support_context":
			return GoldSupportContextSelector()
		return FullCorpusUpperBoundSelector()
	if spec.edge_scorer in {"link_context_llm", "link_context_llm_controller"}:
		_validate_selector_llm_config(llm_config)
	selector = _build_canonical_selector(
		spec,
		llm_config=llm_config,
		backend_factory=selector_backend_factory,
		sentence_transformer_config=sentence_transformer_config,
		sentence_transformer_embedder_factory=sentence_transformer_embedder_factory,
		cross_encoder_config=cross_encoder_config,
		cross_encoder_factory=cross_encoder_factory,
	)
	if spec.budget_fill_mode is None:
		return selector
	return BudgetFillSelector(
		selector,
		spec,
		embedder_config=sentence_transformer_config,
		embedder_factory=sentence_transformer_embedder_factory,
	)


def _build_canonical_selector(
	spec: SelectorSpec,
	*,
	llm_config: SelectorLLMConfig,
	backend_factory: Callable[[SelectorLLMConfig], Any] | None,
	sentence_transformer_config: SentenceTransformerSelectorConfig,
	sentence_transformer_embedder_factory: Callable[
		[SentenceTransformerEmbedderConfig], TextEmbedder
	]
	| None,
	cross_encoder_config: CrossEncoderRerankerConfig | None = None,
	cross_encoder_factory: Callable[[CrossEncoderRerankerConfig], CrossEncoderReranker]
	| None = None,
) -> CorpusSelector:
	if spec.family == "baseline":
		if spec.baseline == "dense":
			return CanonicalDenseSelector(
				spec,
				embedder_config=sentence_transformer_config,
				embedder_factory=sentence_transformer_embedder_factory,
			)
		if spec.baseline == "dense_rerank":
			return CanonicalDenseRerankSelector(
				spec,
				embedder_config=sentence_transformer_config,
				embedder_factory=sentence_transformer_embedder_factory,
				cross_encoder_config=cross_encoder_config,
				cross_encoder_factory=cross_encoder_factory,
			)
		if spec.baseline == "iterative_dense":
			return CanonicalIterativeDenseSelector(
				spec,
				embedder_config=sentence_transformer_config,
				embedder_factory=sentence_transformer_embedder_factory,
			)
		if spec.baseline == "mdr_light":
			return CanonicalMDRLightSelector(
				spec,
				embedder_config=sentence_transformer_config,
				embedder_factory=sentence_transformer_embedder_factory,
			)
		return CanonicalNeighborSelector(
			spec,
			embedder_config=sentence_transformer_config,
			embedder_factory=sentence_transformer_embedder_factory,
		)
	assert spec.search_structure is not None
	if spec.search_structure == "single_path_walk":
		return CanonicalSinglePathSelector(
			spec,
			llm_config=llm_config,
			backend_factory=backend_factory,
			embedder_config=sentence_transformer_config,
			embedder_factory=sentence_transformer_embedder_factory,
		)
	if spec.search_structure == "constrained_multipath":
		return CanonicalConstrainedMultipathSelector(
			spec,
			llm_config=llm_config,
			backend_factory=backend_factory,
			embedder_config=sentence_transformer_config,
			embedder_factory=sentence_transformer_embedder_factory,
		)
	return CanonicalSearchSelector(
		spec,
		llm_config=llm_config,
		backend_factory=backend_factory,
		embedder_config=sentence_transformer_config,
		embedder_factory=sentence_transformer_embedder_factory,
	)


def select_selectors(
	names: Sequence[str] | None = None,
	*,
	preset: SelectorPresetName | str = "full",
	include_diagnostics: bool = True,
	selector_provider: Literal["copilot", "openai", "anthropic", "gemini"]
	| str = "copilot",
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: OpenAIApiMode | str | None = None,
	selector_cache_path: str | None = None,
	selector_backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	sentence_transformer_embedder_factory: Callable[
		[SentenceTransformerEmbedderConfig], TextEmbedder
	]
	| None = None,
	cross_encoder_model: str | None = None,
	cross_encoder_device: str | None = None,
	cross_encoder_factory: Callable[[CrossEncoderRerankerConfig], CrossEncoderReranker]
	| None = None,
	mdr_home: str | Path | None = None,
	mdr_artifact_manifest: str | Path | None = None,
	budget_fill_ratio: float | None = None,
	budget_fill_pool_k: int | None = None,
	walk_score_threshold: float | None = None,
	seed_top_k: int | None = None,
) -> list[CorpusSelector]:
	selector_names = (
		list(names)
		if names is not None
		else selector_names_for_preset(
			preset,
			include_diagnostics=include_diagnostics,
		)
	)
	return [
		build_selector(
			name,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			selector_backend_factory=selector_backend_factory,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			sentence_transformer_embedder_factory=sentence_transformer_embedder_factory,
			cross_encoder_model=cross_encoder_model,
			cross_encoder_device=cross_encoder_device,
			cross_encoder_factory=cross_encoder_factory,
			mdr_home=mdr_home,
			mdr_artifact_manifest=mdr_artifact_manifest,
			budget_fill_ratio=budget_fill_ratio,
			budget_fill_pool_k=budget_fill_pool_k,
			walk_score_threshold=walk_score_threshold,
			seed_top_k=seed_top_k,
		)
		for name in selector_names
	]


def _build_algorithm(search_structure: SearchStructure) -> _PathfindingSelector:
	if search_structure == "beam":
		return SemanticBeamSelector(beam_width=4, allow_revisit=False)
	if search_structure == "astar":
		return SemanticAStarSelector(allow_revisit=False, max_expansions=64)
	if search_structure == "ucs":
		return SemanticUCSSelector(allow_revisit=False, max_expansions=64)
	if search_structure == "beam_ppr":
		return SemanticBeamSelector(
			mode=SelectionMode.HYBRID_WITH_PPR, beam_width=4, allow_revisit=False
		)
	raise ValueError(f"Unsupported search structure: {search_structure}")


def _controller_runtime_hop_cap(spec: SelectorSpec) -> int | None:
	if spec.edge_scorer != "link_context_llm_controller":
		return spec.hop_budget
	return (
		_CONTROLLER_INTERNAL_SAFETY_HOPS if spec.hop_budget is None else spec.hop_budget
	)


def _controller_runtime_max_steps(spec: SelectorSpec) -> int:
	hop_cap = _controller_runtime_hop_cap(spec)
	if hop_cap is None:
		raise ValueError("controller runtime requires a concrete hop cap")
	return hop_cap + 1


def _build_step_scorer_from_spec(
	spec: SelectorSpec,
	llm_config: SelectorLLMConfig,
	backend_factory: Callable[[SelectorLLMConfig], Any] | None,
	*,
	embedder: TextEmbedder | None = None,
) -> StepLinkScorer:
	lookahead_steps = spec.lookahead_depth or 1
	if spec.edge_scorer == "anchor_overlap":
		return AnchorOverlapStepScorer(lookahead_steps=lookahead_steps)
	if spec.edge_scorer == "link_context_overlap":
		profile_name = _resolved_profile_name(spec)
		assert profile_name is not None
		profile = _OVERLAP_PROFILES[profile_name]
		return LinkContextOverlapStepScorer(
			anchor_weight=profile["anchor"],
			sentence_weight=profile["sentence"],
			target_weight=profile["title"],
			novelty_bonus=profile["novelty"],
			lookahead_steps=lookahead_steps,
			profile_name=profile_name,
		)
	if spec.edge_scorer == "link_context_sentence_transformer":
		if embedder is None:
			raise ValueError("Sentence-transformer edge scorers require an embedder.")
		profile_name = _resolved_profile_name(spec)
		assert profile_name is not None
		profile = _SENTENCE_TRANSFORMER_PROFILES[profile_name]
		return SentenceTransformerStepScorer(
			embedder=embedder,
			lookahead_steps=lookahead_steps,
			direct_weight=profile["direct"],
			future_weight=profile["future"],
			novelty_weight=profile["novelty"],
			profile_name=profile_name,
		)
	if spec.edge_scorer == "link_context_llm":
		mode = "two_hop" if lookahead_steps == 2 else "single_hop"
		return LLMStepLinkScorer(
			config=llm_config,
			mode=mode,
			prefilter_scorer=LinkContextOverlapStepScorer(),
			fallback_scorer=LinkContextOverlapStepScorer(
				lookahead_steps=lookahead_steps
			),
			backend_factory=backend_factory,
		)
	if spec.edge_scorer == "link_context_llm_controller":
		mode = "two_hop" if lookahead_steps == 2 else "single_hop"
		semantic_prefilter_scorer = (
			SentenceTransformerStepScorer(
				embedder=embedder,
				lookahead_steps=1,
				profile_name="controller_prefilter_semantic",
			)
			if embedder is not None
			else None
		)
		controller = LLMController(
			config=llm_config,
			mode=mode,
			prefilter_scorer=TitleAwareOverlapStepScorer(),
			semantic_prefilter_scorer=semantic_prefilter_scorer,
			fallback_scorer=LinkContextOverlapStepScorer(
				lookahead_steps=lookahead_steps
			),
			backend_factory=backend_factory,
		)
		return LLMControllerStepScorer(
			controller=controller,
			config=llm_config,
			mode=mode,
			fallback_scorer=LinkContextOverlapStepScorer(
				lookahead_steps=lookahead_steps
			),
		)
	raise ValueError(f"Unsupported edge scorer: {spec.edge_scorer}")


def _dense_seed_nodes(
	graph: LinkContextGraph,
	query: str,
	top_k: int,
	*,
	seed_strategy: SeedStrategyName = "lexical_overlap",
	embedder: TextEmbedder | None = None,
) -> list[str]:
	return [
		node_id
		for node_id, _score in _select_seed_candidates(
			graph,
			query,
			top_k,
			seed_strategy=seed_strategy,
			embedder=embedder,
		)
	]


def _seed_embedder(
	spec: SelectorSpec,
	embedder_getter: Callable[[], TextEmbedder],
) -> TextEmbedder | None:
	if spec.seed_strategy == "sentence_transformer":
		return embedder_getter()
	return None


def _scorer_embedder(
	spec: SelectorSpec,
	embedder_getter: Callable[[], TextEmbedder],
) -> TextEmbedder | None:
	if spec.edge_scorer == "link_context_sentence_transformer":
		return embedder_getter()
	return None


def _seed_backend_metadata(
	spec: SelectorSpec,
	embedder_getter: Callable[[], TextEmbedder],
) -> tuple[str | None, str | None]:
	if spec.seed_strategy == "sentence_transformer":
		embedder = embedder_getter()
		return getattr(embedder, "backend_name", "sentence_transformer"), getattr(
			embedder, "model_name", None
		)
	if spec.seed_strategy == "lexical_overlap":
		return "lexical_overlap", None
	return None, None


def _select_seed_candidates(
	graph: LinkContextGraph,
	query: str,
	top_k: int,
	*,
	seed_strategy: SeedStrategyName,
	embedder: TextEmbedder | None,
	candidate_ids: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
	if top_k <= 0:
		return []
	if seed_strategy == "lexical_overlap":
		return _lexical_seed_candidates(
			graph, query, top_k, candidate_ids=candidate_ids
		)
	if seed_strategy == "sentence_transformer":
		if embedder is None:
			raise ValueError("Sentence-transformer seed strategy requires an embedder.")
		return _sentence_transformer_seed_candidates(
			graph,
			query,
			top_k,
			embedder,
			candidate_ids=candidate_ids,
		)
	raise ValueError(f"Unsupported seed strategy: {seed_strategy}")


def _should_continue_budget_fill(
	spec: SelectorSpec,
	*,
	score: float,
	first_backfill_candidate_score: float,
) -> bool:
	if spec.budget_fill_mode == "always":
		return True
	if spec.budget_fill_mode == "score_floor":
		return score >= (spec.budget_fill_score_floor or 0.05)
	if spec.budget_fill_mode in ("relative_drop", "neighbor", "diverse"):
		threshold = first_backfill_candidate_score * (
			spec.budget_fill_relative_drop_ratio or 0.5
		)
		return score >= threshold
	return True


def _neighbor_fill_candidates(
	graph: LinkContextGraph,
	query: str,
	selected_node_ids: Sequence[str],
	top_k: int,
	*,
	seed_strategy: SeedStrategyName,
	embedder: TextEmbedder | None,
) -> list[tuple[str, float]]:
	"""Build fill pool from hyperlink neighbors of already-selected nodes."""
	selected_set = set(selected_node_ids)
	neighbor_ids: list[str] = []
	seen: set[str] = set()
	for node_id in selected_node_ids:
		for nbr in graph.neighbors(node_id):
			if nbr not in selected_set and nbr not in seen:
				neighbor_ids.append(nbr)
				seen.add(nbr)
	if not neighbor_ids:
		return []
	return _select_seed_candidates(
		graph,
		query,
		min(top_k, len(neighbor_ids)),
		seed_strategy=seed_strategy,
		embedder=embedder,
		candidate_ids=neighbor_ids,
	)


def _mmr_rerank_candidates(
	candidates: list[tuple[str, float]],
	selected_node_ids: Sequence[str],
	graph: LinkContextGraph,
	embedder: TextEmbedder,
	*,
	mmr_lambda: float = _DEFAULT_BUDGET_FILL_MMR_LAMBDA,
) -> list[tuple[str, float]]:
	"""Rerank fill candidates by Maximal Marginal Relevance."""
	if not candidates:
		return []
	all_node_ids = [nid for nid, _ in candidates] + list(selected_node_ids)
	all_texts = [_node_text(graph, nid) for nid in all_node_ids]
	all_embeddings = embedder.encode(all_texts)
	emb_by_id: dict[str, Sequence[float]] = dict(zip(all_node_ids, all_embeddings))
	sim_by_id: dict[str, float] = {nid: score for nid, score in candidates}
	selected_embs = [emb_by_id[nid] for nid in selected_node_ids if nid in emb_by_id]

	remaining = [nid for nid, _ in candidates]
	reranked: list[tuple[str, float]] = []
	while remaining:
		best_mmr = -float("inf")
		best_idx = 0
		for i, nid in enumerate(remaining):
			relevance = sim_by_id[nid]
			if selected_embs:
				max_sim = max(
					_dot_similarity(emb_by_id[nid], s) for s in selected_embs
				)
			else:
				max_sim = 0.0
			mmr = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
			if mmr > best_mmr:
				best_mmr = mmr
				best_idx = i
		chosen_id = remaining.pop(best_idx)
		reranked.append((chosen_id, sim_by_id[chosen_id]))
		selected_embs.append(emb_by_id[chosen_id])
	return reranked


def _lexical_seed_candidates(
	graph: LinkContextGraph,
	query: str,
	top_k: int,
	*,
	candidate_ids: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
	candidate_pool = (
		list(candidate_ids) if candidate_ids is not None else list(graph.nodes)
	)
	scored = [
		(node_id, normalized_token_overlap(query, _node_text(graph, node_id)))
		for node_id in candidate_pool
	]
	scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
	return scored[:top_k]


_ST_PREFILTER_MULTIPLIER = 50
_ST_PREFILTER_MIN = 200


def _sentence_transformer_seed_candidates(
	graph: LinkContextGraph,
	query: str,
	top_k: int,
	embedder: TextEmbedder,
	*,
	candidate_ids: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
	node_ids = list(candidate_ids) if candidate_ids is not None else list(graph.nodes)
	if not node_ids:
		return []
	# Pre-filter large candidate sets via lexical/FTS retrieval before encoding
	prefilter_k = max(top_k * _ST_PREFILTER_MULTIPLIER, _ST_PREFILTER_MIN)
	if len(node_ids) > prefilter_k and hasattr(graph, "topk_similar"):
		original_count = len(node_ids)
		prefiltered = graph.topk_similar(query, node_ids, k=prefilter_k)
		node_ids = [node_id for node_id, _score in prefiltered]
		logger.info(
			"ST seed pre-filtered %d -> %d candidates via topk_similar",
			original_count,
			len(node_ids),
		)
	query_embedding = embedder.encode([query])[0]
	node_embeddings = embedder.encode(
		[_node_text(graph, node_id) for node_id in node_ids]
	)
	scored = [
		(node_id, _clamp_score(_dot_similarity(query_embedding, embedding)))
		for node_id, embedding in zip(node_ids, node_embeddings)
	]
	scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
	return scored[:top_k]


def _trace_from_seed_candidates(
	seed_candidates: Sequence[tuple[str, float]],
) -> list[SelectionTraceStep]:
	return [
		SelectionTraceStep(index=index, node_id=node_id, score=score)
		for index, (node_id, score) in enumerate(seed_candidates)
	]


def _baseline_selector_metadata(
	spec: SelectorSpec,
	*,
	seed_backend: str | None,
	seed_model: str | None,
) -> SelectorMetadata:
	backend = str(spec.baseline or "baseline")
	return SelectorMetadata(
		scorer_kind="baseline",
		backend=backend,
		profile_name=_resolved_profile_name(spec),
		seed_strategy=spec.seed_strategy,
		seed_backend=seed_backend,
		seed_model=seed_model,
		seed_top_k=spec.seed_top_k,
		hop_budget=spec.hop_budget,
		search_structure=spec.search_structure,
		edge_scorer=spec.edge_scorer,
		lookahead_depth=spec.lookahead_depth,
		budget_fill_mode=spec.budget_fill_mode,
		budget_fill_pool_k=spec.budget_fill_pool_k,
		budget_fill_score_floor=spec.budget_fill_score_floor,
		budget_fill_relative_drop_ratio=spec.budget_fill_relative_drop_ratio,
		walk_score_threshold=spec.walk_score_threshold,
	)


def _dense_rerank_metadata(
	spec: SelectorSpec,
	*,
	seed_backend: str | None,
	seed_model: str | None,
	rerank_model: str | None,
) -> SelectorMetadata:
	return SelectorMetadata(
		scorer_kind="baseline",
		backend="dense_rerank",
		profile_name=_resolved_profile_name(spec),
		seed_strategy=spec.seed_strategy,
		seed_backend=seed_backend,
		seed_model=seed_model,
		seed_top_k=spec.seed_top_k,
		hop_budget=spec.hop_budget,
		budget_fill_mode=spec.budget_fill_mode,
		budget_fill_pool_k=spec.budget_fill_pool_k,
		budget_fill_score_floor=spec.budget_fill_score_floor,
		budget_fill_relative_drop_ratio=spec.budget_fill_relative_drop_ratio,
		rerank_model=rerank_model,
	)


def _apply_selector_metadata(
	result: CorpusSelectionResult,
	spec: SelectorSpec,
	*,
	seed_backend: str | None,
	seed_model: str | None,
) -> CorpusSelectionResult:
	metadata = result.selector_metadata
	if metadata is None:
		metadata = _baseline_selector_metadata(
			spec, seed_backend=seed_backend, seed_model=seed_model
		)
	else:
		metadata = replace(
			metadata,
			profile_name=metadata.profile_name or _resolved_profile_name(spec),
			seed_strategy=spec.seed_strategy,
			seed_backend=seed_backend,
			seed_model=seed_model,
			seed_top_k=spec.seed_top_k
			if spec.seed_top_k is not None
			else metadata.seed_top_k,
			hop_budget=spec.hop_budget
			if spec.hop_budget is not None
			else metadata.hop_budget,
			search_structure=spec.search_structure or metadata.search_structure,
			edge_scorer=spec.edge_scorer or metadata.edge_scorer,
			lookahead_depth=spec.lookahead_depth
			if spec.lookahead_depth is not None
			else metadata.lookahead_depth,
			budget_fill_mode=spec.budget_fill_mode or metadata.budget_fill_mode,
			budget_fill_pool_k=spec.budget_fill_pool_k
			if spec.budget_fill_pool_k is not None
			else metadata.budget_fill_pool_k,
			budget_fill_score_floor=(
				spec.budget_fill_score_floor
				if spec.budget_fill_score_floor is not None
				else metadata.budget_fill_score_floor
			),
			budget_fill_relative_drop_ratio=(
				spec.budget_fill_relative_drop_ratio
				if spec.budget_fill_relative_drop_ratio is not None
				else metadata.budget_fill_relative_drop_ratio
			),
			walk_score_threshold=(
				spec.walk_score_threshold
				if spec.walk_score_threshold is not None
				else metadata.walk_score_threshold
			),
		)
	result.selector_metadata = metadata
	return result


def _apply_budget_fill_metadata(
	metadata: SelectorMetadata | None,
	spec: SelectorSpec,
) -> SelectorMetadata:
	if metadata is None:
		metadata = SelectorMetadata(
			scorer_kind="budget_fill",
			backend="budget_fill",
		)
	return replace(
		metadata,
		budget_fill_mode=spec.budget_fill_mode,
		budget_fill_pool_k=spec.budget_fill_pool_k,
		budget_fill_score_floor=spec.budget_fill_score_floor,
		budget_fill_relative_drop_ratio=spec.budget_fill_relative_drop_ratio,
	)


def _validate_selector_llm_config(config: SelectorLLMConfig) -> None:
	if not config.model:
		raise ValueError("selector_model must be configured for LLM selectors.")
	if config.provider == "copilot":
		return
	if not config.api_key_env:
		raise ValueError("selector_api_key_env must be configured for LLM selectors.")
	if not os.environ.get(config.api_key_env):
		raise ValueError(
			f"Missing API key in environment variable {config.api_key_env}"
		)


def _query_tokens(query: str) -> frozenset[str]:
	return frozenset(content_tokens(query))


def _node_text(graph: LinkContextGraph, node_id: str) -> str:
	attr = graph.node_attr.get(node_id, {})
	return f"{attr.get('title', '')} {attr.get('text', '')}".strip()


def _edge_text(graph: LinkContextGraph, link: LinkContext) -> str:
	target = _node_text(graph, link.target)
	return f"{link.anchor_text} {link.sentence} {target}".strip()


def _dot_similarity(left: Sequence[float], right: Sequence[float]) -> float:
	if len(left) != len(right):
		raise ValueError("Embedding vectors must have the same length.")
	return float(
		sum(left_value * right_value for left_value, right_value in zip(left, right))
	)


def _link_query_tokens(
	query_tokens: frozenset[str], link: LinkContext | ScoredLink
) -> set[str]:
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


def _runtime_budget_token_limit(graph: LinkContextGraph, budget: RuntimeBudget) -> int:
	total_tokens = _graph_token_estimate(graph)
	if total_tokens <= 0:
		return 0
	if budget.token_budget_tokens is not None:
		return min(total_tokens, budget.token_budget_tokens)
	minimum_doc = _minimum_document_tokens(graph)
	assert budget.token_budget_ratio is not None
	scaled = math.ceil(total_tokens * budget.token_budget_ratio)
	return min(total_tokens, max(minimum_doc, scaled))


def _minimum_document_tokens(graph: LinkContextGraph) -> int:
	token_counts = [
		c for node_id in graph.nodes if (c := _node_token_cost(graph, node_id)) > 0
	]
	return min(token_counts) if token_counts else 0


def _within_path_budget(state: PathState, budget: SelectorBudget) -> bool:
	if (
		budget.max_nodes is not None
		and len(set(state.visited_nodes)) > budget.max_nodes
	):
		return False
	if budget.max_hops is not None and state.depth > budget.max_hops:
		return False
	if budget.max_tokens is not None and state.token_cost_estimate > budget.max_tokens:
		return False
	return True


def _seed_weights_from_graph(
	graph: LinkContextGraph, query: str, start_nodes: Sequence[str]
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


def _selection_coverage_ratio(
	query_tokens: frozenset[str], selected_links: Sequence[ScoredLink]
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


def _rank_nodes(
	node_scores: dict[str, float], strategy: str, selected_reason: str
) -> list[ScoredNode]:
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
	budget: SelectorBudget,
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
		if (
			budget.max_tokens is not None
			and token_cost_estimate + node_tokens > budget.max_tokens
		):
			continue
		selected.append(node.node_id)
		selected_set.add(node.node_id)
		token_cost_estimate += node_tokens
	return selected, token_cost_estimate


def _trace_from_ranked_nodes(node_scores: dict[str, float]) -> list[SelectionTraceStep]:
	ranked = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
	return [
		SelectionTraceStep(index=index, node_id=node_id, score=score)
		for index, (node_id, score) in enumerate(ranked)
	]


def _trace_from_ordered_nodes(
	graph: LinkContextGraph,
	query: str,
	ordered_node_ids: Sequence[str],
	*,
	fixed_score: float | None = None,
) -> list[SelectionTraceStep]:
	return [
		SelectionTraceStep(
			index=index,
			node_id=node_id,
			score=fixed_score
			if fixed_score is not None
			else _node_score(graph, query, node_id),
		)
		for index, node_id in enumerate(ordered_node_ids)
	]


def _build_selection_from_states(
	*,
	selector_name: str,
	graph: LinkContextGraph,
	query: str,
	query_tokens: frozenset[str],
	states: list[PathState],
	budget: SelectorBudget,
	strategy: str,
	mode: SelectionMode,
	debug_trace: list[str],
	root_node_ids: Sequence[str],
	selector_metadata: SelectorMetadata | None,
	selector_usage: SelectorUsage | None,
	selector_logs: Sequence[WalkStepLog],
	stop_reason: str | None,
) -> CorpusSelectionResult:
	node_scores: dict[str, float] = defaultdict(float)
	link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = {}
	ordered_states = sorted(
		states,
		key=lambda item: (
			item.reward_score,
			item.coverage_ratio(len(query_tokens)),
			-item.g_score,
		),
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
	trace = _trace_from_ranked_nodes(node_scores)
	return _build_corpus_selection_result(
		selector_name=selector_name,
		graph=graph,
		query=query,
		node_scores=node_scores,
		link_scores=link_scores,
		budget=budget,
		strategy=strategy,
		mode=mode,
		debug_trace=debug_trace,
		root_node_ids=root_node_ids,
		trace=trace,
		selector_metadata=selector_metadata,
		selector_usage=selector_usage,
		selector_logs=selector_logs,
		stop_reason=stop_reason,
	)


def _build_hybrid_result(
	*,
	selector_name: str,
	graph: LinkContextGraph,
	query: str,
	query_tokens: frozenset[str],
	path_result: CorpusSelectionResult,
	ppr_node_scores: dict[str, float],
	ppr_link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink],
	budget: SelectorBudget,
	strategy: str,
	debug_trace: list[str],
	selector_metadata: SelectorMetadata | None,
	selector_usage: SelectorUsage | None,
	selector_logs: Sequence[WalkStepLog],
) -> CorpusSelectionResult:
	path_scores = {node.node_id: node.score for node in path_result.ranked_nodes}
	path_norm = _normalize_scores(path_scores)
	ppr_norm = _normalize_scores(ppr_node_scores)

	combined_node_scores: dict[str, float] = {}
	for node_id in set(path_scores) | set(ppr_node_scores):
		combined_node_scores[node_id] = 0.5 * path_norm.get(
			node_id, 0.0
		) + 0.5 * ppr_norm.get(node_id, 0.0)

	ranked_nodes = _rank_nodes(combined_node_scores, strategy, "hybrid")
	selected_node_ids, token_cost_estimate = _apply_budget_to_ranked_nodes(
		graph, ranked_nodes, budget
	)
	selected_node_set = set(selected_node_ids)

	combined_link_scores: dict[
		tuple[str, str, str, str, int, str | None], ScoredLink
	] = {_link_key(link): link for link in path_result.ranked_links}
	for key, link in ppr_link_scores.items():
		if (
			link.source not in selected_node_set
			or link.target not in selected_node_set
			or link.score <= 0
		):
			continue
		existing = combined_link_scores.get(key)
		if existing is None or link.score > existing.score:
			combined_link_scores[key] = link

	ranked_links = sorted(
		combined_link_scores.values(), key=lambda item: item.score, reverse=True
	)
	selected_links = [
		link
		for link in ranked_links
		if link.source in selected_node_set and link.target in selected_node_set
	]
	return CorpusSelectionResult(
		selector_name=selector_name,
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
		root_node_ids=list(path_result.root_node_ids),
		trace=_trace_from_ranked_nodes(combined_node_scores),
		stop_reason=path_result.stop_reason,
		selector_metadata=selector_metadata,
		selector_usage=selector_usage,
		selector_logs=list(selector_logs),
	)


def _build_corpus_selection_result(
	*,
	selector_name: str,
	graph: LinkContextGraph,
	query: str,
	node_scores: dict[str, float],
	link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink],
	budget: SelectorBudget,
	strategy: str,
	mode: SelectionMode,
	debug_trace: list[str],
	root_node_ids: Sequence[str],
	trace: Sequence[SelectionTraceStep],
	selector_metadata: SelectorMetadata | None = None,
	selector_usage: SelectorUsage | None = None,
	selector_logs: Sequence[WalkStepLog] | None = None,
	stop_reason: str | None = None,
	force_full_corpus: bool = False,
	ignore_budget: bool = False,
) -> CorpusSelectionResult:
	ranked_nodes = _rank_nodes(node_scores, strategy, "selection")
	if force_full_corpus:
		selected_node_ids = list(graph.nodes)
		token_cost_estimate = _graph_token_estimate(graph)
	elif ignore_budget:
		selected_node_ids = [step.node_id for step in trace]
		token_cost_estimate = sum(
			_node_token_cost(graph, node_id) for node_id in selected_node_ids
		)
	else:
		selected_node_ids, token_cost_estimate = _apply_budget_to_ranked_nodes(
			graph, ranked_nodes, budget
		)
	selected_node_set = set(selected_node_ids)
	ranked_links = sorted(
		link_scores.values(), key=lambda item: item.score, reverse=True
	)
	selected_links = [
		link
		for link in ranked_links
		if link.source in selected_node_set and link.target in selected_node_set
	]
	return CorpusSelectionResult(
		selector_name=selector_name,
		query=query,
		ranked_nodes=ranked_nodes,
		ranked_links=ranked_links,
		selected_node_ids=selected_node_ids,
		selected_links=selected_links,
		token_cost_estimate=token_cost_estimate,
		strategy=strategy,
		mode=mode,
		debug_trace=debug_trace,
		coverage_ratio=_selection_coverage_ratio(_query_tokens(query), selected_links),
		root_node_ids=[
			node_id
			for node_id in root_node_ids
			if node_id in selected_node_set or node_id in set(root_node_ids)
		],
		trace=[
			step
			for step in trace
			if step.node_id in selected_node_set or step.index == 0
		],
		stop_reason=stop_reason,
		selector_metadata=selector_metadata,
		selector_usage=selector_usage or SelectorUsage(),
		selector_logs=list(selector_logs or []),
	)


def _corpus_selection_from_walk(
	*,
	selector_name: str,
	graph: LinkContextGraph,
	query: str,
	walk: WalkResult,
	root_node_ids: Sequence[str],
	token_budget_limit: int,
	runtime_s: float,
	spec: SelectorSpec,
) -> CorpusSelectionResult:
	node_scores: dict[str, float] = {}
	link_scores: dict[tuple[str, str, str, str, int, str | None], ScoredLink] = {}
	trace: list[SelectionTraceStep] = []
	for step in walk.steps:
		node_scores[step.node_id] = max(node_scores.get(step.node_id, 0.0), step.score)
		trace.append(
			SelectionTraceStep(
				index=step.index,
				node_id=step.node_id,
				score=step.score,
				source_node_id=step.source_node_id,
				anchor_text=step.anchor_text,
				sentence=step.sentence,
			)
		)
		if (
			step.source_node_id is not None
			and step.anchor_text is not None
			and step.sentence is not None
		):
			for link in graph.links_between(step.source_node_id, step.node_id):
				if (
					link.anchor_text == step.anchor_text
					and link.sentence == step.sentence
				):
					scored_link = ScoredLink.from_link(
						link,
						score=step.score,
						source_strategy=selector_name,
						selected_reason="walk",
					)
					link_scores[_link_key(scored_link)] = scored_link
					break
	result = _build_corpus_selection_result(
		selector_name=selector_name,
		graph=graph,
		query=query,
		node_scores=node_scores,
		link_scores=link_scores,
		budget=SelectorBudget(
			max_nodes=None, max_hops=spec.hop_budget, max_tokens=token_budget_limit
		),
		strategy="single_path_walk",
		mode=SelectionMode.STANDALONE,
		debug_trace=[f"walk:{step.node_id}:{step.score:.4f}" for step in walk.steps],
		root_node_ids=root_node_ids,
		trace=trace,
		selector_metadata=_selector_metadata_from_walk(
			walk,
			seed_top_k=spec.seed_top_k,
			hop_budget=spec.hop_budget,
			search_structure="single_path_walk",
			edge_scorer=spec.edge_scorer,
			lookahead_depth=spec.lookahead_depth,
			profile_name=_resolved_profile_name(spec),
		),
		selector_usage=_selector_usage_from_logs(
			walk.selector_logs, runtime_override=runtime_s
		),
		selector_logs=walk.selector_logs,
		stop_reason=walk.stop_reason.value,
	)
	return result


def _graph_token_estimate(graph: LinkContextGraph) -> int:
	if hasattr(graph, "total_token_estimate"):
		return graph.total_token_estimate()  # ty: ignore[call-non-callable] # dynamic method check
	return sum(_node_token_cost(graph, node_id) for node_id in graph.nodes)


def _target_title(graph: LinkContextGraph, node_id: str) -> str:
	return str(graph.node_attr.get(node_id, {}).get("title", node_id))


def _node_score(graph: LinkContextGraph, query: str, node_id: str) -> float:
	attr = graph.node_attr.get(node_id, {})
	text = f"{attr.get('title', '')} {attr.get('text', '')}".strip()
	return normalized_token_overlap(query, text)


def _selector_metadata_from_step_scorer(
	scorer: StepLinkScorer | _WalkScorerMetadataAdapter,
	*,
	seed_top_k: int | None,
	hop_budget: int | None,
	search_structure: str | None,
	edge_scorer: str | None = None,
	lookahead_depth: int | None = None,
	profile_name: str | None = None,
) -> SelectorMetadata:
	metadata: StepScorerMetadata = scorer.metadata
	derived_lookahead = lookahead_depth
	if derived_lookahead is None and metadata.two_hop_prefilter_top_n is not None:
		derived_lookahead = 2
	elif derived_lookahead is None:
		derived_lookahead = 1
	return SelectorMetadata(
		scorer_kind=metadata.scorer_kind,
		backend=metadata.backend,
		profile_name=profile_name or metadata.profile_name,
		provider=metadata.provider,
		model=metadata.model,
		prompt_version=metadata.prompt_version,
		candidate_prefilter_top_n=metadata.candidate_prefilter_top_n,
		two_hop_prefilter_top_n=metadata.two_hop_prefilter_top_n,
		seed_top_k=seed_top_k,
		hop_budget=hop_budget,
		search_structure=search_structure,
		edge_scorer=edge_scorer,
		lookahead_depth=derived_lookahead,
	)


def _selector_metadata_from_walk(
	walk: WalkResult,
	*,
	seed_top_k: int | None,
	hop_budget: int | None,
	search_structure: str | None,
	edge_scorer: str | None,
	lookahead_depth: int | None,
	profile_name: str | None = None,
) -> SelectorMetadata:
	return _selector_metadata_from_step_scorer(
		scorer=_WalkScorerMetadataAdapter(walk.scorer_metadata),
		seed_top_k=seed_top_k,
		hop_budget=hop_budget,
		search_structure=search_structure,
		edge_scorer=edge_scorer,
		lookahead_depth=lookahead_depth,
		profile_name=profile_name,
	)


class _WalkScorerMetadataAdapter:
	def __init__(self, metadata: StepScorerMetadata):
		self.metadata = metadata


def _selector_usage_from_logs(
	logs: Sequence[WalkStepLog], *, runtime_override: float | None = None
) -> SelectorUsage:
	if not logs:
		return SelectorUsage(runtime_s=runtime_override or 0.0)
	controller_logs = [
		log
		for log in logs
		if log.controller is not None and log.controller.kind != "backtrack"
	]
	backtrack_logs = [
		log
		for log in logs
		if log.controller is not None and log.controller.kind == "backtrack"
	]
	explicit_stop_logs = [
		log for log in controller_logs if log.stop_reason == "controller_stop"
	]
	budget_pacing_logs = [
		log for log in controller_logs if log.stop_reason == "budget_pacing_stop"
	]
	return SelectorUsage(
		runtime_s=runtime_override
		if runtime_override is not None
		else sum(log.latency_s for log in logs),
		llm_calls=sum(
			log.llm_calls if log.llm_calls is not None else 1
			for log in logs
			if log.provider is not None
			and (log.controller is None or log.controller.kind != "backtrack")
		),
		prompt_tokens=sum(log.prompt_tokens or 0 for log in logs),
		completion_tokens=sum(log.completion_tokens or 0 for log in logs),
		total_tokens=sum(log.total_tokens or 0 for log in logs),
		cache_hits=sum(1 for log in logs if log.cache_hit),
		step_count=len(logs),
		fallback_steps=sum(
			1 for log in logs if _is_selector_fallback(log.fallback_reason)
		),
		parse_failure_steps=sum(
			1 for log in logs if _is_selector_parse_failure(log.fallback_reason)
		),
		controller_calls=len(controller_logs),
		controller_stop_actions=len(explicit_stop_logs) + len(budget_pacing_logs),
		controller_explicit_stop_actions=len(explicit_stop_logs),
		controller_budget_pacing_stop_actions=len(budget_pacing_logs),
		controller_fork_actions=sum(
			1
			for log in controller_logs
			if log.controller is not None
			and log.controller.effective_action == "choose_two"
		),
		controller_backtrack_actions=len(backtrack_logs),
		controller_prefiltered_candidates=sum(
			sum(
				1
				for candidate in (
					log.controller.candidates if log.controller is not None else []
				)
				if candidate.exposure_status != "visible"
			)
			for log in controller_logs
		),
	)


def _is_selector_fallback(reason: str | None) -> bool:
	return reason is not None and reason != "prefiltered_out"


def _is_selector_parse_failure(reason: str | None) -> bool:
	if reason is None:
		return False
	return (
		reason == "empty_response"
		or reason.startswith("json_parse_error")
		or reason.startswith("schema_error")
	)


__all__ = [
	"CorpusSelectionResult",
	"CorpusSelector",
	"CoverageSemanticHeuristic",
	"PathState",
	"ScoredLink",
	"ScoredNode",
	"SelectionMode",
	"SelectionTraceStep",
	"SelectorBudget",
	"SelectorMetadata",
	"SelectorSpec",
	"SelectorUsage",
	"SemanticAStarSelector",
	"SemanticBeamSelector",
	"SemanticPPRSelector",
	"SemanticUCSSelector",
	"available_selector_names",
	"available_selector_presets",
	"build_selector",
	"parse_selector_spec",
	"selector_names_for_preset",
	"select_selectors",
]
