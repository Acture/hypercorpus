"""Controller candidate exposure planning and prompt-bundle construction.

This module owns the deterministic pre-LLM stage for controller selectors:
- candidate hygiene such as dangling-target filtering,
- small-page bypass and hybrid prefiltering,
- prompt-bundle construction for the LLM backend.

Traversal code should consume the resulting typed exposure/bundle objects rather
than recomputing candidate visibility or rebuilding prompt payloads ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from hypercorpus.graph import LinkContext, LinkContextGraph
from hypercorpus.text import content_tokens
from hypercorpus.walker import (
	LinkContextOverlapStepScorer,
	StepScoreCard,
	_clamp_score,
)


@dataclass(slots=True)
class ControllerExposurePlan:
	"""Deterministic candidate-visibility decision for one controller step."""

	raw_candidate_count: int
	valid_candidate_count: int
	small_page_bypass: bool
	valid_indices: list[int]
	dangling_indices: list[int]
	lexical_prefilter_edge_ids: list[str]
	semantic_prefilter_edge_ids: list[str]
	bonus_rescued_edge_ids: list[str]
	visible_indices: list[int]


@dataclass(slots=True)
class ControllerFutureCandidateBundle:
	"""One future-link preview shown inside a two-hop controller candidate card."""

	edge_id: str
	target_title: str
	anchor_text: str
	sentence: str
	prefilter_score: float
	query_anchor_overlap: float
	query_sentence_overlap: float
	query_target_overlap: float
	answer_bearing_link_bonus: float


@dataclass(slots=True)
class ControllerCandidateBundleEntry:
	"""One visible controller candidate with optional future-link previews."""

	edge_id: str
	source_title: str
	target_title: str
	anchor_text: str
	sentence: str
	prefilter_score: float
	query_anchor_overlap: float
	query_sentence_overlap: float
	query_target_overlap: float
	answer_bearing_link_bonus: float
	source_sentence_mentions_target_title: bool
	semantic_prefilter_score: float
	future_candidates: list[ControllerFutureCandidateBundle] = field(default_factory=list)


@dataclass(slots=True)
class ControllerCandidateBundle:
	"""Typed prompt payload for a single controller decision call."""

	query: str
	current_node_id: str
	path_titles: list[str]
	raw_candidate_count: int
	valid_candidate_count: int
	small_page_bypass: bool
	dangling_edge_ids: list[str]
	lexical_prefilter_edge_ids: list[str]
	semantic_prefilter_edge_ids: list[str]
	bonus_rescued_edge_ids: list[str]
	visible_edge_ids: list[str]
	candidates: list[ControllerCandidateBundleEntry]

	def to_prompt_payload(self) -> dict[str, Any]:
		"""Convert the typed bundle into the JSON-ready prompt payload."""

		return {
			"query": self.query,
			"current_node_id": self.current_node_id,
			"path_titles": list(self.path_titles),
			"raw_candidate_count": self.raw_candidate_count,
			"valid_candidate_count": self.valid_candidate_count,
			"small_page_bypass": self.small_page_bypass,
			"dangling_edge_ids": list(self.dangling_edge_ids),
			"lexical_prefilter_edge_ids": list(self.lexical_prefilter_edge_ids),
			"semantic_prefilter_edge_ids": list(self.semantic_prefilter_edge_ids),
			"bonus_rescued_edge_ids": list(self.bonus_rescued_edge_ids),
			"visible_edge_ids": list(self.visible_edge_ids),
			"candidates": [
				{
					"edge_id": entry.edge_id,
					"source_title": entry.source_title,
					"target_title": entry.target_title,
					"anchor_text": entry.anchor_text,
					"sentence": entry.sentence,
					"prefilter_score": entry.prefilter_score,
					"query_anchor_overlap": entry.query_anchor_overlap,
					"query_sentence_overlap": entry.query_sentence_overlap,
					"query_target_overlap": entry.query_target_overlap,
					"answer_bearing_link_bonus": entry.answer_bearing_link_bonus,
					"source_sentence_mentions_target_title": entry.source_sentence_mentions_target_title,
					"semantic_prefilter_score": entry.semantic_prefilter_score,
					"future_candidates": [
						{
							"edge_id": future.edge_id,
							"target_title": future.target_title,
							"anchor_text": future.anchor_text,
							"sentence": future.sentence,
							"prefilter_score": future.prefilter_score,
							"query_anchor_overlap": future.query_anchor_overlap,
							"query_sentence_overlap": future.query_sentence_overlap,
							"query_target_overlap": future.query_target_overlap,
							"answer_bearing_link_bonus": future.answer_bearing_link_bonus,
						}
						for future in entry.future_candidates
					],
				}
				for entry in self.candidates
			],
		}


def build_controller_exposure_plan(
	*,
	query: str,
	graph: LinkContextGraph,
	candidate_links: Sequence[LinkContext],
	lexical_cards: Sequence[StepScoreCard],
	semantic_cards: Sequence[StepScoreCard] | None,
	small_page_bypass_n: int,
	lexical_top_n: int,
	semantic_top_n: int,
	bonus_keep_n: int,
	visible_cap: int,
) -> ControllerExposurePlan:
	"""Plan which candidate edges are visible to the controller LLM."""

	raw_candidate_count = len(candidate_links)
	valid_indices: list[int] = []
	dangling_indices: list[int] = []
	for index, link in enumerate(candidate_links):
		if has_backing_document_target(graph=graph, link=link):
			valid_indices.append(index)
		else:
			dangling_indices.append(index)
	valid_candidate_count = len(valid_indices)
	if valid_candidate_count == 0:
		return ControllerExposurePlan(
			raw_candidate_count=raw_candidate_count,
			valid_candidate_count=0,
			small_page_bypass=False,
			valid_indices=[],
			dangling_indices=dangling_indices,
			lexical_prefilter_edge_ids=[],
			semantic_prefilter_edge_ids=[],
			bonus_rescued_edge_ids=[],
			visible_indices=[],
		)
	if valid_candidate_count <= small_page_bypass_n:
		return ControllerExposurePlan(
			raw_candidate_count=raw_candidate_count,
			valid_candidate_count=valid_candidate_count,
			small_page_bypass=True,
			valid_indices=valid_indices,
			dangling_indices=dangling_indices,
			lexical_prefilter_edge_ids=[],
			semantic_prefilter_edge_ids=[],
			bonus_rescued_edge_ids=[],
			visible_indices=list(valid_indices),
		)

	lexical_ranked = rank_indices_by_score(valid_indices, lexical_cards)
	lexical_indices = lexical_ranked[:lexical_top_n]
	semantic_indices: list[int] = []
	if semantic_cards is not None:
		semantic_ranked = rank_indices_by_score(valid_indices, semantic_cards)
		semantic_indices = semantic_ranked[:semantic_top_n]

	selected: set[int] = set(lexical_indices) | set(semantic_indices)
	bonus_ranked = sorted(
		valid_indices,
		key=lambda index: (
			answer_bearing_link_bonus(
				query=query,
				graph=graph,
				link=candidate_links[index],
				card=lexical_cards[index],
			),
			lexical_cards[index].total_score,
			semantic_cards[index].total_score if semantic_cards is not None else 0.0,
			-index,
		),
		reverse=True,
	)
	bonus_rescued_indices: list[int] = []
	for index in bonus_ranked:
		if index in selected:
			continue
		selected.add(index)
		bonus_rescued_indices.append(index)
		if len(bonus_rescued_indices) >= bonus_keep_n:
			break

	def combined_order(index: int) -> tuple[float, float, float, int]:
		semantic_score = (
			semantic_cards[index].total_score if semantic_cards is not None else 0.0
		)
		bonus = answer_bearing_link_bonus(
			query=query,
			graph=graph,
			link=candidate_links[index],
			card=lexical_cards[index],
		)
		return (
			max(lexical_cards[index].total_score, semantic_score),
			bonus,
			lexical_cards[index].total_score + semantic_score,
			-index,
		)

	visible_indices = sorted(selected, key=combined_order, reverse=True)[:visible_cap]
	return ControllerExposurePlan(
		raw_candidate_count=raw_candidate_count,
		valid_candidate_count=valid_candidate_count,
		small_page_bypass=False,
		valid_indices=valid_indices,
		dangling_indices=dangling_indices,
		lexical_prefilter_edge_ids=[str(index) for index in lexical_indices],
		semantic_prefilter_edge_ids=[str(index) for index in semantic_indices],
		bonus_rescued_edge_ids=[str(index) for index in bonus_rescued_indices],
		visible_indices=visible_indices,
	)


def build_controller_candidate_bundle(
	*,
	graph: LinkContextGraph,
	candidate_links: Sequence[LinkContext],
	score_cards: Sequence[StepScoreCard],
	semantic_score_cards: Sequence[StepScoreCard] | None,
	exposure_plan: ControllerExposurePlan,
	query: str,
	current_node_id: str,
	path_node_ids: Sequence[str],
	visited_nodes: set[str],
	mode: str,
	future_top_n: int,
) -> ControllerCandidateBundle:
	"""Build the typed prompt bundle for the already-planned visible candidates."""

	entries: list[ControllerCandidateBundleEntry] = []
	future_scorer = LinkContextOverlapStepScorer()
	for index in exposure_plan.visible_indices:
		link = candidate_links[index]
		score_card = score_cards[index]
		target_title = node_title(graph, link.target)
		sentence_mentions_target = title_coverage_in_sentence(
			title=target_title,
			sentence=link.sentence,
		)
		answer_bonus = answer_bearing_link_bonus(
			query=query,
			graph=graph,
			link=link,
			card=score_card,
		)
		entry = ControllerCandidateBundleEntry(
			edge_id=str(index),
			source_title=node_title(graph, link.source),
			target_title=target_title,
			anchor_text=link.anchor_text,
			sentence=link.sentence,
			prefilter_score=score_card.total_score,
			query_anchor_overlap=_clamp_score(
				score_card.subscores.get("anchor_overlap", 0.0)
			),
			query_sentence_overlap=_clamp_score(
				score_card.subscores.get("sentence_overlap", 0.0)
			),
			query_target_overlap=_clamp_score(
				score_card.subscores.get("target_overlap", 0.0)
			),
			answer_bearing_link_bonus=answer_bonus,
			source_sentence_mentions_target_title=sentence_mentions_target >= 0.75,
			semantic_prefilter_score=(
				semantic_score_cards[index].total_score
				if semantic_score_cards is not None
				else 0.0
			),
		)
		if mode == "two_hop":
			next_links = [
				next_link
				for next_link in graph.links_from(link.target)
				if next_link.target not in visited_nodes
				and next_link.target != link.source
			]
			future_cards = future_scorer.score_candidates(
				query=query,
				graph=graph,
				current_node_id=link.target,
				candidate_links=next_links,
				visited_nodes=visited_nodes | {link.target},
				path_node_ids=[*path_node_ids, link.target],
				remaining_steps=1,
			)
			future_indices = prefilter_indices(
				future_cards,
				top_n=min(future_top_n, len(future_cards)),
				query=query,
				graph=graph,
				candidate_links=next_links,
				bonus_keep_n=1,
			)
			entry.future_candidates = [
				ControllerFutureCandidateBundle(
					edge_id=f"{index}-{future_index}",
					target_title=node_title(graph, next_links[future_index].target),
					anchor_text=next_links[future_index].anchor_text,
					sentence=next_links[future_index].sentence,
					prefilter_score=future_cards[future_index].total_score,
					query_anchor_overlap=_clamp_score(
						future_cards[future_index].subscores.get("anchor_overlap", 0.0)
					),
					query_sentence_overlap=_clamp_score(
						future_cards[future_index].subscores.get("sentence_overlap", 0.0)
					),
					query_target_overlap=_clamp_score(
						future_cards[future_index].subscores.get("target_overlap", 0.0)
					),
					answer_bearing_link_bonus=answer_bearing_link_bonus(
						query=query,
						graph=graph,
						link=next_links[future_index],
						card=future_cards[future_index],
					),
				)
				for future_index in future_indices
			]
		entries.append(entry)

	return ControllerCandidateBundle(
		query=query,
		current_node_id=current_node_id,
		path_titles=[node_title(graph, node_id) for node_id in path_node_ids],
		raw_candidate_count=exposure_plan.raw_candidate_count,
		valid_candidate_count=exposure_plan.valid_candidate_count,
		small_page_bypass=exposure_plan.small_page_bypass,
		dangling_edge_ids=[str(index) for index in exposure_plan.dangling_indices],
		lexical_prefilter_edge_ids=list(exposure_plan.lexical_prefilter_edge_ids),
		semantic_prefilter_edge_ids=list(exposure_plan.semantic_prefilter_edge_ids),
		bonus_rescued_edge_ids=list(exposure_plan.bonus_rescued_edge_ids),
		visible_edge_ids=[str(index) for index in exposure_plan.visible_indices],
		candidates=entries,
	)


def prefilter_indices(
	cards: Sequence[StepScoreCard],
	*,
	top_n: int,
	query: str | None = None,
	graph: LinkContextGraph | None = None,
	candidate_links: Sequence[LinkContext] | None = None,
	bonus_keep_n: int = 2,
) -> list[int]:
	ranked = sorted(
		enumerate(cards),
		key=lambda item: (item[1].total_score, -item[0]),
		reverse=True,
	)
	selected: set[int] = {index for index, _card in ranked[:top_n]}
	if query is None or graph is None or candidate_links is None or bonus_keep_n <= 0:
		return [index for index, _card in ranked[:top_n]]
	bonus_ranked = sorted(
		enumerate(cards),
		key=lambda item: (
			answer_bearing_link_bonus(
				query=query,
				graph=graph,
				link=candidate_links[item[0]],
				card=item[1],
			),
			item[1].total_score,
			-item[0],
		),
		reverse=True,
	)
	added = 0
	for index, card in bonus_ranked:
		if index in selected:
			continue
		answer_bonus = answer_bearing_link_bonus(
			query=query,
			graph=graph,
			link=candidate_links[index],
			card=card,
		)
		if answer_bonus < 0.55:
			continue
		selected.add(index)
		added += 1
		if added >= bonus_keep_n:
			break
	return [index for index, _card in ranked if index in selected]


def has_backing_document_target(*, graph: LinkContextGraph, link: LinkContext) -> bool:
	document = graph.get_document(link.target)
	if document is None:
		return False
	return not bool(document.metadata.get("placeholder", False))


def rank_indices_by_score(
	indices: Sequence[int], cards: Sequence[StepScoreCard]
) -> list[int]:
	return sorted(
		indices,
		key=lambda index: (cards[index].total_score, -index),
		reverse=True,
	)


def title_coverage_in_sentence(*, title: str, sentence: str) -> float:
	title_tokens = content_tokens(title)
	if not title_tokens:
		return 0.0
	sentence_tokens = set(content_tokens(sentence))
	shared = sum(1 for token in title_tokens if token in sentence_tokens)
	return shared / max(len(title_tokens), 1)


def answer_bearing_link_bonus(
	*,
	query: str,
	graph: LinkContextGraph,
	link: LinkContext,
	card: StepScoreCard,
) -> float:
	target_title = node_title(graph, link.target)
	title_coverage = title_coverage_in_sentence(title=target_title, sentence=link.sentence)
	sentence_overlap = _clamp_score(card.subscores.get("sentence_overlap", 0.0))
	target_overlap = _clamp_score(card.subscores.get("target_overlap", 0.0))
	anchor_overlap = _clamp_score(card.subscores.get("anchor_overlap", 0.0))
	query_tokens = set(content_tokens(query))
	target_tokens = content_tokens(target_title)
	target_query_overlap = (
		sum(1 for token in target_tokens if token in query_tokens)
		/ max(len(target_tokens), 1)
		if target_tokens
		else 0.0
	)
	return _clamp_score(
		0.45 * sentence_overlap
		+ 0.25 * title_coverage
		+ 0.20 * target_overlap
		+ 0.05 * anchor_overlap
		+ 0.05 * target_query_overlap
	)


def node_title(graph: LinkContextGraph, node_id: str) -> str:
	attr = graph.node_attr.get(node_id, {})
	title = str(attr.get("title", "")).strip()
	return title or node_id
