from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from webwalker.graph import LinkContext, LinkContextGraph
from webwalker.text import normalized_token_overlap


class StopReason(StrEnum):
	DEAD_END = "dead_end"
	BUDGET_EXHAUSTED = "budget_exhausted"
	SCORE_BELOW_THRESHOLD = "score_below_threshold"


@dataclass(slots=True)
class WalkBudget:
	max_steps: int = 3
	min_score: float = 0.0
	allow_revisit: bool = False


@dataclass(slots=True)
class WalkStep:
	index: int
	node_id: str
	score: float
	source_node_id: str | None = None
	anchor_text: str | None = None
	sentence: str | None = None


@dataclass(slots=True)
class WalkResult:
	query: str
	steps: list[WalkStep]
	visited_nodes: list[str]
	stop_reason: StopReason


class LinkScorer(Protocol):
	def score(
		self,
		query: str,
		graph: LinkContextGraph,
		link: LinkContext,
		visited_nodes: set[str],
	) -> float:
		...


class OverlapLinkScorer:
	def __init__(
		self,
		*,
		anchor_weight: float = 0.45,
		sentence_weight: float = 0.35,
		target_weight: float = 0.20,
		novelty_bonus: float = 0.05,
	):
		self.anchor_weight = anchor_weight
		self.sentence_weight = sentence_weight
		self.target_weight = target_weight
		self.novelty_bonus = novelty_bonus

	def score(
		self,
		query: str,
		graph: LinkContextGraph,
		link: LinkContext,
		visited_nodes: set[str],
	) -> float:
		target_attr = graph.node_attr.get(link.target, {})
		target_title = str(target_attr.get("title", ""))
		score = (
			normalized_token_overlap(query, link.anchor_text) * self.anchor_weight
			+ normalized_token_overlap(query, link.sentence) * self.sentence_weight
			+ normalized_token_overlap(query, target_title) * self.target_weight
		)
		if link.target not in visited_nodes:
			score += self.novelty_bonus
		return score


class DynamicWalker:
	def __init__(self, graph: LinkContextGraph, scorer: LinkScorer | None = None):
		self.graph = graph
		self.scorer = scorer or OverlapLinkScorer()

	def walk(
		self,
		query: str,
		start_nodes: list[str],
		budget: WalkBudget | None = None,
	) -> WalkResult:
		budget = budget or WalkBudget()
		if budget.max_steps <= 0:
			raise ValueError("Walk budget must allow at least one step.")
		if not start_nodes:
			raise ValueError("walk requires at least one start node.")

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

		stop_reason = StopReason.BUDGET_EXHAUSTED
		while len(steps) < budget.max_steps:
			candidates: list[tuple[float, LinkContext]] = []
			for neighbor in self.graph.neighbors(current):
				if not budget.allow_revisit and neighbor in visited_set:
					continue
				for link in self.graph.links_between(current, neighbor):
					candidates.append(
						(self.scorer.score(query, self.graph, link, visited_set), link)
					)

			if not candidates:
				stop_reason = StopReason.DEAD_END
				break

			candidates.sort(key=lambda item: item[0], reverse=True)
			best_score, best_link = candidates[0]
			if best_score < budget.min_score:
				stop_reason = StopReason.SCORE_BELOW_THRESHOLD
				break

			current = best_link.target
			visited_nodes.append(current)
			visited_set.add(current)
			steps.append(
				WalkStep(
					index=len(steps),
					node_id=current,
					score=best_score,
					source_node_id=best_link.source,
					anchor_text=best_link.anchor_text,
					sentence=best_link.sentence,
				)
			)
		else:
			stop_reason = StopReason.BUDGET_EXHAUSTED

		return WalkResult(
			query=query,
			steps=steps,
			visited_nodes=visited_nodes,
			stop_reason=stop_reason,
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
