from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, Sequence

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
class StepScorerMetadata:
    scorer_kind: str
    backend: str
    profile_name: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    candidate_prefilter_top_n: int | None = None
    two_hop_prefilter_top_n: int | None = None


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
    ) -> list[StepScoreCard]:
        ...


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
            immediate_subscores = self._immediate_subscores(query, graph, link, visited_nodes)
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
                total_score = _clamp_score(immediate_score + self.lookahead_gamma * future_score)
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
                    subscores={key: _clamp_score(value) for key, value in subscores.items()},
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
    def __init__(self, *, novelty_bonus: float = 0.05, lookahead_steps: int = 1, lookahead_gamma: float = 0.6):
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
        selector_logs: list[WalkStepLog] = []

        stop_reason = StopReason.BUDGET_EXHAUSTED
        while len(steps) < budget.max_steps:
            remaining_steps = budget.max_steps - len(steps)
            candidate_links: list[LinkContext] = []
            for neighbor in self.graph.neighbors(current):
                if not budget.allow_revisit and neighbor in visited_set:
                    continue
                candidate_links.extend(self.graph.links_between(current, neighbor))

            if not candidate_links:
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
                stop_reason = StopReason.DEAD_END
                break

            best_index, best_card, best_link = max(
                (
                    (index, score_cards[index], candidate_links[index])
                    for index in range(min(len(candidate_links), len(score_cards)))
                ),
                key=lambda item: (item[1].total_score, item[1].subscores.get("future_score", 0.0), -item[0]),
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
                    text=best_card.text,
                    raw_response=best_card.raw_response,
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
                        for link, card in zip(candidate_links, score_cards, strict=False)
                    ],
                )
            )

            if best_card.total_score < budget.min_score:
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


def _target_title(graph: LinkContextGraph, node_id: str) -> str:
    return str(graph.node_attr.get(node_id, {}).get("title", node_id))


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
