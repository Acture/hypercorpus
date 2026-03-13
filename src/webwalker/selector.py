from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import StrEnum
import heapq
import itertools
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Literal, Protocol, Sequence

from webwalker.embeddings import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderConfig,
    TextEmbedder,
)
from webwalker.graph import LinkContext, LinkContextGraph
from webwalker.selector_llm import LLMStepLinkScorer, SelectorLLMConfig
from webwalker.text import approx_token_count, content_tokens, normalized_token_overlap
from webwalker.walker import (
    AnchorOverlapStepScorer,
    DynamicWalker,
    LinkContextOverlapStepScorer,
    StepCandidateTrace,
    StepLinkScorer,
    StepScoreCard,
    StepScorerMetadata,
    StopReason,
    TitleAwareOverlapStepScorer,
    WalkBudget,
    WalkResult,
    WalkStepLog,
    _clamp_score,
)

SeedStrategyName = Literal["sentence_transformer", "lexical_overlap"]
BudgetFillMode = Literal["score_floor", "always", "relative_drop"]
BaselineName = Literal[
    "dense",
    "iterative_dense",
    "mdr_light",
    "topology_neighbors",
    "anchor_neighbors",
    "link_context_neighbors",
]
SearchStructure = Literal["single_path_walk", "beam", "astar", "ucs", "beam_ppr"]
EdgeScorerName = Literal["link_context_overlap", "link_context_llm", "anchor_overlap", "link_context_sentence_transformer"]
LookaheadName = Literal["lookahead_1", "lookahead_2"]
SelectorFamily = Literal["baseline", "path_search", "diagnostic"]
SelectorPresetName = Literal["full", "paper_recommended"]

_SEED_STRATEGIES: set[str] = {"sentence_transformer", "lexical_overlap"}
_BASELINES: set[str] = {
    "dense",
    "iterative_dense",
    "mdr_light",
    "topology_neighbors",
    "anchor_neighbors",
    "link_context_neighbors",
}
_SEARCH_STRUCTURES: set[str] = {"single_path_walk", "beam", "astar", "ucs", "beam_ppr"}
_EDGE_SCORERS: set[str] = {"link_context_overlap", "link_context_llm", "anchor_overlap", "link_context_sentence_transformer"}
_LOOKAHEADS: set[str] = {"lookahead_1", "lookahead_2"}
_BUDGET_FILL_SUFFIXES: dict[str, BudgetFillMode] = {
    "budget_fill_score_floor": "score_floor",
    "budget_fill_always": "always",
    "budget_fill_relative_drop": "relative_drop",
}
_SELECTOR_PATTERN = re.compile(
    r"^top_(?P<seed_top_k>\d+)_seed__(?P<seed_strategy>sentence_transformer|lexical_overlap)__hop_(?P<hop_budget>\d+)__(?P<rest>.+)$"
)
_DIAGNOSTIC_SELECTORS = ("gold_support_context", "full_corpus_upper_bound")
_OVERLAP_PROFILE_DEFAULT = "overlap_balanced"
_SENTENCE_TRANSFORMER_PROFILE_DEFAULT = "st_balanced"
_OVERLAP_PROFILES: dict[str, dict[str, float]] = {
    "overlap_balanced": {"anchor": 0.60, "sentence": 0.40, "title": 0.00, "novelty": 0.05},
    "overlap_anchor_heavy": {"anchor": 0.80, "sentence": 0.20, "title": 0.00, "novelty": 0.05},
    "overlap_title_aware": {"anchor": 0.45, "sentence": 0.35, "title": 0.20, "novelty": 0.05},
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


class SelectionMode(StrEnum):
    STANDALONE = "standalone"
    HYBRID_WITH_PPR = "hybrid_with_ppr"


@dataclass(slots=True)
class SelectorBudget:
    max_nodes: int | None = None
    max_hops: int | None = 3
    max_tokens: int | None = 256


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
    ) -> CorpusSelectionResult:
        ...


class _SentenceTransformerSupport:
    def __init__(
        self,
        *,
        embedder_config: SentenceTransformerSelectorConfig | None = None,
        embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
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
        for index, (link, edge_embedding) in enumerate(zip(candidate_links, edge_embeddings)):
            direct_similarity = _clamp_score(_dot_similarity(query_embedding, edge_embedding))
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
                total_score = _clamp_score(self.direct_weight * direct_similarity + self.novelty_weight * novelty)
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
            for inner_index, next_link in enumerate(graph.links_between(link.target, neighbor)):
                next_edge_offset = len(next_links)
                next_links.append(next_link)
                next_edge_ids.append(f"{edge_id}-{next_edge_offset}")
        if not next_links:
            return 0.0, None
        next_embeddings = self.embedder.encode([_edge_text(graph, next_link) for next_link in next_links])
        scored = [
            (_clamp_score(_dot_similarity(query_embedding, embedding)), next_edge_ids[index])
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
        node_scores, link_scores, debug_trace = self._run_ppr(graph, query, start_nodes, scorer)
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
    ) -> tuple[dict[str, float], dict[tuple[str, str, str, str, int, str | None], ScoredLink], list[str]]:
        if not start_nodes:
            raise ValueError("semantic_ppr requires at least one start node.")
        personalization = _normalize_map(seed_weights or _seed_weights_from_graph(graph, query, start_nodes))
        if not personalization:
            personalization = {node_id: 1 / len(start_nodes) for node_id in start_nodes}

        transitions: dict[str, dict[str, float]] = {}
        representative_links: dict[tuple[str, str], ScoredLink] = {}
        debug_trace = [f"seed:{node_id}:{weight:.4f}" for node_id, weight in personalization.items()]

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
                best_score, _best_link = target_best_link.get(link.target, (float("-inf"), scored))
                if card.total_score > best_score:
                    target_best_link[link.target] = (card.total_score, scored)
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
    ) -> CorpusSelectionResult:
        if not start_nodes:
            raise ValueError(f"{self.strategy_name} requires at least one start node.")
        budget = budget or SelectorBudget()
        scorer = scorer or LinkContextOverlapStepScorer()
        heuristic = heuristic or CoverageSemanticHeuristic()
        query_tokens = _query_tokens(query)
        states, debug_trace, selector_logs = self._run_path_search(
            graph=graph,
            query=query,
            query_tokens=query_tokens,
            start_nodes=start_nodes,
            budget=budget,
            scorer=scorer,
            heuristic=heuristic,
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

        frontier_scores = {node.node_id: node.score for node in path_result.ranked_nodes}
        personalization = dict(seed_weights or _seed_weights_from_graph(graph, query, start_nodes))
        for node_id, score in frontier_scores.items():
            personalization[node_id] = personalization.get(node_id, 0.0) + score
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
            key=lambda item: (item[1].total_score, item[1].subscores.get("future_potential", 0.0), -item[0]),
        )
        next_states: list[PathState] = []
        for link, card in zip(candidate_links, score_cards):
            if card.total_score <= 0:
                continue
            link_tokens = _link_query_tokens(query_tokens, link)
            new_covered_tokens = tuple(sorted(covered_set | link_tokens))
            coverage_gain = (len(new_covered_tokens) - len(covered_set)) / query_token_count
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
            next_state.goal_reached = next_state.coverage_ratio(len(query_tokens)) >= 1.0
            next_states.append(next_state)

        log = WalkStepLog(
            step_index=state.depth,
            current_node_id=state.current_node,
            path_node_ids=list(state.visited_nodes),
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
                for link, card in zip(candidate_links, score_cards)
            ],
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
    ) -> tuple[list[PathState], list[str], list[WalkStepLog]]:
        debug_trace: list[str] = []
        selector_logs: list[WalkStepLog] = []
        seeds = self._seed_states(graph, query_tokens, start_nodes, heuristic)
        for seed in seeds:
            debug_trace.append(f"seed:{seed.current_node}:h={seed.h_score:.4f}")

        all_states = list(seeds)
        frontier = list(seeds)
        while frontier:
            next_frontier: list[PathState] = []
            for state in frontier:
                debug_trace.append(f"expand:{state.current_node}:reward={state.reward_score:.4f}:depth={state.depth}")
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
            if not next_frontier:
                break
            next_frontier.sort(
                key=lambda item: (item.reward_score, item.coverage_ratio(len(query_tokens)), -item.g_score),
                reverse=True,
            )
            frontier = next_frontier[: self.beam_width]
            all_states.extend(frontier)
            debug_trace.append("beam:" + ",".join(f"{state.current_node}:{state.reward_score:.4f}" for state in frontier))
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
    ) -> tuple[list[PathState], list[str], list[WalkStepLog]]:
        counter = itertools.count()
        debug_trace: list[str] = []
        selector_logs: list[WalkStepLog] = []
        seeds = self._seed_states(graph, query_tokens, start_nodes, heuristic)
        all_states = list(seeds)
        heap: list[tuple[float, int, PathState]] = []
        for seed in seeds:
            priority = self._priority(seed)
            heapq.heappush(heap, (priority, next(counter), seed))
            debug_trace.append(f"seed:{seed.current_node}:g={seed.g_score:.4f}:h={seed.h_score:.4f}:f={seed.f_score:.4f}")

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
                heapq.heappush(heap, (child_priority, next(counter), child))
                debug_trace.append(
                    f"push:{state.current_node}->{child.current_node}:{self.priority_label}={child_priority:.4f}:g={child.g_score:.4f}:h={child.h_score:.4f}:f={child.f_score:.4f}"
                )
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
        embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
    ):
        super().__init__(embedder_config=embedder_config, embedder_factory=embedder_factory)
        self.spec = spec
        self.name = spec.canonical_name

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
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
            budget=SelectorBudget(max_nodes=None, max_hops=0, max_tokens=_runtime_budget_token_limit(graph, budget)),
            strategy="dense",
            mode=SelectionMode.STANDALONE,
            debug_trace=[f"seed:{node_id}:{score:.4f}" for node_id, score in seed_candidates],
            root_node_ids=root_candidates,
            trace=_trace_from_seed_candidates(seed_candidates),
            selector_metadata=_baseline_selector_metadata(self.spec, seed_backend=seed_backend, seed_model=seed_model),
            selector_usage=SelectorUsage(runtime_s=runtime_s),
            stop_reason="top_k_retrieval",
        )


class CanonicalIterativeDenseSelector(_SentenceTransformerSupport):
    def __init__(
        self,
        spec: SelectorSpec,
        *,
        embedder_config: SentenceTransformerSelectorConfig | None = None,
        embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
    ):
        super().__init__(embedder_config=embedder_config, embedder_factory=embedder_factory)
        self.spec = spec
        self.name = spec.canonical_name

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
        started_at = time.perf_counter()
        embedder = _seed_embedder(self.spec, self._get_embedder)
        hop_budget = self.spec.hop_budget or 1
        per_hop_top_k = self.spec.seed_top_k or 1
        token_budget_limit = _runtime_budget_token_limit(graph, budget)

        selected_order: list[str] = []
        selected_set: set[str] = set()
        node_scores: dict[str, float] = {}
        trace: list[SelectionTraceStep] = []
        debug_trace: list[str] = []

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

        for hop_index in range(1, hop_budget + 1):
            remaining_node_ids = [node_id for node_id in graph.nodes if node_id not in selected_set]
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

        runtime_s = time.perf_counter() - started_at
        seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
        result = _build_corpus_selection_result(
            selector_name=self.name,
            graph=graph,
            query=case.query,
            node_scores=node_scores,
            link_scores={},
            budget=SelectorBudget(max_nodes=None, max_hops=hop_budget, max_tokens=token_budget_limit),
            strategy="iterative_dense",
            mode=SelectionMode.STANDALONE,
            debug_trace=debug_trace,
            root_node_ids=root_candidates,
            trace=trace,
            selector_metadata=_baseline_selector_metadata(self.spec, seed_backend=seed_backend, seed_model=seed_model),
            selector_usage=SelectorUsage(runtime_s=runtime_s),
            stop_reason="iterative_dense_retrieval",
        )
        selected_node_ids = self._trim_to_budget(graph, selected_order, token_budget_limit)
        result.selected_node_ids = selected_node_ids
        result.token_cost_estimate = sum(_node_token_cost(graph, node_id) for node_id in selected_node_ids)
        selected_node_set = set(selected_node_ids)
        result.trace = [step for step in trace if step.node_id in selected_node_set]
        return result

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

    def _expansion_query(self, graph: LinkContextGraph, base_query: str, frontier_node_ids: Sequence[str]) -> str:
        context = " ".join(self._frontier_context(graph, node_id) for node_id in frontier_node_ids)
        return f"{base_query} {context}".strip()

    def _frontier_context(self, graph: LinkContextGraph, node_id: str) -> str:
        document = graph.get_document(node_id)
        if document is None:
            return node_id
        lead_sentence = document.sentences[0] if document.sentences else ""
        return f"{document.title} {lead_sentence}".strip()

    def _trim_to_budget(self, graph: LinkContextGraph, selected_order: Sequence[str], token_budget_limit: int) -> list[str]:
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
    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
        started_at = time.perf_counter()
        embedder = _seed_embedder(self.spec, self._get_embedder)
        if embedder is None:
            raise ValueError("mdr_light requires sentence-transformer seed retrieval.")
        hop_budget = self.spec.hop_budget or 1
        per_hop_top_k = self.spec.seed_top_k or 1
        token_budget_limit = _runtime_budget_token_limit(graph, budget)

        selected_order: list[str] = []
        selected_set: set[str] = set()
        node_scores: dict[str, float] = {}
        trace: list[SelectionTraceStep] = []
        debug_trace: list[str] = []

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

        for hop_index in range(1, hop_budget + 1):
            remaining_node_ids = [node_id for node_id in graph.nodes if node_id not in selected_set]
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
                    debug_trace.append(f"hop:{hop_index}:frontier:{frontier_node_id}:no_candidates")
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

        runtime_s = time.perf_counter() - started_at
        seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
        result = _build_corpus_selection_result(
            selector_name=self.name,
            graph=graph,
            query=case.query,
            node_scores=node_scores,
            link_scores={},
            budget=SelectorBudget(max_nodes=None, max_hops=hop_budget, max_tokens=token_budget_limit),
            strategy="mdr_light",
            mode=SelectionMode.STANDALONE,
            debug_trace=debug_trace,
            root_node_ids=root_candidates,
            trace=trace,
            selector_metadata=_baseline_selector_metadata(self.spec, seed_backend=seed_backend, seed_model=seed_model),
            selector_usage=SelectorUsage(runtime_s=runtime_s),
            stop_reason="mdr_light_retrieval",
        )
        selected_node_ids = self._trim_to_budget(graph, selected_order, token_budget_limit)
        result.selected_node_ids = selected_node_ids
        result.token_cost_estimate = sum(_node_token_cost(graph, node_id) for node_id in selected_node_ids)
        selected_node_set = set(selected_node_ids)
        result.trace = [step for step in trace if step.node_id in selected_node_set]
        return result


class CanonicalNeighborSelector(_SentenceTransformerSupport):
    def __init__(
        self,
        spec: SelectorSpec,
        *,
        embedder_config: SentenceTransformerSelectorConfig | None = None,
        embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
    ):
        super().__init__(embedder_config=embedder_config, embedder_factory=embedder_factory)
        self.spec = spec
        self.name = spec.canonical_name

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
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
                    ((self._score_link(graph, case.query, link), link) for link in links),
                    key=lambda item: (item[0], item[1].target),
                )
                candidates.append((best_score, best_link))
            if not candidates:
                continue
            score, link = max(candidates, key=lambda item: (item[0], item[1].target))
            if link.target in node_scores:
                continue
            ordered_node_ids.append(link.target)
            node_scores[link.target] = max(score, _node_score(graph, case.query, link.target))
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
            budget=SelectorBudget(max_nodes=None, max_hops=1, max_tokens=_runtime_budget_token_limit(graph, budget)),
            strategy=str(self.spec.baseline),
            mode=SelectionMode.STANDALONE,
            debug_trace=[f"seed:{node_id}:{score:.4f}" for node_id, score in seed_candidates],
            root_node_ids=root_candidates,
            trace=trace,
            selector_metadata=_baseline_selector_metadata(self.spec, seed_backend=seed_backend, seed_model=seed_model),
            selector_usage=SelectorUsage(runtime_s=runtime_s),
            stop_reason="neighbor_expansion",
        )

    def _score_link(self, graph: LinkContextGraph, query: str, link: LinkContext) -> float:
        assert self.spec.baseline is not None
        if self.spec.baseline == "topology_neighbors":
            return normalized_token_overlap(query, _target_title(graph, link.target)) + len(graph.neighbors(link.target)) * 0.01
        if self.spec.baseline == "anchor_neighbors":
            return normalized_token_overlap(query, link.anchor_text)
        if self.spec.baseline == "link_context_neighbors":
            return normalized_token_overlap(query, link.anchor_text) * 0.6 + normalized_token_overlap(query, link.sentence) * 0.4
        raise ValueError(f"Unsupported neighbor baseline: {self.spec.baseline}")


class CanonicalSinglePathSelector(_SentenceTransformerSupport):
    def __init__(
        self,
        spec: SelectorSpec,
        *,
        llm_config: SelectorLLMConfig | None = None,
        backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
        embedder_config: SentenceTransformerSelectorConfig | None = None,
        embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
    ):
        super().__init__(embedder_config=embedder_config, embedder_factory=embedder_factory)
        self.spec = spec
        self.name = spec.canonical_name
        self.llm_config = llm_config or SelectorLLMConfig()
        self.backend_factory = backend_factory

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
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
            embedder=_scorer_embedder(self.spec, self._get_embedder),
        )
        walk = DynamicWalker(graph, scorer=scorer).walk(
            case.query,
            start_nodes,
            WalkBudget(max_steps=(self.spec.hop_budget or 0) + 1, min_score=0.05, allow_revisit=False),
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
        return _apply_selector_metadata(result, self.spec, seed_backend=seed_backend, seed_model=seed_model)


class CanonicalSearchSelector(_SentenceTransformerSupport):
    def __init__(
        self,
        spec: SelectorSpec,
        *,
        llm_config: SelectorLLMConfig | None = None,
        backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
        embedder_config: SentenceTransformerSelectorConfig | None = None,
        embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
    ):
        super().__init__(embedder_config=embedder_config, embedder_factory=embedder_factory)
        self.spec = spec
        self.name = spec.canonical_name
        self.llm_config = llm_config or SelectorLLMConfig()
        self.backend_factory = backend_factory

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
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
            embedder=_scorer_embedder(self.spec, self._get_embedder),
        )
        heuristic = CoverageSemanticHeuristic()
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
        )
        seed_backend, seed_model = _seed_backend_metadata(self.spec, self._get_embedder)
        return _apply_selector_metadata(result, self.spec, seed_backend=seed_backend, seed_model=seed_model)


class BudgetFillSelector(_SentenceTransformerSupport):
    def __init__(
        self,
        base_selector: CorpusSelector,
        spec: SelectorSpec,
        *,
        embedder_config: SentenceTransformerSelectorConfig | None = None,
        embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
    ):
        super().__init__(embedder_config=embedder_config, embedder_factory=embedder_factory)
        self.base_selector = base_selector
        self.spec = spec
        self.name = spec.canonical_name

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
        result = self.base_selector.select(graph, case, budget)
        budget_token_limit = _runtime_budget_token_limit(graph, budget)
        seed_candidates = _select_seed_candidates(
            graph,
            case.query,
            self.spec.budget_fill_pool_k or 64,
            seed_strategy=self.spec.seed_strategy or "lexical_overlap",
            embedder=_seed_embedder(self.spec, self._get_embedder),
        )
        selected_node_ids = list(result.selected_node_ids)
        selected_node_set = set(selected_node_ids)
        ranked_nodes = list(result.ranked_nodes)
        trace = list(result.trace)
        debug_trace = list(result.debug_trace)
        token_cost_estimate = result.token_cost_estimate
        first_backfill_score: float | None = None

        for node_id, score in seed_candidates:
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
            if budget_token_limit > 0 and token_cost_estimate + node_tokens > budget_token_limit:
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
            debug_trace.append(f"fill_add:{self.spec.budget_fill_mode}:{node_id}:{score:.4f}")

        selector_metadata = _apply_budget_fill_metadata(result.selector_metadata, self.spec)
        selected_links = [
            link for link in result.ranked_links if link.source in selected_node_set and link.target in selected_node_set
        ]
        return replace(
            result,
            selector_name=self.name,
            ranked_nodes=ranked_nodes,
            selected_node_ids=selected_node_ids,
            selected_links=selected_links,
            token_cost_estimate=token_cost_estimate,
            debug_trace=debug_trace,
            coverage_ratio=_selection_coverage_ratio(_query_tokens(case.query), selected_links),
            trace=trace,
            selector_metadata=selector_metadata,
        )


class GoldSupportContextSelector:
    name = "gold_support_context"
    spec = SelectorSpec(canonical_name="gold_support_context", family="diagnostic")

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
        del budget
        started_at = time.perf_counter()
        ordered_nodes = list(dict.fromkeys(case.gold_support_nodes))
        node_scores = {node_id: 1.0 for node_id in ordered_nodes}
        runtime_s = time.perf_counter() - started_at
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
            root_node_ids=list(dict.fromkeys(case.gold_start_nodes or case.gold_support_nodes)),
            trace=_trace_from_ordered_nodes(graph, case.query, ordered_nodes, fixed_score=1.0),
            stop_reason="gold_support_context",
            ignore_budget=True,
        )


class FullCorpusUpperBoundSelector:
    name = "full_corpus_upper_bound"
    spec = SelectorSpec(canonical_name="full_corpus_upper_bound", family="diagnostic")

    def select(self, graph: LinkContextGraph, case: SelectionCase, budget: RuntimeBudget) -> CorpusSelectionResult:
        del budget
        started_at = time.perf_counter()
        ordered_nodes = list(graph.nodes)
        runtime_s = time.perf_counter() - started_at
        node_scores = {node_id: _node_score(graph, case.query, node_id) for node_id in ordered_nodes}
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
    hop_budget = int(match.group("hop_budget"))
    rest = match.group("rest")
    parts = rest.split("__")
    if len(parts) == 1:
        baseline = parts[0]
        if baseline not in _BASELINES:
            raise ValueError(f"Unknown selector: {name}")
        if baseline == "dense" and hop_budget != 0:
            raise ValueError(f"Unknown selector: {name}")
        if baseline == "iterative_dense" and hop_budget < 1:
            raise ValueError(f"Unknown selector: {name}")
        if baseline == "mdr_light":
            if seed_strategy != "sentence_transformer" or hop_budget < 1:
                raise ValueError(f"Unknown selector: {name}")
        if baseline not in {"dense", "iterative_dense", "mdr_light"} and hop_budget != 1:
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
            budget_fill_relative_drop_ratio=0.5 if budget_fill_mode == "relative_drop" else None,
        )
        _validate_profile_for_spec(spec, raw_name=name)
        return spec
    if len(parts) != 3:
        raise ValueError(f"Unknown selector: {name}")
    search_structure, edge_scorer, lookahead = parts
    if search_structure not in _SEARCH_STRUCTURES or edge_scorer not in _EDGE_SCORERS or lookahead not in _LOOKAHEADS:
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
        budget_fill_relative_drop_ratio=0.5 if budget_fill_mode == "relative_drop" else None,
    )
    _validate_profile_for_spec(spec, raw_name=name)
    return spec


def available_selector_names(*, include_diagnostics: bool = True) -> list[str]:
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
                f"top_3_seed__{seed_strategy}__hop_0__dense",
                f"top_3_seed__{seed_strategy}__hop_1__topology_neighbors",
                f"top_3_seed__{seed_strategy}__hop_1__anchor_neighbors",
                f"top_3_seed__{seed_strategy}__hop_1__link_context_neighbors",
            ]
        )
        if seed_strategy == "sentence_transformer":
            names.extend(
                [
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
    return names


def available_selector_presets() -> list[str]:
    return ["full", "paper_recommended"]


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
    raise ValueError(f"Unknown selector preset: {preset}")


def build_selector(
    name: str,
    *,
    selector_provider: Literal["openai", "anthropic", "gemini"] = "openai",
    selector_model: str | None = None,
    selector_api_key_env: str | None = None,
    selector_base_url: str | None = None,
    selector_cache_path: str | None = None,
    selector_backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
    sentence_transformer_model: str | None = None,
    sentence_transformer_cache_path: str | Path | None = None,
    sentence_transformer_device: str | None = None,
    sentence_transformer_embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
) -> CorpusSelector:
    spec = parse_selector_spec(name)
    llm_config = SelectorLLMConfig(
        provider=selector_provider,
        model=selector_model,
        api_key_env=selector_api_key_env,
        base_url=selector_base_url,
        cache_path=None if selector_cache_path is None else Path(selector_cache_path),
    )
    sentence_transformer_config = SentenceTransformerSelectorConfig(
        model_name=sentence_transformer_model or DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        cache_path=None if sentence_transformer_cache_path is None else Path(sentence_transformer_cache_path),
        device=sentence_transformer_device,
    )
    if spec.family == "diagnostic":
        if name == "gold_support_context":
            return GoldSupportContextSelector()
        return FullCorpusUpperBoundSelector()
    if spec.edge_scorer == "link_context_llm":
        _validate_selector_llm_config(llm_config)
    selector = _build_canonical_selector(
        spec,
        llm_config=llm_config,
        backend_factory=selector_backend_factory,
        sentence_transformer_config=sentence_transformer_config,
        sentence_transformer_embedder_factory=sentence_transformer_embedder_factory,
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
    sentence_transformer_embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None,
) -> CorpusSelector:
    if spec.family == "baseline":
        if spec.baseline == "dense":
            return CanonicalDenseSelector(
                spec,
                embedder_config=sentence_transformer_config,
                embedder_factory=sentence_transformer_embedder_factory,
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
    selector_provider: Literal["openai", "anthropic", "gemini"] = "openai",
    selector_model: str | None = None,
    selector_api_key_env: str | None = None,
    selector_base_url: str | None = None,
    selector_cache_path: str | None = None,
    selector_backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
    sentence_transformer_model: str | None = None,
    sentence_transformer_cache_path: str | Path | None = None,
    sentence_transformer_device: str | None = None,
    sentence_transformer_embedder_factory: Callable[[SentenceTransformerEmbedderConfig], TextEmbedder] | None = None,
) -> list[CorpusSelector]:
    selector_names = list(names) if names is not None else selector_names_for_preset(
        preset,
        include_diagnostics=include_diagnostics,
    )
    return [
        build_selector(
            name,
            selector_provider=selector_provider,
            selector_model=selector_model,
            selector_api_key_env=selector_api_key_env,
            selector_base_url=selector_base_url,
            selector_cache_path=selector_cache_path,
            selector_backend_factory=selector_backend_factory,
            sentence_transformer_model=sentence_transformer_model,
            sentence_transformer_cache_path=sentence_transformer_cache_path,
            sentence_transformer_device=sentence_transformer_device,
            sentence_transformer_embedder_factory=sentence_transformer_embedder_factory,
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
        return SemanticBeamSelector(mode=SelectionMode.HYBRID_WITH_PPR, beam_width=4, allow_revisit=False)
    raise ValueError(f"Unsupported search structure: {search_structure}")


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
            fallback_scorer=LinkContextOverlapStepScorer(lookahead_steps=lookahead_steps),
            backend_factory=backend_factory,
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
        return getattr(embedder, "backend_name", "sentence_transformer"), getattr(embedder, "model_name", None)
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
        return _lexical_seed_candidates(graph, query, top_k, candidate_ids=candidate_ids)
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
    if spec.budget_fill_mode == "relative_drop":
        threshold = first_backfill_candidate_score * (spec.budget_fill_relative_drop_ratio or 0.5)
        return score >= threshold
    return True


def _lexical_seed_candidates(
    graph: LinkContextGraph,
    query: str,
    top_k: int,
    *,
    candidate_ids: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    candidate_pool = list(candidate_ids) if candidate_ids is not None else list(graph.nodes)
    scored = [
        (node_id, normalized_token_overlap(query, _node_text(graph, node_id)))
        for node_id in candidate_pool
    ]
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scored[:top_k]


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
    query_embedding = embedder.encode([query])[0]
    node_embeddings = embedder.encode([_node_text(graph, node_id) for node_id in node_ids])
    scored = [
        (node_id, _clamp_score(_dot_similarity(query_embedding, embedding)))
        for node_id, embedding in zip(node_ids, node_embeddings)
    ]
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scored[:top_k]


def _trace_from_seed_candidates(seed_candidates: Sequence[tuple[str, float]]) -> list[SelectionTraceStep]:
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
        metadata = _baseline_selector_metadata(spec, seed_backend=seed_backend, seed_model=seed_model)
    else:
        metadata = replace(
            metadata,
            profile_name=metadata.profile_name or _resolved_profile_name(spec),
            seed_strategy=spec.seed_strategy,
            seed_backend=seed_backend,
            seed_model=seed_model,
            seed_top_k=spec.seed_top_k if spec.seed_top_k is not None else metadata.seed_top_k,
            hop_budget=spec.hop_budget if spec.hop_budget is not None else metadata.hop_budget,
            search_structure=spec.search_structure or metadata.search_structure,
            edge_scorer=spec.edge_scorer or metadata.edge_scorer,
            lookahead_depth=spec.lookahead_depth if spec.lookahead_depth is not None else metadata.lookahead_depth,
            budget_fill_mode=spec.budget_fill_mode or metadata.budget_fill_mode,
            budget_fill_pool_k=spec.budget_fill_pool_k if spec.budget_fill_pool_k is not None else metadata.budget_fill_pool_k,
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
    if not config.api_key_env:
        raise ValueError("selector_api_key_env must be configured for LLM selectors.")
    if not config.model:
        raise ValueError("selector_model must be configured for LLM selectors.")
    if not os.environ.get(config.api_key_env):
        raise ValueError(f"Missing API key in environment variable {config.api_key_env}")


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
    return float(sum(left_value * right_value for left_value, right_value in zip(left, right)))


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
    token_counts = [_node_token_cost(graph, node_id) for node_id in graph.nodes if _node_token_cost(graph, node_id) > 0]
    return min(token_counts) if token_counts else 0


def _within_path_budget(state: PathState, budget: SelectorBudget) -> bool:
    if budget.max_nodes is not None and len(set(state.visited_nodes)) > budget.max_nodes:
        return False
    if budget.max_hops is not None and state.depth > budget.max_hops:
        return False
    if budget.max_tokens is not None and state.token_cost_estimate > budget.max_tokens:
        return False
    return True


def _seed_weights_from_graph(graph: LinkContextGraph, query: str, start_nodes: Sequence[str]) -> dict[str, float]:
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
    return (link.source, link.target, link.anchor_text, link.sentence, link.sent_idx, link.ref_id)


def _selection_coverage_ratio(query_tokens: frozenset[str], selected_links: Sequence[ScoredLink]) -> float:
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
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


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
        if budget.max_tokens is not None and token_cost_estimate + node_tokens > budget.max_tokens:
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
        SelectionTraceStep(index=index, node_id=node_id, score=fixed_score if fixed_score is not None else _node_score(graph, query, node_id))
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
    selected_links = [link for link in ranked_links if link.source in selected_node_set and link.target in selected_node_set]
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
        token_cost_estimate = sum(_node_token_cost(graph, node_id) for node_id in selected_node_ids)
    else:
        selected_node_ids, token_cost_estimate = _apply_budget_to_ranked_nodes(graph, ranked_nodes, budget)
    selected_node_set = set(selected_node_ids)
    ranked_links = sorted(link_scores.values(), key=lambda item: item.score, reverse=True)
    selected_links = [link for link in ranked_links if link.source in selected_node_set and link.target in selected_node_set]
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
        root_node_ids=[node_id for node_id in root_node_ids if node_id in selected_node_set or node_id in set(root_node_ids)],
        trace=[step for step in trace if step.node_id in selected_node_set or step.index == 0],
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
        if step.source_node_id is not None and step.anchor_text is not None and step.sentence is not None:
            for link in graph.links_between(step.source_node_id, step.node_id):
                if link.anchor_text == step.anchor_text and link.sentence == step.sentence:
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
        budget=SelectorBudget(max_nodes=None, max_hops=spec.hop_budget, max_tokens=token_budget_limit),
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
        selector_usage=_selector_usage_from_logs(walk.selector_logs, runtime_override=runtime_s),
        selector_logs=walk.selector_logs,
        stop_reason=walk.stop_reason.value,
    )
    return result


def _graph_token_estimate(graph: LinkContextGraph) -> int:
    return sum(_node_token_cost(graph, node_id) for node_id in graph.nodes)


def _target_title(graph: LinkContextGraph, node_id: str) -> str:
    return str(graph.node_attr.get(node_id, {}).get("title", node_id))


def _node_score(graph: LinkContextGraph, query: str, node_id: str) -> float:
    attr = graph.node_attr.get(node_id, {})
    text = f"{attr.get('title', '')} {attr.get('text', '')}".strip()
    return normalized_token_overlap(query, text)


def _selector_metadata_from_step_scorer(
    scorer: StepLinkScorer,
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


def _selector_usage_from_logs(logs: Sequence[WalkStepLog], *, runtime_override: float | None = None) -> SelectorUsage:
    if not logs:
        return SelectorUsage(runtime_s=runtime_override or 0.0)
    return SelectorUsage(
        runtime_s=runtime_override if runtime_override is not None else sum(log.latency_s for log in logs),
        llm_calls=sum(1 for log in logs if log.provider is not None),
        prompt_tokens=sum(log.prompt_tokens or 0 for log in logs),
        completion_tokens=sum(log.completion_tokens or 0 for log in logs),
        total_tokens=sum(log.total_tokens or 0 for log in logs),
        cache_hits=sum(1 for log in logs if log.cache_hit),
        step_count=len(logs),
        fallback_steps=sum(1 for log in logs if _is_selector_fallback(log.fallback_reason)),
        parse_failure_steps=sum(1 for log in logs if _is_selector_parse_failure(log.fallback_reason)),
    )


def _is_selector_fallback(reason: str | None) -> bool:
    return reason is not None and reason != "prefiltered_out"


def _is_selector_parse_failure(reason: str | None) -> bool:
    if reason is None:
        return False
    return reason == "empty_response" or reason.startswith("json_parse_error") or reason.startswith("schema_error")


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
