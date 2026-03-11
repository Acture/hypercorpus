from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

from webwalker.answering import Answerer, SupportsAnswer
from webwalker.candidate.policy import SelectByCosTopK, StartPolicy
from webwalker.graph import LinkContext, LinkContextGraph
from webwalker.selector_llm import LLMStepLinkScorer, SelectorLLMConfig
from webwalker.subgraph import SubgraphExtractor
from webwalker.text import answer_f1, approx_token_count, normalize_answer, normalized_token_overlap
from webwalker.walker import (
    AnchorOverlapStepScorer,
    DynamicWalker,
    LinkContextOverlapStepScorer,
    StepScorerMetadata,
    StopReason,
    TitleAwareOverlapStepScorer,
    WalkBudget,
    WalkResult,
    WalkStep,
    WalkStepLog,
)

DEFAULT_TOKEN_BUDGETS: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 1.0)


@dataclass(slots=True)
class EvaluationCase:
    case_id: str
    query: str
    expected_answer: str | None = None
    dataset_name: str = "synthetic"
    gold_support_nodes: list[str] = field(default_factory=list)
    gold_start_nodes: list[str] = field(default_factory=list)
    gold_path_nodes: list[str] | None = None

    def __post_init__(self) -> None:
        self.gold_support_nodes = _dedupe(self.gold_support_nodes)
        self.gold_start_nodes = _dedupe(self.gold_start_nodes or self.gold_support_nodes)
        if self.gold_path_nodes is not None:
            self.gold_path_nodes = _dedupe(self.gold_path_nodes)


@dataclass(slots=True)
class SelectionBudget:
    max_steps: int = 3
    top_k: int = 2
    token_budget_tokens: int | None = None
    token_budget_ratio: float | None = None
    budget_mode: Literal["tokens", "ratio"] = field(init=False)
    budget_value: int | float = field(init=False)
    budget_label: str = field(init=False)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("SelectionBudget.max_steps must be positive.")
        if self.top_k <= 0:
            raise ValueError("SelectionBudget.top_k must be positive.")
        if self.token_budget_tokens is None and self.token_budget_ratio is None:
            self.token_budget_tokens = DEFAULT_TOKEN_BUDGETS[0]
        if (self.token_budget_tokens is None) == (self.token_budget_ratio is None):
            raise ValueError("SelectionBudget requires exactly one of token_budget_tokens or token_budget_ratio.")
        if self.token_budget_tokens is not None:
            if self.token_budget_tokens <= 0:
                raise ValueError("SelectionBudget.token_budget_tokens must be positive.")
            self.budget_mode = "tokens"
            self.budget_value = int(self.token_budget_tokens)
            self.budget_label = f"tokens-{self.token_budget_tokens}"
            return
        assert self.token_budget_ratio is not None
        if self.token_budget_ratio <= 0 or self.token_budget_ratio > 1:
            raise ValueError("SelectionBudget.token_budget_ratio must be in (0, 1].")
        self.budget_mode = "ratio"
        self.budget_value = float(self.token_budget_ratio)
        self.budget_label = f"ratio-{self.token_budget_ratio:.4f}"


@dataclass(slots=True)
class SelectedEdgeContext:
    source: str
    target: str
    anchor_text: str
    sentence: str
    score: float


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
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    candidate_prefilter_top_n: int | None = None
    two_hop_prefilter_top_n: int | None = None


@dataclass(slots=True)
class SelectorUsage:
    runtime_s: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hits: int = 0


@dataclass(slots=True)
class SelectedCorpus:
    node_ids: list[str]
    edge_contexts: list[SelectedEdgeContext]
    token_estimate: int
    root_node_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SelectionMetrics:
    budget_mode: Literal["tokens", "ratio"]
    budget_value: int | float
    budget_label: str
    token_budget_ratio: float | None
    token_budget_tokens: int | None
    budget_token_limit: int
    selection_runtime_s: float
    selected_nodes_count: int
    selected_token_estimate: int
    compression_ratio: float
    budget_adherence: bool
    start_hit: bool | None = None
    support_recall: float | None = None
    support_precision: float | None = None
    support_f1: float | None = None
    path_hit: bool | None = None


@dataclass(slots=True)
class EndToEndResult:
    mode: str
    model: str | None
    answer: str
    confidence: float
    evidence_count: int
    em: float | None
    f1: float | None
    runtime_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(slots=True)
class SelectionResult:
    selector_name: str
    budget: SelectionBudget
    corpus: SelectedCorpus
    metrics: SelectionMetrics
    trace: list[SelectionTraceStep]
    end_to_end: EndToEndResult | None = None
    stop_reason: str | None = None
    graphrag_input_path: str | None = None
    selector_metadata: SelectorMetadata | None = None
    selector_usage: SelectorUsage | None = None
    selector_logs: list[WalkStepLog] = field(default_factory=list)


@dataclass(slots=True)
class CaseEvaluation:
    case: EvaluationCase
    selections: list[SelectionResult]


@dataclass(slots=True)
class SelectorBudgetSummary:
    name: str
    selector_provider: str | None
    selector_model: str | None
    budget_mode: Literal["tokens", "ratio"]
    budget_value: int | float
    budget_label: str
    token_budget_ratio: float | None
    token_budget_tokens: int | None
    num_cases: int
    avg_start_hit: float | None
    avg_support_recall: float | None
    avg_support_precision: float | None
    avg_support_f1: float | None
    avg_path_hit: float | None
    avg_selected_nodes: float
    avg_selected_token_estimate: float
    avg_compression_ratio: float
    avg_budget_adherence: float
    avg_selection_runtime_s: float
    avg_selector_prompt_tokens: float | None
    avg_selector_completion_tokens: float | None
    avg_selector_total_tokens: float | None
    avg_selector_runtime_s: float | None
    avg_selector_llm_calls: float | None
    avg_answer_em: float | None
    avg_answer_f1: float | None


@dataclass(slots=True)
class ExperimentSummary:
    dataset_name: str
    total_cases: int
    selector_budgets: list[SelectorBudgetSummary]


class CorpusSelector(Protocol):
    name: str

    def select(
        self,
        graph: LinkContextGraph,
        case: EvaluationCase,
        budget: SelectionBudget,
    ) -> SelectionResult:
        ...


class SeedRerankSelector:
    name = "seed_rerank"

    def __init__(self, *, start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def select(
        self,
        graph: LinkContextGraph,
        case: EvaluationCase,
        budget: SelectionBudget,
    ) -> SelectionResult:
        started_at = time.perf_counter()
        root_candidates = self.start_policy_factory(budget.top_k).select_start(graph, case.query)
        ordered_nodes = _dedupe(root_candidates)
        trace = [
            SelectionTraceStep(index=index, node_id=node_id, score=_node_score(graph, case.query, node_id))
            for index, node_id in enumerate(ordered_nodes)
        ]
        runtime_s = time.perf_counter() - started_at
        return _build_selection_result(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            ordered_node_ids=ordered_nodes,
            root_candidates=ordered_nodes,
            edge_contexts=[],
            trace=trace,
            runtime_s=runtime_s,
            stop_reason="top_k_retrieval",
        )


class SeedPlusTopologyNeighborsSelector:
    name = "seed_plus_topology_neighbors"

    def __init__(self, *, start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        return _expand_from_roots(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            start_policy_factory=self.start_policy_factory,
            score_link=lambda current, link: _topology_link_score(graph, case.query, current, link),
            tie_break_key=lambda link: (
                normalized_token_overlap(case.query, _target_title(graph, link.target)),
                len(graph.neighbors(link.target)),
                link.target,
            ),
        )


class SeedPlusAnchorNeighborsSelector:
    name = "seed_plus_anchor_neighbors"

    def __init__(self, *, start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        return _expand_from_roots(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            start_policy_factory=self.start_policy_factory,
            score_link=lambda _current, link: normalized_token_overlap(case.query, link.anchor_text),
            tie_break_key=lambda link: (link.target,),
        )


class SeedPlusLinkContextNeighborsSelector:
    name = "seed_plus_link_context_neighbors"

    def __init__(
        self,
        *,
        start_policy_factory: Callable[[int], StartPolicy[str]] | None = None,
        anchor_weight: float = 0.6,
        sentence_weight: float = 0.4,
    ):
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))
        self.anchor_weight = anchor_weight
        self.sentence_weight = sentence_weight

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        return _expand_from_roots(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            start_policy_factory=self.start_policy_factory,
            score_link=lambda _current, link: (
                normalized_token_overlap(case.query, link.anchor_text) * self.anchor_weight
                + normalized_token_overlap(case.query, link.sentence) * self.sentence_weight
            ),
            tie_break_key=lambda link: (link.target,),
        )


class _SinglePathWalkSelector:
    name = "single_path_walk"
    min_score = 0.05

    def __init__(self, *, start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def _select_start_nodes(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> list[str]:
        return self.start_policy_factory(budget.top_k).select_start(graph, case.query)

    def _build_step_scorer(self):
        raise NotImplementedError

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        started_at = time.perf_counter()
        start_nodes = self._select_start_nodes(graph, case, budget)
        scorer = self._build_step_scorer()
        walk = DynamicWalker(graph, scorer=scorer).walk(
            case.query,
            start_nodes,
            WalkBudget(max_steps=budget.max_steps, min_score=self.min_score),
        )
        runtime_s = time.perf_counter() - started_at
        return _selection_from_walk(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            walk=walk,
            runtime_s=runtime_s,
        )


class SeedAnchorOverlapSinglePathWalkSelector(_SinglePathWalkSelector):
    name = "seed__anchor_overlap__single_path_walk"

    def _build_step_scorer(self):
        return AnchorOverlapStepScorer()


class SeedLinkContextOverlapSinglePathWalkSelector(_SinglePathWalkSelector):
    name = "seed__link_context_overlap__single_path_walk"

    def _build_step_scorer(self):
        return LinkContextOverlapStepScorer()


class SeedAnchorOverlapTwoHopSinglePathWalkSelector(_SinglePathWalkSelector):
    name = "seed__anchor_overlap__two_hop_single_path_walk"

    def _build_step_scorer(self):
        return AnchorOverlapStepScorer(lookahead_steps=2)


class SeedLinkContextOverlapTwoHopSinglePathWalkSelector(_SinglePathWalkSelector):
    name = "seed__link_context_overlap__two_hop_single_path_walk"

    def _build_step_scorer(self):
        return LinkContextOverlapStepScorer(lookahead_steps=2)


class SeedTitleAwareSinglePathWalkSelector(_SinglePathWalkSelector):
    name = "seed__title_aware__single_path_walk"

    def _build_step_scorer(self):
        return TitleAwareOverlapStepScorer()


class _LLMSinglePathWalkSelector(_SinglePathWalkSelector):
    llm_mode: Literal["single_hop", "two_hop"] = "single_hop"

    def __init__(
        self,
        *,
        llm_config: SelectorLLMConfig | None = None,
        start_policy_factory: Callable[[int], StartPolicy[str]] | None = None,
        backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
    ):
        super().__init__(start_policy_factory=start_policy_factory)
        self.llm_config = llm_config or SelectorLLMConfig()
        self.backend_factory = backend_factory

    def _build_step_scorer(self):
        return LLMStepLinkScorer(
            config=self.llm_config,
            mode=self.llm_mode,
            prefilter_scorer=LinkContextOverlapStepScorer(),
            fallback_scorer=(
                LinkContextOverlapStepScorer(lookahead_steps=2)
                if self.llm_mode == "two_hop"
                else LinkContextOverlapStepScorer()
            ),
            backend_factory=self.backend_factory,
        )


class SeedLinkContextLLMSinglePathWalkSelector(_LLMSinglePathWalkSelector):
    name = "seed__link_context_llm__single_path_walk"
    llm_mode = "single_hop"


class SeedLinkContextLLMTwoHopSinglePathWalkSelector(_LLMSinglePathWalkSelector):
    name = "seed__link_context_llm__two_hop_single_path_walk"
    llm_mode = "two_hop"


class OracleSeedLinkContextOverlapSinglePathWalkSelector(_SinglePathWalkSelector):
    name = "oracle_seed__link_context_overlap__single_path_walk"

    def __init__(self, *, fallback_start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.fallback_start_policy_factory = fallback_start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def _select_start_nodes(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> list[str]:
        return case.gold_start_nodes or self.fallback_start_policy_factory(budget.top_k).select_start(graph, case.query)

    def _build_step_scorer(self):
        return LinkContextOverlapStepScorer()


class OracleSeedLinkContextLLMSinglePathWalkSelector(_LLMSinglePathWalkSelector):
    name = "oracle_seed__link_context_llm__single_path_walk"
    llm_mode = "single_hop"

    def __init__(
        self,
        *,
        llm_config: SelectorLLMConfig | None = None,
        fallback_start_policy_factory: Callable[[int], StartPolicy[str]] | None = None,
        backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
    ):
        super().__init__(llm_config=llm_config, start_policy_factory=None, backend_factory=backend_factory)
        self.fallback_start_policy_factory = fallback_start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def _select_start_nodes(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> list[str]:
        return case.gold_start_nodes or self.fallback_start_policy_factory(budget.top_k).select_start(graph, case.query)


class RandomWalkSelector:
    name = "random__single_path_walk"

    def __init__(
        self,
        *,
        seed: int = 0,
        start_policy_factory: Callable[[int], StartPolicy[str]] | None = None,
    ):
        self.seed = seed
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        started_at = time.perf_counter()
        start_nodes = self.start_policy_factory(budget.top_k).select_start(graph, case.query)
        walk = _random_walk(
            graph,
            case,
            start_nodes,
            WalkBudget(max_steps=budget.max_steps, min_score=0.0),
            self.seed,
        )
        runtime_s = time.perf_counter() - started_at
        return _selection_from_walk(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            walk=walk,
            runtime_s=runtime_s,
        )


class FullCorpusUpperBoundSelector:
    name = "full_corpus_upper_bound"

    def __init__(self, *, start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        started_at = time.perf_counter()
        root_candidates = self.start_policy_factory(budget.top_k).select_start(graph, case.query)
        ordered_nodes = list(graph.nodes)
        trace = [
            SelectionTraceStep(index=index, node_id=node_id, score=_node_score(graph, case.query, node_id))
            for index, node_id in enumerate(root_candidates)
        ]
        runtime_s = time.perf_counter() - started_at
        return _build_selection_result(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            ordered_node_ids=ordered_nodes,
            root_candidates=root_candidates,
            edge_contexts=[],
            trace=trace,
            runtime_s=runtime_s,
            stop_reason="full_corpus_proxy",
            force_full_corpus=True,
        )


class GoldSupportContextSelector:
    name = "gold_support_context"

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        started_at = time.perf_counter()
        ordered_nodes = list(case.gold_support_nodes)
        root_candidates = list(case.gold_start_nodes or case.gold_support_nodes)
        trace = [
            SelectionTraceStep(
                index=index,
                node_id=node_id,
                score=1.0 if node_id in set(case.gold_support_nodes) else _node_score(graph, case.query, node_id),
            )
            for index, node_id in enumerate(ordered_nodes)
        ]
        runtime_s = time.perf_counter() - started_at
        return _build_selection_result(
            selector_name=self.name,
            graph=graph,
            case=case,
            budget=budget,
            ordered_node_ids=ordered_nodes,
            root_candidates=root_candidates,
            edge_contexts=[],
            trace=trace,
            runtime_s=runtime_s,
            stop_reason="gold_support_context",
            ignore_budget=True,
        )


def build_default_selectors(*, seed: int = 0) -> list[CorpusSelector]:
    del seed
    return [
        SeedRerankSelector(),
        SeedPlusTopologyNeighborsSelector(),
        SeedPlusAnchorNeighborsSelector(),
        SeedPlusLinkContextNeighborsSelector(),
        SeedAnchorOverlapSinglePathWalkSelector(),
        SeedLinkContextOverlapSinglePathWalkSelector(),
        SeedAnchorOverlapTwoHopSinglePathWalkSelector(),
        SeedLinkContextOverlapTwoHopSinglePathWalkSelector(),
    ]


def build_diagnostic_selectors(*, seed: int = 0) -> list[CorpusSelector]:
    return [
        *build_default_selectors(seed=seed),
        SeedTitleAwareSinglePathWalkSelector(),
        OracleSeedLinkContextOverlapSinglePathWalkSelector(),
        GoldSupportContextSelector(),
        RandomWalkSelector(seed=seed),
        FullCorpusUpperBoundSelector(),
    ]


def available_selector_names(*, include_diagnostics: bool = True) -> list[str]:
    default_names = [
        "seed_rerank",
        "seed_plus_topology_neighbors",
        "seed_plus_anchor_neighbors",
        "seed_plus_link_context_neighbors",
        "seed__anchor_overlap__single_path_walk",
        "seed__link_context_overlap__single_path_walk",
        "seed__anchor_overlap__two_hop_single_path_walk",
        "seed__link_context_overlap__two_hop_single_path_walk",
    ]
    if not include_diagnostics:
        return default_names
    return [
        *default_names,
        "seed__title_aware__single_path_walk",
        "oracle_seed__link_context_overlap__single_path_walk",
        "gold_support_context",
        "random__single_path_walk",
        "full_corpus_upper_bound",
        "seed__link_context_llm__single_path_walk",
        "seed__link_context_llm__two_hop_single_path_walk",
        "oracle_seed__link_context_llm__single_path_walk",
    ]


def select_selectors(
    names: Sequence[str] | None = None,
    *,
    seed: int = 0,
    include_diagnostics: bool = True,
    selector_provider: Literal["openai", "anthropic", "gemini"] = "openai",
    selector_model: str | None = None,
    selector_api_key_env: str | None = None,
    selector_base_url: str | None = None,
    selector_cache_path: str | None = None,
    selector_backend_factory: Callable[[SelectorLLMConfig], Any] | None = None,
) -> list[CorpusSelector]:
    llm_config = SelectorLLMConfig(
        provider=selector_provider,
        model=selector_model,
        api_key_env=selector_api_key_env,
        base_url=selector_base_url,
        cache_path=None if selector_cache_path is None else Path(selector_cache_path),
    )
    registry = _selector_registry(
        seed=seed,
        include_diagnostics=include_diagnostics,
        llm_config=llm_config,
        selector_backend_factory=selector_backend_factory,
    )
    if names is None:
        return build_diagnostic_selectors(seed=seed) if include_diagnostics else build_default_selectors(seed=seed)

    selected: list[CorpusSelector] = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown selector: {name}")
        if _selector_name_uses_llm(name):
            _validate_selector_llm_config(llm_config)
        selected.append(registry[name]())
    return selected


def _selector_registry(
    *,
    seed: int,
    include_diagnostics: bool,
    llm_config: SelectorLLMConfig,
    selector_backend_factory: Callable[[SelectorLLMConfig], Any] | None,
) -> dict[str, Callable[[], CorpusSelector]]:
    default_builders: list[tuple[str, Callable[[], CorpusSelector]]] = [
        ("seed_rerank", SeedRerankSelector),
        ("seed_plus_topology_neighbors", SeedPlusTopologyNeighborsSelector),
        ("seed_plus_anchor_neighbors", SeedPlusAnchorNeighborsSelector),
        ("seed_plus_link_context_neighbors", SeedPlusLinkContextNeighborsSelector),
        ("seed__anchor_overlap__single_path_walk", SeedAnchorOverlapSinglePathWalkSelector),
        ("seed__link_context_overlap__single_path_walk", SeedLinkContextOverlapSinglePathWalkSelector),
        ("seed__anchor_overlap__two_hop_single_path_walk", SeedAnchorOverlapTwoHopSinglePathWalkSelector),
        ("seed__link_context_overlap__two_hop_single_path_walk", SeedLinkContextOverlapTwoHopSinglePathWalkSelector),
    ]
    registry = {name: builder for name, builder in default_builders}
    if not include_diagnostics:
        return registry
    registry.update(
        {
            "seed__title_aware__single_path_walk": SeedTitleAwareSinglePathWalkSelector,
            "oracle_seed__link_context_overlap__single_path_walk": OracleSeedLinkContextOverlapSinglePathWalkSelector,
            "gold_support_context": GoldSupportContextSelector,
            "random__single_path_walk": lambda: RandomWalkSelector(seed=seed),
            "full_corpus_upper_bound": FullCorpusUpperBoundSelector,
            "seed__link_context_llm__single_path_walk": lambda: SeedLinkContextLLMSinglePathWalkSelector(
                llm_config=llm_config,
                backend_factory=selector_backend_factory,
            ),
            "seed__link_context_llm__two_hop_single_path_walk": lambda: SeedLinkContextLLMTwoHopSinglePathWalkSelector(
                llm_config=llm_config,
                backend_factory=selector_backend_factory,
            ),
            "oracle_seed__link_context_llm__single_path_walk": lambda: OracleSeedLinkContextLLMSinglePathWalkSelector(
                llm_config=llm_config,
                backend_factory=selector_backend_factory,
            ),
        }
    )
    return registry


def _validate_selector_llm_config(config: SelectorLLMConfig) -> None:
    if not config.api_key_env:
        raise ValueError("selector_api_key_env must be configured for LLM selectors.")
    if not config.model:
        raise ValueError("selector_model must be configured for LLM selectors.")
    if not os.environ.get(config.api_key_env):
        raise ValueError(f"Missing API key in environment variable {config.api_key_env}")


def _selector_name_uses_llm(name: str) -> bool:
    return "_llm__" in name


class Evaluator:
    def __init__(
        self,
        selectors: list[CorpusSelector] | None = None,
        *,
        budget: SelectionBudget | None = None,
        with_e2e: bool = False,
        extractor: SubgraphExtractor | None = None,
        answerer: SupportsAnswer | None = None,
    ):
        self.selectors = selectors or build_default_selectors()
        self.budget = budget or SelectionBudget()
        self.with_e2e = with_e2e
        self.extractor = extractor or SubgraphExtractor()
        self.answerer = answerer or Answerer()

    def evaluate_case(self, graph: LinkContextGraph, case: EvaluationCase) -> CaseEvaluation:
        selections: list[SelectionResult] = []
        for selector in self.selectors:
            result = selector.select(graph, case, self.budget)
            if self.with_e2e:
                result.end_to_end = _run_end_to_end(
                    graph=graph,
                    case=case,
                    node_ids=result.corpus.node_ids,
                    extractor=self.extractor,
                    answerer=self.answerer,
                )
            selections.append(result)
        return CaseEvaluation(case=case, selections=selections)

    def summarize(self, evaluations: Sequence[CaseEvaluation]) -> ExperimentSummary:
        if not evaluations:
            raise ValueError("Cannot summarize an empty experiment.")
        dataset_name = evaluations[0].case.dataset_name
        return summarize_evaluations(evaluations, dataset_name=dataset_name)


def summarize_evaluations(
    evaluations: Sequence[CaseEvaluation],
    *,
    dataset_name: str,
) -> ExperimentSummary:
    if not evaluations:
        raise ValueError("Cannot summarize an empty experiment.")

    ordered_keys = list(
        dict.fromkeys(
            (
                selection.selector_name,
                selection.budget.budget_mode,
                selection.budget.budget_value,
                selection.selector_metadata.provider if selection.selector_metadata is not None else None,
                selection.selector_metadata.model if selection.selector_metadata is not None else None,
            )
            for evaluation in evaluations
            for selection in evaluation.selections
        )
    )

    selector_budgets: list[SelectorBudgetSummary] = []
    for name, budget_mode, budget_value, selector_provider, selector_model in ordered_keys:
        results = [
            selection
            for evaluation in evaluations
            for selection in evaluation.selections
            if selection.selector_name == name
            and _matches_budget(selection.budget, budget_mode=budget_mode, budget_value=budget_value)
            and (selection.selector_metadata.provider if selection.selector_metadata is not None else None)
            == selector_provider
            and (selection.selector_metadata.model if selection.selector_metadata is not None else None) == selector_model
        ]
        exemplar = results[0].budget
        selector_budgets.append(
            SelectorBudgetSummary(
                name=name,
                selector_provider=selector_provider,
                selector_model=selector_model,
                budget_mode=budget_mode,
                budget_value=budget_value,
                budget_label=exemplar.budget_label,
                token_budget_ratio=exemplar.token_budget_ratio,
                token_budget_tokens=exemplar.token_budget_tokens,
                num_cases=len(results),
                avg_start_hit=_average(
                    [1.0 if result.metrics.start_hit else 0.0 for result in results if result.metrics.start_hit is not None]
                ),
                avg_support_recall=_average(
                    [result.metrics.support_recall for result in results if result.metrics.support_recall is not None]
                ),
                avg_support_precision=_average(
                    [result.metrics.support_precision for result in results if result.metrics.support_precision is not None]
                ),
                avg_support_f1=_average(
                    [result.metrics.support_f1 for result in results if result.metrics.support_f1 is not None]
                ),
                avg_path_hit=_average(
                    [1.0 if result.metrics.path_hit else 0.0 for result in results if result.metrics.path_hit is not None]
                ),
                avg_selected_nodes=_average([float(result.metrics.selected_nodes_count) for result in results]) or 0.0,
                avg_selected_token_estimate=_average(
                    [float(result.metrics.selected_token_estimate) for result in results]
                )
                or 0.0,
                avg_compression_ratio=_average([result.metrics.compression_ratio for result in results]) or 0.0,
                avg_budget_adherence=_average([1.0 if result.metrics.budget_adherence else 0.0 for result in results]) or 0.0,
                avg_selection_runtime_s=_average([result.metrics.selection_runtime_s for result in results]) or 0.0,
                avg_selector_prompt_tokens=_average(
                    [float(result.selector_usage.prompt_tokens) for result in results if result.selector_usage is not None]
                ),
                avg_selector_completion_tokens=_average(
                    [float(result.selector_usage.completion_tokens) for result in results if result.selector_usage is not None]
                ),
                avg_selector_total_tokens=_average(
                    [float(result.selector_usage.total_tokens) for result in results if result.selector_usage is not None]
                ),
                avg_selector_runtime_s=_average(
                    [result.selector_usage.runtime_s for result in results if result.selector_usage is not None]
                ),
                avg_selector_llm_calls=_average(
                    [float(result.selector_usage.llm_calls) for result in results if result.selector_usage is not None]
                ),
                avg_answer_em=_average(
                    [
                        result.end_to_end.em
                        for result in results
                        if result.end_to_end is not None and result.end_to_end.em is not None
                    ]
                ),
                avg_answer_f1=_average(
                    [
                        result.end_to_end.f1
                        for result in results
                        if result.end_to_end is not None and result.end_to_end.f1 is not None
                    ]
                ),
            )
        )

    return ExperimentSummary(
        dataset_name=dataset_name,
        total_cases=len(evaluations),
        selector_budgets=selector_budgets,
    )


def _build_selection_result(
    *,
    selector_name: str,
    graph: LinkContextGraph,
    case: EvaluationCase,
    budget: SelectionBudget,
    ordered_node_ids: Sequence[str],
    root_candidates: Sequence[str],
    edge_contexts: Sequence[SelectedEdgeContext],
    trace: Sequence[SelectionTraceStep],
    runtime_s: float,
    stop_reason: str | None,
    force_full_corpus: bool = False,
    ignore_budget: bool = False,
    selector_metadata: SelectorMetadata | None = None,
    selector_usage: SelectorUsage | None = None,
    selector_logs: Sequence[WalkStepLog] | None = None,
) -> SelectionResult:
    total_graph_tokens = _graph_token_estimate(graph)
    budget_token_limit = _budget_token_limit(graph, budget)
    if force_full_corpus:
        selected_node_ids = list(dict.fromkeys(ordered_node_ids))
        selected_token_estimate = total_graph_tokens
        selected_root_ids = [node_id for node_id in root_candidates if node_id in set(selected_node_ids)]
    elif ignore_budget:
        selected_node_ids = list(dict.fromkeys(ordered_node_ids))
        selected_token_estimate = sum(_node_token_cost(graph, node_id) for node_id in selected_node_ids)
        selected_root_ids = [node_id for node_id in root_candidates if node_id in set(selected_node_ids)]
    else:
        selected_node_ids, selected_token_estimate = _fit_nodes_in_order(
            graph,
            ordered_node_ids,
            budget_token_limit,
        )
        selected_root_ids = [node_id for node_id in root_candidates if node_id in set(selected_node_ids)]

    selected_node_set = set(selected_node_ids)
    corpus = SelectedCorpus(
        node_ids=selected_node_ids,
        edge_contexts=[
            context
            for context in edge_contexts
            if context.source in selected_node_set and context.target in selected_node_set
        ],
        token_estimate=selected_token_estimate,
        root_node_ids=selected_root_ids,
    )
    metrics = SelectionMetrics(
        budget_mode=budget.budget_mode,
        budget_value=budget.budget_value,
        budget_label=budget.budget_label,
        token_budget_ratio=budget.token_budget_ratio,
        token_budget_tokens=budget.token_budget_tokens,
        budget_token_limit=budget_token_limit,
        selection_runtime_s=runtime_s,
        selected_nodes_count=len(corpus.node_ids),
        selected_token_estimate=corpus.token_estimate,
        compression_ratio=_compression_ratio(corpus.token_estimate, total_graph_tokens),
        budget_adherence=corpus.token_estimate <= budget_token_limit,
        start_hit=_start_hit(corpus.root_node_ids, case.gold_start_nodes),
        support_recall=_recall(corpus.node_ids, case.gold_support_nodes),
        support_precision=_precision(corpus.node_ids, case.gold_support_nodes),
        support_f1=_support_f1(corpus.node_ids, case.gold_support_nodes),
        path_hit=_path_hit(corpus.node_ids, case.gold_path_nodes),
    )
    return SelectionResult(
        selector_name=selector_name,
        budget=budget,
        corpus=corpus,
        metrics=metrics,
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


def _expand_from_roots(
    *,
    selector_name: str,
    graph: LinkContextGraph,
    case: EvaluationCase,
    budget: SelectionBudget,
    start_policy_factory: Callable[[int], StartPolicy[str]],
    score_link: Callable[[str, LinkContext], float],
    tie_break_key: Callable[[LinkContext], tuple[object, ...]],
) -> SelectionResult:
    started_at = time.perf_counter()
    root_candidates = start_policy_factory(budget.top_k).select_start(graph, case.query)
    ordered_node_ids = list(_dedupe(root_candidates))
    trace = [
        SelectionTraceStep(index=index, node_id=node_id, score=_node_score(graph, case.query, node_id))
        for index, node_id in enumerate(ordered_node_ids)
    ]
    edge_contexts: list[SelectedEdgeContext] = []
    expansion_slots = max(1, budget.max_steps - 1)

    for root_node_id in root_candidates:
        candidates: list[tuple[float, LinkContext]] = []
        for neighbor in graph.neighbors(root_node_id):
            links = graph.links_between(root_node_id, neighbor)
            if not links:
                continue
            best_score, best_link = max(
                ((score_link(root_node_id, link), link) for link in links),
                key=lambda item: (item[0], *tie_break_key(item[1])),
            )
            candidates.append((best_score, best_link))

        candidates.sort(
            key=lambda item: (item[0], *tie_break_key(item[1])),
            reverse=True,
        )

        for score, link in candidates[:expansion_slots]:
            if link.target in ordered_node_ids:
                continue
            ordered_node_ids.append(link.target)
            edge_contexts.append(
                SelectedEdgeContext(
                    source=link.source,
                    target=link.target,
                    anchor_text=link.anchor_text,
                    sentence=link.sentence,
                    score=score,
                )
            )
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
    return _build_selection_result(
        selector_name=selector_name,
        graph=graph,
        case=case,
        budget=budget,
        ordered_node_ids=ordered_node_ids,
        root_candidates=root_candidates,
        edge_contexts=edge_contexts,
        trace=trace,
        runtime_s=runtime_s,
        stop_reason="neighbor_expansion",
    )


def _selection_from_walk(
    *,
    selector_name: str,
    graph: LinkContextGraph,
    case: EvaluationCase,
    budget: SelectionBudget,
    walk: WalkResult,
    runtime_s: float,
) -> SelectionResult:
    trace = [
        SelectionTraceStep(
            index=step.index,
            node_id=step.node_id,
            score=step.score,
            source_node_id=step.source_node_id,
            anchor_text=step.anchor_text,
            sentence=step.sentence,
        )
        for step in walk.steps
    ]
    edge_contexts = [
        SelectedEdgeContext(
            source=step.source_node_id or "",
            target=step.node_id,
            anchor_text=step.anchor_text or "",
            sentence=step.sentence or "",
            score=step.score,
        )
        for step in walk.steps[1:]
    ]
    root_candidates = [walk.steps[0].node_id] if walk.steps else []
    return _build_selection_result(
        selector_name=selector_name,
        graph=graph,
        case=case,
        budget=budget,
        ordered_node_ids=walk.visited_nodes,
        root_candidates=root_candidates,
        edge_contexts=edge_contexts,
        trace=trace,
        runtime_s=runtime_s,
        stop_reason=walk.stop_reason.value,
        selector_metadata=_selector_metadata_from_walk(walk),
        selector_usage=_selector_usage_from_logs(walk.selector_logs),
        selector_logs=walk.selector_logs,
    )


def _run_end_to_end(
    *,
    graph: LinkContextGraph,
    case: EvaluationCase,
    node_ids: Sequence[str],
    extractor: SubgraphExtractor,
    answerer: SupportsAnswer,
) -> EndToEndResult:
    subgraph = extractor.extract(case.query, graph, list(node_ids))
    answer = answerer.answer(case.query, subgraph)
    return EndToEndResult(
        mode=answer.mode,
        model=answer.model,
        answer=answer.answer,
        confidence=answer.confidence,
        evidence_count=len(answer.evidence),
        em=_em(answer.answer, case.expected_answer),
        f1=answer_f1(answer.answer, case.expected_answer),
        runtime_s=answer.runtime_s,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        total_tokens=answer.total_tokens,
    )


def _fit_nodes_in_order(
    graph: LinkContextGraph,
    ordered_node_ids: Sequence[str],
    budget_token_limit: int,
) -> tuple[list[str], int]:
    selected: list[str] = []
    seen: set[str] = set()
    token_estimate = 0
    for node_id in ordered_node_ids:
        if node_id in seen:
            continue
        node_tokens = _node_token_cost(graph, node_id)
        if token_estimate + node_tokens > budget_token_limit:
            continue
        selected.append(node_id)
        seen.add(node_id)
        token_estimate += node_tokens
    return selected, token_estimate


def _budget_token_limit(graph: LinkContextGraph, budget: SelectionBudget) -> int:
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
        _node_token_cost(graph, node_id)
        for node_id in graph.nodes
        if _node_token_cost(graph, node_id) > 0
    ]
    return min(token_counts) if token_counts else 0


def _random_walk(
    graph: LinkContextGraph,
    case: EvaluationCase,
    start_nodes: Sequence[str],
    budget: WalkBudget,
    seed: int,
) -> WalkResult:
    rng = random.Random(f"{seed}:{case.case_id}")
    return _walk_with_neighbor_selector(
        graph,
        case.query,
        list(start_nodes),
        budget,
        lambda _current, eligible: rng.choice(eligible),
    )


def _walk_with_neighbor_selector(
    graph: LinkContextGraph,
    query: str,
    start_nodes: list[str],
    budget: WalkBudget,
    selector: Callable[[str, list[str]], str],
) -> WalkResult:
    if budget.max_steps <= 0:
        raise ValueError("Walk budget must allow at least one step.")
    if not start_nodes:
        raise ValueError("walk requires at least one start node.")

    current = start_nodes[0]
    visited_nodes = [current]
    visited_set = {current}
    steps = [WalkStep(index=0, node_id=current, score=_node_score(graph, query, current))]
    stop_reason = StopReason.BUDGET_EXHAUSTED

    while len(steps) < budget.max_steps:
        eligible = [
            neighbor
            for neighbor in graph.neighbors(current)
            if budget.allow_revisit or neighbor not in visited_set
        ]
        if not eligible:
            stop_reason = StopReason.DEAD_END
            break

        next_node = selector(current, eligible)
        link = graph.links_between(current, next_node)[0]
        score = _node_score(graph, query, next_node)
        if score < budget.min_score:
            stop_reason = StopReason.SCORE_BELOW_THRESHOLD
            break

        current = next_node
        visited_nodes.append(current)
        visited_set.add(current)
        steps.append(
            WalkStep(
                index=len(steps),
                node_id=current,
                score=score,
                source_node_id=link.source,
                anchor_text=link.anchor_text,
                sentence=link.sentence,
            )
        )
    else:
        stop_reason = StopReason.BUDGET_EXHAUSTED

    return WalkResult(
        query=query,
        steps=steps,
        visited_nodes=visited_nodes,
        stop_reason=stop_reason,
        scorer_metadata=StepScorerMetadata(
            scorer_kind="overlap",
            backend="overlap",
        ),
        selector_logs=[],
    )


def _topology_link_score(
    graph: LinkContextGraph,
    query: str,
    _current: str,
    link: LinkContext,
) -> float:
    return (
        normalized_token_overlap(query, _target_title(graph, link.target))
        + len(graph.neighbors(link.target)) * 0.01
    )


def _target_title(graph: LinkContextGraph, node_id: str) -> str:
    return str(graph.node_attr.get(node_id, {}).get("title", node_id))


def _node_score(graph: LinkContextGraph, query: str, node_id: str) -> float:
    attr = graph.node_attr.get(node_id, {})
    text = f"{attr.get('title', '')} {attr.get('text', '')}".strip()
    return normalized_token_overlap(query, text)


def _graph_token_estimate(graph: LinkContextGraph) -> int:
    total = 0
    for node_id in graph.nodes:
        total += _node_token_cost(graph, node_id)
    return total


def _node_token_cost(graph: LinkContextGraph, node_id: str) -> int:
    document = graph.get_document(node_id)
    if document is None:
        return 0
    return approx_token_count(document.text)


def _compression_ratio(selected_tokens: int, total_tokens: int) -> float:
    if total_tokens <= 0:
        return 0.0
    return selected_tokens / total_tokens


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _start_hit(root_node_ids: Sequence[str], gold_start_nodes: Sequence[str]) -> bool | None:
    if not gold_start_nodes:
        return None
    if not root_node_ids:
        return False
    return bool(set(root_node_ids) & set(gold_start_nodes))


def _recall(selected_nodes: Sequence[str], gold_nodes: Sequence[str]) -> float | None:
    if not gold_nodes:
        return None
    selected = set(selected_nodes)
    gold = set(gold_nodes)
    return len(selected & gold) / len(gold)


def _precision(selected_nodes: Sequence[str], gold_nodes: Sequence[str]) -> float | None:
    if not selected_nodes:
        return None
    selected = set(selected_nodes)
    gold = set(gold_nodes)
    return len(selected & gold) / len(selected)


def _support_f1(selected_nodes: Sequence[str], gold_nodes: Sequence[str]) -> float | None:
    precision = _precision(selected_nodes, gold_nodes)
    recall = _recall(selected_nodes, gold_nodes)
    if precision is None or recall is None:
        return None
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _path_hit(selected_nodes: Sequence[str], gold_path_nodes: Sequence[str] | None) -> bool | None:
    if not gold_path_nodes:
        return None
    return set(gold_path_nodes).issubset(set(selected_nodes))


def _selector_metadata_from_walk(walk: WalkResult) -> SelectorMetadata:
    metadata: StepScorerMetadata = walk.scorer_metadata
    return SelectorMetadata(
        scorer_kind=metadata.scorer_kind,
        backend=metadata.backend,
        provider=metadata.provider,
        model=metadata.model,
        prompt_version=metadata.prompt_version,
        candidate_prefilter_top_n=metadata.candidate_prefilter_top_n,
        two_hop_prefilter_top_n=metadata.two_hop_prefilter_top_n,
    )


def _selector_usage_from_logs(logs: Sequence[WalkStepLog]) -> SelectorUsage:
    if not logs:
        return SelectorUsage()
    return SelectorUsage(
        runtime_s=sum(log.latency_s for log in logs),
        llm_calls=sum(1 for log in logs if log.provider is not None),
        prompt_tokens=sum(log.prompt_tokens or 0 for log in logs),
        completion_tokens=sum(log.completion_tokens or 0 for log in logs),
        total_tokens=sum(log.total_tokens or 0 for log in logs),
        cache_hits=sum(1 for log in logs if log.cache_hit),
    )


def _matches_budget(
    budget: SelectionBudget,
    *,
    budget_mode: Literal["tokens", "ratio"],
    budget_value: int | float,
) -> bool:
    if budget.budget_mode != budget_mode:
        return False
    if budget_mode == "tokens":
        return budget.token_budget_tokens == int(budget_value)
    assert budget.token_budget_ratio is not None
    return math.isclose(budget.token_budget_ratio, float(budget_value))


def _em(answer: str, expected_answer: str | None) -> float | None:
    if expected_answer is None:
        return None
    return 1.0 if normalize_answer(answer) == normalize_answer(expected_answer) else 0.0
