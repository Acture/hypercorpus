from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from webwalker.answering import Answerer
from webwalker.candidate.policy import SelectByCosTopK, StartPolicy
from webwalker.graph import LinkContext, LinkContextGraph
from webwalker.subgraph import SubgraphExtractor
from webwalker.text import approx_token_count, normalize_answer, normalized_token_overlap
from webwalker.walker import DynamicWalker, StopReason, WalkBudget, WalkResult, WalkStep

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
    token_budget_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("SelectionBudget.max_steps must be positive.")
        if self.top_k <= 0:
            raise ValueError("SelectionBudget.top_k must be positive.")
        if self.token_budget_ratio <= 0 or self.token_budget_ratio > 1:
            raise ValueError("SelectionBudget.token_budget_ratio must be in (0, 1].")


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
class SelectedCorpus:
    node_ids: list[str]
    edge_contexts: list[SelectedEdgeContext]
    token_estimate: int
    root_node_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SelectionMetrics:
    token_budget_ratio: float
    budget_token_limit: int
    selection_runtime_s: float
    selected_nodes_count: int
    selected_token_estimate: int
    compression_ratio: float
    budget_adherence: bool
    start_hit: bool | None = None
    support_recall: float | None = None
    support_precision: float | None = None
    path_hit: bool | None = None


@dataclass(slots=True)
class EndToEndResult:
    answer: str
    confidence: float
    evidence_count: int
    em: float | None


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


@dataclass(slots=True)
class CaseEvaluation:
    case: EvaluationCase
    selections: list[SelectionResult]


@dataclass(slots=True)
class SelectorBudgetSummary:
    name: str
    token_budget_ratio: float
    num_cases: int
    avg_start_hit: float | None
    avg_support_recall: float | None
    avg_support_precision: float | None
    avg_path_hit: float | None
    avg_selected_nodes: float
    avg_selected_token_estimate: float
    avg_compression_ratio: float
    avg_budget_adherence: float
    avg_selection_runtime_s: float
    avg_e2e_em: float | None


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


class DenseTopKSelector:
    name = "dense_topk"

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


class ExpandTopologySelector:
    name = "expand_topology"

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


class ExpandAnchorSelector:
    name = "expand_anchor"

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


class ExpandLinkContextSelector:
    name = "expand_link_context"

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


class WebWalkerSelector:
    name = "webwalker_selector"

    def __init__(self, *, start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.start_policy_factory = start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        started_at = time.perf_counter()
        start_nodes = self.start_policy_factory(budget.top_k).select_start(graph, case.query)
        walk = DynamicWalker(graph).walk(
            case.query,
            start_nodes,
            WalkBudget(max_steps=budget.max_steps, min_score=0.05),
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


class OracleStartWebWalkerSelector:
    name = "oracle_start_webwalker"

    def __init__(self, *, fallback_start_policy_factory: Callable[[int], StartPolicy[str]] | None = None):
        self.fallback_start_policy_factory = fallback_start_policy_factory or (lambda top_k: SelectByCosTopK(k=top_k))

    def select(self, graph: LinkContextGraph, case: EvaluationCase, budget: SelectionBudget) -> SelectionResult:
        started_at = time.perf_counter()
        start_nodes = case.gold_start_nodes or self.fallback_start_policy_factory(budget.top_k).select_start(graph, case.query)
        walk = DynamicWalker(graph).walk(
            case.query,
            start_nodes,
            WalkBudget(max_steps=budget.max_steps, min_score=0.05),
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


class RandomWalkSelector:
    name = "random_walk"

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


class EagerFullCorpusProxySelector:
    name = "eager_full_corpus_proxy"

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


def build_default_selectors(*, seed: int = 0) -> list[CorpusSelector]:
    return [
        DenseTopKSelector(),
        ExpandTopologySelector(),
        ExpandAnchorSelector(),
        ExpandLinkContextSelector(),
        WebWalkerSelector(),
        OracleStartWebWalkerSelector(),
        EagerFullCorpusProxySelector(),
    ]


def build_diagnostic_selectors(*, seed: int = 0) -> list[CorpusSelector]:
    return [*build_default_selectors(seed=seed), RandomWalkSelector(seed=seed)]


def available_selector_names(*, include_diagnostics: bool = True) -> list[str]:
    selectors = build_diagnostic_selectors() if include_diagnostics else build_default_selectors()
    return [selector.name for selector in selectors]


def select_selectors(
    names: Sequence[str] | None = None,
    *,
    seed: int = 0,
    include_diagnostics: bool = True,
) -> list[CorpusSelector]:
    registry = {
        selector.name: selector
        for selector in (
            build_diagnostic_selectors(seed=seed)
            if include_diagnostics
            else build_default_selectors(seed=seed)
        )
    }
    if names is None:
        return list(registry.values())

    selected: list[CorpusSelector] = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown selector: {name}")
        selected.append(registry[name])
    return selected


class Evaluator:
    def __init__(
        self,
        selectors: list[CorpusSelector] | None = None,
        *,
        budget: SelectionBudget | None = None,
        with_e2e: bool = True,
        extractor: SubgraphExtractor | None = None,
        answerer: Answerer | None = None,
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
            (selection.selector_name, selection.budget.token_budget_ratio)
            for evaluation in evaluations
            for selection in evaluation.selections
        )
    )

    selector_budgets: list[SelectorBudgetSummary] = []
    for name, token_budget_ratio in ordered_keys:
        results = [
            selection
            for evaluation in evaluations
            for selection in evaluation.selections
            if selection.selector_name == name
            and math.isclose(selection.budget.token_budget_ratio, token_budget_ratio)
        ]
        selector_budgets.append(
            SelectorBudgetSummary(
                name=name,
                token_budget_ratio=token_budget_ratio,
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
                avg_e2e_em=_average(
                    [
                        result.end_to_end.em
                        for result in results
                        if result.end_to_end is not None and result.end_to_end.em is not None
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
) -> SelectionResult:
    total_graph_tokens = _graph_token_estimate(graph)
    budget_token_limit = _budget_token_limit(graph, budget)
    if force_full_corpus:
        selected_node_ids = list(dict.fromkeys(ordered_node_ids))
        selected_token_estimate = total_graph_tokens
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
        token_budget_ratio=budget.token_budget_ratio,
        budget_token_limit=budget_token_limit,
        selection_runtime_s=runtime_s,
        selected_nodes_count=len(corpus.node_ids),
        selected_token_estimate=corpus.token_estimate,
        compression_ratio=_compression_ratio(corpus.token_estimate, total_graph_tokens),
        budget_adherence=corpus.token_estimate <= budget_token_limit,
        start_hit=_start_hit(corpus.root_node_ids, case.gold_start_nodes),
        support_recall=_recall(corpus.node_ids, case.gold_support_nodes),
        support_precision=_precision(corpus.node_ids, case.gold_support_nodes),
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
    )


def _run_end_to_end(
    *,
    graph: LinkContextGraph,
    case: EvaluationCase,
    node_ids: Sequence[str],
    extractor: SubgraphExtractor,
    answerer: Answerer,
) -> EndToEndResult:
    subgraph = extractor.extract(case.query, graph, list(node_ids))
    answer = answerer.answer(case.query, subgraph)
    return EndToEndResult(
        answer=answer.answer,
        confidence=answer.confidence,
        evidence_count=len(answer.evidence),
        em=_em(answer.answer, case.expected_answer),
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
    minimum_doc = _minimum_document_tokens(graph)
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


def _path_hit(selected_nodes: Sequence[str], gold_path_nodes: Sequence[str] | None) -> bool | None:
    if not gold_path_nodes:
        return None
    return set(gold_path_nodes).issubset(set(selected_nodes))


def _em(answer: str, expected_answer: str | None) -> float | None:
    if expected_answer is None:
        return None
    return 1.0 if normalize_answer(answer) == normalize_answer(expected_answer) else 0.0
