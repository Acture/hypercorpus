from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

from hypercorpus.answering import Answerer, SupportsAnswer
from hypercorpus.graph import LinkContextGraph
from hypercorpus.selector import (
    CorpusSelectionResult,
    CorpusSelector,
    ScoredLink,
    SelectionTraceStep,
    SelectorMetadata,
    SelectorUsage,
)
from hypercorpus.subgraph import SubgraphExtractor
from hypercorpus.text import answer_f1, approx_token_count, normalize_answer
from hypercorpus.walker import WalkStepLog

DEFAULT_TOKEN_BUDGETS: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_BUDGET_RATIOS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 1.0)
QuestionType = Literal["bridge", "comparison", "unknown"]


@dataclass(slots=True)
class EvaluationCase:
    case_id: str
    query: str
    expected_answer: str | None = None
    dataset_name: str = "synthetic"
    gold_support_nodes: list[str] = field(default_factory=list)
    gold_start_nodes: list[str] = field(default_factory=list)
    gold_path_nodes: list[str] | None = None
    question_type: QuestionType | None = None

    def __post_init__(self) -> None:
        self.gold_support_nodes = _dedupe(self.gold_support_nodes)
        self.gold_start_nodes = _dedupe(self.gold_start_nodes or self.gold_support_nodes)
        if self.gold_path_nodes is not None:
            self.gold_path_nodes = _dedupe(self.gold_path_nodes)
        self.question_type = _normalize_or_infer_question_type(
            self.question_type,
            gold_support_nodes=self.gold_support_nodes,
            gold_path_nodes=self.gold_path_nodes,
        )


@dataclass(slots=True)
class EvaluationBudget:
    token_budget_tokens: int | None = None
    token_budget_ratio: float | None = None
    budget_mode: Literal["tokens", "ratio"] = field(init=False)
    budget_value: int | float = field(init=False)
    budget_label: str = field(init=False)

    def __post_init__(self) -> None:
        if self.token_budget_tokens is None and self.token_budget_ratio is None:
            self.token_budget_tokens = DEFAULT_TOKEN_BUDGETS[0]
        if (self.token_budget_tokens is None) == (self.token_budget_ratio is None):
            raise ValueError("EvaluationBudget requires exactly one of token_budget_tokens or token_budget_ratio.")
        if self.token_budget_tokens is not None:
            if self.token_budget_tokens <= 0:
                raise ValueError("EvaluationBudget.token_budget_tokens must be positive.")
            self.budget_mode = "tokens"
            self.budget_value = int(self.token_budget_tokens)
            self.budget_label = f"tokens-{self.token_budget_tokens}"
            return
        assert self.token_budget_ratio is not None
        if self.token_budget_ratio <= 0 or self.token_budget_ratio > 1:
            raise ValueError("EvaluationBudget.token_budget_ratio must be in (0, 1].")
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
    budget_utilization: float
    empty_selection: bool
    start_hit: bool | None = None
    support_recall: float | None = None
    support_precision: float | None = None
    support_f1: float | None = None
    support_f1_zero_on_empty: float | None = None
    support_set_em: float | None = None
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
    budget: EvaluationBudget
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
    avg_budget_utilization: float
    avg_empty_selection_rate: float
    avg_selection_runtime_s: float
    avg_support_f1_zero_on_empty: float | None
    avg_selector_prompt_tokens: float | None
    avg_selector_completion_tokens: float | None
    avg_selector_total_tokens: float | None
    avg_selector_runtime_s: float | None
    avg_selector_llm_calls: float | None
    avg_selector_fallback_rate: float | None
    avg_selector_parse_failure_rate: float | None
    avg_answer_em: float | None
    avg_answer_f1: float | None
    avg_support_set_em: float | None = None


@dataclass(slots=True)
class ExperimentSummary:
    dataset_name: str
    total_cases: int
    selector_budgets: list[SelectorBudgetSummary]


@dataclass(slots=True)
class _AverageAccumulator:
    total: float = 0.0
    count: int = 0

    def add(self, value: float | None) -> None:
        if value is None:
            return
        self.total += value
        self.count += 1

    def average(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    def average_or_zero(self) -> float:
        value = self.average()
        return value if value is not None else 0.0


@dataclass(slots=True)
class _SelectorBudgetAccumulator:
    name: str
    selector_provider: str | None
    selector_model: str | None
    budget: EvaluationBudget
    num_cases: int = 0
    avg_start_hit: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_support_recall: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_support_precision: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_support_f1: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_path_hit: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selected_nodes: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selected_token_estimate: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_compression_ratio: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_budget_adherence: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_budget_utilization: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_empty_selection_rate: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selection_runtime_s: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_support_f1_zero_on_empty: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_support_set_em: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selector_prompt_tokens: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selector_completion_tokens: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selector_total_tokens: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selector_runtime_s: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selector_llm_calls: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selector_fallback_rate: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_selector_parse_failure_rate: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_answer_em: _AverageAccumulator = field(default_factory=_AverageAccumulator)
    avg_answer_f1: _AverageAccumulator = field(default_factory=_AverageAccumulator)

    def add_result(self, result: SelectionResult) -> None:
        self.num_cases += 1
        metrics = result.metrics
        usage = result.selector_usage
        self.avg_start_hit.add(_bool_metric(metrics.start_hit))
        self.avg_support_recall.add(metrics.support_recall)
        self.avg_support_precision.add(metrics.support_precision)
        self.avg_support_f1.add(metrics.support_f1)
        self.avg_path_hit.add(_bool_metric(metrics.path_hit))
        self.avg_selected_nodes.add(float(metrics.selected_nodes_count))
        self.avg_selected_token_estimate.add(float(metrics.selected_token_estimate))
        self.avg_compression_ratio.add(metrics.compression_ratio)
        self.avg_budget_adherence.add(1.0 if metrics.budget_adherence else 0.0)
        self.avg_budget_utilization.add(metrics.budget_utilization)
        self.avg_empty_selection_rate.add(1.0 if metrics.empty_selection else 0.0)
        self.avg_selection_runtime_s.add(metrics.selection_runtime_s)
        self.avg_support_f1_zero_on_empty.add(metrics.support_f1_zero_on_empty)
        self.avg_support_set_em.add(metrics.support_set_em)
        if usage is not None:
            self.avg_selector_prompt_tokens.add(float(usage.prompt_tokens))
            self.avg_selector_completion_tokens.add(float(usage.completion_tokens))
            self.avg_selector_total_tokens.add(float(usage.total_tokens))
            self.avg_selector_runtime_s.add(usage.runtime_s)
            self.avg_selector_llm_calls.add(float(usage.llm_calls))
            if usage.step_count > 0:
                self.avg_selector_fallback_rate.add(usage.fallback_steps / usage.step_count)
                self.avg_selector_parse_failure_rate.add(usage.parse_failure_steps / usage.step_count)
        if result.end_to_end is not None:
            self.avg_answer_em.add(result.end_to_end.em)
            self.avg_answer_f1.add(result.end_to_end.f1)

    def to_summary(self) -> SelectorBudgetSummary:
        return SelectorBudgetSummary(
            name=self.name,
            selector_provider=self.selector_provider,
            selector_model=self.selector_model,
            budget_mode=self.budget.budget_mode,
            budget_value=self.budget.budget_value,
            budget_label=self.budget.budget_label,
            token_budget_ratio=self.budget.token_budget_ratio,
            token_budget_tokens=self.budget.token_budget_tokens,
            num_cases=self.num_cases,
            avg_start_hit=self.avg_start_hit.average(),
            avg_support_recall=self.avg_support_recall.average(),
            avg_support_precision=self.avg_support_precision.average(),
            avg_support_f1=self.avg_support_f1.average(),
            avg_path_hit=self.avg_path_hit.average(),
            avg_selected_nodes=self.avg_selected_nodes.average_or_zero(),
            avg_selected_token_estimate=self.avg_selected_token_estimate.average_or_zero(),
            avg_compression_ratio=self.avg_compression_ratio.average_or_zero(),
            avg_budget_adherence=self.avg_budget_adherence.average_or_zero(),
            avg_budget_utilization=self.avg_budget_utilization.average_or_zero(),
            avg_empty_selection_rate=self.avg_empty_selection_rate.average_or_zero(),
            avg_selection_runtime_s=self.avg_selection_runtime_s.average_or_zero(),
            avg_support_f1_zero_on_empty=self.avg_support_f1_zero_on_empty.average(),
            avg_support_set_em=self.avg_support_set_em.average(),
            avg_selector_prompt_tokens=self.avg_selector_prompt_tokens.average(),
            avg_selector_completion_tokens=self.avg_selector_completion_tokens.average(),
            avg_selector_total_tokens=self.avg_selector_total_tokens.average(),
            avg_selector_runtime_s=self.avg_selector_runtime_s.average(),
            avg_selector_llm_calls=self.avg_selector_llm_calls.average(),
            avg_selector_fallback_rate=self.avg_selector_fallback_rate.average(),
            avg_selector_parse_failure_rate=self.avg_selector_parse_failure_rate.average(),
            avg_answer_em=self.avg_answer_em.average(),
            avg_answer_f1=self.avg_answer_f1.average(),
        )


class IncrementalExperimentAggregator:
    def __init__(self, *, dataset_name: str):
        self.dataset_name = dataset_name
        self.total_cases = 0
        self._buckets: dict[tuple[str, str, int | float, str | None, str | None], _SelectorBudgetAccumulator] = {}

    def add_case_evaluation(self, evaluation: CaseEvaluation) -> None:
        self.total_cases += 1
        for selection in evaluation.selections:
            self.add_selection_result(selection)

    def add_selection_result(self, selection: SelectionResult) -> None:
        key = (
            selection.selector_name,
            selection.budget.budget_mode,
            selection.budget.budget_value,
            selection.selector_metadata.provider if selection.selector_metadata is not None else None,
            selection.selector_metadata.model if selection.selector_metadata is not None else None,
        )
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _SelectorBudgetAccumulator(
                name=selection.selector_name,
                selector_provider=selection.selector_metadata.provider if selection.selector_metadata is not None else None,
                selector_model=selection.selector_metadata.model if selection.selector_metadata is not None else None,
                budget=selection.budget,
            )
            self._buckets[key] = bucket
        bucket.add_result(selection)

    def to_summary(self) -> ExperimentSummary:
        return ExperimentSummary(
            dataset_name=self.dataset_name,
            total_cases=self.total_cases,
            selector_budgets=[bucket.to_summary() for bucket in self._buckets.values()],
        )

    @classmethod
    def from_evaluations(
        cls,
        evaluations: Sequence[CaseEvaluation],
        *,
        dataset_name: str,
    ) -> IncrementalExperimentAggregator:
        aggregator = cls(dataset_name=dataset_name)
        for evaluation in evaluations:
            aggregator.add_case_evaluation(evaluation)
        return aggregator


class Evaluator:
    def __init__(
        self,
        selectors: Sequence[CorpusSelector],
        *,
        budget: EvaluationBudget | None = None,
        with_e2e: bool = False,
        extractor: SubgraphExtractor | None = None,
        answerer: SupportsAnswer | None = None,
    ):
        self.selectors = list(selectors)
        self.budget = budget or EvaluationBudget()
        self.with_e2e = with_e2e
        self.extractor = extractor or SubgraphExtractor()
        self.answerer = answerer or Answerer()

    def evaluate_case(self, graph: LinkContextGraph, case: EvaluationCase) -> CaseEvaluation:
        selections: list[SelectionResult] = []
        for selector in self.selectors:
            raw_selection = selector.select(graph, case, self.budget)
            result = _selection_result_from_raw(graph=graph, case=case, budget=self.budget, raw=raw_selection)
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
        return summarize_evaluations(evaluations, dataset_name=evaluations[0].case.dataset_name)


def summarize_evaluations(
    evaluations: Sequence[CaseEvaluation],
    *,
    dataset_name: str,
) -> ExperimentSummary:
    if not evaluations:
        raise ValueError("Cannot summarize an empty experiment.")
    return IncrementalExperimentAggregator.from_evaluations(
        evaluations,
        dataset_name=dataset_name,
    ).to_summary()


def _selection_result_from_raw(
    *,
    graph: LinkContextGraph,
    case: EvaluationCase,
    budget: EvaluationBudget,
    raw: CorpusSelectionResult,
) -> SelectionResult:
    total_graph_tokens = _graph_token_estimate(graph)
    budget_token_limit = _budget_token_limit(graph, budget)
    corpus = SelectedCorpus(
        node_ids=list(raw.selected_node_ids),
        edge_contexts=[_selected_edge_from_link(link) for link in raw.selected_links],
        token_estimate=raw.token_cost_estimate,
        root_node_ids=list(raw.root_node_ids),
    )
    metrics = SelectionMetrics(
        budget_mode=budget.budget_mode,
        budget_value=budget.budget_value,
        budget_label=budget.budget_label,
        token_budget_ratio=budget.token_budget_ratio,
        token_budget_tokens=budget.token_budget_tokens,
        budget_token_limit=budget_token_limit,
        selection_runtime_s=raw.selector_usage.runtime_s if raw.selector_usage is not None else 0.0,
        selected_nodes_count=len(corpus.node_ids),
        selected_token_estimate=corpus.token_estimate,
        compression_ratio=_compression_ratio(corpus.token_estimate, total_graph_tokens),
        budget_adherence=corpus.token_estimate <= budget_token_limit,
        budget_utilization=_budget_utilization(corpus.token_estimate, budget_token_limit),
        empty_selection=not corpus.node_ids,
        start_hit=_start_hit(corpus.root_node_ids, case.gold_start_nodes),
        support_recall=_recall(corpus.node_ids, case.gold_support_nodes),
        support_precision=_precision(corpus.node_ids, case.gold_support_nodes),
        support_f1=_support_f1(corpus.node_ids, case.gold_support_nodes),
        support_f1_zero_on_empty=_support_f1_zero_on_empty(corpus.node_ids, case.gold_support_nodes),
        support_set_em=_support_set_em(corpus.node_ids, case.gold_support_nodes),
        path_hit=_path_hit(corpus.node_ids, case.gold_path_nodes),
    )
    return SelectionResult(
        selector_name=raw.selector_name,
        budget=budget,
        corpus=corpus,
        metrics=metrics,
        trace=list(raw.trace),
        stop_reason=raw.stop_reason,
        selector_metadata=raw.selector_metadata,
        selector_usage=raw.selector_usage or SelectorUsage(),
        selector_logs=list(raw.selector_logs),
    )


def _selected_edge_from_link(link: ScoredLink) -> SelectedEdgeContext:
    return SelectedEdgeContext(
        source=link.source,
        target=link.target,
        anchor_text=link.anchor_text,
        sentence=link.sentence,
        score=link.score,
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


def _budget_token_limit(graph: LinkContextGraph, budget: EvaluationBudget) -> int:
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
    token_counts = [c for node_id in graph.nodes if (c := _node_token_cost(graph, node_id)) > 0]
    return min(token_counts) if token_counts else 0


def _graph_token_estimate(graph: LinkContextGraph) -> int:
    if hasattr(graph, "total_token_estimate"):
        return graph.total_token_estimate()
    return sum(_node_token_cost(graph, node_id) for node_id in graph.nodes)


def _node_token_cost(graph: LinkContextGraph, node_id: str) -> int:
    document = graph.get_document(node_id)
    if document is None:
        return 0
    return approx_token_count(document.text)


def _compression_ratio(selected_tokens: int, total_tokens: int) -> float:
    if total_tokens <= 0:
        return 0.0
    return selected_tokens / total_tokens


def _budget_utilization(selected_tokens: int, budget_token_limit: int) -> float:
    if budget_token_limit <= 0:
        return 0.0
    return selected_tokens / budget_token_limit


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _bool_metric(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


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


def _support_f1_zero_on_empty(selected_nodes: Sequence[str], gold_nodes: Sequence[str]) -> float | None:
    support_f1 = _support_f1(selected_nodes, gold_nodes)
    if support_f1 is not None:
        return support_f1
    if not gold_nodes:
        return None
    if not selected_nodes:
        return 0.0
    return None


def _support_set_em(selected_nodes: Sequence[str], gold_nodes: Sequence[str]) -> float | None:
    if not gold_nodes:
        return None
    return 1.0 if set(gold_nodes).issubset(set(selected_nodes)) else 0.0


def _path_hit(selected_nodes: Sequence[str], gold_path_nodes: Sequence[str] | None) -> bool | None:
    if not gold_path_nodes:
        return None
    return set(gold_path_nodes).issubset(set(selected_nodes))


def _matches_budget(
    budget: EvaluationBudget,
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


def _normalize_or_infer_question_type(
    value: QuestionType | str | None,
    *,
    gold_support_nodes: Sequence[str],
    gold_path_nodes: Sequence[str] | None,
) -> QuestionType:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"bridge", "comparison", "unknown"}:
            return normalized  # type: ignore[return-value]
    if gold_path_nodes:
        return "bridge"
    if len(set(gold_support_nodes)) >= 2:
        return "comparison"
    return "unknown"


__all__ = [
    "CaseEvaluation",
    "EndToEndResult",
    "EvaluationBudget",
    "EvaluationCase",
    "Evaluator",
    "ExperimentSummary",
    "SelectedCorpus",
    "SelectedEdgeContext",
    "SelectionMetrics",
    "SelectionResult",
    "SelectorBudgetSummary",
    "summarize_evaluations",
]
