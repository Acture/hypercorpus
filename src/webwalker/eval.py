from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from webwalker.answering import AnswerWithEvidence, Answerer
from webwalker.candidate.policy import SelectByCosTopK, StartPolicy
from webwalker.graph import LinkContextGraph
from webwalker.selector import (
	CorpusSelectionResult,
	CorpusSelector,
	SELECTION_BUDGET_PRESETS,
	SelectionBudget,
	SelectionMode,
	SemanticAStarSelector,
	SemanticBeamSelector,
	SemanticGBFSSelector,
	SemanticPPRSelector,
	SemanticUCSSelector,
)
from webwalker.subgraph import QuerySubgraph, SubgraphExtractor
from webwalker.text import content_tokens, normalize_answer
from webwalker.walker import DynamicWalker, WalkBudget, WalkResult


@dataclass(slots=True)
class EvaluationCase:
	case_id: str
	query: str
	expected_answer: str | None = None
	supporting_node_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineMetrics:
	runtime_s: float
	time_to_first_answer_s: float
	token_cost_estimate: int
	visited_steps: int
	support_recall: float | None = None
	support_precision: float | None = None
	selected_nodes: int | None = None
	selected_links: int | None = None
	tokens_per_recalled_support: float | None = None
	coverage_ratio: float | None = None
	answer_em: bool | None = None


@dataclass(slots=True)
class PipelineResult:
	name: str
	answer: AnswerWithEvidence
	subgraph: QuerySubgraph
	visited_nodes: list[str]
	metrics: PipelineMetrics
	correct: bool | None
	walk: WalkResult | None = None
	selection: CorpusSelectionResult | None = None


@dataclass(slots=True)
class CaseEvaluation:
	case: EvaluationCase
	results: list[PipelineResult]

	def selection_report(self) -> "SelectionReport":
		rows = [_selection_report_row(self.case, result) for result in self.results]
		rows.sort(key=_selection_report_sort_key)
		return SelectionReport(case_id=self.case.case_id, rows=rows)


@dataclass(slots=True)
class SelectionReportRow:
	name: str
	support_recall: float | None
	support_precision: float | None
	token_cost_estimate: int
	tokens_per_recalled_support: float | None
	coverage_ratio: float | None
	answer_em: bool | None
	selected_nodes: int | None
	selected_links: int | None


@dataclass(slots=True)
class SelectionReport:
	case_id: str
	rows: list[SelectionReportRow]


class QueryPipeline(Protocol):
	name: str

	def run(self, graph: LinkContextGraph, case: EvaluationCase) -> PipelineResult:
		...


class DenseRAGPipeline:
	name = "dense_rag"

	def __init__(self, *, top_k: int = 2, extractor: SubgraphExtractor | None = None, answerer: Answerer | None = None):
		self.top_k = top_k
		self.extractor = extractor or SubgraphExtractor()
		self.answerer = answerer or Answerer()

	def run(self, graph: LinkContextGraph, case: EvaluationCase) -> PipelineResult:
		started_at = time.perf_counter()
		visited_nodes = [node for node, _score in graph.topk_similar(case.query, graph.nodes, self.top_k)]
		subgraph = self.extractor.extract(case.query, graph, visited_nodes)
		answer = self.answerer.answer(case.query, subgraph)
		runtime = time.perf_counter() - started_at
		correct = _is_correct(answer.answer, case.expected_answer)
		return PipelineResult(
			name=self.name,
			answer=answer,
			subgraph=subgraph,
			visited_nodes=visited_nodes,
			metrics=_metrics_from_nodes(
				case=case,
				selected_nodes=visited_nodes,
				selected_links=len(subgraph.relations),
				token_cost_estimate=subgraph.token_cost_estimate,
				coverage_ratio=_subgraph_coverage_ratio(case.query, subgraph),
				runtime=runtime,
				answer=answer,
			),
			correct=correct,
		)


class BaselineGraphRAGPipeline:
	name = "baseline_graphrag"

	def __init__(
		self,
		*,
		top_k: int = 2,
		expand_neighbors: int = 2,
		extractor: SubgraphExtractor | None = None,
		answerer: Answerer | None = None,
	):
		self.top_k = top_k
		self.expand_neighbors = expand_neighbors
		self.extractor = extractor or SubgraphExtractor()
		self.answerer = answerer or Answerer()

	def run(self, graph: LinkContextGraph, case: EvaluationCase) -> PipelineResult:
		started_at = time.perf_counter()
		roots = [node for node, _score in graph.topk_similar(case.query, graph.nodes, self.top_k)]
		visited_nodes = list(roots)
		for node_id in roots:
			for neighbor in graph.neighbors(node_id)[: self.expand_neighbors]:
				if neighbor not in visited_nodes:
					visited_nodes.append(neighbor)
		subgraph = self.extractor.extract(case.query, graph, visited_nodes)
		answer = self.answerer.answer(case.query, subgraph)
		runtime = time.perf_counter() - started_at
		correct = _is_correct(answer.answer, case.expected_answer)
		return PipelineResult(
			name=self.name,
			answer=answer,
			subgraph=subgraph,
			visited_nodes=visited_nodes,
			metrics=_metrics_from_nodes(
				case=case,
				selected_nodes=visited_nodes,
				selected_links=len(subgraph.relations),
				token_cost_estimate=subgraph.token_cost_estimate,
				coverage_ratio=_subgraph_coverage_ratio(case.query, subgraph),
				runtime=runtime,
				answer=answer,
			),
			correct=correct,
		)


class WebWalkerPipeline:
	name = "webwalker"

	def __init__(
		self,
		*,
		start_policy: StartPolicy[str] | None = None,
		walk_budget: WalkBudget | None = None,
		extractor: SubgraphExtractor | None = None,
		answerer: Answerer | None = None,
	):
		self.start_policy = start_policy or SelectByCosTopK(k=2)
		self.walk_budget = walk_budget or WalkBudget(max_steps=3, min_score=0.05)
		self.extractor = extractor or SubgraphExtractor()
		self.answerer = answerer or Answerer()

	def run(self, graph: LinkContextGraph, case: EvaluationCase) -> PipelineResult:
		started_at = time.perf_counter()
		start_nodes = self.start_policy.select_start(graph, case.query)
		walk = DynamicWalker(graph).walk(case.query, start_nodes, self.walk_budget)
		subgraph = self.extractor.extract(case.query, graph, walk.visited_nodes)
		answer = self.answerer.answer(case.query, subgraph)
		runtime = time.perf_counter() - started_at
		correct = _is_correct(answer.answer, case.expected_answer)
		return PipelineResult(
			name=self.name,
			answer=answer,
			subgraph=subgraph,
			visited_nodes=walk.visited_nodes,
			metrics=_metrics_from_nodes(
				case=case,
				selected_nodes=walk.visited_nodes,
				selected_links=len(subgraph.relations),
				token_cost_estimate=subgraph.token_cost_estimate,
				coverage_ratio=_subgraph_coverage_ratio(case.query, subgraph),
				runtime=runtime,
				answer=answer,
				visited_steps=len(walk.steps),
			),
			correct=correct,
			walk=walk,
		)


class SemanticSelectorPipeline:
	def __init__(
		self,
		selector: CorpusSelector,
		*,
		start_policy: StartPolicy[str] | None = None,
		selection_budget: SelectionBudget | None = None,
		extractor: SubgraphExtractor | None = None,
		answerer: Answerer | None = None,
	):
		self.selector = selector
		self.name = _selector_pipeline_name(selector)
		self.start_policy = start_policy or SelectByCosTopK(k=3)
		self.selection_budget = selection_budget or _copy_budget(SELECTION_BUDGET_PRESETS["combined_default"])
		self.extractor = extractor or SubgraphExtractor()
		self.answerer = answerer or Answerer()

	def run(self, graph: LinkContextGraph, case: EvaluationCase) -> PipelineResult:
		started_at = time.perf_counter()
		start_nodes = self.start_policy.select_start(graph, case.query)
		selection = self.selector.select(
			graph,
			case.query,
			start_nodes,
			self.selection_budget,
		)
		subgraph = self.extractor.extract(case.query, graph, selection.selected_node_ids)
		answer = self.answerer.answer(case.query, subgraph)
		runtime = time.perf_counter() - started_at
		correct = _is_correct(answer.answer, case.expected_answer)
		return PipelineResult(
			name=self.name,
			answer=answer,
			subgraph=subgraph,
			visited_nodes=selection.selected_node_ids,
			metrics=_metrics_from_selection(
				case=case,
				selection=selection,
				runtime=runtime,
				answer=answer,
			),
			correct=correct,
			selection=selection,
		)


class Evaluator:
	def __init__(self, pipelines: list[QueryPipeline] | None = None):
		self.pipelines = pipelines or [
			BaselineGraphRAGPipeline(),
			DenseRAGPipeline(),
			WebWalkerPipeline(),
			SemanticSelectorPipeline(SemanticBeamSelector()),
			SemanticSelectorPipeline(SemanticAStarSelector()),
			SemanticSelectorPipeline(SemanticGBFSSelector()),
			SemanticSelectorPipeline(SemanticUCSSelector()),
			SemanticSelectorPipeline(SemanticPPRSelector()),
			SemanticSelectorPipeline(SemanticBeamSelector(mode=SelectionMode.HYBRID_WITH_PPR)),
			SemanticSelectorPipeline(SemanticAStarSelector(mode=SelectionMode.HYBRID_WITH_PPR)),
			SemanticSelectorPipeline(SemanticGBFSSelector(mode=SelectionMode.HYBRID_WITH_PPR)),
			SemanticSelectorPipeline(SemanticUCSSelector(mode=SelectionMode.HYBRID_WITH_PPR)),
		]

	def evaluate_case(self, graph: LinkContextGraph, case: EvaluationCase) -> CaseEvaluation:
		return CaseEvaluation(
			case=case,
			results=[pipeline.run(graph, case) for pipeline in self.pipelines],
		)


def _selector_pipeline_name(selector: CorpusSelector) -> str:
	mode = getattr(selector, "mode", SelectionMode.STANDALONE)
	if mode == SelectionMode.HYBRID_WITH_PPR:
		return f"{selector.strategy_name}_ppr"
	return selector.strategy_name


def _selection_report_row(case: EvaluationCase, result: PipelineResult) -> SelectionReportRow:
	return SelectionReportRow(
		name=result.name,
		support_recall=result.metrics.support_recall,
		support_precision=result.metrics.support_precision,
		token_cost_estimate=result.metrics.token_cost_estimate,
		tokens_per_recalled_support=result.metrics.tokens_per_recalled_support,
		coverage_ratio=result.metrics.coverage_ratio,
		answer_em=_is_correct(result.answer.answer, case.expected_answer),
		selected_nodes=result.metrics.selected_nodes,
		selected_links=result.metrics.selected_links,
	)


def _selection_report_sort_key(row: SelectionReportRow) -> tuple[float, float, float, float]:
	recall = row.support_recall if row.support_recall is not None else float("-inf")
	token_cost = float(row.token_cost_estimate)
	coverage = row.coverage_ratio if row.coverage_ratio is not None else float("-inf")
	answer_em = 1.0 if row.answer_em else 0.0
	return (
		-recall,
		token_cost,
		-coverage,
		-answer_em,
	)


def _copy_budget(budget: SelectionBudget) -> SelectionBudget:
	return SelectionBudget(
		max_nodes=budget.max_nodes,
		max_hops=budget.max_hops,
		max_tokens=budget.max_tokens,
	)


def _metrics_from_selection(
	*,
	case: EvaluationCase,
	selection: CorpusSelectionResult,
	runtime: float,
	answer: AnswerWithEvidence,
) -> PipelineMetrics:
	return _build_metrics(
		case=case,
		selected_nodes=selection.selected_node_ids,
		selected_links=len(selection.selected_links),
		token_cost_estimate=selection.token_cost_estimate,
		coverage_ratio=selection.coverage_ratio,
		runtime=runtime,
		answer=answer,
		visited_steps=len(selection.selected_node_ids),
	)


def _metrics_from_nodes(
	*,
	case: EvaluationCase,
	selected_nodes: Sequence[str],
	selected_links: int,
	token_cost_estimate: int,
	coverage_ratio: float | None,
	runtime: float,
	answer: AnswerWithEvidence,
	visited_steps: int | None = None,
) -> PipelineMetrics:
	return _build_metrics(
		case=case,
		selected_nodes=selected_nodes,
		selected_links=selected_links,
		token_cost_estimate=token_cost_estimate,
		coverage_ratio=coverage_ratio,
		runtime=runtime,
		answer=answer,
		visited_steps=visited_steps or len(selected_nodes),
	)


def _build_metrics(
	*,
	case: EvaluationCase,
	selected_nodes: Sequence[str],
	selected_links: int,
	token_cost_estimate: int,
	coverage_ratio: float | None,
	runtime: float,
	answer: AnswerWithEvidence,
	visited_steps: int,
) -> PipelineMetrics:
	support_set = set(case.supporting_node_ids)
	selected_set = set(selected_nodes)
	recalled = support_set & selected_set
	support_recall = None
	support_precision = None
	tokens_per_recalled_support = None
	if support_set:
		support_recall = len(recalled) / len(support_set)
		support_precision = len(recalled) / len(selected_set) if selected_set else 0.0
		if recalled:
			tokens_per_recalled_support = token_cost_estimate / len(recalled)

	answer_em = _is_correct(answer.answer, case.expected_answer)
	return PipelineMetrics(
		runtime_s=runtime,
		time_to_first_answer_s=runtime,
		token_cost_estimate=token_cost_estimate,
		visited_steps=visited_steps,
		support_recall=support_recall,
		support_precision=support_precision,
		selected_nodes=len(selected_set),
		selected_links=selected_links,
		tokens_per_recalled_support=tokens_per_recalled_support,
		coverage_ratio=coverage_ratio,
		answer_em=answer_em,
	)


def _subgraph_coverage_ratio(query: str, subgraph: QuerySubgraph) -> float:
	query_tokens = set(content_tokens(query))
	if not query_tokens:
		return 0.0
	covered_tokens: set[str] = set()
	for relation in subgraph.relations:
		covered_tokens |= query_tokens & set(content_tokens(f"{relation.anchor_text} {relation.sentence}"))
	return len(covered_tokens) / len(query_tokens)


def _is_correct(answer: str, expected_answer: str | None) -> bool | None:
	if expected_answer is None:
		return None
	return normalize_answer(answer) == normalize_answer(expected_answer)
