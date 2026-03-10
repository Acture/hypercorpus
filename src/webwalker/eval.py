from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from webwalker.answering import AnswerWithEvidence, Answerer
from webwalker.candidate.policy import SelectByCosTopK, StartPolicy
from webwalker.graph import LinkContextGraph
from webwalker.subgraph import QuerySubgraph, SubgraphExtractor
from webwalker.text import normalize_answer
from webwalker.walker import DynamicWalker, WalkBudget, WalkResult


@dataclass(slots=True)
class EvaluationCase:
	case_id: str
	query: str
	expected_answer: str | None = None


@dataclass(slots=True)
class PipelineMetrics:
	runtime_s: float
	time_to_first_answer_s: float
	token_cost_estimate: int
	visited_steps: int


@dataclass(slots=True)
class PipelineResult:
	name: str
	answer: AnswerWithEvidence
	subgraph: QuerySubgraph
	visited_nodes: list[str]
	metrics: PipelineMetrics
	correct: bool | None
	walk: WalkResult | None = None


@dataclass(slots=True)
class CaseEvaluation:
	case: EvaluationCase
	results: list[PipelineResult]


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
		return PipelineResult(
			name=self.name,
			answer=answer,
			subgraph=subgraph,
			visited_nodes=visited_nodes,
			metrics=PipelineMetrics(
				runtime_s=runtime,
				time_to_first_answer_s=runtime,
				token_cost_estimate=subgraph.token_cost_estimate,
				visited_steps=len(visited_nodes),
			),
			correct=_is_correct(answer.answer, case.expected_answer),
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
		return PipelineResult(
			name=self.name,
			answer=answer,
			subgraph=subgraph,
			visited_nodes=visited_nodes,
			metrics=PipelineMetrics(
				runtime_s=runtime,
				time_to_first_answer_s=runtime,
				token_cost_estimate=subgraph.token_cost_estimate,
				visited_steps=len(visited_nodes),
			),
			correct=_is_correct(answer.answer, case.expected_answer),
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
		return PipelineResult(
			name=self.name,
			answer=answer,
			subgraph=subgraph,
			visited_nodes=walk.visited_nodes,
			metrics=PipelineMetrics(
				runtime_s=runtime,
				time_to_first_answer_s=runtime,
				token_cost_estimate=subgraph.token_cost_estimate,
				visited_steps=len(walk.steps),
			),
			correct=_is_correct(answer.answer, case.expected_answer),
			walk=walk,
		)


class Evaluator:
	def __init__(self, pipelines: list[QueryPipeline] | None = None):
		self.pipelines = pipelines or [
			BaselineGraphRAGPipeline(),
			DenseRAGPipeline(),
			WebWalkerPipeline(),
		]

	def evaluate_case(self, graph: LinkContextGraph, case: EvaluationCase) -> CaseEvaluation:
		return CaseEvaluation(
			case=case,
			results=[pipeline.run(graph, case) for pipeline in self.pipelines],
		)


def _is_correct(answer: str, expected_answer: str | None) -> bool | None:
	if expected_answer is None:
		return None
	return normalize_answer(answer) == normalize_answer(expected_answer)
