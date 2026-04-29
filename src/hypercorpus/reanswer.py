"""Re-run the LLM answerer over selections produced by prior experiments.

This module reads `results.jsonl` rows produced by `run_iirc_store_experiment`
(and other store-backed runners), reconstructs the selected subgraph from
`selected_node_ids` against a `ShardedDocumentStore`, and feeds the subgraph
through `LLMAnswerer` to compute answer F1/EM. It does **not** re-run any
selectors, which makes it cheap enough to scale up the end-to-end QA evaluation
to N=100 without paying selector LLM cost again.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, cast

from hypercorpus.answering import (
	Answerer,
	AnswererProvider,
	LLMAnswerer,
	LLMAnswererConfig,
	SupportsAnswer,
)
from hypercorpus.datasets.store import ShardedDocumentStore
from hypercorpus.eval import EvaluationCase, _em
from hypercorpus.subgraph import FullDocumentExtractor, SubgraphExtractor
from hypercorpus.text import answer_f1

ExtractorLike = SubgraphExtractor | FullDocumentExtractor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReanswerInputRow:
	"""A subset of fields from a prior `results.jsonl` row that reanswer needs."""

	source_run: str
	dataset_name: str
	case_id: str
	query: str
	expected_answer: str | None
	gold_support_nodes: list[str]
	gold_start_nodes: list[str]
	gold_path_nodes: list[str] | None
	question_type: str | None
	selector: str
	budget_label: str
	selected_node_ids: list[str]
	selected_token_estimate: int
	support_f1: float | None


def _coerce_str_list(value: Any) -> list[str]:
	if value is None:
		return []
	return [str(item) for item in value]


def _row_from_record(record: dict[str, Any], *, source_run: str) -> ReanswerInputRow:
	selection = record.get("selection") or {}
	corpus = selection.get("corpus") or {}
	metrics = selection.get("metrics") or {}
	gold_path_raw = record.get("gold_path_nodes")
	return ReanswerInputRow(
		source_run=source_run,
		dataset_name=str(record.get("dataset_name", "")),
		case_id=str(record["case_id"]),
		query=str(record["query"]),
		expected_answer=(
			None
			if record.get("expected_answer") is None
			else str(record["expected_answer"])
		),
		gold_support_nodes=_coerce_str_list(record.get("gold_support_nodes")),
		gold_start_nodes=_coerce_str_list(record.get("gold_start_nodes")),
		gold_path_nodes=(
			None if gold_path_raw is None else _coerce_str_list(gold_path_raw)
		),
		question_type=(
			None
			if record.get("question_type") is None
			else str(record["question_type"])
		),
		selector=str(record["selector"]),
		budget_label=str(
			record.get("budget_label")
			or selection.get("budget", {}).get("budget_label", "")
		),
		selected_node_ids=_coerce_str_list(corpus.get("node_ids")),
		selected_token_estimate=int(corpus.get("token_estimate") or 0),
		support_f1=(
			None if metrics.get("support_f1") is None else float(metrics["support_f1"])
		),
	)


def iter_input_rows(
	run_dirs: Sequence[Path],
	*,
	selector_filter: Sequence[str] | None = None,
	budget_label_filter: Sequence[str] | None = None,
) -> Iterator[ReanswerInputRow]:
	"""Yield filtered, deduplicated `ReanswerInputRow`s from results.jsonl files.

	When the same `(case_id, selector, budget_label)` triple appears in more
	than one input run, only the first occurrence is yielded. This lets a
	caller pass overlapping run directories (for example, several runs that
	all include the same dense baseline) without inflating the per-row count.
	"""

	selector_set = set(selector_filter) if selector_filter else None
	budget_set = set(budget_label_filter) if budget_label_filter else None
	seen: set[tuple[str, str, str]] = set()
	for run_dir in run_dirs:
		run_dir = Path(run_dir)
		chunks_dir = run_dir / "chunks"
		if not chunks_dir.is_dir():
			logger.warning("Skipping %s: no chunks/ subdir", run_dir)
			continue
		for results_path in sorted(chunks_dir.glob("*/results.jsonl")):
			with results_path.open("r", encoding="utf-8") as handle:
				for line in handle:
					line = line.strip()
					if not line:
						continue
					record = json.loads(line)
					row = _row_from_record(record, source_run=run_dir.name)
					if selector_set is not None and row.selector not in selector_set:
						continue
					if budget_set is not None and row.budget_label not in budget_set:
						continue
					key = (row.case_id, row.selector, row.budget_label)
					if key in seen:
						continue
					seen.add(key)
					yield row


def _case_from_row(row: ReanswerInputRow) -> EvaluationCase:
	return EvaluationCase(
		case_id=row.case_id,
		query=row.query,
		expected_answer=row.expected_answer,
		dataset_name=row.dataset_name or "iirc",
		gold_support_nodes=list(row.gold_support_nodes),
		gold_start_nodes=list(row.gold_start_nodes),
		gold_path_nodes=(
			None if row.gold_path_nodes is None else list(row.gold_path_nodes)
		),
		question_type=cast(Any, row.question_type),
	)


def build_answerer(
	*,
	answerer_mode: str,
	provider: str | None,
	model: str | None,
	api_key_env: str | None,
	base_url: str | None,
	cache_path: Path | None,
) -> SupportsAnswer:
	if answerer_mode == "heuristic":
		return Answerer()
	if answerer_mode != "llm_fixed":
		raise ValueError(f"Unknown answerer_mode: {answerer_mode}")
	resolved_provider = cast(AnswererProvider, provider or "copilot")
	config = LLMAnswererConfig(
		provider=resolved_provider,
		model=model,
		api_key_env=api_key_env,
		base_url=base_url,
		cache_path=cache_path,
	)
	return LLMAnswerer(config=config)


@dataclass(slots=True)
class ReanswerOutcome:
	row: ReanswerInputRow
	answer: str
	answer_em: float | None
	answer_f1: float | None
	mode: str
	model: str | None
	confidence: float
	evidence_count: int
	runtime_s: float
	prompt_tokens: int | None
	completion_tokens: int | None
	total_tokens: int | None


SYSTEM_PROMPT_FOR_BUDGET = (
	"Answer only from the supplied evidence context. Reply with the shortest "
	"exact answer span — a name, number, date, or short phrase. No full "
	"sentences, no leading articles (a/an/the), no explanations, no units "
	"unless asked. Return JSON: {\"answer\": \"...\"}"
)


def reanswer_row(
	row: ReanswerInputRow,
	*,
	graph: Any,
	answerer: SupportsAnswer,
	extractor: ExtractorLike,
) -> ReanswerOutcome:
	case = _case_from_row(row)
	# When the extractor is real-token-budget aware (FullDocumentExtractor),
	# include the system prompt + question shell in its overhead estimate so
	# the cap reflects the actual prompt size sent to the LLM.
	if isinstance(extractor, FullDocumentExtractor):
		overhead = (
			SYSTEM_PROMPT_FOR_BUDGET
			+ "\nQuestion:\n"
			+ case.query
			+ "\n\nEvidence context:\n"
		)
		subgraph = extractor.extract(
			case.query,
			graph,
			list(row.selected_node_ids),
			prompt_overhead_text=overhead,
		)
	else:
		subgraph = extractor.extract(case.query, graph, list(row.selected_node_ids))
	started = time.perf_counter()
	answer = answerer.answer(case.query, subgraph)
	runtime_s = time.perf_counter() - started
	return ReanswerOutcome(
		row=row,
		answer=answer.answer,
		answer_em=_em(answer.answer, case.expected_answer),
		answer_f1=answer_f1(answer.answer, case.expected_answer),
		mode=answer.mode,
		model=answer.model,
		confidence=answer.confidence,
		evidence_count=len(answer.evidence),
		runtime_s=getattr(answer, "runtime_s", runtime_s) or runtime_s,
		prompt_tokens=answer.prompt_tokens,
		completion_tokens=answer.completion_tokens,
		total_tokens=answer.total_tokens,
	)


def outcome_to_record(outcome: ReanswerOutcome) -> dict[str, Any]:
	row = outcome.row
	return {
		"source_run": row.source_run,
		"dataset_name": row.dataset_name,
		"case_id": row.case_id,
		"query": row.query,
		"expected_answer": row.expected_answer,
		"gold_support_nodes": list(row.gold_support_nodes),
		"selector": row.selector,
		"budget_label": row.budget_label,
		"selected_node_ids": list(row.selected_node_ids),
		"selected_token_estimate": row.selected_token_estimate,
		"selection_support_f1": row.support_f1,
		"end_to_end": {
			"mode": outcome.mode,
			"model": outcome.model,
			"answer": outcome.answer,
			"confidence": outcome.confidence,
			"evidence_count": outcome.evidence_count,
			"em": outcome.answer_em,
			"f1": outcome.answer_f1,
			"runtime_s": outcome.runtime_s,
			"prompt_tokens": outcome.prompt_tokens,
			"completion_tokens": outcome.completion_tokens,
			"total_tokens": outcome.total_tokens,
		},
	}


def summarize_outcomes(outcomes: Sequence[ReanswerOutcome]) -> dict[str, Any]:
	"""Aggregate F1/EM by (selector, budget_label)."""

	groups: dict[tuple[str, str], list[ReanswerOutcome]] = {}
	for outcome in outcomes:
		key = (outcome.row.selector, outcome.row.budget_label)
		groups.setdefault(key, []).append(outcome)

	per_group = []
	for (selector, budget_label), group in sorted(groups.items()):
		f1_values = [o.answer_f1 for o in group if o.answer_f1 is not None]
		em_values = [o.answer_em for o in group if o.answer_em is not None]
		per_group.append(
			{
				"selector": selector,
				"budget_label": budget_label,
				"num_cases": len(group),
				"avg_answer_f1": (
					sum(f1_values) / len(f1_values) if f1_values else None
				),
				"avg_answer_em": (
					sum(em_values) / len(em_values) if em_values else None
				),
			}
		)
	return {"groups": per_group, "total_outcomes": len(outcomes)}


def run_reanswer(
	*,
	input_runs: Sequence[Path],
	store_uri: str | Path,
	output_root: Path,
	exp_name: str,
	answerer_mode: str = "llm_fixed",
	answer_provider: str | None = "copilot",
	answer_model: str | None = "gpt-4.1",
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: Path | None = None,
	selector_filter: Sequence[str] | None = None,
	budget_label_filter: Sequence[str] | None = None,
	max_cases: int | None = None,
	max_snippets_per_node: int = 2,
	max_relations: int = 8,
	evidence_mode: str = "full_document",
	max_input_tokens: int = 50_000,
	store_factory: Any = None,
	progress_log_every: int = 1,
) -> Path:
	"""Drive the full reanswer pipeline, write outputs under output_root/exp_name."""

	output_dir = Path(output_root) / exp_name
	output_dir.mkdir(parents=True, exist_ok=True)
	results_path = output_dir / "results.jsonl"
	summary_path = output_dir / "summary.json"
	manifest_path = output_dir / "run_manifest.json"

	rows: list[ReanswerInputRow] = list(
		iter_input_rows(
			[Path(p) for p in input_runs],
			selector_filter=selector_filter,
			budget_label_filter=budget_label_filter,
		)
	)
	if max_cases is not None:
		rows = _limit_unique_cases(rows, max_cases=max_cases)

	if not rows:
		raise RuntimeError(
			"No input rows matched the given run dirs / selector / budget filters."
		)

	store = (
		store_factory(store_uri)
		if store_factory is not None
		else ShardedDocumentStore(store_uri)
	)
	answerer = build_answerer(
		answerer_mode=answerer_mode,
		provider=answer_provider,
		model=answer_model,
		api_key_env=answer_api_key_env,
		base_url=answer_base_url,
		cache_path=answer_cache_path,
	)
	extractor: ExtractorLike
	if evidence_mode == "full_document":
		extractor = FullDocumentExtractor(max_input_tokens=max_input_tokens)
	elif evidence_mode == "lexical_snippets":
		extractor = SubgraphExtractor(
			max_snippets_per_node=max_snippets_per_node,
			max_relations=max_relations,
		)
	else:
		raise ValueError(
			f"Unknown evidence_mode={evidence_mode!r}; expected 'full_document' or 'lexical_snippets'."
		)

	outcomes: list[ReanswerOutcome] = []
	with results_path.open("w", encoding="utf-8") as handle:
		for index, row in enumerate(rows):
			try:
				outcome = reanswer_row(
					row, graph=store, answerer=answerer, extractor=extractor
				)
			except Exception as exc:  # noqa: BLE001 — keep the run alive
				logger.warning(
					"Reanswer FAILED on case=%s selector=%s budget=%s: %s",
					row.case_id,
					row.selector,
					row.budget_label,
					exc,
				)
				outcome = ReanswerOutcome(
					row=row,
					answer="",
					answer_em=0.0,
					answer_f1=0.0,
					mode="error",
					model=None,
					confidence=0.0,
					evidence_count=0,
					runtime_s=0.0,
					prompt_tokens=None,
					completion_tokens=None,
					total_tokens=None,
				)
			outcomes.append(outcome)
			handle.write(
				json.dumps(outcome_to_record(outcome), ensure_ascii=False) + "\n"
			)
			handle.flush()
			if progress_log_every and (index + 1) % progress_log_every == 0:
				logger.info(
					"Reanswered %s/%s (case=%s selector=%s budget=%s f1=%s)",
					index + 1,
					len(rows),
					row.case_id,
					row.selector,
					row.budget_label,
					outcome.answer_f1,
				)

	summary = summarize_outcomes(outcomes)
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	manifest_path.write_text(
		json.dumps(
			{
				"exp_name": exp_name,
				"input_runs": [str(p) for p in input_runs],
				"store_uri": str(store_uri),
				"answerer_mode": answerer_mode,
				"answer_provider": answer_provider,
				"answer_model": answer_model,
				"selector_filter": list(selector_filter) if selector_filter else None,
				"budget_label_filter": (
					list(budget_label_filter) if budget_label_filter else None
				),
				"max_cases": max_cases,
				"num_outcomes": len(outcomes),
			},
			indent=2,
		),
		encoding="utf-8",
	)
	logger.info(
		"reanswer complete: %s outcomes written to %s", len(outcomes), results_path
	)
	return output_dir


def _limit_unique_cases(
	rows: Iterable[ReanswerInputRow], *, max_cases: int
) -> list[ReanswerInputRow]:
	"""Keep all rows whose case_id is among the first `max_cases` distinct case_ids encountered."""

	seen_cases: list[str] = []
	seen_set: set[str] = set()
	kept: list[ReanswerInputRow] = []
	for row in rows:
		if row.case_id not in seen_set:
			if len(seen_set) >= max_cases:
				continue
			seen_set.add(row.case_id)
			seen_cases.append(row.case_id)
		kept.append(row)
	return kept


__all__ = [
	"ReanswerInputRow",
	"ReanswerOutcome",
	"build_answerer",
	"iter_input_rows",
	"outcome_to_record",
	"reanswer_row",
	"run_reanswer",
	"summarize_outcomes",
]
