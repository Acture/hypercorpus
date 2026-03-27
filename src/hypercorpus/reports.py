from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from hypercorpus.eval import ExperimentSummary, SelectorBudgetSummary
from hypercorpus.selector import parse_selector_spec


SUMMARY_REPORT_FIELDNAMES = [
	"dataset",
	"selector",
	"selector_family",
	"selector_provider",
	"selector_model",
	"seed_strategy",
	"seed_top_k",
	"hop_budget",
	"baseline",
	"search_structure",
	"edge_scorer",
	"lookahead_depth",
	"profile_name",
	"budget_fill_mode",
	"budget_fill_pool_k",
	"budget_fill_score_floor",
	"budget_fill_relative_drop_ratio",
	"budget_mode",
	"budget_value",
	"budget_label",
	"token_budget_tokens",
	"token_budget_ratio",
	"num_cases",
	"support_recall",
	"support_precision",
	"support_f1",
	"support_f1_zero_on_empty",
	"support_set_em",
	"budget_utilization",
	"budget_adherence",
	"empty_selection_rate",
	"selected_nodes",
	"selected_token_estimate",
	"selection_runtime_s",
	"selector_total_tokens",
	"selector_runtime_s",
	"selector_llm_calls",
	"answer_em",
	"answer_f1",
]

STUDY_COMPARISON_FIELDNAMES = [
	"study_preset",
	"dataset_name",
	"selector_name",
	"budget_label",
	"control_selector_name",
	"avg_support_f1",
	"avg_support_set_em",
	"avg_support_precision",
	"avg_support_recall",
	"avg_path_hit",
	"rank_within_budget",
	"delta_support_f1_vs_control",
	"delta_support_precision_vs_control",
	"delta_support_recall_vs_control",
]

SUBSET_COMPARISON_FIELDNAMES = [
	"study_preset",
	"dataset_name",
	"subset_label",
	"selector_name",
	"budget_label",
	"control_selector_name",
	"num_cases",
	"avg_support_f1_zero_on_empty",
	"avg_support_set_em",
	"avg_support_precision",
	"avg_support_recall",
	"avg_path_hit",
	"delta_support_f1_vs_dense_control",
	"delta_support_precision_vs_dense_control",
	"delta_support_recall_vs_dense_control",
	"delta_path_hit_vs_dense_control",
	"avg_stop_rate",
	"avg_explicit_stop_rate",
	"avg_budget_pacing_stop_rate",
	"avg_fork_rate",
	"avg_backtrack_rate",
	"avg_controller_calls",
	"avg_prefiltered_candidates",
]


@dataclass(frozen=True, slots=True)
class ReportBundlePaths:
	summary_rows_path: Path
	study_comparison_rows_path: Path
	subset_comparison_rows_path: Path


def load_experiment_summary(summary_path: str | Path) -> ExperimentSummary:
	payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
	rows = [SelectorBudgetSummary(**record) for record in payload["selector_budgets"]]
	return ExperimentSummary(
		dataset_name=payload["dataset_name"],
		total_cases=payload["total_cases"],
		selector_budgets=rows,
	)


def summary_report_rows(summary: ExperimentSummary) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for selector_budget in summary.selector_budgets:
		row = _selector_budget_report_row(summary.dataset_name, selector_budget)
		rows.append(row)
	return rows


def export_summary_report(summary: ExperimentSummary, output_path: str | Path) -> Path:
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	rows = summary_report_rows(summary)
	with output.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=SUMMARY_REPORT_FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)
	return output


def export_summary_report_from_file(
	summary_path: str | Path, output_path: str | Path | None = None
) -> Path:
	summary_file = Path(summary_path)
	summary = load_experiment_summary(summary_file)
	destination = (
		Path(output_path)
		if output_path is not None
		else summary_file.with_name("summary_rows.csv")
	)
	return export_summary_report(summary, destination)


def export_report_bundle_from_file(
	summary_path: str | Path,
	*,
	summary_rows_output_path: str | Path | None = None,
	study_comparison_output_path: str | Path | None = None,
	subset_comparison_output_path: str | Path | None = None,
) -> ReportBundlePaths:
	summary_file = Path(summary_path)
	summary = load_experiment_summary(summary_file)
	summary_rows_path = (
		Path(summary_rows_output_path)
		if summary_rows_output_path is not None
		else summary_file.with_name("summary_rows.csv")
	)
	study_comparison_path = (
		Path(study_comparison_output_path)
		if study_comparison_output_path is not None
		else summary_rows_path.parent / "study_comparison_rows.csv"
	)
	subset_comparison_path = (
		Path(subset_comparison_output_path)
		if subset_comparison_output_path is not None
		else summary_rows_path.parent / "subset_comparison_rows.csv"
	)
	study_preset, control_selector_name = _report_context_from_manifest(summary_file)
	results_path = summary_file.with_name("results.jsonl")
	return ReportBundlePaths(
		summary_rows_path=export_summary_report(summary, summary_rows_path),
		study_comparison_rows_path=export_study_comparison_report(
			summary,
			study_comparison_path,
			study_preset=study_preset,
			control_selector_name=control_selector_name,
		),
		subset_comparison_rows_path=export_subset_comparison_report(
			results_path if results_path.exists() else [],
			subset_comparison_path,
			study_preset=study_preset,
			control_selector_name=control_selector_name,
		),
	)


def study_comparison_rows(
	summary: ExperimentSummary,
	*,
	study_preset: str | None = None,
	control_selector_name: str | None = None,
) -> list[dict[str, Any]]:
	grouped: dict[str, list[SelectorBudgetSummary]] = {}
	ordered_budget_labels: list[str] = []
	for selector_budget in summary.selector_budgets:
		budget_label = selector_budget.budget_label
		if budget_label not in grouped:
			grouped[budget_label] = []
			ordered_budget_labels.append(budget_label)
		grouped[budget_label].append(selector_budget)

	rows: list[dict[str, Any]] = []
	for budget_label in ordered_budget_labels:
		selector_budgets = grouped[budget_label]
		ranked = sorted(
			selector_budgets,
			key=lambda selector_budget: (
				selector_budget.avg_support_f1 is None,
				-(selector_budget.avg_support_f1 or float("-inf")),
				selector_budget.name,
			),
		)
		control = next(
			(
				selector_budget
				for selector_budget in selector_budgets
				if selector_budget.name == control_selector_name
			),
			None,
		)
		for rank, selector_budget in enumerate(ranked, start=1):
			rows.append(
				{
					"study_preset": study_preset,
					"dataset_name": summary.dataset_name,
					"selector_name": selector_budget.name,
					"budget_label": budget_label,
					"control_selector_name": control_selector_name,
					"avg_support_f1": selector_budget.avg_support_f1,
					"avg_support_set_em": selector_budget.avg_support_set_em,
					"avg_support_precision": selector_budget.avg_support_precision,
					"avg_support_recall": selector_budget.avg_support_recall,
					"avg_path_hit": selector_budget.avg_path_hit,
					"rank_within_budget": rank,
					"delta_support_f1_vs_control": _delta(
						selector_budget.avg_support_f1,
						control.avg_support_f1 if control is not None else None,
					),
					"delta_support_precision_vs_control": _delta(
						selector_budget.avg_support_precision,
						control.avg_support_precision if control is not None else None,
					),
					"delta_support_recall_vs_control": _delta(
						selector_budget.avg_support_recall,
						control.avg_support_recall if control is not None else None,
					),
				}
			)
	return rows


def export_study_comparison_report(
	summary: ExperimentSummary,
	output_path: str | Path,
	*,
	study_preset: str | None = None,
	control_selector_name: str | None = None,
) -> Path:
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	rows = study_comparison_rows(
		summary,
		study_preset=study_preset,
		control_selector_name=control_selector_name,
	)
	with output.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=STUDY_COMPARISON_FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)
	return output


def load_result_records(
	results_path: str | Path | Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
	if isinstance(results_path, Sequence) and not isinstance(results_path, (str, Path)):
		return [dict(record) for record in results_path]
	path = Path(results_path)
	if not path.exists():
		return []
	records: list[dict[str, Any]] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		if not line.strip():
			continue
		records.append(json.loads(line))
	return records


def export_subset_comparison_report(
	results_path: str | Path | Sequence[dict[str, Any]],
	output_path: str | Path,
	*,
	study_preset: str | None = None,
	control_selector_name: str | None = None,
) -> Path:
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	rows = subset_comparison_rows(
		load_result_records(results_path),
		study_preset=study_preset,
		control_selector_name=control_selector_name,
	)
	with output.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=SUBSET_COMPARISON_FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)
	return output


def subset_comparison_rows(
	records: Sequence[dict[str, Any]],
	*,
	study_preset: str | None = None,
	control_selector_name: str | None = None,
) -> list[dict[str, Any]]:
	if not records:
		return []
	dataset_name = str(records[0].get("dataset_name", ""))
	selector_budget_keys: list[tuple[str, str]] = []
	seen_keys: set[tuple[str, str]] = set()
	for record in records:
		key = (str(record["selector"]), str(record["budget_label"]))
		if key in seen_keys:
			continue
		seen_keys.add(key)
		selector_budget_keys.append(key)

	rows: list[dict[str, Any]] = []
	for subset_label, predicate in _subset_predicates():
		subset_records = [record for record in records if predicate(record)]
		if not subset_records:
			continue
		grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
		for record in subset_records:
			grouped.setdefault(
				(str(record["selector"]), str(record["budget_label"])), []
			).append(record)
		_ctrl = control_selector_name or ""
		control_by_budget = {
			budget_label: _aggregate_subset_rows(
				grouped.get((_ctrl, budget_label)) or []
			)
			for _selector_name, budget_label in selector_budget_keys
		}
		emitted: set[tuple[str, str]] = set()
		for selector_name, budget_label in selector_budget_keys:
			if (selector_name, budget_label) in emitted:
				continue
			emitted.add((selector_name, budget_label))
			aggregate = _aggregate_subset_rows(
				grouped.get((selector_name, budget_label), [])
			)
			control = control_by_budget.get(budget_label)
			rows.append(
				{
					"study_preset": study_preset,
					"dataset_name": dataset_name,
					"subset_label": subset_label,
					"selector_name": selector_name,
					"budget_label": budget_label,
					"control_selector_name": control_selector_name,
					"num_cases": aggregate["num_cases"],
					"avg_support_f1_zero_on_empty": aggregate[
						"avg_support_f1_zero_on_empty"
					],
					"avg_support_set_em": aggregate["avg_support_set_em"],
					"avg_support_precision": aggregate["avg_support_precision"],
					"avg_support_recall": aggregate["avg_support_recall"],
					"avg_path_hit": aggregate["avg_path_hit"],
					"delta_support_f1_vs_dense_control": _subset_delta(
						selector_name,
						control_selector_name,
						aggregate["avg_support_f1_zero_on_empty"],
						None
						if control is None
						else control["avg_support_f1_zero_on_empty"],
					),
					"delta_support_precision_vs_dense_control": _subset_delta(
						selector_name,
						control_selector_name,
						aggregate["avg_support_precision"],
						None if control is None else control["avg_support_precision"],
					),
					"delta_support_recall_vs_dense_control": _subset_delta(
						selector_name,
						control_selector_name,
						aggregate["avg_support_recall"],
						None if control is None else control["avg_support_recall"],
					),
					"delta_path_hit_vs_dense_control": _subset_delta(
						selector_name,
						control_selector_name,
						aggregate["avg_path_hit"],
						None if control is None else control["avg_path_hit"],
					),
					"avg_stop_rate": aggregate["avg_stop_rate"],
					"avg_explicit_stop_rate": aggregate[
						"avg_explicit_stop_rate"
					],
					"avg_budget_pacing_stop_rate": aggregate[
						"avg_budget_pacing_stop_rate"
					],
					"avg_fork_rate": aggregate["avg_fork_rate"],
					"avg_backtrack_rate": aggregate["avg_backtrack_rate"],
					"avg_controller_calls": aggregate["avg_controller_calls"],
					"avg_prefiltered_candidates": aggregate[
						"avg_prefiltered_candidates"
					],
				}
			)
	return rows


def _report_context_from_manifest(summary_file: Path) -> tuple[str | None, str | None]:
	manifest_path = summary_file.with_name("run_manifest.json")
	if not manifest_path.exists():
		return None, None
	payload = json.loads(manifest_path.read_text(encoding="utf-8"))
	study_preset = payload.get("study_preset")
	control_selector_name = payload.get("control_selector_name")
	if control_selector_name is None and study_preset is not None:
		try:
			from hypercorpus.experiments import resolve_study_preset

			control_selector_name = resolve_study_preset(
				str(study_preset)
			).control_selector_name
		except Exception:
			control_selector_name = None
	return (
		str(study_preset) if study_preset is not None else None,
		str(control_selector_name) if control_selector_name is not None else None,
	)


def _selector_budget_report_row(
	dataset_name: str, selector_budget: SelectorBudgetSummary
) -> dict[str, Any]:
	spec = _selector_spec_details(selector_budget.name)
	row = {
		"dataset": dataset_name,
		"selector": selector_budget.name,
		"selector_provider": selector_budget.selector_provider,
		"selector_model": selector_budget.selector_model,
		"budget_mode": selector_budget.budget_mode,
		"budget_value": selector_budget.budget_value,
		"budget_label": selector_budget.budget_label,
		"token_budget_tokens": selector_budget.token_budget_tokens,
		"token_budget_ratio": selector_budget.token_budget_ratio,
		"num_cases": selector_budget.num_cases,
		"support_recall": selector_budget.avg_support_recall,
		"support_precision": selector_budget.avg_support_precision,
		"support_f1": selector_budget.avg_support_f1,
		"support_f1_zero_on_empty": selector_budget.avg_support_f1_zero_on_empty,
		"support_set_em": selector_budget.avg_support_set_em,
		"budget_utilization": selector_budget.avg_budget_utilization,
		"budget_adherence": selector_budget.avg_budget_adherence,
		"empty_selection_rate": selector_budget.avg_empty_selection_rate,
		"selected_nodes": selector_budget.avg_selected_nodes,
		"selected_token_estimate": selector_budget.avg_selected_token_estimate,
		"selection_runtime_s": selector_budget.avg_selection_runtime_s,
		"selector_total_tokens": selector_budget.avg_selector_total_tokens,
		"selector_runtime_s": selector_budget.avg_selector_runtime_s,
		"selector_llm_calls": selector_budget.avg_selector_llm_calls,
		"answer_em": selector_budget.avg_answer_em,
		"answer_f1": selector_budget.avg_answer_f1,
	}
	row.update(spec)
	return row


def _selector_spec_details(selector_name: str) -> dict[str, Any]:
	try:
		spec = parse_selector_spec(selector_name)
	except ValueError:
		return {
			"selector_family": "unknown",
			"seed_strategy": None,
			"seed_top_k": None,
			"hop_budget": None,
			"baseline": None,
			"search_structure": None,
			"edge_scorer": None,
			"lookahead_depth": None,
			"profile_name": None,
			"budget_fill_mode": None,
			"budget_fill_pool_k": None,
			"budget_fill_score_floor": None,
			"budget_fill_relative_drop_ratio": None,
		}
	spec_dict = asdict(spec)
	return {
		"selector_family": spec_dict["family"],
		"seed_strategy": spec_dict["seed_strategy"],
		"seed_top_k": spec_dict["seed_top_k"],
		"hop_budget": spec_dict["hop_budget"],
		"baseline": spec_dict["baseline"],
		"search_structure": spec_dict["search_structure"],
		"edge_scorer": spec_dict["edge_scorer"],
		"lookahead_depth": spec_dict["lookahead_depth"],
		"profile_name": spec_dict["profile_name"],
		"budget_fill_mode": spec_dict["budget_fill_mode"],
		"budget_fill_pool_k": spec_dict["budget_fill_pool_k"],
		"budget_fill_score_floor": spec_dict["budget_fill_score_floor"],
		"budget_fill_relative_drop_ratio": spec_dict["budget_fill_relative_drop_ratio"],
	}


def _subset_predicates() -> list[tuple[str, Any]]:
	return [
		("all_cases", lambda _record: True),
		("bridge", lambda record: _record_question_type(record) == "bridge"),
		("comparison", lambda record: _record_question_type(record) == "comparison"),
		(
			"support_count_ge_4",
			lambda record: len(record.get("gold_support_nodes", [])) >= 4,
		),
		("has_gold_path", lambda record: bool(record.get("gold_path_nodes"))),
		(
			"start_not_equal_support",
			lambda record: set(record.get("gold_start_nodes", []))
			!= set(record.get("gold_support_nodes", [])),
		),
	]


def _aggregate_subset_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
	if not rows:
		return {
			"num_cases": 0,
			"avg_support_f1_zero_on_empty": None,
			"avg_support_set_em": None,
			"avg_support_precision": None,
			"avg_support_recall": None,
			"avg_path_hit": None,
			"avg_stop_rate": 0.0,
			"avg_explicit_stop_rate": 0.0,
			"avg_budget_pacing_stop_rate": 0.0,
			"avg_fork_rate": 0.0,
			"avg_backtrack_rate": 0.0,
			"avg_controller_calls": 0.0,
			"avg_prefiltered_candidates": 0.0,
		}
	support_f1 = [
		float(record["selection"]["metrics"]["support_f1_zero_on_empty"])
		for record in rows
		if record["selection"]["metrics"].get("support_f1_zero_on_empty") is not None
	]
	support_set_em = [
		float(record["selection"]["metrics"]["support_set_em"])
		for record in rows
		if record["selection"]["metrics"].get("support_set_em") is not None
	]
	support_precision = [
		float(record["selection"]["metrics"]["support_precision"])
		for record in rows
		if record["selection"]["metrics"].get("support_precision") is not None
	]
	support_recall = [
		float(record["selection"]["metrics"]["support_recall"])
		for record in rows
		if record["selection"]["metrics"].get("support_recall") is not None
	]
	path_hit = [
		1.0 if record["selection"]["metrics"]["path_hit"] else 0.0
		for record in rows
		if record["selection"]["metrics"].get("path_hit") is not None
	]
	stop_rates: list[float] = []
	explicit_stop_rates: list[float] = []
	budget_pacing_stop_rates: list[float] = []
	fork_rates: list[float] = []
	backtrack_rates: list[float] = []
	controller_calls: list[float] = []
	prefiltered_candidates: list[float] = []
	for record in rows:
		usage = dict(record["selection"].get("selector_usage") or {})
		calls = float(usage.get("controller_calls", 0))
		controller_calls.append(calls)
		if calls > 0:
			stop_rates.append(float(usage.get("controller_stop_actions", 0)) / calls)
			explicit_stop_rates.append(
				float(usage.get("controller_explicit_stop_actions", 0)) / calls
			)
			budget_pacing_stop_rates.append(
				float(usage.get("controller_budget_pacing_stop_actions", 0)) / calls
			)
			fork_rates.append(float(usage.get("controller_fork_actions", 0)) / calls)
			backtrack_rates.append(
				float(usage.get("controller_backtrack_actions", 0)) / calls
			)
			prefiltered_candidates.append(
				float(usage.get("controller_prefiltered_candidates", 0)) / calls
			)
		else:
			stop_rates.append(0.0)
			explicit_stop_rates.append(0.0)
			budget_pacing_stop_rates.append(0.0)
			fork_rates.append(0.0)
			backtrack_rates.append(0.0)
			prefiltered_candidates.append(0.0)
	return {
		"num_cases": len(rows),
		"avg_support_f1_zero_on_empty": _mean_or_none(support_f1),
		"avg_support_set_em": _mean_or_none(support_set_em),
		"avg_support_precision": _mean_or_none(support_precision),
		"avg_support_recall": _mean_or_none(support_recall),
		"avg_path_hit": _mean_or_none(path_hit),
		"avg_stop_rate": _mean_or_zero(stop_rates),
		"avg_explicit_stop_rate": _mean_or_zero(explicit_stop_rates),
		"avg_budget_pacing_stop_rate": _mean_or_zero(budget_pacing_stop_rates),
		"avg_fork_rate": _mean_or_zero(fork_rates),
		"avg_backtrack_rate": _mean_or_zero(backtrack_rates),
		"avg_controller_calls": _mean_or_zero(controller_calls),
		"avg_prefiltered_candidates": _mean_or_zero(prefiltered_candidates),
	}


def _mean_or_none(values: Sequence[float]) -> float | None:
	if not values:
		return None
	return sum(values) / len(values)


def _mean_or_zero(values: Sequence[float]) -> float:
	value = _mean_or_none(values)
	return value if value is not None else 0.0


def _record_question_type(record: dict[str, Any]) -> str:
	raw = record.get("question_type")
	if isinstance(raw, str):
		normalized = raw.strip().lower()
		if normalized in {"bridge", "comparison", "unknown"}:
			return normalized
	if record.get("gold_path_nodes"):
		return "bridge"
	if len(set(record.get("gold_support_nodes", []))) >= 2:
		return "comparison"
	return "unknown"


def _subset_delta(
	selector_name: str,
	control_selector_name: str | None,
	value: float | None,
	baseline: float | None,
) -> float | None:
	if selector_name == control_selector_name:
		return 0.0
	return _delta(value, baseline)


def _delta(value: float | None, baseline: float | None) -> float | None:
	if value is None or baseline is None:
		return None
	return value - baseline
