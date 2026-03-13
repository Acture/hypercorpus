from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from webwalker.eval import ExperimentSummary, SelectorBudgetSummary
from webwalker.selector import parse_selector_spec


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
    "avg_support_precision",
    "avg_support_recall",
    "avg_path_hit",
    "rank_within_budget",
    "delta_support_f1_vs_control",
    "delta_support_precision_vs_control",
    "delta_support_recall_vs_control",
]


@dataclass(frozen=True, slots=True)
class ReportBundlePaths:
    summary_rows_path: Path
    study_comparison_rows_path: Path


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


def export_summary_report_from_file(summary_path: str | Path, output_path: str | Path | None = None) -> Path:
    summary_file = Path(summary_path)
    summary = load_experiment_summary(summary_file)
    destination = Path(output_path) if output_path is not None else summary_file.with_name("summary_rows.csv")
    return export_summary_report(summary, destination)


def export_report_bundle_from_file(
    summary_path: str | Path,
    *,
    summary_rows_output_path: str | Path | None = None,
    study_comparison_output_path: str | Path | None = None,
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
    study_preset, control_selector_name = _report_context_from_manifest(summary_file)
    return ReportBundlePaths(
        summary_rows_path=export_summary_report(summary, summary_rows_path),
        study_comparison_rows_path=export_study_comparison_report(
            summary,
            study_comparison_path,
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
            (selector_budget for selector_budget in selector_budgets if selector_budget.name == control_selector_name),
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
                    "avg_support_precision": selector_budget.avg_support_precision,
                    "avg_support_recall": selector_budget.avg_support_recall,
                    "avg_path_hit": selector_budget.avg_path_hit,
                    "rank_within_budget": rank,
                    "delta_support_f1_vs_control": _delta(selector_budget.avg_support_f1, control.avg_support_f1 if control is not None else None),
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


def _report_context_from_manifest(summary_file: Path) -> tuple[str | None, str | None]:
    manifest_path = summary_file.with_name("run_manifest.json")
    if not manifest_path.exists():
        return None, None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    study_preset = payload.get("study_preset")
    control_selector_name = payload.get("control_selector_name")
    if control_selector_name is None and study_preset is not None:
        try:
            from webwalker.experiments import resolve_study_preset

            control_selector_name = resolve_study_preset(str(study_preset)).control_selector_name
        except Exception:
            control_selector_name = None
    return (
        str(study_preset) if study_preset is not None else None,
        str(control_selector_name) if control_selector_name is not None else None,
    )


def _selector_budget_report_row(dataset_name: str, selector_budget: SelectorBudgetSummary) -> dict[str, Any]:
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


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline
