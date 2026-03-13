from __future__ import annotations

import csv
import json
from dataclasses import asdict
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
    }
