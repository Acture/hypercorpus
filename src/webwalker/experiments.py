from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from webwalker.datasets.twowiki import load_2wiki_graph, load_2wiki_questions
from webwalker.eval import (
    DEFAULT_BUDGET_RATIOS,
    CaseEvaluation,
    Evaluator,
    ExperimentSummary,
    SelectionBudget,
    available_selector_names,
    select_selectors,
)


def run_2wiki_experiment(
    *,
    questions_path: str | Path,
    graph_records_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    selector_names: Sequence[str] | None = None,
    budget_ratios: Sequence[float] | None = None,
    seed: int = 0,
    max_steps: int = 3,
    top_k: int = 2,
    with_e2e: bool = True,
    export_graphrag_inputs: bool = True,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
    graph = load_2wiki_graph(graph_records_path)
    cases = load_2wiki_questions(questions_path, limit=limit)
    selectors = select_selectors(
        selector_names,
        seed=seed,
        include_diagnostics=selector_names is not None,
    )
    ratios = list(budget_ratios or DEFAULT_BUDGET_RATIOS)
    evaluators = [
        Evaluator(
            selectors,
            budget=SelectionBudget(
                max_steps=max_steps,
                top_k=top_k,
                token_budget_ratio=ratio,
            ),
            with_e2e=with_e2e,
        )
        for ratio in ratios
    ]

    evaluations: list[CaseEvaluation] = []
    for case in cases:
        selections = []
        for evaluator in evaluators:
            selections.extend(evaluator.evaluate_case(graph, case).selections)
        evaluations.append(CaseEvaluation(case=case, selections=selections))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if export_graphrag_inputs:
        for evaluation in evaluations:
            for selection in evaluation.selections:
                selection.graphrag_input_path = _write_graphrag_input(
                    graph=graph,
                    case_id=evaluation.case.case_id,
                    selection=selection,
                    output_dir=output_path,
                )

    summary = Evaluator().summarize(evaluations)
    results_path = output_path / "results.jsonl"
    summary_path = output_path / "summary.json"

    with results_path.open("w", encoding="utf-8") as handle:
        for evaluation in evaluations:
            for selection in evaluation.selections:
                record = {
                    "dataset_name": evaluation.case.dataset_name,
                    "case_id": evaluation.case.case_id,
                    "query": evaluation.case.query,
                    "expected_answer": evaluation.case.expected_answer,
                    "gold_support_nodes": evaluation.case.gold_support_nodes,
                    "gold_start_nodes": evaluation.case.gold_start_nodes,
                    "gold_path_nodes": evaluation.case.gold_path_nodes,
                    "selector": selection.selector_name,
                    "token_budget_ratio": selection.budget.token_budget_ratio,
                    "selection": {
                        "budget": asdict(selection.budget),
                        "corpus": asdict(selection.corpus),
                        "metrics": asdict(selection.metrics),
                        "trace": [asdict(step) for step in selection.trace],
                        "stop_reason": selection.stop_reason,
                        "graphrag_input_path": selection.graphrag_input_path,
                    },
                    "end_to_end": asdict(selection.end_to_end) if selection.end_to_end is not None else None,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return evaluations, summary


def parse_selector_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        return None
    return names


def parse_budget_ratios(value: str | None) -> list[float] | None:
    if value is None:
        return None
    ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not ratios:
        return None
    for ratio in ratios:
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"Budget ratio must be in (0, 1], got {ratio}")
    return ratios


def selector_choices_help(*, include_diagnostics: bool = True) -> str:
    return ",".join(available_selector_names(include_diagnostics=include_diagnostics))


def budget_ratio_choices_help() -> str:
    return ",".join(f"{ratio:.2f}" for ratio in DEFAULT_BUDGET_RATIOS)


def _write_graphrag_input(
    *,
    graph,
    case_id: str,
    selection,
    output_dir: Path,
) -> str:
    ratio_slug = _budget_ratio_slug(selection.budget.token_budget_ratio)
    export_path = (
        output_dir
        / "graphrag_inputs"
        / selection.selector_name
        / f"budget-{ratio_slug}"
        / f"{case_id}.csv"
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with export_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "text", "url"])
        writer.writeheader()
        for node_id in selection.corpus.node_ids:
            document = graph.get_document(node_id)
            if document is None:
                continue
            url = str(graph.node_attr.get(node_id, {}).get("url", ""))
            writer.writerow(
                {
                    "id": document.node_id,
                    "title": document.title,
                    "text": document.text,
                    "url": url,
                }
            )

    return str(export_path.relative_to(output_dir))


def _budget_ratio_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")
