from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from webwalker.answering import Answerer, LLMAnswerer, LLMAnswererConfig, SupportsAnswer
from webwalker.datasets.common import DatasetAdapter
from webwalker.datasets.docs import DocumentationAdapter
from webwalker.datasets.iirc import IIRCAdapter
from webwalker.datasets.twowiki import TwoWikiAdapter
from webwalker.datasets.twowiki_store import ShardedLinkContextStore
from webwalker.eval import (
    DEFAULT_BUDGET_RATIOS,
    DEFAULT_TOKEN_BUDGETS,
    CaseEvaluation,
    Evaluator,
    ExperimentSummary,
    SelectionBudget,
    SelectorBudgetSummary,
    available_selector_names,
    select_selectors,
    summarize_evaluations,
)
from webwalker.logging import create_progress, should_render_progress

logger = logging.getLogger(__name__)


def run_dataset_experiment(
    *,
    adapter: DatasetAdapter,
    questions_path: str | Path,
    graph_source: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    selector_names: Sequence[str] | None = None,
    token_budgets: Sequence[int] | None = None,
    budget_ratios: Sequence[float] | None = None,
    seed: int = 0,
    max_steps: int = 3,
    top_k: int = 2,
    with_e2e: bool = False,
    answerer_mode: str = "heuristic",
    answer_model: str = "gpt-4.1-mini",
    answer_api_key_env: str = "OPENAI_API_KEY",
    answer_base_url: str | None = None,
    answer_cache_path: str | Path | None = None,
    export_graphrag_inputs: bool = True,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
    dataset_label = getattr(adapter, "dataset_name", "dataset")
    logger.info("Loading %s graph from %s", dataset_label, graph_source)
    graph = adapter.load_graph(graph_source)
    logger.info("Loading %s questions from %s", dataset_label, questions_path)
    cases = adapter.load_cases(questions_path, limit=limit)
    selectors = select_selectors(
        selector_names,
        seed=seed,
        include_diagnostics=selector_names is not None,
    )
    budgets = _resolve_budgets(
        token_budgets=token_budgets,
        budget_ratios=budget_ratios,
        max_steps=max_steps,
        top_k=top_k,
    )
    evaluators = _build_evaluators(
        selectors=selectors,
        budgets=budgets,
        with_e2e=with_e2e,
        answerer_mode=answerer_mode,
        answer_model=answer_model,
        answer_api_key_env=answer_api_key_env,
        answer_base_url=answer_base_url,
        answer_cache_path=answer_cache_path,
    )
    logger.info(
        "Running %s experiment (cases=%s, selectors=%s, budgets=%s, with_e2e=%s, answerer=%s, export_graphrag_inputs=%s)",
        dataset_label,
        len(cases),
        [selector.name for selector in selectors],
        [budget.budget_label for budget in budgets],
        with_e2e,
        answerer_mode,
        export_graphrag_inputs,
    )

    evaluations = _evaluate_cases(
        graph=graph,
        cases=cases,
        evaluators=evaluators,
        description=f"evaluate {dataset_label} cases",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if export_graphrag_inputs:
        _export_graphrag_inputs(
            graph=graph,
            evaluations=evaluations,
            output_dir=output_path,
            description=f"export {dataset_label} graphrag inputs",
        )

    summary = summarize_evaluations(
        evaluations,
        dataset_name=cases[0].dataset_name if cases else dataset_label,
    )
    _write_result_files(chunk_dir=output_path, evaluations=evaluations, summary=summary)
    logger.info("Completed %s experiment; results written to %s", dataset_label, output_path)

    return evaluations, summary


def run_2wiki_experiment(
    *,
    questions_path: str | Path,
    graph_records_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    selector_names: Sequence[str] | None = None,
    token_budgets: Sequence[int] | None = None,
    budget_ratios: Sequence[float] | None = None,
    seed: int = 0,
    max_steps: int = 3,
    top_k: int = 2,
    with_e2e: bool = False,
    answerer_mode: str = "heuristic",
    answer_model: str = "gpt-4.1-mini",
    answer_api_key_env: str = "OPENAI_API_KEY",
    answer_base_url: str | None = None,
    answer_cache_path: str | Path | None = None,
    export_graphrag_inputs: bool = True,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
    return run_dataset_experiment(
        adapter=TwoWikiAdapter(),
        questions_path=questions_path,
        graph_source=graph_records_path,
        output_dir=output_dir,
        limit=limit,
        selector_names=selector_names,
        token_budgets=token_budgets,
        budget_ratios=budget_ratios,
        seed=seed,
        max_steps=max_steps,
        top_k=top_k,
        with_e2e=with_e2e,
        answerer_mode=answerer_mode,
        answer_model=answer_model,
        answer_api_key_env=answer_api_key_env,
        answer_base_url=answer_base_url,
        answer_cache_path=answer_cache_path,
        export_graphrag_inputs=export_graphrag_inputs,
    )


def run_iirc_experiment(
    *,
    questions_path: str | Path,
    graph_records_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    selector_names: Sequence[str] | None = None,
    token_budgets: Sequence[int] | None = None,
    budget_ratios: Sequence[float] | None = None,
    seed: int = 0,
    max_steps: int = 3,
    top_k: int = 2,
    with_e2e: bool = False,
    answerer_mode: str = "heuristic",
    answer_model: str = "gpt-4.1-mini",
    answer_api_key_env: str = "OPENAI_API_KEY",
    answer_base_url: str | None = None,
    answer_cache_path: str | Path | None = None,
    export_graphrag_inputs: bool = True,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
    return run_dataset_experiment(
        adapter=IIRCAdapter(),
        questions_path=questions_path,
        graph_source=graph_records_path,
        output_dir=output_dir,
        limit=limit,
        selector_names=selector_names,
        token_budgets=token_budgets,
        budget_ratios=budget_ratios,
        seed=seed,
        max_steps=max_steps,
        top_k=top_k,
        with_e2e=with_e2e,
        answerer_mode=answerer_mode,
        answer_model=answer_model,
        answer_api_key_env=answer_api_key_env,
        answer_base_url=answer_base_url,
        answer_cache_path=answer_cache_path,
        export_graphrag_inputs=export_graphrag_inputs,
    )


def run_docs_experiment(
    *,
    questions_path: str | Path,
    docs_source: str | Path,
    output_dir: str | Path,
    dataset_name: str = "docs",
    limit: int | None = None,
    selector_names: Sequence[str] | None = None,
    token_budgets: Sequence[int] | None = None,
    budget_ratios: Sequence[float] | None = None,
    seed: int = 0,
    max_steps: int = 3,
    top_k: int = 2,
    with_e2e: bool = False,
    answerer_mode: str = "heuristic",
    answer_model: str = "gpt-4.1-mini",
    answer_api_key_env: str = "OPENAI_API_KEY",
    answer_base_url: str | None = None,
    answer_cache_path: str | Path | None = None,
    export_graphrag_inputs: bool = True,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
    return run_dataset_experiment(
        adapter=DocumentationAdapter(dataset_name=dataset_name),
        questions_path=questions_path,
        graph_source=docs_source,
        output_dir=output_dir,
        limit=limit,
        selector_names=selector_names,
        token_budgets=token_budgets,
        budget_ratios=budget_ratios,
        seed=seed,
        max_steps=max_steps,
        top_k=top_k,
        with_e2e=with_e2e,
        answerer_mode=answerer_mode,
        answer_model=answer_model,
        answer_api_key_env=answer_api_key_env,
        answer_base_url=answer_base_url,
        answer_cache_path=answer_cache_path,
        export_graphrag_inputs=export_graphrag_inputs,
    )


def run_2wiki_store_experiment(
    *,
    store_uri: str | Path,
    output_root: str | Path,
    exp_name: str,
    split: str = "dev",
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    case_start: int = 0,
    case_limit: int | None = None,
    chunk_size: int | None = None,
    chunk_index: int | None = None,
    selector_names: Sequence[str] | None = None,
    token_budgets: Sequence[int] | None = None,
    budget_ratios: Sequence[float] | None = None,
    seed: int = 0,
    max_steps: int = 3,
    top_k: int = 2,
    with_e2e: bool = False,
    answerer_mode: str = "heuristic",
    answer_model: str = "gpt-4.1-mini",
    answer_api_key_env: str = "OPENAI_API_KEY",
    answer_base_url: str | None = None,
    answer_cache_path: str | Path | None = None,
    export_graphrag_inputs: bool = True,
) -> tuple[list[CaseEvaluation], ExperimentSummary, Path]:
    store = ShardedLinkContextStore(store_uri, cache_dir=cache_dir)
    logger.info("Loading %s split questions from sharded store", split)
    cases = store.load_questions(split)
    selected_cases, chunk_meta = _slice_cases(
        cases,
        split=split,
        limit=limit,
        case_start=case_start,
        case_limit=case_limit,
        chunk_size=chunk_size,
        chunk_index=chunk_index,
    )
    selectors = select_selectors(
        selector_names,
        seed=seed,
        include_diagnostics=selector_names is not None,
    )
    budgets = _resolve_budgets(
        token_budgets=token_budgets,
        budget_ratios=budget_ratios,
        max_steps=max_steps,
        top_k=top_k,
    )
    evaluators = _build_evaluators(
        selectors=selectors,
        budgets=budgets,
        with_e2e=with_e2e,
        answerer_mode=answerer_mode,
        answer_model=answer_model,
        answer_api_key_env=answer_api_key_env,
        answer_base_url=answer_base_url,
        answer_cache_path=answer_cache_path,
    )
    logger.info(
        "Running store-backed 2Wiki experiment (split=%s, selected_cases=%s, selectors=%s, budgets=%s, with_e2e=%s, answerer=%s, export_graphrag_inputs=%s)",
        split,
        len(selected_cases),
        [selector.name for selector in selectors],
        [budget.budget_label for budget in budgets],
        with_e2e,
        answerer_mode,
        export_graphrag_inputs,
    )

    evaluations = _evaluate_cases(
        graph=store,
        cases=selected_cases,
        evaluators=evaluators,
        description="evaluate store-backed 2wiki cases",
    )

    chunk_dir = _chunk_output_dir(output_root=Path(output_root), exp_name=exp_name, chunk_meta=chunk_meta)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if export_graphrag_inputs:
        _export_graphrag_inputs(
            graph=store,
            evaluations=evaluations,
            output_dir=chunk_dir,
            description="export store graphrag inputs",
        )

    summary = summarize_evaluations(evaluations, dataset_name=selected_cases[0].dataset_name if selected_cases else "2wikimultihop")
    _write_result_files(chunk_dir=chunk_dir, evaluations=evaluations, summary=summary)
    (chunk_dir / "chunk.json").write_text(
        json.dumps(
            {
                **chunk_meta,
                "store_uri": str(store_uri),
                "selectors": list(selector_names or []),
                "token_budgets": list(token_budgets) if token_budgets is not None else None,
                "budget_ratios": list(budget_ratios) if budget_ratios is not None else None,
                "with_e2e": with_e2e,
                "answerer_mode": answerer_mode,
                "answer_model": answer_model if with_e2e and answerer_mode == "llm_fixed" else None,
                "answer_api_key_env": answer_api_key_env if with_e2e and answerer_mode == "llm_fixed" else None,
                "answer_base_url": answer_base_url if with_e2e and answerer_mode == "llm_fixed" else None,
                "answer_cache_path": str(answer_cache_path) if answer_cache_path is not None else None,
                "export_graphrag_inputs": export_graphrag_inputs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Completed store-backed 2Wiki chunk; results written to %s", chunk_dir)
    return evaluations, summary, chunk_dir


def merge_2wiki_results(
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[ExperimentSummary, list[int]]:
    root = Path(run_dir)
    chunk_root = root / "chunks"
    chunk_dirs = sorted(path for path in chunk_root.iterdir() if path.is_dir())
    logger.info("Merging 2Wiki chunk results from %s (%s chunks)", root, len(chunk_dirs))
    records: list[dict[str, Any]] = []
    chunk_indices: list[int] = []
    total_cases: int | None = None
    chunk_size: int | None = None

    for chunk_dir in _iterate_with_optional_progress(chunk_dirs, description="merge chunk results"):
        chunk_meta_path = chunk_dir / "chunk.json"
        if chunk_meta_path.exists():
            chunk_meta = json.loads(chunk_meta_path.read_text(encoding="utf-8"))
            if chunk_meta.get("chunk_index") is not None:
                chunk_indices.append(int(chunk_meta["chunk_index"]))
            if chunk_meta.get("total_cases") is not None:
                total_cases = int(chunk_meta["total_cases"])
            if chunk_meta.get("chunk_size") is not None:
                chunk_size = int(chunk_meta["chunk_size"])
        results_path = chunk_dir / "results.jsonl"
        if not results_path.exists():
            continue
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    summary = _summarize_result_records(records)
    merged_dir = Path(output_dir) if output_dir is not None else root
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_results = merged_dir / "results.jsonl"
    with merged_results.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (merged_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    missing_chunks = _missing_chunk_indices(chunk_indices, total_cases=total_cases, chunk_size=chunk_size)
    logger.info(
        "Merged %s result records into %s (missing_chunks=%s)",
        len(records),
        merged_dir,
        missing_chunks,
    )
    return summary, missing_chunks


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


def parse_token_budgets(value: str | None) -> list[int] | None:
    if value is None:
        return None
    budgets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not budgets:
        return None
    for budget in budgets:
        if budget <= 0:
            raise ValueError(f"Token budget must be positive, got {budget}")
    return budgets


def selector_choices_help(*, include_diagnostics: bool = True) -> str:
    return ",".join(available_selector_names(include_diagnostics=include_diagnostics))


def budget_ratio_choices_help() -> str:
    return ",".join(f"{ratio:.2f}" for ratio in DEFAULT_BUDGET_RATIOS)


def token_budget_choices_help() -> str:
    return ",".join(str(budget) for budget in DEFAULT_TOKEN_BUDGETS)


def store_budget_ratio_choices_help() -> str:
    return budget_ratio_choices_help()


def store_token_budget_choices_help() -> str:
    return token_budget_choices_help()


def _build_evaluators(
    *,
    selectors,
    budgets: Sequence[SelectionBudget],
    with_e2e: bool,
    answerer_mode: str,
    answer_model: str,
    answer_api_key_env: str,
    answer_base_url: str | None,
    answer_cache_path: str | Path | None,
) -> list[Evaluator]:
    answerer = _build_answerer(
        with_e2e=with_e2e,
        answerer_mode=answerer_mode,
        answer_model=answer_model,
        answer_api_key_env=answer_api_key_env,
        answer_base_url=answer_base_url,
        answer_cache_path=answer_cache_path,
    )
    return [
        Evaluator(
            selectors,
            budget=budget,
            with_e2e=with_e2e,
            answerer=answerer,
        )
        for budget in budgets
    ]


def _resolve_budgets(
    *,
    token_budgets: Sequence[int] | None,
    budget_ratios: Sequence[float] | None,
    max_steps: int,
    top_k: int,
) -> list[SelectionBudget]:
    if token_budgets is not None and budget_ratios is not None:
        raise ValueError("Specify either token_budgets or budget_ratios, not both.")
    if token_budgets is not None:
        return [
            SelectionBudget(
                max_steps=max_steps,
                top_k=top_k,
                token_budget_tokens=budget,
                token_budget_ratio=None,
            )
            for budget in token_budgets
        ]
    if budget_ratios is not None:
        return [
            SelectionBudget(
                max_steps=max_steps,
                top_k=top_k,
                token_budget_tokens=None,
                token_budget_ratio=ratio,
            )
            for ratio in budget_ratios
        ]
    return [
        SelectionBudget(
            max_steps=max_steps,
            top_k=top_k,
            token_budget_tokens=budget,
            token_budget_ratio=None,
        )
        for budget in DEFAULT_TOKEN_BUDGETS
    ]


def _build_answerer(
    *,
    with_e2e: bool,
    answerer_mode: str,
    answer_model: str,
    answer_api_key_env: str,
    answer_base_url: str | None,
    answer_cache_path: str | Path | None,
) -> SupportsAnswer | None:
    if not with_e2e:
        return None
    if answerer_mode == "heuristic":
        return Answerer()
    if answerer_mode != "llm_fixed":
        raise ValueError(f"Unknown answerer_mode: {answerer_mode}")
    if not os.environ.get(answer_api_key_env):
        raise ValueError(f"Missing API key in environment variable {answer_api_key_env}")
    return LLMAnswerer(
        config=LLMAnswererConfig(
            model=answer_model,
            api_key_env=answer_api_key_env,
            base_url=answer_base_url,
            cache_path=Path(answer_cache_path) if answer_cache_path is not None else None,
        )
    )


def _evaluate_cases(
    *,
    graph,
    cases: Sequence[Any],
    evaluators: Sequence[Evaluator],
    description: str,
) -> list[CaseEvaluation]:
    evaluations: list[CaseEvaluation] = []
    total = len(cases)
    if total == 0:
        return evaluations
    logger.info("%s (%s cases)", description.capitalize(), total)
    for case in _iterate_with_optional_progress(cases, description=description):
        selections = []
        for evaluator in evaluators:
            selections.extend(evaluator.evaluate_case(graph, case).selections)
        evaluations.append(CaseEvaluation(case=case, selections=selections))
    return evaluations


def _export_graphrag_inputs(
    *,
    graph,
    evaluations: Sequence[CaseEvaluation],
    output_dir: Path,
    description: str,
) -> None:
    total = sum(len(evaluation.selections) for evaluation in evaluations)
    logger.info("%s (%s exports)", description.capitalize(), total)
    selection_iter = (
        (evaluation.case.case_id, selection)
        for evaluation in evaluations
        for selection in evaluation.selections
    )
    for case_id, selection in _iterate_with_optional_progress(
        selection_iter,
        total=total,
        description=description,
    ):
        selection.graphrag_input_path = _write_graphrag_input(
            graph=graph,
            case_id=case_id,
            selection=selection,
            output_dir=output_dir,
        )


def _iterate_with_optional_progress(
    items,
    *,
    description: str,
    total: int | None = None,
):
    if total is None:
        try:
            total = len(items)
        except TypeError:
            total = None
    if not should_render_progress():
        for item in items:
            yield item
        return

    with create_progress(transient=True) as progress:
        task_id = progress.add_task(description, total=total)
        for item in items:
            yield item
            progress.advance(task_id, 1)


def _write_graphrag_input(
    *,
    graph,
    case_id: str,
    selection,
    output_dir: Path,
) -> str:
    budget_slug = _budget_label_slug(selection.budget.budget_label)
    export_path = (
        output_dir
        / "graphrag_inputs"
        / selection.selector_name
        / f"budget-{budget_slug}"
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


def _budget_label_slug(value: str) -> str:
    return value.replace(".", "_")


def _write_result_files(
    *,
    chunk_dir: Path,
    evaluations: list[CaseEvaluation],
    summary: ExperimentSummary,
) -> None:
    results_path = chunk_dir / "results.jsonl"
    summary_path = chunk_dir / "summary.json"
    with results_path.open("w", encoding="utf-8") as handle:
        for evaluation in evaluations:
            for selection in evaluation.selections:
                record = _selection_record(evaluation, selection)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _selection_record(evaluation: CaseEvaluation, selection) -> dict[str, Any]:
    return {
        "dataset_name": evaluation.case.dataset_name,
        "case_id": evaluation.case.case_id,
        "query": evaluation.case.query,
        "expected_answer": evaluation.case.expected_answer,
        "gold_support_nodes": evaluation.case.gold_support_nodes,
        "gold_start_nodes": evaluation.case.gold_start_nodes,
        "gold_path_nodes": evaluation.case.gold_path_nodes,
        "selector": selection.selector_name,
        "budget_mode": selection.budget.budget_mode,
        "budget_value": selection.budget.budget_value,
        "budget_label": selection.budget.budget_label,
        "token_budget_tokens": selection.budget.token_budget_tokens,
        "token_budget_ratio": selection.budget.token_budget_ratio,
        "answerer_mode": selection.end_to_end.mode if selection.end_to_end is not None else None,
        "answer_model": selection.end_to_end.model if selection.end_to_end is not None else None,
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


def _slice_cases(
    cases,
    *,
    split: str,
    limit: int | None,
    case_start: int,
    case_limit: int | None,
    chunk_size: int | None,
    chunk_index: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    if limit is not None:
        cases = cases[:limit]
    total_cases = len(cases)
    effective_chunk_size = chunk_size if chunk_size is not None else (100 if chunk_index is not None else None)
    if chunk_index is not None:
        if effective_chunk_size is None:
            raise ValueError("chunk_size must be set when chunk_index is provided")
        case_start = chunk_index * effective_chunk_size
        case_limit = effective_chunk_size
    end = total_cases if case_limit is None else min(total_cases, case_start + case_limit)
    selected_cases = cases[case_start:end]
    return selected_cases, {
        "split": split,
        "total_cases": total_cases,
        "case_start": case_start,
        "case_limit": end - case_start,
        "chunk_size": effective_chunk_size,
        "chunk_index": chunk_index,
    }


def _chunk_output_dir(*, output_root: Path, exp_name: str, chunk_meta: dict[str, Any]) -> Path:
    chunk_index = chunk_meta.get("chunk_index")
    if chunk_index is not None:
        label = f"chunk-{int(chunk_index):05d}"
    else:
        start = int(chunk_meta.get("case_start", 0))
        end = start + int(chunk_meta.get("case_limit", 0))
        label = f"range-{start:05d}-{max(start, end - 1):05d}"
    return output_root / exp_name / "chunks" / label


def _summarize_result_records(records: Sequence[dict[str, Any]]) -> ExperimentSummary:
    if not records:
        raise ValueError("Cannot summarize empty result records.")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for record in records:
        key = (
            str(record["selector"]),
            str(record["budget_mode"]),
            str(record["budget_value"]),
        )
        if key not in groups:
            groups[key] = []
            ordered_keys.append(key)
        groups[key].append(record)

    selector_budgets: list[dict[str, Any]] = []
    for key in ordered_keys:
        rows = groups[key]
        name = str(rows[0]["selector"])
        budget_mode = str(rows[0]["budget_mode"])
        budget_value_raw = rows[0]["budget_value"]
        budget_value: int | float = int(budget_value_raw) if budget_mode == "tokens" else float(budget_value_raw)
        budget_label = str(rows[0]["budget_label"])
        token_budget_tokens = rows[0].get("token_budget_tokens")
        token_budget_ratio = rows[0].get("token_budget_ratio")
        start_hits = [
            1.0 if row["selection"]["metrics"]["start_hit"] else 0.0
            for row in rows
            if row["selection"]["metrics"]["start_hit"] is not None
        ]
        support_recall = [
            float(row["selection"]["metrics"]["support_recall"])
            for row in rows
            if row["selection"]["metrics"]["support_recall"] is not None
        ]
        support_precision = [
            float(row["selection"]["metrics"]["support_precision"])
            for row in rows
            if row["selection"]["metrics"]["support_precision"] is not None
        ]
        support_f1 = [
            float(row["selection"]["metrics"]["support_f1"])
            for row in rows
            if row["selection"]["metrics"]["support_f1"] is not None
        ]
        path_hits = [
            1.0 if row["selection"]["metrics"]["path_hit"] else 0.0
            for row in rows
            if row["selection"]["metrics"]["path_hit"] is not None
        ]
        selected_nodes = [float(row["selection"]["metrics"]["selected_nodes_count"]) for row in rows]
        selected_tokens = [float(row["selection"]["metrics"]["selected_token_estimate"]) for row in rows]
        compression = [float(row["selection"]["metrics"]["compression_ratio"]) for row in rows]
        adherence = [1.0 if row["selection"]["metrics"]["budget_adherence"] else 0.0 for row in rows]
        runtime = [float(row["selection"]["metrics"]["selection_runtime_s"]) for row in rows]
        answer_em = [
            float(row["end_to_end"]["em"])
            for row in rows
            if row["end_to_end"] is not None and row["end_to_end"]["em"] is not None
        ]
        answer_f1 = [
            float(row["end_to_end"]["f1"])
            for row in rows
            if row["end_to_end"] is not None and row["end_to_end"]["f1"] is not None
        ]
        selector_budgets.append(
            {
                "name": name,
                "budget_mode": budget_mode,
                "budget_value": budget_value,
                "budget_label": budget_label,
                "token_budget_ratio": float(token_budget_ratio) if token_budget_ratio is not None else None,
                "token_budget_tokens": int(token_budget_tokens) if token_budget_tokens is not None else None,
                "num_cases": len(rows),
                "avg_start_hit": _average_or_none(start_hits),
                "avg_support_recall": _average_or_none(support_recall),
                "avg_support_precision": _average_or_none(support_precision),
                "avg_support_f1": _average_or_none(support_f1),
                "avg_path_hit": _average_or_none(path_hits),
                "avg_selected_nodes": _average_or_none(selected_nodes) or 0.0,
                "avg_selected_token_estimate": _average_or_none(selected_tokens) or 0.0,
                "avg_compression_ratio": _average_or_none(compression) or 0.0,
                "avg_budget_adherence": _average_or_none(adherence) or 0.0,
                "avg_selection_runtime_s": _average_or_none(runtime) or 0.0,
                "avg_answer_em": _average_or_none(answer_em),
                "avg_answer_f1": _average_or_none(answer_f1),
            }
        )
    return ExperimentSummary(
        dataset_name=str(records[0]["dataset_name"]),
        total_cases=len({record["case_id"] for record in records}),
        selector_budgets=[
            SelectorBudgetSummary(**row) for row in selector_budgets
        ],
    )


def _average_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _missing_chunk_indices(
    chunk_indices: Sequence[int],
    *,
    total_cases: int | None,
    chunk_size: int | None,
) -> list[int]:
    if not chunk_indices or total_cases is None or chunk_size is None or chunk_size <= 0:
        return []
    expected_chunks = (total_cases + chunk_size - 1) // chunk_size
    expected = set(range(expected_chunks))
    return sorted(expected.difference(chunk_indices))
