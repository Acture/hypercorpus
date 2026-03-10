from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from webwalker.experiments import (
    budget_ratio_choices_help,
    parse_budget_ratios,
    parse_selector_names,
    run_2wiki_experiment,
    selector_choices_help,
)

experiments_app = typer.Typer(
    name="webwalker experiments",
    help="webwalker experiment runners",
    add_completion=False,
)


@experiments_app.command("run-2wiki")
def run_2wiki(
    questions: Path = typer.Option(..., "--questions", exists=True, dir_okay=False, help="Path to 2Wiki questions JSON"),
    graph_records: Path = typer.Option(..., "--graph-records", exists=True, dir_okay=False, help="Path to 2Wiki paragraph hyperlink JSONL"),
    output: Path = typer.Option(..., "--output", file_okay=False, help="Output directory"),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Optional number of questions to evaluate"),
    selectors: str | None = typer.Option(
        None,
        "--selectors",
        help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
    ),
    budget_ratios: str | None = typer.Option(
        None,
        "--budget-ratios",
        help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for stochastic selectors"),
    max_steps: int = typer.Option(3, "--max-steps", min=1, help="Maximum walk or expansion steps"),
    top_k: int = typer.Option(2, "--top-k", min=1, help="Top-k start candidates / dense retrieval depth"),
    with_e2e: bool = typer.Option(True, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"),
    export_graphrag_inputs: bool = typer.Option(
        True,
        "--export-graphrag-inputs/--no-export-graphrag-inputs",
        help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
    ),
) -> None:
    console = Console()
    _, summary = run_2wiki_experiment(
        questions_path=questions,
        graph_records_path=graph_records,
        output_dir=output,
        limit=limit,
        selector_names=parse_selector_names(selectors),
        budget_ratios=parse_budget_ratios(budget_ratios),
        seed=seed,
        max_steps=max_steps,
        top_k=top_k,
        with_e2e=with_e2e,
        export_graphrag_inputs=export_graphrag_inputs,
    )

    table = Table(title=f"{summary.dataset_name} summary")
    table.add_column("selector")
    table.add_column("budget", justify="right")
    table.add_column("support_recall", justify="right")
    table.add_column("support_precision", justify="right")
    table.add_column("selected_tokens", justify="right")
    table.add_column("compression_ratio", justify="right")
    table.add_column("budget_adherence", justify="right")
    table.add_column("runtime_s", justify="right")
    table.add_column("e2e_em", justify="right")

    for row in summary.selector_budgets:
        table.add_row(
            row.name,
            _format_metric(row.token_budget_ratio),
            _format_metric(row.avg_support_recall),
            _format_metric(row.avg_support_precision),
            _format_metric(row.avg_selected_token_estimate),
            _format_metric(row.avg_compression_ratio),
            _format_metric(row.avg_budget_adherence),
            _format_metric(row.avg_selection_runtime_s),
            _format_metric(row.avg_e2e_em),
        )

    console.print(table)
    console.print(
        "columns: selector, budget, support_recall, support_precision, selected_tokens, "
        "compression_ratio, budget_adherence, runtime_s, e2e_em"
    )
    console.print(f"results.jsonl -> {output / 'results.jsonl'}")
    console.print(f"summary.json -> {output / 'summary.json'}")
    if export_graphrag_inputs:
        console.print(f"graphrag_inputs -> {output / 'graphrag_inputs'}")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"
