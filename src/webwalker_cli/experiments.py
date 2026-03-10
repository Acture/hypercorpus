from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from webwalker.experiments import (
    budget_ratio_choices_help,
    merge_2wiki_results,
    parse_budget_ratios,
    parse_selector_names,
    run_2wiki_experiment,
    run_2wiki_store_experiment,
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
    _print_summary(console, summary)
    console.print(f"results.jsonl -> {output / 'results.jsonl'}")
    console.print(f"summary.json -> {output / 'summary.json'}")
    if export_graphrag_inputs:
        console.print(f"graphrag_inputs -> {output / 'graphrag_inputs'}")


@experiments_app.command("run-2wiki-store")
def run_2wiki_store(
    store: str = typer.Option(..., "--store", help="Path or s3:// URI to a prepared 2Wiki store"),
    exp_name: str = typer.Option(..., "--exp-name", help="Experiment name under the output root"),
    output_root: Path = typer.Option(Path("runs"), "--output-root", file_okay=False, help="Root directory for chunk outputs"),
    split: str = typer.Option("dev", "--split", help="Question split inside the prepared store"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", file_okay=False, help="Local cache directory for remote stores"),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Optional total number of questions to consider before slicing"),
    case_start: int = typer.Option(0, "--case-start", min=0, help="Start offset after split selection"),
    case_limit: int | None = typer.Option(None, "--case-limit", min=1, help="Maximum number of questions for this run"),
    chunk_size: int | None = typer.Option(None, "--chunk-size", min=1, help="Questions per chunk"),
    chunk_index: int | None = typer.Option(None, "--chunk-index", min=0, help="Chunk index to run"),
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
    _evaluations, summary, chunk_dir = run_2wiki_store_experiment(
        store_uri=store,
        output_root=output_root,
        exp_name=exp_name,
        split=split,
        cache_dir=cache_dir,
        limit=limit,
        case_start=case_start,
        case_limit=case_limit,
        chunk_size=chunk_size,
        chunk_index=chunk_index,
        selector_names=parse_selector_names(selectors),
        budget_ratios=parse_budget_ratios(budget_ratios),
        seed=seed,
        max_steps=max_steps,
        top_k=top_k,
        with_e2e=with_e2e,
        export_graphrag_inputs=export_graphrag_inputs,
    )
    _print_summary(console, summary)
    console.print(f"chunk_dir -> {chunk_dir}")
    console.print(f"results.jsonl -> {chunk_dir / 'results.jsonl'}")
    console.print(f"summary.json -> {chunk_dir / 'summary.json'}")
    if export_graphrag_inputs:
        console.print(f"graphrag_inputs -> {chunk_dir / 'graphrag_inputs'}")


@experiments_app.command("merge-2wiki-results")
def merge_2wiki_store_results(
    run_dir: Path = typer.Option(..., "--run-dir", file_okay=False, help="Run directory containing chunks/"),
    output_dir: Path | None = typer.Option(None, "--output-dir", file_okay=False, help="Optional output directory for merged files"),
) -> None:
    console = Console()
    summary, missing_chunks = merge_2wiki_results(run_dir=run_dir, output_dir=output_dir)
    merged_dir = output_dir or run_dir
    _print_summary(console, summary)
    console.print(f"merged results.jsonl -> {merged_dir / 'results.jsonl'}")
    console.print(f"merged summary.json -> {merged_dir / 'summary.json'}")
    console.print(f"missing_chunks -> {missing_chunks if missing_chunks else '[]'}")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _print_summary(console: Console, summary) -> None:
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
