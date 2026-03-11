from rich.console import Console
from typer.testing import CliRunner

from webwalker.eval import ExperimentSummary, SelectorBudgetSummary
from webwalker_cli import app
from webwalker_cli.experiments import _print_summary


def _make_selector_budget_summary(
    *,
    name: str,
    selector_provider: str | None = None,
    selector_model: str | None = None,
    budget_label: str = "tokens-128",
    avg_selector_total_tokens: float | None = None,
    avg_selector_runtime_s: float | None = None,
    avg_selector_llm_calls: float | None = None,
    avg_selector_fallback_rate: float | None = None,
    avg_selector_parse_failure_rate: float | None = None,
) -> SelectorBudgetSummary:
    return SelectorBudgetSummary(
        name=name,
        selector_provider=selector_provider,
        selector_model=selector_model,
        budget_mode="tokens",
        budget_value=128,
        budget_label=budget_label,
        token_budget_ratio=None,
        token_budget_tokens=128,
        num_cases=1,
        avg_start_hit=None,
        avg_support_recall=0.5,
        avg_support_precision=0.6,
        avg_support_f1=0.545,
        avg_path_hit=None,
        avg_selected_nodes=2.0,
        avg_selected_token_estimate=70.0,
        avg_compression_ratio=0.1,
        avg_budget_adherence=1.0,
        avg_selection_runtime_s=0.01,
        avg_selector_prompt_tokens=None,
        avg_selector_completion_tokens=None,
        avg_selector_total_tokens=avg_selector_total_tokens,
        avg_selector_runtime_s=avg_selector_runtime_s,
        avg_selector_llm_calls=avg_selector_llm_calls,
        avg_selector_fallback_rate=avg_selector_fallback_rate,
        avg_selector_parse_failure_rate=avg_selector_parse_failure_rate,
        avg_answer_em=None,
        avg_answer_f1=None,
    )


def test_run_2wiki_cli_smoke(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "cli-out"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "run-2wiki",
            "--questions",
            str(questions_path),
            "--graph-records",
            str(graph_path),
            "--output",
            str(output_dir),
            "--limit",
            "1",
            "--selectors",
            "seed__link_context_overlap__single_path_walk,full_corpus_upper_bound",
            "--token-budgets",
            "128,256",
            "--seed",
            "7",
            "--no-e2e",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "2wikimultihop summary" in result.stdout
    assert "selector" in result.stdout
    assert "budget" in result.stdout
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "selector_logs.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "graphrag_inputs").exists()


def test_run_2wiki_cli_supports_no_e2e_and_no_export(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "cli-no-e2e"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "run-2wiki",
            "--questions",
            str(questions_path),
            "--graph-records",
            str(graph_path),
            "--output",
            str(output_dir),
            "--limit",
            "1",
            "--selectors",
            "seed_rerank",
            "--token-budgets",
            "128",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "2wikimultihop summary" in result.stdout
    assert not (output_dir / "graphrag_inputs").exists()


def test_run_2wiki_cli_rejects_legacy_selector_ids(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "cli-legacy-selector"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "run-2wiki",
            "--questions",
            str(questions_path),
            "--graph-records",
            str(graph_path),
            "--output",
            str(output_dir),
            "--limit",
            "1",
            "--selectors",
            "dense_topk",
            "--token-budgets",
            "128",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "Unknown selector: dense_topk"


def test_run_2wiki_cli_rejects_conflicting_budget_flags(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "cli-conflicting-budgets"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "run-2wiki",
            "--questions",
            str(questions_path),
            "--graph-records",
            str(graph_path),
            "--output",
            str(output_dir),
            "--limit",
            "1",
            "--selectors",
            "seed_rerank",
            "--token-budgets",
            "128",
            "--budget-ratios",
            "0.10",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "Specify either --token-budgets or --budget-ratios, not both."


def test_run_2wiki_store_cli_smoke(prepared_two_wiki_store, tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "run-2wiki-store",
            "--store",
            str(prepared_two_wiki_store.root),
            "--exp-name",
            "pilot",
            "--output-root",
            str(tmp_path / "runs"),
            "--chunk-size",
            "1",
            "--chunk-index",
            "0",
            "--selectors",
            "seed__link_context_overlap__single_path_walk,full_corpus_upper_bound",
            "--token-budgets",
            "128,256",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "chunk_dir" in result.stdout
    assert (tmp_path / "runs" / "pilot" / "chunks" / "chunk-00000" / "results.jsonl").exists()


def test_run_iirc_cli_smoke(iirc_files, tmp_path):
    questions_path, graph_path = iirc_files
    output_dir = tmp_path / "iirc-cli-out"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "run-iirc",
            "--questions",
            str(questions_path),
            "--graph-records",
            str(graph_path),
            "--output",
            str(output_dir),
            "--selectors",
            "seed_rerank",
            "--token-budgets",
            "128",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "iirc summary" in result.stdout
    assert (output_dir / "results.jsonl").exists()


def test_run_docs_cli_smoke(docs_files, tmp_path):
    questions_path, docs_root = docs_files
    output_dir = tmp_path / "docs-cli-out"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "run-docs",
            "--questions",
            str(questions_path),
            "--docs-source",
            str(docs_root),
            "--output",
            str(output_dir),
            "--dataset-name",
            "python_docs",
            "--selectors",
            "seed_rerank",
            "--token-budgets",
            "128",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "python_docs summary" in result.stdout
    assert (output_dir / "results.jsonl").exists()


def test_merge_2wiki_results_cli_reports_missing_chunks(prepared_two_wiki_store, tmp_path):
    runner = CliRunner()
    run_root = tmp_path / "runs"
    run_dir = run_root / "pilot"

    run_result = runner.invoke(
        app,
        [
            "experiments",
            "run-2wiki-store",
            "--store",
            str(prepared_two_wiki_store.root),
            "--exp-name",
            "pilot",
            "--output-root",
            str(run_root),
            "--chunk-size",
            "1",
            "--chunk-index",
            "0",
            "--selectors",
            "seed_rerank",
            "--token-budgets",
            "128",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout

    merge_result = runner.invoke(
        app,
        [
            "experiments",
            "merge-2wiki-results",
            "--run-dir",
            str(run_dir),
        ],
    )

    assert merge_result.exit_code == 0, merge_result.stdout
    assert "missing_chunks -> [1]" in merge_result.stdout


def test_print_summary_renders_main_table_only_for_non_llm_rows():
    summary = ExperimentSummary(
        dataset_name="2wikimultihop",
        total_cases=1,
        selector_budgets=[
            _make_selector_budget_summary(name="seed_rerank"),
        ],
    )
    console = Console(record=True, width=160)

    _print_summary(console, summary)

    output = console.export_text()
    assert "2wikimultihop summary" in output
    assert "support_recall" in output
    assert "support_precision" in output
    assert "answer_em" in output
    assert "selector health" not in output
    assert "selector_tokens" not in output
    assert "columns:" not in output


def test_print_summary_renders_selector_health_table_for_llm_rows():
    summary = ExperimentSummary(
        dataset_name="2wikimultihop",
        total_cases=1,
        selector_budgets=[
            _make_selector_budget_summary(
                name="seed__link_context_llm__single_path_walk",
                selector_provider="anthropic",
                selector_model="claude-haiku-4-5-20251001",
                avg_selector_total_tokens=1736.75,
                avg_selector_runtime_s=5.173,
                avg_selector_llm_calls=1.45,
                avg_selector_fallback_rate=0.0,
                avg_selector_parse_failure_rate=0.0,
            ),
        ],
    )
    console = Console(record=True, width=220)

    _print_summary(console, summary)

    output = console.export_text()
    assert "2wikimultihop summary" in output
    assert "2wikimultihop selector health" in output
    assert "selector_tokens" in output
    assert "selector_calls" in output
    assert "selector_fallback" in output
    assert "selector_parse_fail" in output
    assert "seed__link_context_llm__single_path_walk@anthropic" in output
    assert "claude-haiku-4-5-202" in output
    assert "columns:" not in output
