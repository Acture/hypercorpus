from typer.testing import CliRunner

from webwalker_cli import app


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
            "webwalker_selector,eager_full_corpus_proxy",
            "--budget-ratios",
            "0.10,1.0",
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "2wikimultihop summary" in result.stdout
    assert "selector" in result.stdout
    assert "budget" in result.stdout
    assert "e2e_em" in result.stdout
    assert (output_dir / "results.jsonl").exists()
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
            "dense_topk",
            "--budget-ratios",
            "0.10",
            "--no-e2e",
            "--no-export-graphrag-inputs",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "e2e_em" in result.stdout
    assert not (output_dir / "graphrag_inputs").exists()


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
            "webwalker_selector,eager_full_corpus_proxy",
            "--budget-ratios",
            "0.10,1.0",
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
            "dense_topk",
            "--budget-ratios",
            "0.10",
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
            "dense_topk",
            "--budget-ratios",
            "0.10",
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
            "dense_topk",
            "--budget-ratios",
            "0.10",
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
