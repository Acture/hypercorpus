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
