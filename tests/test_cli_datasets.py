from typer.testing import CliRunner

from webwalker_cli import app


def test_fetch_2wiki_cli_downloads_question_and_graph(two_wiki_archives, tmp_path, monkeypatch):
    questions_zip, graph_zip = two_wiki_archives
    output_dir = tmp_path / "downloaded"
    runner = CliRunner()

    monkeypatch.setattr(
        "webwalker.datasets.fetch.TWOWIKI_QUESTIONS_URL",
        questions_zip.as_uri(),
    )
    monkeypatch.setattr(
        "webwalker.datasets.TWOWIKI_QUESTIONS_URL",
        questions_zip.as_uri(),
    )
    monkeypatch.setattr(
        "webwalker_cli.datasets.TWOWIKI_QUESTIONS_URL",
        questions_zip.as_uri(),
    )
    monkeypatch.setattr(
        "webwalker.datasets.fetch.TWOWIKI_GRAPH_URL",
        graph_zip.as_uri(),
    )
    monkeypatch.setattr(
        "webwalker.datasets.TWOWIKI_GRAPH_URL",
        graph_zip.as_uri(),
    )
    monkeypatch.setattr(
        "webwalker_cli.datasets.TWOWIKI_GRAPH_URL",
        graph_zip.as_uri(),
    )

    result = runner.invoke(
        app,
        [
            "datasets",
            "fetch-2wiki",
            "--output-dir",
            str(output_dir),
            "--split",
            "dev",
            "--graph",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "run-2wiki" in result.stdout
    assert (output_dir / "questions" / "dev.json").exists()
    assert (output_dir / "graph" / "para_with_hyperlink.jsonl").exists()


def test_write_2wiki_sample_cli_writes_smoke_dataset(tmp_path):
    output_dir = tmp_path / "sample"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "write-2wiki-sample",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "sample source" in result.stdout
    assert (output_dir / "questions" / "dev.json").exists()
    assert (output_dir / "graph" / "para_with_hyperlink.jsonl").exists()


def test_prepare_2wiki_store_cli_creates_store(two_wiki_archives, tmp_path):
    questions_zip, graph_zip = two_wiki_archives
    output_dir = tmp_path / "store"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-2wiki-store",
            "--output-dir",
            str(output_dir),
            "--questions-source",
            questions_zip.as_uri(),
            "--graph-source",
            graph_zip.as_uri(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "index" / "catalog.sqlite").exists()
    assert "run-2wiki-store" in result.stdout


def test_inspect_2wiki_store_cli_reports_recommendation(tmp_path):
    raw_root = tmp_path / "dataset" / "2wikimultihop"
    (raw_root / "data_ids_april7").mkdir(parents=True)
    (raw_root / "data_ids_april7" / "dev.json").write_text("[]", encoding="utf-8")
    (raw_root / "para_with_hyperlink.jsonl").write_text("", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "inspect-2wiki-store",
            "--raw-root",
            str(raw_root),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "recommended_action -> use-local-raw" in result.stdout


def test_prepare_2wiki_store_cli_rejects_existing_partial_output(two_wiki_archives, tmp_path):
    questions_zip, graph_zip = two_wiki_archives
    output_dir = tmp_path / "existing-store"
    (output_dir / "index").mkdir(parents=True)
    (output_dir / "index" / "catalog.sqlite").write_text("partial", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-2wiki-store",
            "--output-dir",
            str(output_dir),
            "--questions-source",
            questions_zip.as_uri(),
            "--graph-source",
            graph_zip.as_uri(),
        ],
    )

    assert result.exit_code != 0
