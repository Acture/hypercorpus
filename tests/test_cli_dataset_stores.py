import json

from typer.testing import CliRunner

from hypercorpus_cli import app


def test_prepare_iirc_store_cli_creates_store(iirc_files, tmp_path):
    questions_path, graph_path = iirc_files
    output_dir = tmp_path / "iirc-store"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-iirc-store",
            "--output-dir",
            str(output_dir),
            "--questions-source",
            str(questions_path),
            "--graph-source",
            str(graph_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "manifest.json").exists()
    assert "run-iirc-store" in result.stdout


def test_prepare_musique_store_cli_creates_store(musique_files, tmp_path):
    questions_path, graph_path = musique_files
    output_dir = tmp_path / "musique-store"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-musique-store",
            "--output-dir",
            str(output_dir),
            "--questions-source",
            str(questions_path),
            "--graph-source",
            str(graph_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "manifest.json").exists()
    assert "run-musique-store" in result.stdout


def test_prepare_musique_store_cli_rejects_iirc_normalized_inputs(iirc_files, tmp_path):
    questions_path, graph_path = iirc_files
    output_dir = tmp_path / "wrong-musique-store"
    (tmp_path / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "iirc",
                "questions": {"dev": questions_path.name},
                "graph": graph_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-musique-store",
            "--output-dir",
            str(output_dir),
            "--questions-source",
            str(questions_path),
            "--graph-source",
            str(graph_path),
        ],
    )

    assert result.exit_code != 0, result.output
    assert "look like dataset 'iirc'" in result.output
    assert "prepare-iirc-store" in result.output


def test_prepare_hotpotqa_store_cli_rejects_distractor_variant(hotpotqa_fullwiki_files, tmp_path):
    questions_path, graph_path = hotpotqa_fullwiki_files
    output_dir = tmp_path / "hotpot-store"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-hotpotqa-store",
            "--output-dir",
            str(output_dir),
            "--questions-source",
            str(questions_path),
            "--graph-source",
            str(graph_path),
            "--variant",
            "distractor",
        ],
    )

    assert result.exit_code != 0


def test_prepare_hotpotqa_store_cli_creates_store(hotpotqa_fullwiki_files, tmp_path):
    questions_path, graph_path = hotpotqa_fullwiki_files
    output_dir = tmp_path / "hotpot-store"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-hotpotqa-store",
            "--output-dir",
            str(output_dir),
            "--questions-source",
            str(questions_path),
            "--graph-source",
            str(graph_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "manifest.json").exists()
    assert "run-hotpotqa-store" in result.stdout
