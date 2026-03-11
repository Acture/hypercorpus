from typer.testing import CliRunner

from webwalker_cli import app


def test_fetch_iirc_cli_downloads_raw_archive(iirc_raw_archive, tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "iirc-fetch"
    result = runner.invoke(
        app,
        [
            "datasets",
            "fetch-iirc",
            "--output-dir",
            str(output_dir),
            "--archive-url",
            iirc_raw_archive.as_uri(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "raw" / "source-manifest.json").exists()


def test_convert_musique_raw_cli_writes_normalized_layout(musique_raw_split_files, tmp_path):
    runner = CliRunner()
    raw_dir = next(iter(musique_raw_split_files.values())).parent
    output_dir = tmp_path / "musique-normalized"
    result = runner.invoke(
        app,
        [
            "datasets",
            "convert-musique-raw",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "questions" / "dev.json").exists()
    assert (output_dir / "graph" / "normalized.jsonl").exists()


def test_prepare_iirc_store_from_raw_cli_creates_store(iirc_raw_archive, tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "iirc-from-raw"
    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-iirc-store-from-raw",
            "--output-dir",
            str(output_dir),
            "--archive-url",
            iirc_raw_archive.as_uri(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "store" / "manifest.json").exists()


def test_prepare_musique_store_from_raw_cli_creates_store(musique_raw_split_files, tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "musique-from-raw"
    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-musique-store-from-raw",
            "--output-dir",
            str(output_dir),
            "--split",
            "dev",
            "--dev-url",
            musique_raw_split_files["dev"].as_uri(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "store" / "manifest.json").exists()


def test_prepare_hotpotqa_store_from_raw_cli_creates_store(hotpotqa_raw_split_files, tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "hotpot-from-raw"
    result = runner.invoke(
        app,
        [
            "datasets",
            "prepare-hotpotqa-store-from-raw",
            "--output-dir",
            str(output_dir),
            "--variant",
            "distractor",
            "--split",
            "dev",
            "--dev-url",
            hotpotqa_raw_split_files["distractor"]["dev"].as_uri(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "store" / "manifest.json").exists()
