import io
import json
from pathlib import Path
import tarfile

from typer.testing import CliRunner

from hypercorpus_cli import app


def _write_iirc_archive(tmp_path: Path, *, include_context: bool) -> Path:
	dev_payload = [
		{
			"title": "Moon Launch Program",
			"text": "Moon Launch Program launches from Cape Canaveral.",
			"links": [{"target": "Cape Canaveral", "indices": [33, 48]}],
			"questions": [
				{
					"qid": "iirc-raw-1",
					"question": "Which state contains the launch city?",
					"answer": {
						"type": "span",
						"answer_spans": [{"text": "Florida", "passage": "Florida"}],
					},
					"question_links": ["Cape Canaveral"],
					"context": [{"passage": "Florida"}],
				}
			],
		}
	]
	context_articles = {
		"Cape Canaveral": {
			"text": "Cape Canaveral is in Florida.",
			"links": [{"target": "Florida", "indices": [22, 29]}],
		},
		"Florida": {
			"text": "Florida is a state in the southeastern United States.",
			"links": [],
		},
	}
	archive_path = tmp_path / (
		"iirc-with-context.tgz" if include_context else "iirc-questions-only.tgz"
	)
	payloads = {
		"dev.json": dev_payload,
		"train.json": dev_payload,
	}
	if include_context:
		payloads["context_articles.json"] = context_articles
	with tarfile.open(archive_path, "w:gz") as archive:
		for name, payload in payloads.items():
			data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
			info = tarfile.TarInfo(name=name)
			info.size = len(data)
			archive.addfile(info, io.BytesIO(data))
	return archive_path


def _write_iirc_context_source(tmp_path: Path) -> Path:
	context_path = tmp_path / "context_articles.json"
	context_payload = {
		"Cape Canaveral": {
			"text": "Cape Canaveral is in Florida.",
			"links": [{"target": "Florida", "indices": [22, 29]}],
		},
		"Florida": {
			"text": "Florida is a state in the southeastern United States.",
			"links": [],
		},
	}
	context_path.write_text(
		json.dumps(context_payload, ensure_ascii=False), encoding="utf-8"
	)
	return context_path


def _write_iirc_context_archive(tmp_path: Path) -> Path:
	context_path = _write_iirc_context_source(tmp_path)
	archive_path = tmp_path / "context_articles.tar.gz"
	with tarfile.open(archive_path, "w:gz") as archive:
		data = context_path.read_bytes()
		info = tarfile.TarInfo(name="context_articles.json")
		info.size = len(data)
		archive.addfile(info, io.BytesIO(data))
	return archive_path


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


def test_fetch_iirc_cli_reports_built_in_context_failures(tmp_path, monkeypatch):
	runner = CliRunner()
	output_dir = tmp_path / "iirc-fetch"
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	missing_context = tmp_path / "missing-context_articles.tar.gz"
	monkeypatch.setattr(
		"hypercorpus.datasets.fetch.IIRC_CONTEXT_ARTICLES_URL", missing_context.as_uri()
	)
	result = runner.invoke(
		app,
		[
			"datasets",
			"fetch-iirc",
			"--output-dir",
			str(output_dir),
			"--archive-url",
			archive_path.as_uri(),
		],
	)

	assert result.exit_code != 0
	assert "--context-source" in result.output
	assert "--question-only" in result.output


def test_fetch_iirc_cli_accepts_context_source(tmp_path):
	runner = CliRunner()
	output_dir = tmp_path / "iirc-fetch"
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_path = _write_iirc_context_source(tmp_path)
	result = runner.invoke(
		app,
		[
			"datasets",
			"fetch-iirc",
			"--output-dir",
			str(output_dir),
			"--archive-url",
			archive_path.as_uri(),
			"--context-source",
			str(context_path),
		],
	)

	assert result.exit_code == 0, result.stdout
	assert (
		output_dir / "raw" / "iirc" / "extracted" / "context_articles.json"
	).exists()


def test_fetch_iirc_cli_uses_built_in_context_source_by_default(tmp_path, monkeypatch):
	runner = CliRunner()
	output_dir = tmp_path / "iirc-fetch"
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_archive = _write_iirc_context_archive(tmp_path)
	monkeypatch.setattr(
		"hypercorpus.datasets.fetch.IIRC_CONTEXT_ARTICLES_URL", context_archive.as_uri()
	)
	result = runner.invoke(
		app,
		[
			"datasets",
			"fetch-iirc",
			"--output-dir",
			str(output_dir),
			"--archive-url",
			archive_path.as_uri(),
		],
	)

	assert result.exit_code == 0, result.stdout
	assert (
		output_dir / "raw" / "iirc" / "extracted" / "context_articles.json"
	).exists()
	assert (
		output_dir / "raw" / "iirc" / "archives" / "context_articles.tar.gz"
	).exists()


def test_convert_iirc_raw_cli_prints_prepare_iirc_follow_up(iirc_raw_archive, tmp_path):
	runner = CliRunner()
	fetched_dir = tmp_path / "iirc-fetch"
	fetch_result = runner.invoke(
		app,
		[
			"datasets",
			"fetch-iirc",
			"--output-dir",
			str(fetched_dir),
			"--archive-url",
			iirc_raw_archive.as_uri(),
		],
	)
	assert fetch_result.exit_code == 0, fetch_result.stdout

	output_dir = tmp_path / "iirc-normalized"
	result = runner.invoke(
		app,
		[
			"datasets",
			"convert-iirc-raw",
			"--raw-dir",
			str(fetched_dir / "raw" / "iirc"),
			"--output-dir",
			str(output_dir),
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "prepare-iirc-store" in result.stdout
	assert "prepare-musique-store" not in result.stdout


def test_convert_musique_raw_cli_writes_normalized_layout(
	musique_raw_split_files, tmp_path
):
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
	assert "prepare-musique-store" in result.stdout


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


def test_prepare_iirc_store_from_raw_cli_accepts_context_source(tmp_path):
	runner = CliRunner()
	output_dir = tmp_path / "iirc-from-raw"
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_path = _write_iirc_context_source(tmp_path)
	result = runner.invoke(
		app,
		[
			"datasets",
			"prepare-iirc-store-from-raw",
			"--output-dir",
			str(output_dir),
			"--archive-url",
			archive_path.as_uri(),
			"--context-source",
			str(context_path),
		],
	)

	assert result.exit_code == 0, result.stdout
	assert (output_dir / "store" / "manifest.json").exists()


def test_prepare_iirc_store_from_raw_cli_uses_built_in_context_source_by_default(
	tmp_path, monkeypatch
):
	runner = CliRunner()
	output_dir = tmp_path / "iirc-from-raw"
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_archive = _write_iirc_context_archive(tmp_path)
	monkeypatch.setattr(
		"hypercorpus.datasets.fetch.IIRC_CONTEXT_ARTICLES_URL", context_archive.as_uri()
	)
	result = runner.invoke(
		app,
		[
			"datasets",
			"prepare-iirc-store-from-raw",
			"--output-dir",
			str(output_dir),
			"--archive-url",
			archive_path.as_uri(),
		],
	)

	assert result.exit_code == 0, result.stdout
	assert (output_dir / "store" / "manifest.json").exists()


def test_prepare_iirc_store_from_raw_cli_allows_question_only(tmp_path):
	runner = CliRunner()
	output_dir = tmp_path / "iirc-from-raw"
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	result = runner.invoke(
		app,
		[
			"datasets",
			"prepare-iirc-store-from-raw",
			"--output-dir",
			str(output_dir),
			"--archive-url",
			archive_path.as_uri(),
			"--question-only",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert (output_dir / "store" / "manifest.json").exists()


def test_prepare_musique_store_from_raw_cli_creates_store(
	musique_raw_split_files, tmp_path
):
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


def test_prepare_hotpotqa_store_from_raw_cli_creates_store(
	hotpotqa_raw_split_files, tmp_path
):
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
