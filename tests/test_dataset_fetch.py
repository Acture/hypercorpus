import io
import json
from pathlib import Path
import tarfile
from typing import Any
import urllib.error

import pytest

from hypercorpus.datasets.fetch import (
	fetch_hotpotqa_dataset,
	fetch_iirc_dataset,
	fetch_musique_dataset,
	fetch_2wiki_dataset,
	write_2wiki_sample_dataset,
)
from hypercorpus.datasets.twowiki_store import prepare_2wiki_store
from hypercorpus.datasets.twowiki import load_2wiki_graph, load_2wiki_questions


def _write_iirc_archive(
	tmp_path: Path, *, include_context: bool, include_appledouble: bool = False
) -> Path:
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
	payloads: dict[str, Any] = {
		"dev.json": dev_payload,
		"train.json": dev_payload,
	}
	if include_context:
		payloads["context_articles.json"] = context_articles
	if include_appledouble:
		payloads["._dev.json"] = {"ignored": True}
		payloads["._train.json"] = {"ignored": True}
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


def test_fetch_2wiki_dataset_extracts_selected_split_and_graph(
	two_wiki_archives, tmp_path
):
	questions_zip, graph_zip = two_wiki_archives
	output_dir = tmp_path / "fetched"

	layout = fetch_2wiki_dataset(
		output_dir,
		split="dev",
		include_graph=True,
		questions_url=questions_zip.as_uri(),
		graph_url=graph_zip.as_uri(),
	)

	assert set(layout.question_paths) == {"dev"}
	assert layout.question_paths["dev"] == output_dir / "questions" / "dev.json"
	assert layout.question_paths["dev"].exists()
	assert layout.graph_path == output_dir / "graph" / "para_with_hyperlink.jsonl"
	assert layout.graph_path is not None
	assert layout.graph_path.exists()
	assert not (output_dir / "archives").exists()

	cases = load_2wiki_questions(layout.question_paths["dev"])
	graph = load_2wiki_graph(layout.graph_path)
	assert cases[0].case_id == "q1"
	assert graph.get_document("Moon Launch Program") is not None


def test_fetch_2wiki_dataset_can_keep_archives_and_fetch_all_splits(
	two_wiki_archives, tmp_path
):
	questions_zip, _graph_zip = two_wiki_archives
	output_dir = tmp_path / "archives"

	layout = fetch_2wiki_dataset(
		output_dir,
		split="all",
		include_graph=False,
		keep_archives=True,
		questions_url=questions_zip.as_uri(),
	)

	assert set(layout.question_paths) == {"dev", "train", "test"}
	assert set(layout.archive_paths) == {"questions"}
	assert layout.archive_paths["questions"].exists()


def test_write_2wiki_sample_dataset_writes_runnable_layout(tmp_path):
	output_dir = tmp_path / "sample"

	layout = write_2wiki_sample_dataset(output_dir)

	assert set(layout.question_paths) == {"dev"}
	assert layout.graph_path is not None
	assert layout.source == "sample"

	questions = json.loads(layout.question_paths["dev"].read_text(encoding="utf-8"))
	assert questions[0]["_id"] == "q1"
	graph = load_2wiki_graph(layout.graph_path)
	assert graph.get_document("Cape Canaveral") is not None


def test_prepare_2wiki_store_rejects_when_min_free_space_is_too_high(
	two_wiki_archives, tmp_path
):
	questions_zip, graph_zip = two_wiki_archives

	with pytest.raises(RuntimeError, match="Refusing to proceed"):
		prepare_2wiki_store(
			tmp_path / "store",
			questions_source=questions_zip.as_uri(),
			graph_source=graph_zip.as_uri(),
			min_free_gib=10_000.0,
		)


def test_fetch_iirc_dataset_extracts_archive_and_writes_source_manifest(
	iirc_raw_archive, tmp_path
):
	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched", archive_url=iirc_raw_archive.as_uri()
	)

	assert layout.source_manifest_path is not None
	assert layout.source_manifest_path.exists()
	assert (layout.raw_dir / "iirc" / "extracted" / "dev.json").exists()
	assert (layout.raw_dir / "iirc" / "extracted" / "context_articles.json").exists()


def test_fetch_iirc_dataset_requires_context_by_default_when_built_in_context_fetch_fails(
	tmp_path, monkeypatch
):
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	missing_context = tmp_path / "missing-context_articles.tar.gz"
	monkeypatch.setattr(
		"hypercorpus.datasets.fetch.IIRC_CONTEXT_ARTICLES_URL", missing_context.as_uri()
	)

	with pytest.raises(ValueError, match="context_articles.json"):
		fetch_iirc_dataset(tmp_path / "iirc-fetched", archive_url=archive_path.as_uri())


def test_fetch_iirc_dataset_can_materialize_context_from_local_path(tmp_path):
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_path = _write_iirc_context_source(tmp_path)

	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched",
		archive_url=archive_path.as_uri(),
		context_source=context_path,
	)

	context_destination = (
		layout.raw_dir / "iirc" / "extracted" / "context_articles.json"
	)
	assert context_destination.exists()
	assert layout.source_manifest_path is not None
	manifest = json.loads(layout.source_manifest_path.read_text(encoding="utf-8"))
	names = {artifact["name"] for artifact in manifest["artifacts"]}
	assert "context_articles.json" in names


def test_fetch_iirc_dataset_uses_built_in_context_source_by_default(
	tmp_path, monkeypatch
):
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_archive = _write_iirc_context_archive(tmp_path)
	monkeypatch.setattr(
		"hypercorpus.datasets.fetch.IIRC_CONTEXT_ARTICLES_URL", context_archive.as_uri()
	)

	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched", archive_url=archive_path.as_uri()
	)

	extracted_context = layout.raw_dir / "iirc" / "extracted" / "context_articles.json"
	archived_context = layout.raw_dir / "iirc" / "archives" / "context_articles.tar.gz"
	assert extracted_context.exists()
	assert archived_context.exists()
	assert layout.source_manifest_path is not None
	manifest = json.loads(layout.source_manifest_path.read_text(encoding="utf-8"))
	by_name = {artifact["name"]: artifact for artifact in manifest["artifacts"]}
	assert by_name["context_archive"]["source_url"] == context_archive.as_uri()


def test_fetch_iirc_dataset_can_materialize_context_from_local_archive(tmp_path):
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_archive = _write_iirc_context_archive(tmp_path)

	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched",
		archive_url=archive_path.as_uri(),
		context_source=context_archive,
	)

	assert (layout.raw_dir / "iirc" / "extracted" / "context_articles.json").exists()
	assert (layout.raw_dir / "iirc" / "archives" / "context_articles.tar.gz").exists()
	assert layout.source_manifest_path is not None
	manifest = json.loads(layout.source_manifest_path.read_text(encoding="utf-8"))
	by_name = {artifact["name"]: artifact for artifact in manifest["artifacts"]}
	assert (
		by_name["context_archive"]["source_url"] == context_archive.resolve().as_uri()
	)


def test_fetch_iirc_dataset_can_materialize_context_from_url_archive(tmp_path):
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	context_archive = _write_iirc_context_archive(tmp_path)

	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched",
		archive_url=archive_path.as_uri(),
		context_source=context_archive.as_uri(),
	)

	assert (layout.raw_dir / "iirc" / "extracted" / "context_articles.json").exists()
	assert layout.source_manifest_path is not None
	manifest = json.loads(layout.source_manifest_path.read_text(encoding="utf-8"))
	by_name = {artifact["name"]: artifact for artifact in manifest["artifacts"]}
	assert by_name["context_archive"]["source_url"] == context_archive.as_uri()


def test_fetch_iirc_dataset_prefers_archive_embedded_context_over_built_in_source(
	iirc_raw_archive, tmp_path, monkeypatch
):
	def _fail_download(source_url: str, destination: Path) -> None:
		if source_url == iirc_raw_archive.as_uri():
			destination.write_bytes(iirc_raw_archive.read_bytes())
			return
		raise AssertionError(f"unexpected download: {source_url} -> {destination}")

	monkeypatch.setattr(
		"hypercorpus.datasets.fetch.IIRC_CONTEXT_ARTICLES_URL",
		"https://example.invalid/context_articles.tar.gz",
	)
	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched",
		archive_url=iirc_raw_archive.as_uri(),
		downloader=_fail_download,
	)

	assert (layout.raw_dir / "iirc" / "extracted" / "context_articles.json").exists()


def test_fetch_iirc_dataset_reports_built_in_context_download_failures(
	tmp_path, monkeypatch
):
	archive_path = _write_iirc_archive(tmp_path, include_context=False)
	monkeypatch.setattr(
		"hypercorpus.datasets.fetch.IIRC_CONTEXT_ARTICLES_URL",
		"https://example.invalid/context_articles.tar.gz",
	)

	def _fail_download(source_url: str, destination: Path) -> None:
		if source_url == archive_path.as_uri():
			destination.write_bytes(archive_path.read_bytes())
			return
		raise urllib.error.URLError("offline")

	with pytest.raises(ValueError, match="--context-source"):
		fetch_iirc_dataset(
			tmp_path / "iirc-fetched",
			archive_url=archive_path.as_uri(),
			downloader=_fail_download,
		)


def test_fetch_iirc_dataset_allows_question_only_when_requested(tmp_path):
	archive_path = _write_iirc_archive(tmp_path, include_context=False)

	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched",
		archive_url=archive_path.as_uri(),
		require_context=False,
	)

	assert (layout.raw_dir / "iirc" / "extracted" / "dev.json").exists()
	assert not (
		layout.raw_dir / "iirc" / "extracted" / "context_articles.json"
	).exists()


def test_fetch_iirc_dataset_ignores_appledouble_json_members(tmp_path):
	archive_path = _write_iirc_archive(
		tmp_path, include_context=True, include_appledouble=True
	)

	layout = fetch_iirc_dataset(
		tmp_path / "iirc-fetched", archive_url=archive_path.as_uri()
	)

	assert (layout.raw_dir / "iirc" / "extracted" / "dev.json").exists()
	assert not (layout.raw_dir / "iirc" / "extracted" / "._dev.json").exists()
	assert layout.source_manifest_path is not None
	manifest = json.loads(layout.source_manifest_path.read_text(encoding="utf-8"))
	assert "._dev.json" not in {artifact["name"] for artifact in manifest["artifacts"]}


def test_fetch_musique_dataset_downloads_requested_split_and_subset(
	musique_raw_split_files, tmp_path
):
	layout = fetch_musique_dataset(
		tmp_path / "musique-fetched",
		split="dev",
		subset="full",
		split_urls={
			name: path.as_uri() for name, path in musique_raw_split_files.items()
		},
	)

	assert set(layout.artifact_paths) == {"dev"}
	assert layout.artifact_paths["dev"].name == "dev.jsonl"
	assert (
		layout.source_manifest_path is not None and layout.source_manifest_path.exists()
	)


def test_fetch_hotpotqa_dataset_downloads_variant_questions(
	hotpotqa_raw_split_files, tmp_path
):
	layout = fetch_hotpotqa_dataset(
		tmp_path / "hotpot-fetched",
		variant="distractor",
		split="dev",
		split_urls={
			name: path.as_uri()
			for name, path in hotpotqa_raw_split_files["distractor"].items()
		},
	)

	assert set(layout.artifact_paths) == {"dev"}
	assert layout.artifact_paths["dev"].name == "dev.json"
	assert layout.source_manifest_path is not None
	manifest = json.loads(layout.source_manifest_path.read_text(encoding="utf-8"))
	assert manifest["dataset_name"] == "hotpotqa"
	assert manifest["variant"] == "distractor"
