from __future__ import annotations

import json
import hashlib
import logging
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from hypercorpus.logging import copy_stream_with_progress

TWOWIKI_QUESTIONS_URL = (
	"https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
)
TWOWIKI_GRAPH_URL = (
	"https://www.dropbox.com/s/wlhw26kik59wbh8/para_with_hyperlink.zip?dl=1"
)
TWOWIKI_GRAPH_BASENAME = "para_with_hyperlink.jsonl"
TWOWIKI_SPLITS = ("train", "dev", "test")
IIRC_ARCHIVE_URL = "https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz"
IIRC_CONTEXT_ARTICLES_URL = (
	"https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz"
)
MUSIQUE_FULL_SPLIT_URLS = {
	"train": "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_full_v1.0_train.jsonl",
	"dev": "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_full_v1.0_dev.jsonl",
	"test": "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_full_v1.0_test.jsonl",
}
MUSIQUE_ANSWERABLE_SPLIT_URLS = {
	"train": "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_ans_v1.0_train.jsonl",
	"dev": "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl",
	"test": "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_ans_v1.0_test.jsonl",
}
HOTPOTQA_DISTRACTOR_SPLIT_URLS = {
	"train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
	"dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}
HOTPOTQA_FULLWIKI_SPLIT_URLS = {
	"train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
	"dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
}

DownloadFn = Callable[[str, Path], None]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TwoWikiDatasetLayout:
	output_dir: Path
	question_paths: dict[str, Path] = field(default_factory=dict)
	graph_path: Path | None = None
	archive_paths: dict[str, Path] = field(default_factory=dict)
	source: str = "remote"


@dataclass(slots=True)
class RawDatasetLayout:
	output_dir: Path
	raw_dir: Path
	dataset_name: str
	variant: str | None = None
	artifact_paths: dict[str, Path] = field(default_factory=dict)
	source_manifest_path: Path | None = None


def fetch_2wiki_dataset(
	output_dir: str | Path,
	*,
	split: str = "dev",
	include_questions: bool = True,
	include_graph: bool = False,
	keep_archives: bool = False,
	overwrite: bool = False,
	questions_url: str | None = None,
	graph_url: str | None = None,
	downloader: DownloadFn = None,
) -> TwoWikiDatasetLayout:
	if not include_questions and not include_graph:
		raise ValueError(
			"At least one of include_questions or include_graph must be enabled."
		)

	logger.info(
		"Preparing 2Wiki dataset under %s (split=%s, questions=%s, graph=%s, keep_archives=%s, overwrite=%s)",
		output_dir,
		split,
		include_questions,
		include_graph,
		keep_archives,
		overwrite,
	)
	resolved_output = Path(output_dir)
	resolved_output.mkdir(parents=True, exist_ok=True)
	download_impl = downloader or _download_url
	selected_splits = _resolve_splits(split)
	resolved_questions_url = questions_url or TWOWIKI_QUESTIONS_URL
	resolved_graph_url = graph_url or TWOWIKI_GRAPH_URL
	layout = TwoWikiDatasetLayout(output_dir=resolved_output)

	if include_questions:
		questions_dir = resolved_output / "questions"
		questions_dir.mkdir(parents=True, exist_ok=True)
		required = {name: questions_dir / f"{name}.json" for name in selected_splits}
		missing = [
			name for name, path in required.items() if overwrite or not path.exists()
		]
		if missing:
			logger.info("Fetching 2Wiki questions for splits: %s", ", ".join(missing))
			archive_path = _prepare_archive(
				resolved_output,
				archive_name="2wiki-questions.zip",
				source_url=resolved_questions_url,
				keep_archives=keep_archives,
				overwrite=overwrite,
				downloader=download_impl,
			)
			if keep_archives:
				layout.archive_paths["questions"] = archive_path
			for name, destination in required.items():
				_extract_zip_member(
					archive_path,
					basename=f"{name}.json",
					destination=destination,
					overwrite=overwrite,
				)
			_cleanup_archive(archive_path, keep_archives=keep_archives)
		layout.question_paths = required

	if include_graph:
		graph_dir = resolved_output / "graph"
		graph_dir.mkdir(parents=True, exist_ok=True)
		destination = graph_dir / TWOWIKI_GRAPH_BASENAME
		if overwrite or not destination.exists():
			logger.info("Fetching shared 2Wiki graph archive into %s", destination)
			archive_path = _prepare_archive(
				resolved_output,
				archive_name="2wiki-graph.zip",
				source_url=resolved_graph_url,
				keep_archives=keep_archives,
				overwrite=overwrite,
				downloader=download_impl,
			)
			if keep_archives:
				layout.archive_paths["graph"] = archive_path
			_extract_zip_member(
				archive_path,
				basename=TWOWIKI_GRAPH_BASENAME,
				destination=destination,
				overwrite=overwrite,
			)
			_cleanup_archive(archive_path, keep_archives=keep_archives)
		layout.graph_path = destination

	logger.info(
		"Prepared 2Wiki dataset under %s (question_files=%s, graph=%s)",
		resolved_output,
		len(layout.question_paths),
		layout.graph_path is not None,
	)
	return layout


def fetch_iirc_dataset(
	output_dir: str | Path,
	*,
	archive_url: str | None = None,
	context_source: str | Path | None = None,
	require_context: bool = True,
	overwrite: bool = False,
	downloader: DownloadFn = None,
) -> RawDatasetLayout:
	resolved_output = Path(output_dir)
	raw_dir = resolved_output / "raw"
	dataset_dir = raw_dir / "iirc"
	archives_dir = dataset_dir / "archives"
	extracted_dir = dataset_dir / "extracted"
	archives_dir.mkdir(parents=True, exist_ok=True)
	extracted_dir.mkdir(parents=True, exist_ok=True)
	download_impl = downloader or _download_url
	source_url = archive_url or IIRC_ARCHIVE_URL
	archive_path = archives_dir / Path(urllib.parse.urlparse(source_url).path).name
	if overwrite or not archive_path.exists():
		logger.info("Fetching IIRC archive from %s into %s", source_url, archive_path)
		download_impl(source_url, archive_path)
	if overwrite:
		for path in extracted_dir.rglob("*"):
			if path.is_file():
				path.unlink()
	_extract_tgz_json_members(
		archive_path, destination_root=extracted_dir, overwrite=overwrite
	)
	context_destination = extracted_dir / "context_articles.json"
	context_source_ref: str | None = None
	context_artifact_name: str | None = None
	if not context_destination.exists():
		if context_source is not None or require_context:
			resolved_context_source = (
				context_source
				if context_source is not None
				else IIRC_CONTEXT_ARTICLES_URL
			)
			try:
				context_source_ref, context_artifact_name = (
					_materialize_iirc_context_source(
						resolved_context_source,
						archives_dir=archives_dir,
						destination=context_destination,
						overwrite=overwrite,
						downloader=download_impl,
					)
				)
			except (
				FileNotFoundError,
				tarfile.TarError,
				urllib.error.URLError,
				ValueError,
			) as exc:
				if context_source is not None:
					raise
				if require_context:
					raise ValueError(
						"IIRC archive did not contain context_articles.json and the built-in context corpus "
						f"download failed from {IIRC_CONTEXT_ARTICLES_URL}. Pass --context-source "
						"/path/or/url/to/context_articles.json or context_articles.tar.gz, or use "
						"--question-only to allow a partial fetch."
					) from exc
		if require_context and not context_destination.exists():
			raise ValueError(
				"IIRC archive did not contain context_articles.json. "
				"Pass --context-source /path/or/url/to/context_articles.json or context_articles.tar.gz, "
				"or use --question-only to allow a partial fetch."
			)
	artifacts = _collect_relative_files(extracted_dir)
	layout = RawDatasetLayout(
		output_dir=resolved_output,
		raw_dir=raw_dir,
		dataset_name="iirc",
		artifact_paths={path.stem: path for path in artifacts},
	)
	source_urls = {"archive": source_url}
	manifest_paths: dict[str, Path] = {
		"archive": archive_path,
		**{path.name: path for path in artifacts},
	}
	if context_source_ref is not None and context_artifact_name is not None:
		source_urls[context_artifact_name] = context_source_ref
		if context_artifact_name == "context_archive":
			manifest_paths["context_archive"] = archives_dir / "context_articles.tar.gz"
	layout.source_manifest_path = _write_source_manifest(
		raw_dir,
		dataset_name="iirc",
		variant=None,
		artifacts=_manifest_artifacts(
			manifest_paths,
			source_urls=source_urls,
		),
	)
	return layout


def fetch_musique_dataset(
	output_dir: str | Path,
	*,
	split: str = "dev",
	subset: str = "full",
	split_urls: dict[str, str] | None = None,
	overwrite: bool = False,
	downloader: DownloadFn = None,
) -> RawDatasetLayout:
	resolved_output = Path(output_dir)
	raw_dir = resolved_output / "raw"
	dataset_dir = raw_dir / "musique" / subset
	questions_dir = dataset_dir / "questions"
	questions_dir.mkdir(parents=True, exist_ok=True)
	selected_splits = _resolve_standard_splits(split)
	source_urls = split_urls or _resolve_musique_split_urls(subset)
	download_impl = downloader or _download_url
	artifact_paths: dict[str, Path] = {}
	artifact_sources: dict[str, str] = {}
	for split_name in selected_splits:
		try:
			source_url = source_urls[split_name]
		except KeyError as exc:
			raise ValueError(
				f"Unsupported MuSiQue split '{split_name}' for subset '{subset}'"
			) from exc
		destination = questions_dir / f"{split_name}.jsonl"
		if overwrite or not destination.exists():
			logger.info(
				"Fetching MuSiQue %s/%s questions from %s",
				subset,
				split_name,
				source_url,
			)
			download_impl(source_url, destination)
		artifact_paths[split_name] = destination
		artifact_sources[split_name] = source_url
	layout = RawDatasetLayout(
		output_dir=resolved_output,
		raw_dir=raw_dir,
		dataset_name="musique",
		variant=subset,
		artifact_paths=artifact_paths,
	)
	layout.source_manifest_path = _write_source_manifest(
		raw_dir,
		dataset_name="musique",
		variant=subset,
		artifacts=_manifest_artifacts(artifact_paths, source_urls=artifact_sources),
	)
	return layout


def fetch_hotpotqa_dataset(
	output_dir: str | Path,
	*,
	variant: str = "distractor",
	split: str = "dev",
	split_urls: dict[str, str] | None = None,
	overwrite: bool = False,
	downloader: DownloadFn = None,
) -> RawDatasetLayout:
	resolved_output = Path(output_dir)
	raw_dir = resolved_output / "raw"
	dataset_dir = raw_dir / "hotpotqa" / variant
	questions_dir = dataset_dir / "questions"
	questions_dir.mkdir(parents=True, exist_ok=True)
	selected_splits = _resolve_standard_splits(split)
	source_urls = split_urls or _resolve_hotpotqa_split_urls(variant)
	download_impl = downloader or _download_url
	artifact_paths: dict[str, Path] = {}
	artifact_sources: dict[str, str] = {}
	for split_name in selected_splits:
		try:
			source_url = source_urls[split_name]
		except KeyError as exc:
			raise ValueError(
				f"Unsupported HotpotQA split '{split_name}' for variant '{variant}'"
			) from exc
		destination = questions_dir / f"{split_name}.json"
		if overwrite or not destination.exists():
			logger.info(
				"Fetching HotpotQA %s/%s questions from %s",
				variant,
				split_name,
				source_url,
			)
			download_impl(source_url, destination)
		artifact_paths[split_name] = destination
		artifact_sources[split_name] = source_url
	layout = RawDatasetLayout(
		output_dir=resolved_output,
		raw_dir=raw_dir,
		dataset_name="hotpotqa",
		variant=variant,
		artifact_paths=artifact_paths,
	)
	layout.source_manifest_path = _write_source_manifest(
		raw_dir,
		dataset_name="hotpotqa",
		variant=variant,
		artifacts=_manifest_artifacts(artifact_paths, source_urls=artifact_sources),
	)
	return layout


def write_2wiki_sample_dataset(
	output_dir: str | Path,
	*,
	overwrite: bool = False,
) -> TwoWikiDatasetLayout:
	logger.info("Writing bundled 2Wiki sample dataset into %s", output_dir)
	resolved_output = Path(output_dir)
	resolved_output.mkdir(parents=True, exist_ok=True)
	questions_dir = resolved_output / "questions"
	graph_dir = resolved_output / "graph"
	questions_dir.mkdir(parents=True, exist_ok=True)
	graph_dir.mkdir(parents=True, exist_ok=True)

	questions_path = questions_dir / "dev.json"
	graph_path = graph_dir / TWOWIKI_GRAPH_BASENAME

	if overwrite or not questions_path.exists():
		questions_path.write_text(
			json.dumps(_sample_questions(), ensure_ascii=False, indent=2),
			encoding="utf-8",
		)
	if overwrite or not graph_path.exists():
		graph_path.write_text(
			"\n".join(
				json.dumps(record, ensure_ascii=False)
				for record in _sample_graph_records()
			)
			+ "\n",
			encoding="utf-8",
		)

	return TwoWikiDatasetLayout(
		output_dir=resolved_output,
		question_paths={"dev": questions_path},
		graph_path=graph_path,
		source="sample",
	)


def _prepare_archive(
	output_dir: Path,
	*,
	archive_name: str,
	source_url: str,
	keep_archives: bool,
	overwrite: bool,
	downloader: DownloadFn,
) -> Path:
	if keep_archives:
		archive_dir = output_dir / "archives"
		archive_dir.mkdir(parents=True, exist_ok=True)
		archive_path = archive_dir / archive_name
		if overwrite or not archive_path.exists():
			logger.info("Downloading archive %s from %s", archive_name, source_url)
			downloader(source_url, archive_path)
		return archive_path

	with tempfile.NamedTemporaryFile(
		prefix="hypercorpus-", suffix=f"-{archive_name}", delete=False
	) as handle:
		archive_path = Path(handle.name)
	logger.info("Downloading temporary archive %s from %s", archive_name, source_url)
	downloader(source_url, archive_path)
	return archive_path


def _cleanup_archive(archive_path: Path, *, keep_archives: bool) -> None:
	if keep_archives:
		return
	archive_path.unlink(missing_ok=True)


def _extract_zip_member(
	archive_path: Path,
	*,
	basename: str,
	destination: Path,
	overwrite: bool,
) -> None:
	if destination.exists() and not overwrite:
		return

	with zipfile.ZipFile(archive_path) as archive:
		member_name = _find_member_by_basename(archive.namelist(), basename)
		if member_name is None:
			raise FileNotFoundError(
				f"Archive {archive_path} does not contain {basename}"
			)
		info = archive.getinfo(member_name)
		logger.info("Extracting %s from %s to %s", basename, archive_path, destination)
		destination.parent.mkdir(parents=True, exist_ok=True)
		with archive.open(member_name) as src, destination.open("wb") as dst:
			copy_stream_with_progress(
				src,
				dst,
				description=f"extract {basename}",
				total=info.file_size,
			)


def _extract_tgz_json_members(
	archive_path: Path, *, destination_root: Path, overwrite: bool
) -> None:
	with tarfile.open(archive_path, "r:*") as archive:
		for member in archive.getmembers():
			if not member.isfile():
				continue
			member_basename = Path(member.name).name
			if member_basename.startswith("._"):
				continue
			suffixes = [suffix.lower() for suffix in Path(member.name).suffixes]
			if not suffixes or suffixes[-1] not in {".json", ".jsonl"}:
				continue
			destination = destination_root / member_basename
			if destination.exists() and not overwrite:
				continue
			destination.parent.mkdir(parents=True, exist_ok=True)
			extracted = archive.extractfile(member)
			if extracted is None:
				continue
			with extracted as src, destination.open("wb") as dst:
				copy_stream_with_progress(
					src,
					dst,
					description=f"extract {member_basename}",
					total=member.size,
				)


def _extract_tgz_member_by_basename(
	archive_path: Path,
	*,
	basename: str,
	destination: Path,
	overwrite: bool,
) -> None:
	if destination.exists() and not overwrite:
		return

	with tarfile.open(archive_path, "r:*") as archive:
		member = next(
			(
				candidate
				for candidate in archive.getmembers()
				if candidate.isfile() and Path(candidate.name).name == basename
			),
			None,
		)
		if member is None:
			raise FileNotFoundError(
				f"Archive {archive_path} does not contain {basename}"
			)
		destination.parent.mkdir(parents=True, exist_ok=True)
		extracted = archive.extractfile(member)
		if extracted is None:
			raise FileNotFoundError(
				f"Archive {archive_path} member {basename} could not be extracted"
			)
		with extracted as src, destination.open("wb") as dst:
			copy_stream_with_progress(
				src,
				dst,
				description=f"extract {basename}",
				total=member.size,
			)


def _find_member_by_basename(names: list[str], basename: str) -> str | None:
	for name in names:
		if Path(name).name == basename:
			return name
	return None


def _resolve_splits(split: str) -> tuple[str, ...]:
	normalized = split.strip().lower()
	if normalized == "all":
		return TWOWIKI_SPLITS
	if normalized not in TWOWIKI_SPLITS:
		valid = ", ".join([*TWOWIKI_SPLITS, "all"])
		raise ValueError(f"Unsupported 2Wiki split '{split}'. Expected one of: {valid}")
	return (normalized,)


def _resolve_standard_splits(split: str) -> tuple[str, ...]:
	normalized = split.strip().lower()
	if normalized == "all":
		return ("train", "dev", "test")
	if normalized not in {"train", "dev", "test"}:
		raise ValueError(
			f"Unsupported split '{split}'. Expected one of: train, dev, test, all"
		)
	return (normalized,)


def _resolve_musique_split_urls(subset: str) -> dict[str, str]:
	normalized = subset.strip().lower()
	if normalized == "full":
		return MUSIQUE_FULL_SPLIT_URLS
	if normalized in {"ans", "answerable"}:
		return MUSIQUE_ANSWERABLE_SPLIT_URLS
	raise ValueError("MuSiQue subset must be one of: full, ans")


def _resolve_hotpotqa_split_urls(variant: str) -> dict[str, str]:
	normalized = variant.strip().lower()
	if normalized == "distractor":
		return HOTPOTQA_DISTRACTOR_SPLIT_URLS
	if normalized == "fullwiki":
		return HOTPOTQA_FULLWIKI_SPLIT_URLS
	raise ValueError("HotpotQA variant must be one of: distractor, fullwiki")


def _download_url(source_url: str, destination: Path) -> None:
	normalized_url = _normalize_dropbox_url(source_url)
	destination.parent.mkdir(parents=True, exist_ok=True)
	parsed = urllib.parse.urlparse(normalized_url)
	request_or_url: str | urllib.request.Request
	expected_bytes: int | None = None
	if "drive.google.com" in parsed.netloc:
		_download_google_drive(normalized_url, destination)
		return
	if parsed.scheme == "file":
		request_or_url = normalized_url
		file_path = Path(parsed.path)
		if file_path.exists():
			expected_bytes = file_path.stat().st_size
	else:
		request_or_url = urllib.request.Request(
			normalized_url,
			headers={
				"User-Agent": "hypercorpus/0.1 (+https://github.com/Acture/hypercorpus)"
			},
		)
	with (
		urllib.request.urlopen(request_or_url) as response,
		destination.open("wb") as handle,
	):
		content_length = response.headers.get("Content-Length")
		if content_length:
			expected_bytes = int(content_length)
		if expected_bytes is not None:
			size_mb = expected_bytes / (1024 * 1024)
			logger.info(
				"Downloading %s (%.1f MB) to %s", normalized_url, size_mb, destination
			)
		else:
			logger.info(
				"Downloading %s (unknown size) to %s", normalized_url, destination
			)
		copy_stream_with_progress(
			response,
			handle,
			description=f"download {destination.name}",
			total=expected_bytes,
		)


def _download_google_drive(source_url: str, destination: Path) -> None:
	import http.cookiejar
	import re

	destination.parent.mkdir(parents=True, exist_ok=True)
	opener = urllib.request.build_opener(
		urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
	)
	request = urllib.request.Request(
		source_url, headers={"User-Agent": "hypercorpus/0.1"}
	)
	with opener.open(request) as response:
		body = response.read()
		content_type = response.headers.get("Content-Type", "")
		if "text/html" in content_type:
			text = body.decode("utf-8", errors="ignore")
			confirm_match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
			if confirm_match:
				parsed = urllib.parse.urlparse(source_url)
				query = urllib.parse.parse_qs(parsed.query)
				query["confirm"] = [confirm_match.group(1)]
				confirmed_url = urllib.parse.urlunparse(
					parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
				)
				request = urllib.request.Request(
					confirmed_url, headers={"User-Agent": "hypercorpus/0.1"}
				)
				with (
					opener.open(request) as confirmed_response,
					destination.open("wb") as handle,
				):
					content_length = confirmed_response.headers.get("Content-Length")
					total = int(content_length) if content_length else None
					copy_stream_with_progress(
						confirmed_response,
						handle,
						description=f"download {destination.name}",
						total=total,
					)
				return
		with destination.open("wb") as handle:
			handle.write(body)


def _materialize_iirc_context_source(
	source: str | Path,
	*,
	archives_dir: Path,
	destination: Path,
	overwrite: bool,
	downloader: DownloadFn,
) -> tuple[str, str]:
	if destination.exists() and not overwrite:
		return destination.resolve().as_uri(), "context_articles.json"

	source_path = (
		source if isinstance(source, Path) else Path(urllib.parse.urlparse(source).path)
	)
	suffixes = [suffix.lower() for suffix in source_path.suffixes]
	is_archive = len(suffixes) >= 2 and suffixes[-2:] == [".tar", ".gz"]

	if isinstance(source, Path):
		resolved_source = source.expanduser().resolve()
		if not resolved_source.exists():
			raise FileNotFoundError(
				f"IIRC context source does not exist: {resolved_source}"
			)
		if is_archive:
			archive_destination = archives_dir / "context_articles.tar.gz"
			archive_destination.parent.mkdir(parents=True, exist_ok=True)
			if overwrite or not archive_destination.exists():
				shutil.copy2(resolved_source, archive_destination)
			_extract_tgz_member_by_basename(
				archive_destination,
				basename="context_articles.json",
				destination=destination,
				overwrite=overwrite,
			)
			return resolved_source.as_uri(), "context_archive"
		destination.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(resolved_source, destination)
		return resolved_source.as_uri(), "context_articles.json"

	parsed = urllib.parse.urlparse(source)
	if parsed.scheme:
		if is_archive:
			archive_destination = archives_dir / "context_articles.tar.gz"
			downloader(source, archive_destination)
			_extract_tgz_member_by_basename(
				archive_destination,
				basename="context_articles.json",
				destination=destination,
				overwrite=overwrite,
			)
			return source, "context_archive"
		downloader(source, destination)
		return source, "context_articles.json"

	resolved_source = Path(source).expanduser().resolve()
	if not resolved_source.exists():
		raise FileNotFoundError(
			f"IIRC context source does not exist: {resolved_source}"
		)
	if is_archive:
		archive_destination = archives_dir / "context_articles.tar.gz"
		archive_destination.parent.mkdir(parents=True, exist_ok=True)
		if overwrite or not archive_destination.exists():
			shutil.copy2(resolved_source, archive_destination)
		_extract_tgz_member_by_basename(
			archive_destination,
			basename="context_articles.json",
			destination=destination,
			overwrite=overwrite,
		)
		return resolved_source.as_uri(), "context_archive"
	destination.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(resolved_source, destination)
	return resolved_source.as_uri(), "context_articles.json"


def _normalize_dropbox_url(source_url: str) -> str:
	parsed = urllib.parse.urlparse(source_url)
	if "dropbox.com" not in parsed.netloc:
		return source_url

	query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
	query["dl"] = ["1"]
	return urllib.parse.urlunparse(
		parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
	)


def _write_source_manifest(
	raw_dir: Path,
	*,
	dataset_name: str,
	variant: str | None,
	artifacts: Iterable[dict[str, Any]],
) -> Path:
	manifest_path = raw_dir / "source-manifest.json"
	payload = {
		"dataset_name": dataset_name,
		"variant": variant,
		"artifacts": list(artifacts),
	}
	manifest_path.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
	)
	return manifest_path


def _manifest_artifacts(
	artifact_paths: dict[str, Path],
	*,
	source_urls: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
	artifacts: list[dict[str, Any]] = []
	for name, path in sorted(artifact_paths.items()):
		artifacts.append(
			{
				"name": name,
				"source_url": source_urls.get(name)
				if source_urls is not None
				else None,
				"path": str(path),
				"size_bytes": path.stat().st_size if path.exists() else None,
				"sha256": _sha256_path(path) if path.exists() else None,
			}
		)
	return artifacts


def _collect_relative_files(root: Path) -> list[Path]:
	return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256_path(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _sample_graph_records() -> list[dict[str, object]]:
	return [
		{
			"id": "100",
			"title": "Moon Launch Program",
			"sentences": [
				"Moon Launch Program uses Cape Canaveral as its launch site.",
				"The program was directed by Alice Johnson.",
			],
			"mentions": [
				{
					"id": 0,
					"start": 25,
					"end": 40,
					"ref_url": "Cape_Canaveral",
					"ref_ids": ["200"],
					"sent_idx": 0,
				},
				{
					"id": 1,
					"start": 28,
					"end": 41,
					"ref_url": "Alice_Johnson",
					"ref_ids": ["300"],
					"sent_idx": 1,
				},
			],
		},
		{
			"id": "200",
			"title": "Cape Canaveral",
			"sentences": ["Cape Canaveral is a city in Florida."],
			"mentions": [
				{
					"id": 0,
					"start": 29,
					"end": 36,
					"ref_url": "Florida",
					"ref_ids": ["400"],
					"sent_idx": 0,
				},
			],
		},
		{
			"id": "300",
			"title": "Alice Johnson",
			"sentences": ["Alice Johnson directed the Moon Launch Program in 1969."],
			"mentions": [],
		},
		{
			"id": "400",
			"title": "Florida",
			"sentences": ["Florida is a state in the southeastern United States."],
			"mentions": [],
		},
	]


def _sample_questions() -> list[dict[str, object]]:
	return [
		{
			"_id": "q1",
			"question": "Which city hosts the launch site?",
			"answer": "Cape Canaveral",
			"supporting_facts": [
				["Moon Launch Program", 0],
				["Cape Canaveral", 0],
			],
		},
		{
			"_id": "q2",
			"question": "Who directed the Moon Launch Program?",
			"answer": "Alice Johnson",
			"supporting_facts": [
				["Moon Launch Program", 1],
				["Alice Johnson", 0],
			],
		},
	]
