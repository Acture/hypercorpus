from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .fetch import TWOWIKI_GRAPH_URL, TWOWIKI_QUESTIONS_URL
from .store import (
	DEFAULT_MIN_FREE_GIB,
	DEFAULT_SHARD_TARGET_SIZE_BYTES,
	DatasetStoreManifest,
	LocalDirectoryStore,
	ObjectStore,
	PreparedDatasetStore,
	PreparedSource,
	S3CompatibleObjectStore,
	ShardedDocumentStore,
	StoreObjectInfo,
	_directory_size_bytes,
	_download_url,
	_download_to_temp,
	_ensure_min_free_space,
	_estimate_prepare_bytes,
	_looks_like_url,
	_materialize_source,
	_probe_remote_size,
	_write_questions,
	ensure_min_free_space,
	estimate_prepare_bytes,
	open_object_store,
	prepare_store_from_records,
	probe_source_size,
)
from .twowiki import (
	TWOWIKI_DEFAULT_GRAPH_PATH,
	TWOWIKI_DEFAULT_QUESTIONS_DIR,
	iter_2wiki_graph_records,
	normalize_2wiki_graph_record,
	normalize_2wiki_title,
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hypercorpus" / "2wiki"

logger = logging.getLogger(__name__)

PreparedTwoWikiStore = PreparedDatasetStore
TwoWikiStoreManifest = DatasetStoreManifest


class ShardedLinkContextStore(ShardedDocumentStore):
	def __init__(
		self,
		store_uri: str | Path,
		*,
		cache_dir: str | Path | None = None,
		object_store: ObjectStore | None = None,
	):
		super().__init__(
			store_uri,
			cache_dir=cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR,
			object_store=object_store,
		)


@dataclass(slots=True)
class TwoWikiStoreInspection:
	raw_questions_path: Path | None
	raw_graph_path: Path | None
	raw_graph_size_bytes: int | None
	store_manifest_path: Path | None
	cache_dir: Path
	cache_size_bytes: int
	free_space_bytes: int
	remote_questions_size_bytes: int | None
	remote_graph_size_bytes: int | None
	recommended_action: str


def prepare_2wiki_store(
	output_dir: str | Path,
	*,
	questions_source: str | Path | None = None,
	graph_source: str | Path | None = None,
	target_shard_size_bytes: int = DEFAULT_SHARD_TARGET_SIZE_BYTES,
	keep_raw: bool = False,
	overwrite: bool = False,
	min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> PreparedTwoWikiStore:
	output_root = Path(output_dir)
	logger.info(
		"Preparing sharded 2Wiki store under %s (questions_source=%s, graph_source=%s, keep_raw=%s, overwrite=%s)",
		output_root,
		questions_source or "auto",
		graph_source or "auto",
		keep_raw,
		overwrite,
	)
	if overwrite and output_root.exists():
		shutil.rmtree(output_root)
	elif _has_existing_store_state(output_root):
		raise RuntimeError(
			f"Output directory already contains a prepared or partial 2Wiki store: {output_root}. "
			"Re-run with --overwrite or choose a different --output-dir."
		)
	output_root.mkdir(parents=True, exist_ok=True)
	(output_root / "questions").mkdir(parents=True, exist_ok=True)
	(output_root / "index").mkdir(parents=True, exist_ok=True)
	(output_root / "shards").mkdir(parents=True, exist_ok=True)

	resolved_questions_source = resolve_2wiki_questions_source(questions_source)
	resolved_graph_source = resolve_2wiki_graph_source(graph_source)
	logger.info(
		"Resolved 2Wiki sources: questions=%s, graph=%s",
		resolved_questions_source.source,
		resolved_graph_source.source,
	)
	_ensure_min_free_space(
		output_root,
		min_free_gib=min_free_gib,
		expected_new_bytes=_estimate_prepare_bytes(resolved_graph_source.size_bytes),
	)

	temp_paths: list[Path] = []
	try:
		questions_local = _materialize_questions_source(
			resolved_questions_source, temp_paths=temp_paths
		)
		graph_local = _materialize_graph_source(
			resolved_graph_source,
			keep_raw=keep_raw,
			output_root=output_root,
			temp_paths=temp_paths,
		)
		questions_paths = _write_questions(output_root / "questions", questions_local)

		logger.info("Phase 1/2: indexing 2Wiki titles from %s", graph_local)
		id_to_title: dict[str, str] = {}
		seen_titles: set[str] = set()
		duplicate_titles: list[str] = []
		for record in iter_2wiki_graph_records(graph_local):
			title = normalize_2wiki_title(record.get("title", ""))
			if title in seen_titles:
				duplicate_titles.append(title)
			else:
				seen_titles.add(title)
			id_to_title[str(record.get("id"))] = title
		if duplicate_titles:
			unique_duplicates = list(dict.fromkeys(duplicate_titles))
			preview = ", ".join(unique_duplicates[:5])
			logger.warning(
				"Skipped %s duplicate 2Wiki node title(s) while preparing the store. Examples: %s",
				len(unique_duplicates),
				preview,
			)
		logger.info("Phase 2/2: writing catalog and shards into %s", output_root)
		prepared = prepare_store_from_records(
			output_root=output_root,
			dataset_name="2wikimultihop",
			normalized_records=_iter_two_wiki_normalized_records(
				graph_local, id_to_title=id_to_title
			),
			target_shard_size_bytes=target_shard_size_bytes,
			questions_source=resolved_questions_source.source,
			graph_source=resolved_graph_source.source,
			questions_paths=questions_paths,
		)
		logger.info(
			"Prepared sharded 2Wiki store under %s (documents=%s, shards=%s, total_tokens=%s)",
			output_root,
			prepared.manifest.total_document_count,
			prepared.manifest.shard_count,
			prepared.manifest.total_token_estimate,
		)
		return prepared
	finally:
		for path in temp_paths:
			path.unlink(missing_ok=True)


def inspect_2wiki_store(
	*,
	store_uri: str | Path | None = None,
	cache_dir: str | Path | None = None,
	raw_root: str | Path | None = None,
	questions_url: str = TWOWIKI_QUESTIONS_URL,
	graph_url: str = TWOWIKI_GRAPH_URL,
) -> TwoWikiStoreInspection:
	raw_root_path = (
		Path(raw_root) if raw_root is not None else TWOWIKI_DEFAULT_GRAPH_PATH.parent
	)
	raw_questions_path = raw_root_path / "data_ids_april7" / "dev.json"
	raw_graph_path = raw_root_path / "para_with_hyperlink.jsonl"
	cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
	cache_root.mkdir(parents=True, exist_ok=True)
	free_space_bytes = shutil.disk_usage(cache_root).free
	cache_size_bytes = _directory_size_bytes(cache_root)

	manifest_path: Path | None = None
	if store_uri is not None and not str(store_uri).startswith("s3://"):
		store_root = Path(store_uri)
		candidate = store_root / "manifest.json"
		if candidate.exists():
			manifest_path = candidate

	raw_graph_size = raw_graph_path.stat().st_size if raw_graph_path.exists() else None
	remote_questions_size = _probe_remote_size(questions_url)
	remote_graph_size = _probe_remote_size(graph_url)

	recommended_action = "prepare-store"
	if raw_questions_path.exists() and raw_graph_path.exists():
		recommended_action = "use-local-raw"
	if free_space_bytes < int(DEFAULT_MIN_FREE_GIB * 1024**3):
		recommended_action = "cleanup-first"
	if (
		store_uri is None
		and not raw_questions_path.exists()
		and not raw_graph_path.exists()
	):
		recommended_action = "safe-to-fetch"

	return TwoWikiStoreInspection(
		raw_questions_path=raw_questions_path if raw_questions_path.exists() else None,
		raw_graph_path=raw_graph_path if raw_graph_path.exists() else None,
		raw_graph_size_bytes=raw_graph_size,
		store_manifest_path=manifest_path,
		cache_dir=cache_root,
		cache_size_bytes=cache_size_bytes,
		free_space_bytes=free_space_bytes,
		remote_questions_size_bytes=remote_questions_size,
		remote_graph_size_bytes=remote_graph_size,
		recommended_action=recommended_action,
	)


def resolve_2wiki_questions_source(source: str | Path | None) -> PreparedSource:
	if source is None:
		local_dir = TWOWIKI_DEFAULT_QUESTIONS_DIR
		if local_dir.exists():
			return PreparedSource(
				source=str(local_dir),
				local_path=local_dir,
				size_bytes=_directory_size_bytes(local_dir),
			)
		return PreparedSource(
			source=TWOWIKI_QUESTIONS_URL,
			local_path=None,
			size_bytes=_probe_remote_size(TWOWIKI_QUESTIONS_URL),
		)
	path = None if _looks_like_url(source) else Path(source)
	if path is not None and path.exists():
		size = _directory_size_bytes(path) if path.is_dir() else path.stat().st_size
		return PreparedSource(source=str(path), local_path=path, size_bytes=size)
	return PreparedSource(
		source=str(source), local_path=None, size_bytes=_probe_remote_size(str(source))
	)


def resolve_2wiki_graph_source(source: str | Path | None) -> PreparedSource:
	if source is None:
		if TWOWIKI_DEFAULT_GRAPH_PATH.exists():
			return PreparedSource(
				source=str(TWOWIKI_DEFAULT_GRAPH_PATH),
				local_path=TWOWIKI_DEFAULT_GRAPH_PATH,
				size_bytes=TWOWIKI_DEFAULT_GRAPH_PATH.stat().st_size,
			)
		return PreparedSource(
			source=TWOWIKI_GRAPH_URL,
			local_path=None,
			size_bytes=_probe_remote_size(TWOWIKI_GRAPH_URL),
		)
	path = None if _looks_like_url(source) else Path(source)
	if path is not None and path.exists():
		return PreparedSource(
			source=str(path), local_path=path, size_bytes=path.stat().st_size
		)
	return PreparedSource(
		source=str(source), local_path=None, size_bytes=_probe_remote_size(str(source))
	)


def _materialize_questions_source(
	source: PreparedSource, *, temp_paths: list[Path]
) -> Path:
	return _materialize_source(
		source,
		keep_raw=False,
		output_root=Path("."),
		temp_paths=temp_paths,
		label="2Wiki questions",
	)


def _materialize_graph_source(
	source: PreparedSource,
	*,
	keep_raw: bool,
	output_root: Path,
	temp_paths: list[Path],
) -> Path:
	return _materialize_source(
		source,
		keep_raw=keep_raw,
		output_root=output_root,
		temp_paths=temp_paths,
		label="2Wiki graph",
	)


def _iter_two_wiki_normalized_records(
	path: Path, *, id_to_title: dict[str, str]
) -> Iterator[dict]:
	for raw_record in iter_2wiki_graph_records(path):
		yield normalize_2wiki_graph_record(raw_record, id_to_title=id_to_title)


def _has_existing_store_state(path: Path) -> bool:
	if not path.exists():
		return False
	manifest = path / "manifest.json"
	catalog = path / "index" / "catalog.sqlite"
	shards_dir = path / "shards"
	questions_dir = path / "questions"
	return (
		manifest.exists()
		or catalog.exists()
		or (shards_dir.exists() and any(shards_dir.iterdir()))
		or (questions_dir.exists() and any(questions_dir.glob("*.json*")))
	)
