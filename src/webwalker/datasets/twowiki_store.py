from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from webwalker.graph import DocumentNode, LinkContext
from webwalker.logging import copy_stream_with_progress, create_progress, should_render_progress
from webwalker.text import approx_token_count, normalized_token_overlap

from .fetch import TWOWIKI_GRAPH_URL, TWOWIKI_QUESTIONS_URL
from .twowiki import (
    TWOWIKI_DEFAULT_GRAPH_PATH,
    TWOWIKI_DEFAULT_QUESTIONS_DIR,
    iter_2wiki_graph_records,
    load_2wiki_questions,
    normalize_2wiki_graph_record,
    normalize_2wiki_title,
)

DEFAULT_SHARD_TARGET_SIZE_BYTES = 128 * 1024 * 1024
DEFAULT_MIN_FREE_GIB = 25.0
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "webwalker" / "2wiki"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StoreObjectInfo:
    path: str
    size_bytes: int | None
    etag: str | None = None


class ObjectStore(Protocol):
    uri: str

    def exists(self, relative_path: str) -> bool:
        ...

    def stat(self, relative_path: str) -> StoreObjectInfo | None:
        ...

    def ensure_local(self, relative_path: str, local_path: Path) -> Path:
        ...


@dataclass(slots=True)
class LocalDirectoryStore:
    root: Path

    @property
    def uri(self) -> str:
        return str(self.root)

    def exists(self, relative_path: str) -> bool:
        return (self.root / relative_path).exists()

    def stat(self, relative_path: str) -> StoreObjectInfo | None:
        path = self.root / relative_path
        if not path.exists():
            return None
        return StoreObjectInfo(path=relative_path, size_bytes=path.stat().st_size)

    def ensure_local(self, relative_path: str, local_path: Path) -> Path:
        source = self.root / relative_path
        if not source.exists():
            raise FileNotFoundError(f"Object store entry missing: {relative_path}")
        return source


@dataclass(slots=True)
class S3CompatibleObjectStore:
    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None
    region_name: str | None = None

    @property
    def uri(self) -> str:
        if self.prefix:
            return f"s3://{self.bucket}/{self.prefix}"
        return f"s3://{self.bucket}"

    def _client(self) -> Any:
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "boto3 is required to use s3:// dataset stores. Install boto3 or use a local path."
            ) from exc
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url or os.getenv("AWS_ENDPOINT_URL") or os.getenv("WEBWALKER_S3_ENDPOINT_URL"),
            region_name=self.region_name or os.getenv("AWS_DEFAULT_REGION"),
        )

    def _key(self, relative_path: str) -> str:
        relative = relative_path.lstrip("/")
        prefix = self.prefix.strip("/")
        if prefix:
            return f"{prefix}/{relative}"
        return relative

    def exists(self, relative_path: str) -> bool:
        return self.stat(relative_path) is not None

    def stat(self, relative_path: str) -> StoreObjectInfo | None:
        client = self._client()
        key = self._key(relative_path)
        try:
            response = client.head_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        return StoreObjectInfo(
            path=relative_path,
            size_bytes=int(response.get("ContentLength", 0)) or None,
            etag=str(response.get("ETag", "")).strip('"') or None,
        )

    def ensure_local(self, relative_path: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            return local_path
        client = self._client()
        logger.info("Downloading dataset store object %s to local cache %s", relative_path, local_path)
        client.download_file(self.bucket, self._key(relative_path), str(local_path))
        return local_path


@dataclass(slots=True)
class ShardInfo:
    path: str
    record_count: int
    uncompressed_bytes: int
    compressed_bytes: int
    sha256: str


@dataclass(slots=True)
class TwoWikiStoreManifest:
    dataset_name: str
    version: str
    shard_count: int
    target_shard_size_bytes: int
    total_document_count: int
    total_token_estimate: int
    questions_source: str
    graph_source: str
    questions_files: list[str]
    shards: list[ShardInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TwoWikiStoreManifest":
        return cls(
            dataset_name=str(payload["dataset_name"]),
            version=str(payload["version"]),
            shard_count=int(payload["shard_count"]),
            target_shard_size_bytes=int(payload["target_shard_size_bytes"]),
            total_document_count=int(payload["total_document_count"]),
            total_token_estimate=int(payload["total_token_estimate"]),
            questions_source=str(payload["questions_source"]),
            graph_source=str(payload["graph_source"]),
            questions_files=[str(item) for item in payload.get("questions_files", [])],
            shards=[
                ShardInfo(
                    path=str(item["path"]),
                    record_count=int(item["record_count"]),
                    uncompressed_bytes=int(item["uncompressed_bytes"]),
                    compressed_bytes=int(item["compressed_bytes"]),
                    sha256=str(item["sha256"]),
                )
                for item in payload.get("shards", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "shard_count": self.shard_count,
            "target_shard_size_bytes": self.target_shard_size_bytes,
            "total_document_count": self.total_document_count,
            "total_token_estimate": self.total_token_estimate,
            "questions_source": self.questions_source,
            "graph_source": self.graph_source,
            "questions_files": list(self.questions_files),
            "shards": [
                {
                    "path": shard.path,
                    "record_count": shard.record_count,
                    "uncompressed_bytes": shard.uncompressed_bytes,
                    "compressed_bytes": shard.compressed_bytes,
                    "sha256": shard.sha256,
                }
                for shard in self.shards
            ],
        }


@dataclass(slots=True)
class PreparedTwoWikiStore:
    root: Path
    manifest_path: Path
    catalog_path: Path
    questions_paths: dict[str, Path]
    shard_paths: list[Path]
    manifest: TwoWikiStoreManifest


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


def open_object_store(store_uri: str | Path) -> ObjectStore:
    if isinstance(store_uri, Path):
        return LocalDirectoryStore(store_uri)
    raw = str(store_uri)
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme == "s3":
        return S3CompatibleObjectStore(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    return LocalDirectoryStore(Path(raw))


class CatalogNodeAttrView:
    def __init__(self, store: "ShardedLinkContextStore"):
        self.store = store

    def get(self, node_id: str, default: Any = None) -> dict[str, Any] | Any:
        row = self.store._document_row(node_id)
        if row is None:
            return default
        return {
            "title": row["title"],
            "text": row["text"],
            "url": row["url"],
            "token_estimate": row["token_estimate"],
            "dataset": "2wikimultihop",
        }


class ShardedLinkContextStore:
    def __init__(
        self,
        store_uri: str | Path,
        *,
        cache_dir: str | Path | None = None,
        object_store: ObjectStore | None = None,
    ):
        self.store = object_store or open_object_store(store_uri)
        logger.info("Opening sharded 2Wiki store from %s", self.store.uri)
        manifest_path = self._ensure_metadata_file("manifest.json", cache_dir=cache_dir)
        self.manifest = TwoWikiStoreManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        self.cache_root = self._resolve_cache_root(cache_dir)
        self.catalog_path = self._ensure_metadata_file("index/catalog.sqlite", cache_dir=cache_dir)
        self._nodes = self._load_nodes()
        self.node_attr = CatalogNodeAttrView(self)
        self._document_cache: dict[str, DocumentNode] = {}
        self._document_row_cache: dict[str, dict[str, Any]] = {}
        self._links_from_cache: dict[str, list[LinkContext]] = {}
        self._neighbors_cache: dict[str, list[str]] = {}
        self._shard_cache: dict[str, dict[str, dict[str, Any]]] = {}
        logger.info(
            "Opened sharded 2Wiki store %s (nodes=%s, shards=%s, cache=%s)",
            self.store.uri,
            len(self._nodes),
            self.manifest.shard_count,
            self.cache_root,
        )

    @property
    def nodes(self) -> list[str]:
        return list(self._nodes)

    def question_path(self, split: str) -> Path:
        relative_path = f"questions/{split}.json"
        return self._ensure_metadata_file(relative_path)

    def load_questions(self, split: str, *, limit: int | None = None):
        return load_2wiki_questions(self.question_path(split), limit=limit)

    def get_document(self, node_id: str) -> DocumentNode | None:
        if node_id in self._document_cache:
            return self._document_cache[node_id]
        row = self._document_row(node_id)
        if row is None:
            return None
        record = self._record_for_node(node_id, row["shard_path"])
        document = DocumentNode(
            node_id=node_id,
            title=str(row["title"]),
            sentences=tuple(str(sentence) for sentence in record.get("sentences", ()) or ()),
            metadata={
                "url": row["url"],
                "dataset": "2wikimultihop",
            },
        )
        self._document_cache[node_id] = document
        return document

    def neighbors(self, node_id: str) -> list[str]:
        if node_id in self._neighbors_cache:
            return list(self._neighbors_cache[node_id])
        with sqlite3.connect(self.catalog_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT target FROM links WHERE source = ? ORDER BY target",
                (node_id,),
            ).fetchall()
        neighbors = [str(row[0]) for row in rows]
        self._neighbors_cache[node_id] = neighbors
        return list(neighbors)

    def links_from(self, source: str) -> list[LinkContext]:
        if source in self._links_from_cache:
            return list(self._links_from_cache[source])
        with sqlite3.connect(self.catalog_path) as conn:
            rows = conn.execute(
                """
                SELECT source, target, anchor_text, sentence, sent_idx, ref_id, ref_url
                FROM links
                WHERE source = ?
                ORDER BY rowid
                """,
                (source,),
            ).fetchall()
        links = [
            LinkContext(
                source=str(row[0]),
                target=str(row[1]),
                anchor_text=str(row[2]),
                sentence=str(row[3]),
                sent_idx=int(row[4]),
                ref_id=str(row[5]) if row[5] is not None else None,
                metadata={"ref_url": str(row[6]) if row[6] is not None else ""},
            )
            for row in rows
        ]
        self._links_from_cache[source] = links
        return list(links)

    def links_between(self, source: str, target: str) -> list[LinkContext]:
        return [link for link in self.links_from(source) if link.target == target]

    def topk_similar(self, q: str, candidates: Sequence[str], k: int) -> list[tuple[str, float]]:
        candidate_set: set[str] | None = None
        if len(candidates) != len(self._nodes):
            candidate_set = set(candidates)
        rows = self._fts_search(q, limit=max(k * 5, k))
        scored: list[tuple[str, float]] = []
        seen: set[str] = set()
        for row in rows:
            node_id = str(row["node_id"])
            if candidate_set is not None and node_id not in candidate_set:
                continue
            score = normalized_token_overlap(q, f"{row['title']} {row['text']}")
            scored.append((node_id, score))
            seen.add(node_id)
            if len(scored) >= k:
                break

        if len(scored) < k:
            fallback_candidates = candidates if candidate_set is not None else self._nodes
            for node_id in fallback_candidates:
                if node_id in seen:
                    continue
                attr = self.node_attr.get(node_id, {})
                text = f"{attr.get('title', '')} {attr.get('text', '')}".strip()
                scored.append((node_id, normalized_token_overlap(q, text)))
            scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def total_token_estimate(self) -> int:
        return self.manifest.total_token_estimate

    def store_questions_locally(self) -> dict[str, Path]:
        return {Path(path).stem: self.question_path(Path(path).stem) for path in self.manifest.questions_files}

    def _resolve_cache_root(self, cache_dir: str | Path | None) -> Path:
        base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        if isinstance(self.store, LocalDirectoryStore):
            return self.store.root
        cache_root = base / self.manifest.version
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _ensure_metadata_file(self, relative_path: str, *, cache_dir: str | Path | None = None) -> Path:
        if isinstance(self.store, LocalDirectoryStore):
            path = self.store.root / relative_path
            if not path.exists():
                raise FileNotFoundError(f"Dataset store entry missing: {path}")
            return path
        base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        local_path = base / relative_path
        if not local_path.exists():
            logger.info("Materializing metadata file %s into %s", relative_path, local_path)
        return self.store.ensure_local(relative_path, local_path)

    def _ensure_shard(self, shard_path: str) -> Path:
        if isinstance(self.store, LocalDirectoryStore):
            path = self.store.root / shard_path
            if not path.exists():
                raise FileNotFoundError(f"Shard missing: {path}")
            return path
        local_path = self.cache_root / shard_path
        if not local_path.exists():
            logger.info("Materializing shard %s into %s", shard_path, local_path)
        return self.store.ensure_local(shard_path, local_path)

    def _load_nodes(self) -> list[str]:
        with sqlite3.connect(self.catalog_path) as conn:
            rows = conn.execute("SELECT node_id FROM documents ORDER BY node_id").fetchall()
        return [str(row[0]) for row in rows]

    def _document_row(self, node_id: str) -> dict[str, Any] | None:
        if node_id in self._document_row_cache:
            return self._document_row_cache[node_id]
        with sqlite3.connect(self.catalog_path) as conn:
            row = conn.execute(
                """
                SELECT node_id, title, text, url, shard_path, token_estimate
                FROM documents
                WHERE node_id = ?
                """,
                (node_id,),
            ).fetchone()
        if row is None:
            return None
        payload = {
            "node_id": str(row[0]),
            "title": str(row[1]),
            "text": str(row[2]),
            "url": str(row[3] or ""),
            "shard_path": str(row[4]),
            "token_estimate": int(row[5]),
        }
        self._document_row_cache[node_id] = payload
        return payload

    def _record_for_node(self, node_id: str, shard_path: str) -> dict[str, Any]:
        shard_records = self._load_shard(shard_path)
        if node_id not in shard_records:
            raise KeyError(f"Node {node_id} not found in shard {shard_path}")
        return shard_records[node_id]

    def _load_shard(self, shard_path: str) -> dict[str, dict[str, Any]]:
        if shard_path in self._shard_cache:
            return self._shard_cache[shard_path]
        local_path = self._ensure_shard(shard_path)
        records: dict[str, dict[str, Any]] = {}
        with gzip.open(local_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records[str(record["node_id"])] = record
        self._shard_cache[shard_path] = records
        return records

    def _fts_search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        tokens = [token for token in _tokenize_query(query) if token]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{token}"' for token in tokens)
        with sqlite3.connect(self.catalog_path) as conn:
            rows = conn.execute(
                """
                SELECT node_id, title, text
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts)
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [
            {
                "node_id": str(row[0]),
                "title": str(row[1]),
                "text": str(row[2]),
            }
            for row in rows
        ]


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
            resolved_questions_source,
            temp_paths=temp_paths,
        )
        graph_local = _materialize_graph_source(
            resolved_graph_source,
            keep_raw=keep_raw,
            output_root=output_root,
            temp_paths=temp_paths,
        )
        questions_paths = _write_questions(output_root / "questions", questions_local)
        manifest, catalog_path, shard_paths = _write_store_from_raw_graph(
            graph_local,
            output_root=output_root,
            target_shard_size_bytes=target_shard_size_bytes,
            questions_source=resolved_questions_source.source,
            graph_source=resolved_graph_source.source,
            questions_paths=questions_paths,
        )
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Prepared sharded 2Wiki store under %s (documents=%s, shards=%s, total_tokens=%s)",
            output_root,
            manifest.total_document_count,
            manifest.shard_count,
            manifest.total_token_estimate,
        )
        return PreparedTwoWikiStore(
            root=output_root,
            manifest_path=manifest_path,
            catalog_path=catalog_path,
            questions_paths=questions_paths,
            shard_paths=shard_paths,
            manifest=manifest,
        )
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
    raw_root_path = Path(raw_root) if raw_root is not None else TWOWIKI_DEFAULT_GRAPH_PATH.parent
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
    if store_uri is None and not raw_questions_path.exists() and not raw_graph_path.exists():
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


@dataclass(slots=True)
class PreparedSource:
    source: str
    local_path: Path | None
    size_bytes: int | None
    cleanup_when_done: bool = False


def resolve_2wiki_questions_source(source: str | Path | None) -> PreparedSource:
    if source is None:
        local_dir = TWOWIKI_DEFAULT_QUESTIONS_DIR
        if local_dir.exists():
            return PreparedSource(source=str(local_dir), local_path=local_dir, size_bytes=_directory_size_bytes(local_dir))
        return PreparedSource(source=TWOWIKI_QUESTIONS_URL, local_path=None, size_bytes=_probe_remote_size(TWOWIKI_QUESTIONS_URL))
    path = None if _looks_like_url(source) else Path(source)
    if path is not None and path.exists():
        size = _directory_size_bytes(path) if path.is_dir() else path.stat().st_size
        return PreparedSource(source=str(path), local_path=path, size_bytes=size)
    return PreparedSource(source=str(source), local_path=None, size_bytes=_probe_remote_size(str(source)))


def resolve_2wiki_graph_source(source: str | Path | None) -> PreparedSource:
    if source is None:
        if TWOWIKI_DEFAULT_GRAPH_PATH.exists():
            return PreparedSource(
                source=str(TWOWIKI_DEFAULT_GRAPH_PATH),
                local_path=TWOWIKI_DEFAULT_GRAPH_PATH,
                size_bytes=TWOWIKI_DEFAULT_GRAPH_PATH.stat().st_size,
            )
        return PreparedSource(source=TWOWIKI_GRAPH_URL, local_path=None, size_bytes=_probe_remote_size(TWOWIKI_GRAPH_URL))
    path = None if _looks_like_url(source) else Path(source)
    if path is not None and path.exists():
        return PreparedSource(source=str(path), local_path=path, size_bytes=path.stat().st_size)
    return PreparedSource(source=str(source), local_path=None, size_bytes=_probe_remote_size(str(source)))


def _materialize_questions_source(source: PreparedSource, *, temp_paths: list[Path]) -> Path:
    if source.local_path is not None and source.local_path.exists():
        logger.info("Using local 2Wiki questions source at %s", source.local_path)
        return source.local_path
    logger.info("Downloading 2Wiki questions source from %s", source.source)
    temp_path = _download_to_temp(source.source)
    temp_paths.append(temp_path)
    return temp_path


def _materialize_graph_source(
    source: PreparedSource,
    *,
    keep_raw: bool,
    output_root: Path,
    temp_paths: list[Path],
) -> Path:
    if source.local_path is not None and source.local_path.exists():
        logger.info("Using local 2Wiki graph source at %s", source.local_path)
        return source.local_path
    if keep_raw:
        raw_dir = output_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        destination = raw_dir / _basename_from_source(source.source, default="para_with_hyperlink.zip")
        logger.info("Downloading 2Wiki graph source from %s into %s", source.source, destination)
        _download_url(source.source, destination)
        return destination
    logger.info("Downloading temporary 2Wiki graph source from %s", source.source)
    temp_path = _download_to_temp(source.source)
    temp_paths.append(temp_path)
    return temp_path


def _write_questions(output_dir: Path, questions_source_path: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    question_paths: dict[str, Path] = {}
    if questions_source_path.is_dir():
        logger.info("Copying 2Wiki question files from directory %s", questions_source_path)
        for split_name in ("train", "dev", "test"):
            candidate = questions_source_path / f"{split_name}.json"
            if not candidate.exists():
                continue
            destination = output_dir / candidate.name
            shutil.copyfile(candidate, destination)
            question_paths[split_name] = destination
            logger.info("Wrote question split %s to %s", split_name, destination)
        return question_paths

    suffixes = [suffix.lower() for suffix in questions_source_path.suffixes]
    if suffixes and suffixes[-1] == ".json":
        destination = output_dir / questions_source_path.name
        shutil.copyfile(questions_source_path, destination)
        question_paths[destination.stem] = destination
        logger.info("Copied single question file to %s", destination)
        return question_paths
    if suffixes and suffixes[-1] == ".zip":
        import zipfile

        logger.info("Extracting 2Wiki question files from archive %s", questions_source_path)
        with zipfile.ZipFile(questions_source_path) as archive:
            for split_name in ("train", "dev", "test"):
                member = _find_member_by_basename(archive.namelist(), f"{split_name}.json")
                if member is None:
                    continue
                destination = output_dir / f"{split_name}.json"
                with archive.open(member) as src, destination.open("wb") as dst:
                    copy_stream_with_progress(
                        src,
                        dst,
                        description=f"extract {split_name}.json",
                        total=archive.getinfo(member).file_size,
                    )
                question_paths[split_name] = destination
                logger.info("Wrote question split %s to %s", split_name, destination)
        return question_paths

    raise ValueError(f"Unsupported 2Wiki questions source: {questions_source_path}")


def _write_store_from_raw_graph(
    graph_source_path: Path,
    *,
    output_root: Path,
    target_shard_size_bytes: int,
    questions_source: str,
    graph_source: str,
    questions_paths: dict[str, Path],
) -> tuple[TwoWikiStoreManifest, Path, list[Path]]:
    id_to_title: dict[str, str] = {}
    logger.info("Phase 1/2: indexing 2Wiki titles from %s", graph_source_path)
    indexed_count = 0
    for indexed_count, record in enumerate(
        _iter_records_with_progress(
            graph_source_path,
            description="index graph titles",
        ),
        start=1,
    ):
        id_to_title[str(record.get("id"))] = normalize_2wiki_title(record.get("title", ""))
    logger.info("Indexed %s 2Wiki graph titles", indexed_count)

    catalog_path = output_root / "index" / "catalog.sqlite"
    shard_paths: list[Path] = []
    shard_infos: list[ShardInfo] = []
    total_documents = 0
    total_tokens = 0
    version = _store_version(questions_source, graph_source, target_shard_size_bytes)

    with sqlite3.connect(catalog_path) as conn:
        _create_catalog_schema(conn)
        writer = _ShardWriter(output_root / "shards", target_shard_size_bytes=target_shard_size_bytes)
        try:
            logger.info("Phase 2/2: writing catalog and shards into %s", output_root)
            for total_documents, raw_record in enumerate(
                _iter_records_with_progress(
                    graph_source_path,
                    description="write catalog and shards",
                ),
                start=1,
            ):
                normalized = normalize_2wiki_graph_record(raw_record, id_to_title=id_to_title)
                normalized["url"] = str(raw_record.get("url", ""))
                shard_relpath = writer.write_record(normalized)
                text = " ".join(str(sentence) for sentence in normalized["sentences"])
                token_estimate = approx_token_count(text)
                total_tokens += token_estimate
                conn.execute(
                    """
                    INSERT INTO documents(node_id, title, text, url, shard_path, token_estimate)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized["node_id"],
                        normalized["title"],
                        text,
                        normalized.get("url", ""),
                        shard_relpath,
                        token_estimate,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO documents_fts(node_id, title, text)
                    VALUES (?, ?, ?)
                    """,
                    (
                        normalized["node_id"],
                        normalized["title"],
                        text,
                    ),
                )
                for mention in normalized.get("mentions", []) or []:
                    target = _resolve_link_target(mention)
                    conn.execute(
                        """
                        INSERT INTO links(source, target, anchor_text, sentence, sent_idx, ref_id, ref_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            normalized["node_id"],
                            target,
                            _anchor_text_from_mention(normalized["sentences"], mention),
                            _sentence_from_mention(normalized["sentences"], mention),
                            int(mention.get("sent_idx", 0)),
                            str((mention.get("ref_ids") or [None])[0]) if mention.get("ref_ids") else None,
                            str(mention.get("ref_url", "")),
                        ),
                    )
            shard_infos = writer.close()
        finally:
            conn.commit()
            shard_paths = [output_root / shard.path for shard in shard_infos]
    logger.info(
        "Wrote catalog.sqlite and %s shard(s) from %s 2Wiki records",
        len(shard_infos),
        total_documents,
    )

    manifest = TwoWikiStoreManifest(
        dataset_name="2wikimultihop",
        version=version,
        shard_count=len(shard_infos),
        target_shard_size_bytes=target_shard_size_bytes,
        total_document_count=total_documents,
        total_token_estimate=total_tokens,
        questions_source=questions_source,
        graph_source=graph_source,
        questions_files=[str(Path("questions") / path.name) for path in sorted(questions_paths.values())],
        shards=shard_infos,
    )
    return manifest, catalog_path, shard_paths


class _ShardWriter:
    def __init__(self, shard_dir: Path, *, target_shard_size_bytes: int):
        self.shard_dir = shard_dir
        self.target_shard_size_bytes = target_shard_size_bytes
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self._current_handle: gzip.GzipFile | None = None
        self._current_text_handle: Any = None
        self._current_uncompressed_bytes = 0
        self._current_record_count = 0
        self._current_path: Path | None = None
        self._shards: list[ShardInfo] = []

    def write_record(self, record: dict[str, Any]) -> str:
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if (
            self._current_path is None
            or (
                self._current_uncompressed_bytes >= self.target_shard_size_bytes
                and self._current_record_count > 0
            )
        ):
            self._open_new_shard()
        assert self._current_text_handle is not None
        self._current_text_handle.write(payload.decode("utf-8"))
        self._current_uncompressed_bytes += len(payload)
        self._current_record_count += 1
        assert self._current_path is not None
        return str(Path("shards") / self._current_path.name)

    def close(self) -> list[ShardInfo]:
        self._close_current_shard()
        return list(self._shards)

    def _open_new_shard(self) -> None:
        self._close_current_shard()
        self._current_path = self.shard_dir / f"part-{self.index:05d}.jsonl.gz"
        self._current_handle = gzip.open(self._current_path, "wt", encoding="utf-8")
        self._current_text_handle = self._current_handle
        self._current_uncompressed_bytes = 0
        self._current_record_count = 0
        self.index += 1

    def _close_current_shard(self) -> None:
        if self._current_handle is None or self._current_path is None:
            return
        self._current_handle.close()
        compressed_bytes = self._current_path.stat().st_size
        sha256 = hashlib.sha256(self._current_path.read_bytes()).hexdigest()
        self._shards.append(
            ShardInfo(
                path=str(Path("shards") / self._current_path.name),
                record_count=self._current_record_count,
                uncompressed_bytes=self._current_uncompressed_bytes,
                compressed_bytes=compressed_bytes,
                sha256=sha256,
            )
        )
        self._current_handle = None
        self._current_text_handle = None
        self._current_path = None


def _create_catalog_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            node_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            url TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            token_estimate INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS links (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            anchor_text TEXT NOT NULL,
            sentence TEXT NOT NULL,
            sent_idx INTEGER NOT NULL,
            ref_id TEXT,
            ref_url TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON links(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source_target ON links(source, target)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
        USING fts5(node_id UNINDEXED, title, text)
        """
    )
    conn.commit()


def _resolve_link_target(mention: dict[str, Any]) -> str:
    ref_ids = mention.get("ref_ids") or []
    if ref_ids:
        return str(ref_ids[0])
    return normalize_2wiki_title(mention.get("ref_url", ""))


def _anchor_text_from_mention(sentences: Sequence[str], mention: dict[str, Any]) -> str:
    sentence = _sentence_from_mention(sentences, mention)
    start = int(mention.get("start", 0))
    end = int(mention.get("end", start))
    anchor_text = sentence[start:end].strip()
    if anchor_text:
        return anchor_text
    return normalize_2wiki_title(mention.get("ref_url", ""))


def _sentence_from_mention(sentences: Sequence[str], mention: dict[str, Any]) -> str:
    sent_idx = int(mention.get("sent_idx", 0))
    if 0 <= sent_idx < len(sentences):
        return str(sentences[sent_idx])
    return ""


def _download_to_temp(source_url: str) -> Path:
    suffix = Path(urllib.parse.urlparse(source_url).path).suffix or ".bin"
    with tempfile.NamedTemporaryFile(prefix="webwalker-", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
    logger.info("Downloading %s into temporary file %s", source_url, temp_path)
    _download_url(source_url, temp_path)
    return temp_path


def _download_url(source_url: str, destination: Path) -> None:
    normalized_url = _normalize_dropbox_url(source_url)
    parsed = urllib.parse.urlparse(normalized_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request_or_url: str | urllib.request.Request
    expected_bytes: int | None = None
    if parsed.scheme == "file":
        request_or_url = normalized_url
        file_path = Path(parsed.path)
        if file_path.exists():
            expected_bytes = file_path.stat().st_size
    else:
        request_or_url = urllib.request.Request(
            normalized_url,
            headers={"User-Agent": "webwalker/0.1 (+https://github.com/Acture/webwalker)"},
        )
    with urllib.request.urlopen(request_or_url) as response, destination.open("wb") as handle:
        content_length = response.headers.get("Content-Length")
        if content_length:
            expected_bytes = int(content_length)
        logger.info("Downloading %s to %s", normalized_url, destination)
        copy_stream_with_progress(
            response,
            handle,
            description=f"download {destination.name}",
            total=expected_bytes,
        )


def _probe_remote_size(source: str) -> int | None:
    if not source:
        return None
    if not _looks_like_url(source):
        path = Path(source)
        if not path.exists():
            return None
        return _directory_size_bytes(path) if path.is_dir() else path.stat().st_size
    parsed = urllib.parse.urlparse(_normalize_dropbox_url(source))
    if parsed.scheme == "file":
        path = Path(parsed.path)
        if not path.exists():
            return None
        return path.stat().st_size
    request = urllib.request.Request(parsed.geturl(), method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            content_length = response.headers.get("Content-Length")
    except Exception:
        return None
    if not content_length:
        return None
    return int(content_length)


def _ensure_min_free_space(path: Path, *, min_free_gib: float, expected_new_bytes: int | None) -> None:
    probe_path = path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    free_bytes = shutil.disk_usage(probe_path).free
    min_free_bytes = int(min_free_gib * 1024**3)
    if free_bytes < min_free_bytes:
        raise RuntimeError(
            f"Refusing to proceed: free space {free_bytes / 1024**3:.1f} GiB is below the configured minimum "
            f"{min_free_gib:.1f} GiB."
        )
    if expected_new_bytes is None:
        return
    if free_bytes - expected_new_bytes < min_free_bytes:
        raise RuntimeError(
            f"Refusing to proceed: estimated write {expected_new_bytes / 1024**3:.1f} GiB would leave less than "
            f"{min_free_gib:.1f} GiB free."
        )


def _estimate_prepare_bytes(graph_size_bytes: int | None) -> int | None:
    if graph_size_bytes is None:
        return None
    return int(graph_size_bytes * 1.6)


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _looks_like_url(value: str | Path | None) -> bool:
    if value is None:
        return False
    raw = str(value)
    parsed = urllib.parse.urlparse(raw)
    return parsed.scheme in {"http", "https", "file", "s3"}


def _normalize_dropbox_url(source_url: str) -> str:
    parsed = urllib.parse.urlparse(source_url)
    if "dropbox.com" not in parsed.netloc:
        return source_url
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query["dl"] = ["1"]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def _basename_from_source(source: str, *, default: str) -> str:
    parsed = urllib.parse.urlparse(source)
    name = Path(parsed.path).name
    return name or default


def _find_member_by_basename(names: Iterable[str], basename: str) -> str | None:
    for name in names:
        if Path(name).name == basename:
            return name
    return None


def _iter_records_with_progress(
    path: Path,
    *,
    description: str,
    update_every: int = 10_000,
):
    if not should_render_progress():
        for record in iter_2wiki_graph_records(path):
            yield record
        return

    count = 0
    with create_progress(transient=True) as progress:
        task_id = progress.add_task(description, total=None)
        for count, record in enumerate(iter_2wiki_graph_records(path), start=1):
            if count == 1 or count % update_every == 0:
                progress.update(task_id, description=f"{description} [{count:,} records]")
            yield record
        progress.update(task_id, description=f"{description} [{count:,} records]")


def _store_version(questions_source: str, graph_source: str, target_shard_size_bytes: int) -> str:
    digest = hashlib.sha1(
        f"{questions_source}|{graph_source}|{target_shard_size_bytes}".encode("utf-8")
    ).hexdigest()[:12]
    return f"2wikimultihop-{digest}"


def _tokenize_query(query: str) -> list[str]:
    cleaned = []
    token = []
    for char in query.lower():
        if char.isalnum():
            token.append(char)
            continue
        if token:
            cleaned.append("".join(token))
            token = []
    if token:
        cleaned.append("".join(token))
    return cleaned


def probe_source_size(source: str | Path) -> int | None:
    return _probe_remote_size(str(source))


def ensure_min_free_space(path: Path, *, min_free_gib: float, expected_new_bytes: int | None) -> None:
    _ensure_min_free_space(path, min_free_gib=min_free_gib, expected_new_bytes=expected_new_bytes)


def estimate_prepare_bytes(graph_size_bytes: int | None) -> int | None:
    return _estimate_prepare_bytes(graph_size_bytes)
