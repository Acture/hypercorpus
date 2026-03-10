import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from webwalker.datasets import (
    ShardedLinkContextStore,
    StoreObjectInfo,
    inspect_2wiki_store,
    prepare_2wiki_store,
)


def test_prepare_2wiki_store_writes_manifest_catalog_and_shards(prepared_two_wiki_store):
    assert prepared_two_wiki_store.manifest_path.exists()
    assert prepared_two_wiki_store.catalog_path.exists()
    assert prepared_two_wiki_store.manifest.shard_count >= 1
    assert set(prepared_two_wiki_store.questions_paths) == {"dev", "train", "test"}

    manifest_payload = json.loads(prepared_two_wiki_store.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["dataset_name"] == "2wikimultihop"

    with sqlite3.connect(prepared_two_wiki_store.catalog_path) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        link_count = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]

    assert doc_count == 4
    assert link_count == 3


def test_sharded_link_context_store_uses_catalog_before_loading_shards(prepared_two_wiki_store, tmp_path):
    store = ShardedLinkContextStore(prepared_two_wiki_store.root, cache_dir=tmp_path / "cache")

    assert store._shard_cache == {}
    ranked = store.topk_similar("launch site city", store.nodes, k=2)
    assert ranked
    assert store.neighbors("Moon Launch Program") == ["Alice Johnson", "Cape Canaveral"]
    assert len(store.links_from("Moon Launch Program")) == 2
    assert store._shard_cache == {}

    document = store.get_document("Moon Launch Program")
    assert document is not None
    assert "Cape Canaveral" in document.text
    assert len(store._shard_cache) == 1


def test_sharded_link_context_store_remote_store_downloads_shard_once(prepared_two_wiki_store, tmp_path):
    @dataclass
    class MockRemoteStore:
        root: Path
        downloads: dict[str, int]

        def exists(self, relative_path: str) -> bool:
            return (self.root / relative_path).exists()

        def stat(self, relative_path: str) -> StoreObjectInfo | None:
            target = self.root / relative_path
            if not target.exists():
                return None
            return StoreObjectInfo(path=relative_path, size_bytes=target.stat().st_size)

        def ensure_local(self, relative_path: str, local_path: Path) -> Path:
            self.downloads[relative_path] = self.downloads.get(relative_path, 0) + 1
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                local_path.write_bytes((self.root / relative_path).read_bytes())
            return local_path

        @property
        def uri(self) -> str:
            return "s3://mock-bucket/2wiki-store"

    remote = MockRemoteStore(root=prepared_two_wiki_store.root, downloads={})
    cache_dir = tmp_path / "remote-cache"
    store = ShardedLinkContextStore("s3://mock-bucket/2wiki-store", cache_dir=cache_dir, object_store=remote)

    assert remote.downloads["manifest.json"] == 1
    assert remote.downloads["index/catalog.sqlite"] == 1

    document = store.get_document("Moon Launch Program")
    assert document is not None
    shard_key = next(iter(remote.downloads.keys() - {"manifest.json", "index/catalog.sqlite"}))
    assert remote.downloads[shard_key] == 1

    again = store.get_document("Moon Launch Program")
    assert again is not None
    assert remote.downloads[shard_key] == 1


def test_inspect_2wiki_store_reports_use_local_raw(tmp_path, monkeypatch):
    raw_root = tmp_path / "dataset" / "2wikimultihop"
    (raw_root / "data_ids_april7").mkdir(parents=True)
    (raw_root / "data_ids_april7" / "dev.json").write_text("[]", encoding="utf-8")
    (raw_root / "para_with_hyperlink.jsonl").write_text("", encoding="utf-8")

    inspection = inspect_2wiki_store(raw_root=raw_root, cache_dir=tmp_path / "cache")
    assert inspection.recommended_action == "use-local-raw"


def test_prepare_2wiki_store_emits_phase_logs(two_wiki_archives, tmp_path, caplog):
    questions_zip, graph_zip = two_wiki_archives
    caplog.set_level("INFO")

    prepare_2wiki_store(
        tmp_path / "logged-store",
        questions_source=questions_zip.as_uri(),
        graph_source=graph_zip.as_uri(),
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Preparing sharded 2Wiki store" in message for message in messages)
    assert any("Phase 1/2: indexing 2Wiki titles" in message for message in messages)
    assert any("Phase 2/2: writing catalog and shards" in message for message in messages)


def test_prepare_2wiki_store_skips_duplicate_titles(tmp_path, two_wiki_questions, caplog):
    graph_path = tmp_path / "duplicate-graph.jsonl"
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    duplicate_records = [
        {
            "id": "1",
            "title": "Unconquered",
            "sentences": ["First record."],
            "mentions": [],
        },
        {
            "id": "2",
            "title": "Unconquered",
            "sentences": ["Second record."],
            "mentions": [],
        },
    ]
    graph_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in duplicate_records) + "\n",
        encoding="utf-8",
    )
    for split in ("train", "dev", "test"):
        (questions_dir / f"{split}.json").write_text(json.dumps(two_wiki_questions, ensure_ascii=False), encoding="utf-8")

    caplog.set_level("WARNING")
    prepared = prepare_2wiki_store(
        tmp_path / "duplicate-store",
        questions_source=questions_dir,
        graph_source=graph_path,
    )

    with sqlite3.connect(prepared.catalog_path) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    assert doc_count == 1
    assert prepared.manifest.total_document_count == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("Skipped 1 duplicate 2Wiki node title" in message for message in messages)


def test_prepare_2wiki_store_rejects_existing_partial_output(two_wiki_archives, tmp_path):
    questions_zip, graph_zip = two_wiki_archives
    output_dir = tmp_path / "existing-store"
    (output_dir / "index").mkdir(parents=True)
    (output_dir / "index" / "catalog.sqlite").write_text("partial", encoding="utf-8")

    try:
        prepare_2wiki_store(
            output_dir,
            questions_source=questions_zip.as_uri(),
            graph_source=graph_zip.as_uri(),
        )
    except RuntimeError as exc:
        assert "already contains a prepared or partial 2Wiki store" in str(exc)
    else:
        raise AssertionError("prepare_2wiki_store should reject an existing partial output directory")
