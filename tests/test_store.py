import json
from pathlib import Path

from hypercorpus.datasets import (
	PreparedDatasetStore,
	ShardedDocumentStore,
	prepare_normalized_graph_store,
)


def test_prepare_normalized_graph_store_round_trips_non_twowiki_dataset(
	iirc_files, tmp_path
):
	questions_path, graph_path = iirc_files
	output_dir = tmp_path / "generic-store"

	prepared = prepare_normalized_graph_store(
		output_dir,
		dataset_name="iirc",
		questions_source=questions_path,
		graph_source=graph_path,
	)

	assert isinstance(prepared, PreparedDatasetStore)
	assert prepared.manifest.dataset_name == "iirc"
	manifest_payload = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
	assert manifest_payload["dataset_name"] == "iirc"
	assert {path.stem for path in prepared.questions_paths.values()} == {"dev"}

	store = ShardedDocumentStore(prepared.root)
	assert store.manifest.dataset_name == "iirc"
	assert set(store.nodes) == {"Moon Launch Program", "Cape Canaveral", "Florida"}

	cases = store.load_questions("dev")
	assert len(cases) == 2
	assert cases[0].dataset_name == "iirc"

	document = store.get_document("Moon Launch Program")
	assert document is not None
	assert "Cape Canaveral" in document.text


def test_prepare_normalized_graph_store_accepts_graph_source_directory(
	iirc_questions, iirc_graph_records, tmp_path
):
	questions_path = tmp_path / "iirc-questions.json"
	questions_path.write_text(
		json.dumps(iirc_questions, ensure_ascii=False), encoding="utf-8"
	)
	graph_dir = tmp_path / "graph-bundle"
	graph_dir.mkdir()
	shard_a = graph_dir / "part-00000.jsonl"
	shard_b = graph_dir / "part-00001.jsonl"
	shard_a.write_text(
		json.dumps(iirc_graph_records[0], ensure_ascii=False) + "\n", encoding="utf-8"
	)
	shard_b.write_text(
		"\n".join(
			json.dumps(record, ensure_ascii=False) for record in iirc_graph_records[1:]
		)
		+ "\n",
		encoding="utf-8",
	)

	prepared = prepare_normalized_graph_store(
		tmp_path / "directory-store",
		dataset_name="iirc",
		questions_source=questions_path,
		graph_source=graph_dir,
	)

	store = ShardedDocumentStore(prepared.root)
	assert set(store.nodes) == {"Moon Launch Program", "Cape Canaveral", "Florida"}
