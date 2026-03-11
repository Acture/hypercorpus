import json

from webwalker.datasets import PreparedDatasetStore, ShardedDocumentStore, prepare_normalized_graph_store


def test_prepare_normalized_graph_store_round_trips_non_twowiki_dataset(iirc_files, tmp_path):
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

