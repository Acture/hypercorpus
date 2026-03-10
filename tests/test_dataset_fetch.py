import json

import pytest

from webwalker.datasets.fetch import fetch_2wiki_dataset, write_2wiki_sample_dataset
from webwalker.datasets.twowiki_store import prepare_2wiki_store
from webwalker.datasets.twowiki import load_2wiki_graph, load_2wiki_questions


def test_fetch_2wiki_dataset_extracts_selected_split_and_graph(two_wiki_archives, tmp_path):
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
    assert layout.graph_path.exists()
    assert not (output_dir / "archives").exists()

    cases = load_2wiki_questions(layout.question_paths["dev"])
    graph = load_2wiki_graph(layout.graph_path)
    assert cases[0].case_id == "q1"
    assert graph.get_document("Moon Launch Program") is not None


def test_fetch_2wiki_dataset_can_keep_archives_and_fetch_all_splits(two_wiki_archives, tmp_path):
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


def test_prepare_2wiki_store_rejects_when_min_free_space_is_too_high(two_wiki_archives, tmp_path):
    questions_zip, graph_zip = two_wiki_archives

    with pytest.raises(RuntimeError, match="Refusing to proceed"):
        prepare_2wiki_store(
            tmp_path / "store",
            questions_source=questions_zip.as_uri(),
            graph_source=graph_zip.as_uri(),
            min_free_gib=10_000.0,
        )
