import json
from pathlib import Path

from hypercorpus.datasets import (
    convert_hotpotqa_raw_dataset,
    convert_iirc_raw_dataset,
    convert_musique_raw_dataset,
    fetch_iirc_dataset,
    load_hotpotqa_graph,
    load_hotpotqa_questions,
    load_iirc_graph,
    load_iirc_questions,
    load_musique_graph,
    load_musique_questions,
)


def _write_iirc_html_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "iirc-html-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "title": "Moon Launch Program",
            "text": "Moon Launch Program launches from Cape Canaveral.",
            "links": [{"target": "Cape Canaveral", "indices": [33, 48]}],
            "questions": [
                {
                    "qid": "iirc-html-1",
                    "question": "Which state contains the launch city?",
                    "answer": {"type": "span", "answer_spans": [{"text": "Florida", "passage": "florida"}]},
                    "question_links": ["Cape Canaveral", "cape canaveral"],
                    "context": [
                        {"passage": "main", "text": "Moon Launch Program launches from Cape Canaveral."},
                        {"passage": "cape canaveral", "text": "Cape Canaveral is in Florida."},
                        {"passage": "Florida", "text": "Florida is a state in the United States."},
                    ],
                }
            ],
        }
    ]
    context_articles = {
        "cape canaveral": 'Cape Canaveral is in <a href="Florida">Florida</a>.',
        "florida": 'Florida is a state in the <a href="United%20States">United States</a>.',
    }
    (raw_dir / "train.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (raw_dir / "dev.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (raw_dir / "context_articles.json").write_text(json.dumps(context_articles, ensure_ascii=False), encoding="utf-8")
    return raw_dir


def test_convert_hotpotqa_raw_distractor_builds_case_local_graph(hotpotqa_raw_split_files, tmp_path):
    raw_dir = hotpotqa_raw_split_files["distractor"]["dev"].parent
    layout = convert_hotpotqa_raw_dataset(raw_dir, tmp_path / "normalized-hotpot", variant="distractor")

    cases = load_hotpotqa_questions(layout.question_paths["dev"], variant="distractor")
    graph = load_hotpotqa_graph(layout.graph_path, variant="fullwiki")

    assert cases[0].gold_support_nodes == ["h1::Apollo Program", "h1::Kennedy Space Center"]
    assert set(graph.nodes) == {"h1::Apollo Program", "h1::Kennedy Space Center", "h1::Florida"}


def test_convert_hotpotqa_raw_fullwiki_uses_supplied_graph(hotpotqa_raw_split_files, hotpotqa_fullwiki_files, tmp_path):
    _questions_path, graph_path = hotpotqa_fullwiki_files
    raw_dir = hotpotqa_raw_split_files["fullwiki"]["dev"].parent
    layout = convert_hotpotqa_raw_dataset(
        raw_dir,
        tmp_path / "normalized-hotpot-fullwiki",
        variant="fullwiki",
        graph_source=graph_path,
    )

    cases = load_hotpotqa_questions(layout.question_paths["dev"], variant="fullwiki")
    graph = load_hotpotqa_graph(layout.graph_path, variant="fullwiki")

    assert cases[0].gold_support_nodes == ["Apollo Program", "Kennedy Space Center", "Florida"]
    assert len(graph.nodes) == 3


def test_convert_musique_raw_preserves_support_start_and_path(musique_raw_split_files, tmp_path):
    raw_dir = next(iter(musique_raw_split_files.values())).parent
    layout = convert_musique_raw_dataset(raw_dir, tmp_path / "normalized-musique")

    cases = load_musique_questions(layout.question_paths["dev"])
    graph = load_musique_graph(layout.graph_path)

    assert cases[0].gold_start_nodes == ["m-raw-1::p0::Apollo Program"]
    assert cases[0].gold_path_nodes == [
        "m-raw-1::p0::Apollo Program",
        "m-raw-1::p1::Kennedy Space Center",
        "m-raw-1::p2::Florida",
    ]
    assert len(graph.nodes) == 3


def test_convert_iirc_raw_flattens_nested_questions_and_context(iirc_raw_archive, tmp_path):
    raw_layout = fetch_iirc_dataset(tmp_path / "iirc-raw", archive_url=iirc_raw_archive.as_uri())
    layout = convert_iirc_raw_dataset(raw_layout.raw_dir / "iirc", tmp_path / "normalized-iirc")

    cases = load_iirc_questions(layout.question_paths["dev"])
    graph = load_iirc_graph(layout.graph_path)

    assert cases[0].gold_start_nodes == ["Moon Launch Program"]
    assert cases[0].gold_support_nodes == ["Moon Launch Program", "Cape Canaveral", "Florida"]
    assert "main" not in cases[0].gold_support_nodes
    assert len({node.lower() for node in cases[0].gold_support_nodes}) == len(cases[0].gold_support_nodes)
    assert set(graph.nodes) == {"Moon Launch Program", "Cape Canaveral", "Florida"}


def test_convert_iirc_raw_supports_html_context_articles_and_title_aliases(tmp_path):
    raw_dir = _write_iirc_html_raw_dir(tmp_path)

    layout = convert_iirc_raw_dataset(raw_dir, tmp_path / "normalized-iirc-html")

    cases = load_iirc_questions(layout.question_paths["dev"])
    graph = load_iirc_graph(layout.graph_path)

    assert cases[0].gold_start_nodes == ["Moon Launch Program"]
    assert cases[0].gold_support_nodes == ["Moon Launch Program", "Cape Canaveral", "Florida"]
    assert cases[0].gold_path_nodes == ["Moon Launch Program", "Cape Canaveral"]
    assert "main" not in cases[0].gold_support_nodes
    assert "cape canaveral" not in cases[0].gold_support_nodes
    assert set(graph.nodes) >= {"Moon Launch Program", "Cape Canaveral", "Florida"}
    assert graph.links_between("Cape Canaveral", "Florida")[0].anchor_text == "Florida"
    assert graph.links_between("Florida", "United States")[0].anchor_text == "United States"
