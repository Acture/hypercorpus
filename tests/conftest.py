import bz2
import io
import json
import tarfile
import zipfile
from pathlib import Path

from pytest import fixture

from webwalker.datasets import prepare_2wiki_store
from webwalker.graph import DocumentNode, LinkContext, LinkContextGraph


@fixture
def test_dir() -> Path:
	return Path(__file__).parent


@fixture
def project_dir(test_dir: Path) -> Path:
	return test_dir.parent


@fixture
def dataset_dir(project_dir: Path) -> Path:
	return project_dir / "dataset"


@fixture
def hotqa_dir(dataset_dir: Path) -> Path:
	return dataset_dir / "hotqa"


@fixture
def sample_graph() -> LinkContextGraph:
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				node_id="mission",
				title="Moon Launch Program",
				sentences=(
					"Moon Launch Program uses Cape Canaveral as its launch site.",
					"The program was directed by Alice Johnson.",
				),
				metadata={"phi": 0.6},
			),
			DocumentNode(
				node_id="cape",
				title="Cape Canaveral",
				sentences=("Cape Canaveral is a city in Florida.",),
				metadata={"phi": 0.8},
			),
			DocumentNode(
				node_id="director",
				title="Alice Johnson",
				sentences=("Alice Johnson directed the Moon Launch Program in 1969.",),
				metadata={"phi": 0.2},
			),
			DocumentNode(
				node_id="florida",
				title="Florida",
				sentences=("Florida is a state in the southeastern United States.",),
				metadata={"phi": 0.1},
			),
		]
	)
	graph.add_link(
		LinkContext(
			source="mission",
			target="cape",
			anchor_text="Cape Canaveral",
			sentence="Moon Launch Program uses Cape Canaveral as its launch site.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="mission",
			target="director",
			anchor_text="Alice Johnson",
			sentence="The program was directed by Alice Johnson.",
			sent_idx=1,
		)
	)
	graph.add_link(
		LinkContext(
			source="cape",
			target="florida",
			anchor_text="Florida",
			sentence="Cape Canaveral is a city in Florida.",
			sent_idx=0,
		)
	)
	return graph


@fixture
def cyclic_graph() -> LinkContextGraph:
	graph = LinkContextGraph(
		documents=[
			DocumentNode("a", "Launch Root", ("Launch Root links to Relay Node.",)),
			DocumentNode("b", "Relay Node", ("Relay Node links to Final Target.",)),
			DocumentNode("c", "Final Target", ("Final Target holds the answer.",)),
		]
	)
	graph.add_link(
		LinkContext(
			source="a",
			target="b",
			anchor_text="Relay Node",
			sentence="Launch Root links to Relay Node.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="b",
			target="a",
			anchor_text="Launch Root",
			sentence="Relay Node can also revisit Launch Root.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="b",
			target="c",
			anchor_text="Final Target",
			sentence="Relay Node links to Final Target.",
			sent_idx=0,
		)
	)
	return graph


@fixture
def tiny_tar_bz2(tmp_path: Path) -> Path:
	inner_jsonl = "\n".join(
		[
			json.dumps({"title": "Moon Launch Program", "text": "Uses Cape Canaveral."}),
			json.dumps({"title": "Cape Canaveral", "text": "A city in Florida."}),
		]
	).encode("utf-8")
	inner_bz2 = bz2.compress(inner_jsonl)
	archive_path = tmp_path / "sample.tar.bz2"

	with tarfile.open(archive_path, "w:bz2") as archive:
		info = tarfile.TarInfo(name="sample/wiki_00.bz2")
		info.size = len(inner_bz2)
		archive.addfile(info, io.BytesIO(inner_bz2))

	return archive_path


@fixture
def two_wiki_graph_records() -> list[dict]:
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
			"sentences": [
				"Cape Canaveral is a city in Florida.",
			],
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
			"sentences": [
				"Alice Johnson directed the Moon Launch Program in 1969.",
			],
			"mentions": [],
		},
		{
			"id": "400",
			"title": "Florida",
			"sentences": [
				"Florida is a state in the southeastern United States.",
			],
			"mentions": [],
		},
	]


@fixture
def two_wiki_questions() -> list[dict]:
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


@fixture
def two_wiki_files(tmp_path: Path, two_wiki_graph_records: list[dict], two_wiki_questions: list[dict]) -> tuple[Path, Path]:
	graph_path = tmp_path / "para_with_hyperlink.jsonl"
	questions_path = tmp_path / "dev.json"

	graph_path.write_text(
		"\n".join(json.dumps(record, ensure_ascii=False) for record in two_wiki_graph_records) + "\n",
		encoding="utf-8",
	)
	questions_path.write_text(
		json.dumps(two_wiki_questions, ensure_ascii=False),
		encoding="utf-8",
	)

	return questions_path, graph_path


@fixture
def two_wiki_archives(tmp_path: Path, two_wiki_graph_records: list[dict], two_wiki_questions: list[dict]) -> tuple[Path, Path]:
	questions_zip = tmp_path / "data_ids_april7.zip"
	graph_zip = tmp_path / "para_with_hyperlink.zip"

	with zipfile.ZipFile(questions_zip, "w") as archive:
		archive.writestr("data_ids_april7/dev.json", json.dumps(two_wiki_questions, ensure_ascii=False))
		archive.writestr("data_ids_april7/train.json", json.dumps(two_wiki_questions, ensure_ascii=False))
		archive.writestr("data_ids_april7/test.json", json.dumps(two_wiki_questions, ensure_ascii=False))

	with zipfile.ZipFile(graph_zip, "w") as archive:
		archive.writestr(
			"para_with_hyperlink.jsonl",
			"\n".join(json.dumps(record, ensure_ascii=False) for record in two_wiki_graph_records) + "\n",
		)

	return questions_zip, graph_zip


@fixture
def prepared_two_wiki_store(two_wiki_archives, tmp_path: Path):
	questions_zip, graph_zip = two_wiki_archives
	output_dir = tmp_path / "2wiki-store"
	return prepare_2wiki_store(
		output_dir,
		questions_source=questions_zip.as_uri(),
		graph_source=graph_zip.as_uri(),
	)
