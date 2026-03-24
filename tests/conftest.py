import bz2
import io
import json
import tarfile
import zipfile
from pathlib import Path

from pytest import fixture

from hypercorpus.datasets import prepare_2wiki_store, prepare_normalized_graph_store
from hypercorpus.graph import DocumentNode, LinkContext, LinkContextGraph


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
			"type": "bridge",
			"supporting_facts": [
				["Moon Launch Program", 0],
				["Cape Canaveral", 0],
			],
		},
		{
			"_id": "q2",
			"question": "Who directed the Moon Launch Program?",
			"answer": "Alice Johnson",
			"type": "bridge",
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


@fixture
def iirc_graph_records() -> list[dict]:
	return [
		{
			"node_id": "Moon Launch Program",
			"title": "Moon Launch Program",
			"sentences": [
				"Moon Launch Program launches from Cape Canaveral.",
			],
			"links": [
				{
					"target": "Cape Canaveral",
					"anchor_text": "Cape Canaveral",
					"sentence": "Moon Launch Program launches from Cape Canaveral.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Cape Canaveral",
			"title": "Cape Canaveral",
			"sentences": [
				"Cape Canaveral is in Florida.",
			],
			"links": [
				{
					"target": "Florida",
					"anchor_text": "Florida",
					"sentence": "Cape Canaveral is in Florida.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Florida",
			"title": "Florida",
			"sentences": [
				"Florida is a state in the southeastern United States.",
			],
			"links": [],
		},
	]


@fixture
def iirc_questions() -> list[dict]:
	return [
		{
			"case_id": "i1",
			"question": "Which state contains the launch city?",
			"answer": "Florida",
			"gold_support_nodes": ["Moon Launch Program", "Cape Canaveral", "Florida"],
			"gold_start_nodes": ["Moon Launch Program"],
		},
		{
			"case_id": "i2",
			"question": "Which city is the launch site?",
			"answer": "Cape Canaveral",
			"gold_support_nodes": ["Moon Launch Program", "Cape Canaveral"],
			"gold_start_nodes": ["Moon Launch Program"],
			"gold_path_nodes": ["Moon Launch Program", "Cape Canaveral"],
		},
	]


@fixture
def iirc_files(tmp_path: Path, iirc_graph_records: list[dict], iirc_questions: list[dict]) -> tuple[Path, Path]:
	graph_path = tmp_path / "iirc-graph.json"
	questions_path = tmp_path / "iirc-questions.json"
	graph_path.write_text(json.dumps(iirc_graph_records, ensure_ascii=False), encoding="utf-8")
	questions_path.write_text(json.dumps(iirc_questions, ensure_ascii=False), encoding="utf-8")
	return questions_path, graph_path


@fixture
def prepared_iirc_store(iirc_files, tmp_path: Path):
	questions_path, graph_path = iirc_files
	output_dir = tmp_path / "iirc-store"
	return prepare_normalized_graph_store(
		output_dir,
		dataset_name="iirc",
		questions_source=questions_path,
		graph_source=graph_path,
	)


@fixture
def musique_graph_records() -> list[dict]:
	return [
		{
			"node_id": "Apollo Program",
			"title": "Apollo Program",
			"sentences": ["Apollo Program launched from Kennedy Space Center."],
			"links": [
				{
					"target": "Kennedy Space Center",
					"anchor_text": "Kennedy Space Center",
					"sentence": "Apollo Program launched from Kennedy Space Center.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Kennedy Space Center",
			"title": "Kennedy Space Center",
			"sentences": ["Kennedy Space Center is in Florida."],
			"links": [
				{
					"target": "Florida",
					"anchor_text": "Florida",
					"sentence": "Kennedy Space Center is in Florida.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Florida",
			"title": "Florida",
			"sentences": ["Florida is a state in the United States."],
			"links": [],
		},
	]


@fixture
def musique_questions() -> list[dict]:
	return [
		{
			"id": "m1",
			"question": "Which state contains Kennedy Space Center?",
			"answer": "Florida",
			"supporting_pages": ["Apollo Program", "Kennedy Space Center", "Florida"],
			"start_nodes": ["Apollo Program"],
			"reasoning_path": ["Apollo Program", "Kennedy Space Center", "Florida"],
		}
	]


@fixture
def musique_files(tmp_path: Path, musique_graph_records: list[dict], musique_questions: list[dict]) -> tuple[Path, Path]:
	graph_path = tmp_path / "musique-graph.json"
	questions_path = tmp_path / "musique-questions.json"
	graph_path.write_text(json.dumps(musique_graph_records, ensure_ascii=False), encoding="utf-8")
	questions_path.write_text(json.dumps(musique_questions, ensure_ascii=False), encoding="utf-8")
	return questions_path, graph_path


@fixture
def prepared_musique_store(musique_files, tmp_path: Path):
	questions_path, graph_path = musique_files
	output_dir = tmp_path / "musique-store"
	return prepare_normalized_graph_store(
		output_dir,
		dataset_name="musique",
		questions_source=questions_path,
		graph_source=graph_path,
	)


@fixture
def hotpotqa_distractor_questions() -> list[dict]:
	return [
		{
			"_id": "h1",
			"question": "Which state contains the launch facility used by Apollo Program?",
			"answer": "Florida",
			"supporting_facts": [["Apollo Program", 0], ["Kennedy Space Center", 0]],
			"context": [
				["Apollo Program", ["Apollo Program launched from Kennedy Space Center."]],
				["Kennedy Space Center", ["Kennedy Space Center is in Florida."]],
				["Florida", ["Florida is a state in the United States."]],
			],
		}
	]


@fixture
def hotpotqa_distractor_file(tmp_path: Path, hotpotqa_distractor_questions: list[dict]) -> Path:
	questions_path = tmp_path / "hotpotqa-distractor.json"
	questions_path.write_text(json.dumps(hotpotqa_distractor_questions, ensure_ascii=False), encoding="utf-8")
	return questions_path


@fixture
def hotpotqa_fullwiki_graph_records() -> list[dict]:
	return [
		{
			"node_id": "Apollo Program",
			"title": "Apollo Program",
			"sentences": ["Apollo Program launched from Kennedy Space Center."],
			"links": [
				{
					"target": "Kennedy Space Center",
					"anchor_text": "Kennedy Space Center",
					"sentence": "Apollo Program launched from Kennedy Space Center.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Kennedy Space Center",
			"title": "Kennedy Space Center",
			"sentences": ["Kennedy Space Center is in Florida."],
			"links": [
				{
					"target": "Florida",
					"anchor_text": "Florida",
					"sentence": "Kennedy Space Center is in Florida.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Florida",
			"title": "Florida",
			"sentences": ["Florida is a state in the United States."],
			"links": [],
		},
	]


@fixture
def hotpotqa_fullwiki_questions() -> list[dict]:
	return [
		{
			"_id": "hf1",
			"question": "Which state contains Kennedy Space Center?",
			"answer": "Florida",
			"supporting_facts": [["Apollo Program", 0], ["Kennedy Space Center", 0], ["Florida", 0]],
		}
	]


@fixture
def hotpotqa_fullwiki_files(
	tmp_path: Path,
	hotpotqa_fullwiki_graph_records: list[dict],
	hotpotqa_fullwiki_questions: list[dict],
) -> tuple[Path, Path]:
	graph_path = tmp_path / "hotpotqa-fullwiki-graph.json"
	questions_path = tmp_path / "hotpotqa-fullwiki-questions.json"
	graph_path.write_text(json.dumps(hotpotqa_fullwiki_graph_records, ensure_ascii=False), encoding="utf-8")
	questions_path.write_text(json.dumps(hotpotqa_fullwiki_questions, ensure_ascii=False), encoding="utf-8")
	return questions_path, graph_path


@fixture
def prepared_hotpotqa_store(hotpotqa_fullwiki_files, tmp_path: Path):
	questions_path, graph_path = hotpotqa_fullwiki_files
	output_dir = tmp_path / "hotpotqa-store"
	return prepare_normalized_graph_store(
		output_dir,
		dataset_name="hotpotqa-fullwiki",
		questions_source=questions_path,
		graph_source=graph_path,
	)


@fixture
def iirc_raw_archive(tmp_path: Path) -> Path:
	dev_payload = [
		{
			"title": "Moon Launch Program",
			"text": "Moon Launch Program launches from Cape Canaveral.",
			"links": [{"target": "Cape Canaveral", "indices": [33, 48]}],
			"questions": [
				{
					"qid": "iirc-raw-1",
					"question": "Which state contains the launch city?",
					"answer": {"type": "span", "answer_spans": [{"text": "Florida", "passage": "Florida"}]},
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
	archive_path = tmp_path / "iirc_train_dev.tgz"
	with tarfile.open(archive_path, "w:gz") as archive:
		for name, payload in {
			"dev.json": dev_payload,
			"train.json": dev_payload,
			"context_articles.json": context_articles,
		}.items():
			data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
			info = tarfile.TarInfo(name=name)
			info.size = len(data)
			archive.addfile(info, io.BytesIO(data))
	return archive_path


@fixture
def musique_raw_split_files(tmp_path: Path) -> dict[str, Path]:
	record = {
		"id": "m-raw-1",
		"question": "Which state contains Kennedy Space Center?",
		"answer": "Florida",
		"paragraphs": [
			{"title": "Apollo Program", "paragraph_text": "Apollo Program launched from Kennedy Space Center."},
			{"title": "Kennedy Space Center", "paragraph_text": "Kennedy Space Center is in Florida."},
			{"title": "Florida", "paragraph_text": "Florida is a state in the United States."},
		],
		"paragraph_support_idx": [0, 1, 2],
		"question_decomposition": [{"paragraph_idx": 0}, {"paragraph_idx": 1}, {"paragraph_idx": 2}],
	}
	paths: dict[str, Path] = {}
	for split in ("train", "dev", "test"):
		path = tmp_path / f"musique_full_v1.0_{split}.jsonl"
		path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
		paths[split] = path
	return paths


@fixture
def hotpotqa_raw_split_files(
	tmp_path: Path,
	hotpotqa_distractor_questions: list[dict],
	hotpotqa_fullwiki_questions: list[dict],
) -> dict[str, dict[str, Path]]:
	distractor_dev = tmp_path / "hotpot_dev_distractor_v1.json"
	distractor_train = tmp_path / "hotpot_train_v1.1.json"
	fullwiki_dev = tmp_path / "hotpot_dev_fullwiki_v1.json"
	distractor_dev.write_text(json.dumps(hotpotqa_distractor_questions, ensure_ascii=False), encoding="utf-8")
	distractor_train.write_text(json.dumps(hotpotqa_distractor_questions, ensure_ascii=False), encoding="utf-8")
	fullwiki_dev.write_text(json.dumps(hotpotqa_fullwiki_questions, ensure_ascii=False), encoding="utf-8")
	return {
		"distractor": {"dev": distractor_dev, "train": distractor_train},
		"fullwiki": {"dev": fullwiki_dev, "train": distractor_train},
	}


@fixture
def docs_files(tmp_path: Path) -> tuple[Path, Path]:
	docs_root = tmp_path / "python-docs"
	docs_root.mkdir()
	(docs_root / "index.html").write_text(
		"""
		<html>
		  <head><title>Python Docs</title></head>
		  <body>
		    <p>Start with the <a href="guide.html">guide</a>.</p>
		  </body>
		</html>
		""",
		encoding="utf-8",
	)
	(docs_root / "guide.html").write_text(
		"""
		<html>
		  <head><title>TLS Guide</title></head>
		  <body>
		    <p>The guide points to the <a href="api.html">API reference</a>.</p>
		    <p>Enable TLS by configuring the client.</p>
		  </body>
		</html>
		""",
		encoding="utf-8",
	)
	(docs_root / "api.html").write_text(
		"""
		<html>
		  <head><title>API Reference</title></head>
		  <body>
		    <p>The <a href="index.html">documentation home</a> links here.</p>
		    <p>Client.connect enables TLS mode.</p>
		  </body>
		</html>
		""",
		encoding="utf-8",
	)
	questions_path = tmp_path / "docs-questions.json"
	questions_path.write_text(
		json.dumps(
			[
				{
					"case_id": "d1",
					"question": "Which page explains TLS mode?",
					"answer": "API Reference",
					"gold_support_nodes": ["guide", "api"],
					"gold_start_nodes": ["guide"],
					"gold_path_nodes": ["guide", "api"],
				}
			],
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	return questions_path, docs_root
