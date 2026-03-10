import bz2
import io
import json
import tarfile
from pathlib import Path

from pytest import fixture

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
