from webwalker.datasets.docs import load_docs_graph, load_docs_questions


def test_load_docs_graph_parses_internal_links(docs_files):
	_questions_path, docs_root = docs_files
	graph = load_docs_graph(docs_root, dataset_name="python_docs")

	assert graph.neighbors("index") == ["guide"]
	assert graph.neighbors("guide") == ["api"]
	assert graph.links_between("guide", "api")[0].anchor_text == "API reference"
	assert graph.node_attr["api"]["dataset"] == "python_docs"


def test_load_docs_questions_uses_dataset_name_override(docs_files):
	questions_path, _docs_root = docs_files
	cases = load_docs_questions(questions_path, dataset_name="python_docs")

	assert cases[0].dataset_name == "python_docs"
	assert cases[0].gold_support_nodes == ["guide", "api"]
