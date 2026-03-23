from hypercorpus.datasets.iirc import load_iirc_graph, load_iirc_questions


def test_load_iirc_graph_builds_link_graph(iirc_files):
	_questions_path, graph_path = iirc_files
	graph = load_iirc_graph(graph_path)

	assert graph.neighbors("Moon Launch Program") == ["Cape Canaveral"]
	assert graph.neighbors("Cape Canaveral") == ["Florida"]


def test_load_iirc_questions_supports_optional_path_nodes(iirc_files):
	questions_path, _graph_path = iirc_files
	cases = load_iirc_questions(questions_path)

	assert [case.case_id for case in cases] == ["i1", "i2"]
	assert cases[0].gold_path_nodes is None
	assert cases[1].gold_path_nodes == ["Moon Launch Program", "Cape Canaveral"]
	assert cases[0].dataset_name == "iirc"
