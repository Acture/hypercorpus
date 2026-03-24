from hypercorpus.datasets.twowiki import load_2wiki_graph, load_2wiki_questions


def test_load_2wiki_graph_uses_titles_as_node_ids(two_wiki_files):
	_questions_path, graph_path = two_wiki_files
	graph = load_2wiki_graph(graph_path)

	assert "Moon Launch Program" in graph.nodes
	assert graph.neighbors("Moon Launch Program") == ["Cape Canaveral", "Alice Johnson"]
	assert graph.links_between("Moon Launch Program", "Cape Canaveral")[0].anchor_text == "Cape Canaveral"


def test_load_2wiki_questions_maps_supporting_facts(two_wiki_files):
	questions_path, _graph_path = two_wiki_files
	cases = load_2wiki_questions(questions_path)

	assert [case.case_id for case in cases] == ["q1", "q2"]
	assert cases[0].gold_support_nodes == ["Moon Launch Program", "Cape Canaveral"]
	assert cases[0].gold_start_nodes == ["Moon Launch Program", "Cape Canaveral"]
	assert cases[0].dataset_name == "2wikimultihop"
	assert cases[0].question_type == "bridge"
