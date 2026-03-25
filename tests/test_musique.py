from hypercorpus.datasets import (
	MuSiQueAdapter,
	load_musique_graph,
	load_musique_questions,
)


def test_load_musique_graph_and_questions(musique_files):
	questions_path, graph_path = musique_files

	graph = load_musique_graph(graph_path)
	cases = load_musique_questions(questions_path)

	assert len(graph.nodes) == 3
	assert len(cases) == 1
	assert cases[0].dataset_name == "musique"
	assert cases[0].gold_support_nodes == [
		"Apollo Program",
		"Kennedy Space Center",
		"Florida",
	]
	assert cases[0].gold_start_nodes == ["Apollo Program"]
	assert cases[0].gold_path_nodes == [
		"Apollo Program",
		"Kennedy Space Center",
		"Florida",
	]


def test_musique_adapter_loads_graph_and_cases(musique_files):
	questions_path, graph_path = musique_files
	adapter = MuSiQueAdapter()

	graph = adapter.load_graph(graph_path)
	cases = adapter.load_cases(questions_path)

	assert len(graph.nodes) == 3
	assert cases[0].query == "Which state contains Kennedy Space Center?"
