from hypercorpus.datasets import (
	HotpotQAAdapter,
	build_hotpotqa_distractor_graph_for_case,
	load_hotpotqa_graph,
	load_hotpotqa_questions,
)


def test_hotpotqa_distractor_builds_case_local_graph(
	hotpotqa_distractor_questions, hotpotqa_distractor_file
):
	record = hotpotqa_distractor_questions[0]
	graph = build_hotpotqa_distractor_graph_for_case(record)
	cases = load_hotpotqa_questions(hotpotqa_distractor_file, variant="distractor")

	assert set(graph.nodes) == {
		"h1::Apollo Program",
		"h1::Kennedy Space Center",
		"h1::Florida",
	}
	assert cases[0].dataset_name == "hotpotqa-distractor"
	assert cases[0].gold_support_nodes == [
		"h1::Apollo Program",
		"h1::Kennedy Space Center",
	]


def test_hotpotqa_fullwiki_adapter_loads_graph_and_cases(hotpotqa_fullwiki_files):
	questions_path, graph_path = hotpotqa_fullwiki_files
	adapter = HotpotQAAdapter(variant="fullwiki")

	graph = load_hotpotqa_graph(graph_path, variant="fullwiki")
	cases = adapter.load_cases(questions_path)

	assert len(graph.nodes) == 3
	assert cases[0].dataset_name == "hotpotqa-fullwiki"
	assert cases[0].gold_support_nodes == [
		"Apollo Program",
		"Kennedy Space Center",
		"Florida",
	]
