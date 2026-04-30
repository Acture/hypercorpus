from __future__ import annotations

from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.graph import DocumentNode, LinkContext, LinkContextGraph
from hypercorpus.selector import (
	available_selector_names,
	build_selector,
	parse_selector_spec,
)


BFS_2HOP = "top_1_seed__sentence_transformer__hop_0__bfs_khop_2"


class FakeEmbedder:
	backend_name = "sentence_transformer"
	model_name = "multi-qa-MiniLM-L6-cos-v1"

	def __init__(self, vectors: dict[str, list[float]]):
		self.vectors = vectors

	def encode(self, texts):
		return [self.vectors.get(text, [0.0, 0.0]) for text in texts]


def _selector_with_seed_vector(query: str = "launch seed"):
	return build_selector(
		BFS_2HOP,
		sentence_transformer_embedder_factory=lambda _config: FakeEmbedder(
			{
				query: [1.0, 0.0],
				"Seed Seed": [1.0, 0.0],
				"Seed s": [1.0, 0.0],
			}
		),
	)


def _shortest_depths(
	graph: LinkContextGraph, seed: str, max_depth: int
) -> dict[str, int]:
	depths = {seed: 0}
	queue = [seed]
	queue_index = 0
	while queue_index < len(queue):
		source = queue[queue_index]
		queue_index += 1
		if depths[source] >= max_depth:
			continue
		for target in graph.neighbors(source):
			if target in depths:
				continue
			depths[target] = depths[source] + 1
			queue.append(target)
	return depths


def test_bfs_khop_selector_is_registered_and_parseable():
	assert BFS_2HOP in available_selector_names(include_diagnostics=False)

	spec = parse_selector_spec(BFS_2HOP)
	future_spec = parse_selector_spec(
		"top_1_seed__sentence_transformer__hop_0__bfs_khop_3"
	)
	selector = build_selector(
		BFS_2HOP,
		sentence_transformer_embedder_factory=lambda _config: FakeEmbedder({}),
	)

	assert spec.family == "baseline"
	assert spec.seed_strategy == "sentence_transformer"
	assert spec.seed_top_k == 1
	assert spec.hop_budget == 0
	assert spec.baseline == "bfs_khop"
	assert spec.bfs_khop_depth == 2
	assert future_spec.bfs_khop_depth == 3
	assert selector.name == BFS_2HOP


def test_bfs_khop_selector_selects_nonempty_nodes_within_two_hops():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("seed", "Seed", ("Seed",)),
			DocumentNode("hop1", "Hop One", ("one",)),
			DocumentNode("hop2", "Hop Two", ("two",)),
			DocumentNode("hop3", "Hop Three", ("three",)),
		]
	)
	graph.add_link(LinkContext("seed", "hop1", "to hop1", "Seed links to hop1.", 0))
	graph.add_link(LinkContext("hop1", "hop2", "to hop2", "Hop1 links to hop2.", 0))
	graph.add_link(LinkContext("hop2", "hop3", "to hop3", "Hop2 links to hop3.", 0))
	selector = _selector_with_seed_vector()

	result = selector.select(
		graph,
		EvaluationCase(case_id="bfs-depth", query="launch seed"),
		EvaluationBudget(token_budget_tokens=32),
	)
	depths = _shortest_depths(graph, result.root_node_ids[0], max_depth=2)

	assert result.selected_node_ids
	assert result.root_node_ids == ["seed"]
	assert "hop3" not in result.selected_node_ids
	assert all(depths[node_id] <= 2 for node_id in result.selected_node_ids)


def test_bfs_khop_selector_tie_breaks_by_depth_then_degree():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("seed", "Seed", ("s",)),
			DocumentNode("low", "Low", ("l",)),
			DocumentNode("high", "High", ("h",)),
			DocumentNode("deep_high", "Deep", ("d",)),
			DocumentNode("h1", "H One", ("a",)),
			DocumentNode("h2", "H Two", ("b",)),
			DocumentNode("h3", "H Three", ("c",)),
			DocumentNode("d1", "D One", ("e",)),
			DocumentNode("d2", "D Two", ("f",)),
			DocumentNode("d3", "D Three", ("g",)),
			DocumentNode("d4", "D Four", ("i",)),
		]
	)
	graph.add_link(LinkContext("seed", "low", "low", "Seed links to low.", 0))
	graph.add_link(LinkContext("seed", "high", "high", "Seed links to high.", 0))
	graph.add_link(LinkContext("low", "deep_high", "deep", "Low links deep.", 0))
	for index, target in enumerate(("h1", "h2", "h3")):
		graph.add_link(LinkContext("high", target, target, "High fans out.", index))
	for index, target in enumerate(("d1", "d2", "d3", "d4")):
		graph.add_link(
			LinkContext("deep_high", target, target, "Deep fans out.", index)
		)
	selector = _selector_with_seed_vector(query="tie query")

	result = selector.select(
		graph,
		EvaluationCase(case_id="bfs-tie", query="tie query"),
		EvaluationBudget(token_budget_tokens=3),
	)

	assert result.selected_node_ids == ["seed", "high", "low"]
	assert "deep_high" not in result.selected_node_ids
