from webwalker.graph import DocumentNode, LinkContext, LinkContextGraph
from webwalker.selector import (
	SELECTION_BUDGET_PRESETS,
	SelectionBudget,
	SelectionMode,
	SemanticAStarSelector,
	SemanticBeamSelector,
	SemanticGBFSSelector,
	SemanticPPRSelector,
	SemanticUCSSelector,
)


def test_selector_returns_weighted_subgraph_contract(sample_graph):
	selection = SemanticBeamSelector().select(
		sample_graph,
		"Which city hosts the launch site?",
		["mission", "cape"],
		SelectionBudget(max_nodes=4, max_hops=3, max_tokens=256),
	)
	assert selection.strategy == "semantic_beam"
	assert selection.mode == SelectionMode.STANDALONE
	assert selection.ranked_nodes
	assert selection.selected_node_ids
	assert all(link.source in set(selection.selected_node_ids) for link in selection.selected_links)
	assert all(link.target in set(selection.selected_node_ids) for link in selection.selected_links)
	assert selection.token_cost_estimate <= 256


def test_selection_budget_presets_are_supported(sample_graph):
	for budget in SELECTION_BUDGET_PRESETS.values():
		selection = SemanticBeamSelector().select(
			sample_graph,
			"Which city hosts the launch site?",
			["mission", "cape"],
			budget,
		)
		if budget.max_tokens is not None:
			assert selection.token_cost_estimate <= budget.max_tokens


def test_semantic_beam_keeps_bridge_path():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start", "Start", ("Start references several launch site notes.",)),
			DocumentNode("flashy", "Flashy City", ("Flashy City has launch-site trivia.",)),
			DocumentNode("bridge", "Bridge Doc", ("Bridge Doc points to Florida for launch-site evidence.",)),
			DocumentNode("florida", "Florida", ("Florida contains the launch site.",)),
		]
	)
	graph.add_link(
		LinkContext(
			source="start",
			target="flashy",
			anchor_text="launch site",
			sentence="Start references a launch site showcase.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="start",
			target="bridge",
			anchor_text="launch site records",
			sentence="Bridge Doc keeps launch site records for Florida.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="bridge",
			target="florida",
			anchor_text="Florida",
			sentence="Bridge Doc says the launch site is in Florida.",
			sent_idx=0,
		)
	)

	selection = SemanticBeamSelector().select(
		graph,
		"Which state contains the launch site?",
		["start"],
		SelectionBudget(max_nodes=4, max_hops=2, max_tokens=256),
	)
	assert "bridge" in selection.selected_node_ids
	assert "florida" in selection.selected_node_ids


def test_astar_gbfs_and_ucs_diverge_on_priority_functions():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start", "Start", ("Start connects to two launch clues.",)),
			DocumentNode("a", "Route A", ("Route A eventually reaches the state clue.",)),
			DocumentNode("b", "Route B", ("Route B eventually reaches the state clue.",)),
			DocumentNode("goal", "Goal", ("Goal confirms the launch state.",)),
		]
	)
	graph.add_link(
		LinkContext(
			source="start",
			target="a",
			anchor_text="launch",
			sentence="noise",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="start",
			target="b",
			anchor_text="launch",
			sentence="launch",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="a",
			target="goal",
			anchor_text="state",
			sentence="state proof",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="b",
			target="goal",
			anchor_text="state",
			sentence="state proof",
			sent_idx=0,
		)
	)

	budget = SelectionBudget(max_nodes=4, max_hops=2, max_tokens=256)
	astar = SemanticAStarSelector().select(graph, "launch state", ["start"], budget)
	gbfs = SemanticGBFSSelector().select(graph, "launch state", ["start"], budget)
	ucs = SemanticUCSSelector().select(graph, "launch state", ["start"], budget)

	assert _second_pop(astar.debug_trace) == "b"
	assert _second_pop(gbfs.debug_trace) == "a"
	assert _second_pop(ucs.debug_trace) == "b"


def test_semantic_ppr_prefers_relevant_branch_over_noise():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start", "Start", ("Start links broadly.",)),
			DocumentNode("relevant", "Relevant", ("Relevant page discusses the launch site.",)),
			DocumentNode("support", "Support", ("Support page mentions Florida.",)),
			DocumentNode("noise1", "Noise One", ("Noise one is unrelated.",)),
			DocumentNode("noise2", "Noise Two", ("Noise two is unrelated.",)),
			DocumentNode("noise3", "Noise Three", ("Noise three is unrelated.",)),
		]
	)
	graph.add_link(
		LinkContext(
			source="start",
			target="relevant",
			anchor_text="launch site",
			sentence="Relevant branch contains launch site evidence.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="relevant",
			target="support",
			anchor_text="Florida",
			sentence="The launch site is in Florida.",
			sent_idx=0,
		)
	)
	for noise in ("noise1", "noise2", "noise3"):
		graph.add_link(
			LinkContext(
				source="start",
				target=noise,
				anchor_text="misc",
				sentence="unrelated",
				sent_idx=0,
			)
		)

	selection = SemanticPPRSelector().select(
		graph,
		"launch florida",
		["start"],
		SelectionBudget(max_nodes=4, max_hops=3, max_tokens=256),
	)
	assert _rank_of(selection.selected_node_ids, "relevant") < _rank_of(selection.selected_node_ids, "noise1")
	assert _rank_of(selection.selected_node_ids, "support") < _rank_of(selection.selected_node_ids, "noise1")


def test_hybrid_beam_ppr_improves_support_recall():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start", "Start", ("Mission references a launch site.",)),
			DocumentNode("bridge", "Bridge", ("Bridge knows the launch site state.",)),
			DocumentNode("florida", "Florida", ("Florida contains the launch site.",)),
			DocumentNode("county", "Brevard County", ("Brevard County contains the Florida launch area.",)),
		]
	)
	graph.add_link(
		LinkContext(
			source="start",
			target="bridge",
			anchor_text="launch site",
			sentence="Bridge tracks the launch site.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="bridge",
			target="florida",
			anchor_text="Florida",
			sentence="The launch site is in Florida.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="florida",
			target="county",
			anchor_text="county",
			sentence="Florida points to Brevard County for the launch region.",
			sent_idx=0,
		)
	)

	budget = SelectionBudget(max_nodes=4, max_hops=2, max_tokens=256)
	standalone = SemanticBeamSelector().select(graph, "launch florida county", ["start"], budget)
	hybrid = SemanticBeamSelector(mode=SelectionMode.HYBRID_WITH_PPR).select(
		graph,
		"launch florida county",
		["start"],
		budget,
	)
	supporting = {"florida", "county"}
	assert _support_recall(hybrid.selected_node_ids, supporting) > _support_recall(
		standalone.selected_node_ids,
		supporting,
	)


def _second_pop(debug_trace: list[str]) -> str:
	pops = [line for line in debug_trace if line.startswith("pop:")]
	return pops[1].split(":")[1]


def _rank_of(node_ids: list[str], node_id: str) -> int:
	return node_ids.index(node_id)


def _support_recall(selected_node_ids: list[str], supporting_node_ids: set[str]) -> float:
	return len(set(selected_node_ids) & supporting_node_ids) / len(supporting_node_ids)
