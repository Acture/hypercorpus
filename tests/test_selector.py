import pytest

from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.graph import DocumentNode, LinkContext, LinkContextGraph
from hypercorpus.selector import (
	_controller_runtime_max_steps,
	SelectorBudget,
	SelectionMode,
	SemanticAStarSelector,
	SemanticBeamSelector,
	SemanticPPRSelector,
	SemanticUCSSelector,
	SentenceTransformerStepScorer,
	available_selector_names,
	build_selector,
	parse_selector_spec,
)
from hypercorpus.walker import DynamicWalker, LinkContextOverlapStepScorer, WalkBudget


class FakeEmbedder:
	backend_name = "sentence_transformer"
	model_name = "multi-qa-MiniLM-L6-cos-v1"

	def __init__(self, vectors: dict[str, list[float]]):
		self.vectors = vectors

	def encode(self, texts):
		return [self.vectors[text] for text in texts]


def test_parse_selector_spec_handles_canonical_forms():
	dense = parse_selector_spec("top_1_seed__lexical_overlap__hop_0__dense")
	iterative = parse_selector_spec(
		"top_1_seed__sentence_transformer__hop_2__iterative_dense"
	)
	beam = parse_selector_spec(
		"top_3_seed__sentence_transformer__hop_3__beam__link_context_llm__lookahead_2"
	)
	controller = parse_selector_spec(
		"top_1_seed__sentence_transformer__hop_adaptive__single_path_walk__link_context_llm_controller__lookahead_2"
	)
	multipath = parse_selector_spec(
		"top_1_seed__sentence_transformer__hop_adaptive__constrained_multipath__link_context_llm_controller__lookahead_2"
	)

	assert dense.family == "baseline"
	assert dense.seed_strategy == "lexical_overlap"
	assert dense.seed_top_k == 1
	assert dense.hop_budget == 0
	assert dense.baseline == "dense"
	assert iterative.family == "baseline"
	assert iterative.seed_strategy == "sentence_transformer"
	assert iterative.hop_budget == 2
	assert iterative.baseline == "iterative_dense"
	assert beam.family == "path_search"
	assert beam.seed_strategy == "sentence_transformer"
	assert beam.seed_top_k == 3
	assert beam.hop_budget == 3
	assert beam.search_structure == "beam"
	assert beam.edge_scorer == "link_context_llm"
	assert beam.lookahead_depth == 2
	assert controller.search_structure == "single_path_walk"
	assert controller.edge_scorer == "link_context_llm_controller"
	assert controller.lookahead_depth == 2
	assert controller.hop_budget is None
	assert multipath.search_structure == "constrained_multipath"
	assert multipath.edge_scorer == "link_context_llm_controller"
	assert multipath.hop_budget is None


@pytest.mark.parametrize(
	"selector_name",
	[
		"top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_overlap__lookahead_2",
		"top_1_seed__sentence_transformer__hop_adaptive__constrained_multipath__link_context_llm_controller__lookahead_1",
		"top_1_seed__sentence_transformer__hop_adaptive__beam__link_context_llm_controller__lookahead_2",
		"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm_controller__lookahead_2",
		"top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2",
		"top_1_seed__sentence_transformer__hop_adaptive__single_path_walk__link_context_overlap__lookahead_1",
	],
)
def test_parse_selector_spec_rejects_invalid_controller_search_combinations(
	selector_name,
):
	with pytest.raises(ValueError) as exc:
		parse_selector_spec(selector_name)

	assert str(exc.value) == f"Unknown selector: {selector_name}"


def test_controller_only_hop_adaptive_uses_internal_safety_cap():
	names = available_selector_names(include_diagnostics=False)
	spec = parse_selector_spec(
		"top_1_seed__sentence_transformer__hop_adaptive__single_path_walk__link_context_llm_controller__lookahead_2"
	)

	assert (
		"top_1_seed__sentence_transformer__hop_adaptive__single_path_walk__link_context_llm_controller__lookahead_2"
		in names
	)
	assert (
		"top_1_seed__sentence_transformer__hop_adaptive__constrained_multipath__link_context_llm_controller__lookahead_2"
		in names
	)
	assert (
		"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm_controller__lookahead_2"
		not in names
	)
	assert _controller_runtime_max_steps(spec) == 11


def test_parse_selector_spec_accepts_budget_fill_suffixes_and_rejects_diagnostics():
	filled = parse_selector_spec(
		"top_1_seed__lexical_overlap__hop_0__dense__budget_fill_always"
	)

	assert (
		filled.canonical_name
		== "top_1_seed__lexical_overlap__hop_0__dense__budget_fill_always"
	)
	assert filled.base_canonical_name == "top_1_seed__lexical_overlap__hop_0__dense"
	assert filled.budget_fill_mode == "always"
	assert filled.budget_fill_pool_k == 64

	try:
		parse_selector_spec("gold_support_context__budget_fill_always")
	except ValueError as exc:
		assert str(exc) == "Unknown selector: gold_support_context__budget_fill_always"
	else:
		raise AssertionError("expected diagnostic fill suffix to be rejected")


def test_parse_selector_spec_accepts_profile_suffix_before_budget_fill():
	spec = parse_selector_spec(
		"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop"
	)

	assert spec.canonical_name == (
		"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop"
	)
	assert spec.base_canonical_name == (
		"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2"
	)
	assert spec.profile_name == "st_future_heavy"
	assert spec.budget_fill_mode == "relative_drop"
	assert spec.budget_fill_relative_drop_ratio == 0.5


@pytest.mark.parametrize(
	"selector_name",
	[
		"top_1_seed__lexical_overlap__hop_2__mdr_light",
		"top_1_seed__sentence_transformer__hop_0__mdr_light",
		"top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_unknown",
	],
)
def test_parse_selector_spec_rejects_invalid_mdr_light_and_profile_combinations(
	selector_name,
):
	with pytest.raises(ValueError) as exc:
		parse_selector_spec(selector_name)

	assert str(exc.value) == f"Unknown selector: {selector_name}"


def test_semantic_beam_returns_weighted_subgraph_contract(sample_graph):
	selection = SemanticBeamSelector().select(
		sample_graph,
		"Which city hosts the launch site?",
		["mission", "cape"],
		SelectorBudget(max_nodes=4, max_hops=3, max_tokens=256),
	)
	assert selection.strategy == "semantic_beam"
	assert selection.mode == SelectionMode.STANDALONE
	assert selection.ranked_nodes
	assert selection.selected_node_ids
	assert all(
		link.source in set(selection.selected_node_ids)
		for link in selection.selected_links
	)
	assert all(
		link.target in set(selection.selected_node_ids)
		for link in selection.selected_links
	)
	assert selection.token_cost_estimate <= 256


def test_dynamic_walker_resumes_from_checkpointed_step(sample_graph):
	walker = DynamicWalker(sample_graph, scorer=LinkContextOverlapStepScorer())
	checkpoints: list[dict] = []

	full = walker.walk(
		"Which state contains Cape Canaveral?",
		["mission"],
		WalkBudget(max_steps=3, min_score=0.0, allow_revisit=False),
		checkpoint_callback=lambda payload: checkpoints.append(payload),
	)

	resumed = walker.walk(
		"Which state contains Cape Canaveral?",
		["mission"],
		WalkBudget(max_steps=3, min_score=0.0, allow_revisit=False),
		resume_state=checkpoints[0],
	)

	assert checkpoints
	assert [step.node_id for step in resumed.steps] == [
		step.node_id for step in full.steps
	]
	assert [log.current_node_id for log in resumed.selector_logs] == [
		log.current_node_id for log in full.selector_logs
	]


def test_semantic_beam_resumes_from_checkpointed_expansion(sample_graph):
	selector = SemanticBeamSelector()
	checkpoints: list[dict] = []

	full = selector.select(
		sample_graph,
		"Which state contains Cape Canaveral?",
		["mission"],
		SelectorBudget(max_nodes=4, max_hops=3, max_tokens=256),
		checkpoint_callback=lambda payload: checkpoints.append(payload),
	)
	resumed = selector.select(
		sample_graph,
		"Which state contains Cape Canaveral?",
		["mission"],
		SelectorBudget(max_nodes=4, max_hops=3, max_tokens=256),
		resume_state=checkpoints[0],
	)

	assert checkpoints
	assert resumed.selected_node_ids == full.selected_node_ids
	assert len(resumed.selector_logs) == len(full.selector_logs)


def test_iterative_dense_resumes_from_checkpointed_hop(sample_graph):
	selector = build_selector("top_1_seed__lexical_overlap__hop_2__iterative_dense")
	case = EvaluationCase(
		case_id="q-resume",
		query="Which state contains Cape Canaveral?",
		gold_support_nodes=["mission", "cape", "florida"],
		gold_start_nodes=["mission"],
		dataset_name="synthetic",
	)
	budget = EvaluationBudget(token_budget_tokens=128)
	checkpoints: list[dict] = []

	full = selector.select(
		sample_graph,
		case,
		budget,  # ty: ignore[invalid-argument-type] # Protocol vs concrete
		checkpoint_callback=lambda payload: checkpoints.append(payload),  # ty: ignore[unknown-argument] # concrete impl kwargs
	)
	resumed = selector.select(sample_graph, case, budget, resume_state=checkpoints[0])  # ty: ignore[invalid-argument-type,unknown-argument] # concrete impl kwargs

	assert checkpoints
	assert resumed.selected_node_ids == full.selected_node_ids


def test_budget_fill_resumes_from_checkpointed_fill_state(sample_graph):
	selector = build_selector(
		"top_1_seed__lexical_overlap__hop_0__dense__budget_fill_always"
	)
	case = EvaluationCase(
		case_id="q-fill",
		query="Which state contains Cape Canaveral?",
		gold_support_nodes=["mission", "cape", "florida"],
		gold_start_nodes=["mission"],
		dataset_name="synthetic",
	)
	budget = EvaluationBudget(token_budget_tokens=128)
	checkpoints: list[dict] = []

	full = selector.select(
		sample_graph,
		case,
		budget,  # ty: ignore[invalid-argument-type] # Protocol vs concrete
		checkpoint_callback=lambda payload: checkpoints.append(payload),  # ty: ignore[unknown-argument] # concrete impl kwargs
	)
	resumed = selector.select(sample_graph, case, budget, resume_state=checkpoints[0])  # ty: ignore[invalid-argument-type,unknown-argument] # concrete impl kwargs

	assert checkpoints
	assert resumed.selected_node_ids == full.selected_node_ids


def test_semantic_beam_keeps_bridge_path():
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"start", "Start", ("Start references several launch site notes.",)
			),
			DocumentNode(
				"flashy", "Flashy City", ("Flashy City has launch-site trivia.",)
			),
			DocumentNode(
				"bridge",
				"Bridge Doc",
				("Bridge Doc points to Florida for launch-site evidence.",),
			),
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
		SelectorBudget(max_nodes=4, max_hops=2, max_tokens=256),
	)
	assert "bridge" in selection.selected_node_ids
	assert "florida" in selection.selected_node_ids


def test_astar_and_ucs_run_on_same_contract():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start", "Start", ("Start connects to two launch clues.",)),
			DocumentNode(
				"a", "Route A", ("Route A eventually reaches the state clue.",)
			),
			DocumentNode(
				"b", "Route B", ("Route B eventually reaches the state clue.",)
			),
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

	budget = SelectorBudget(max_nodes=4, max_hops=2, max_tokens=256)
	astar = SemanticAStarSelector().select(graph, "launch state", ["start"], budget)
	ucs = SemanticUCSSelector().select(graph, "launch state", ["start"], budget)

	assert astar.selected_node_ids
	assert ucs.selected_node_ids
	assert any(line.startswith("pop:") for line in astar.debug_trace)
	assert any(line.startswith("pop:") for line in ucs.debug_trace)


def test_semantic_ppr_prefers_relevant_branch_over_noise():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start", "Start", ("Start links broadly.",)),
			DocumentNode(
				"relevant", "Relevant", ("Relevant page discusses the launch site.",)
			),
			DocumentNode("support", "Support", ("Support page mentions Florida.",)),
			DocumentNode("noise1", "Noise One", ("Noise one is unrelated.",)),
			DocumentNode("noise2", "Noise Two", ("Noise two is unrelated.",)),
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
	for noise in ("noise1", "noise2"):
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
		SelectorBudget(max_nodes=4, max_hops=3, max_tokens=256),
	)
	assert selection.selected_node_ids.index(
		"relevant"
	) < selection.selected_node_ids.index("noise1")
	assert selection.selected_node_ids.index(
		"support"
	) < selection.selected_node_ids.index("noise1")


def test_canonical_dense_selector_uses_seed_top_k(sample_graph):
	selector = build_selector("top_1_seed__lexical_overlap__hop_0__dense")
	result = selector.select(
		sample_graph,
		EvaluationCase(case_id="q1", query="Which city hosts the launch site?"),
		EvaluationBudget(token_budget_tokens=256),  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	)

	assert result.selected_node_ids == ["mission"]


def test_sentence_transformer_seed_strategy_uses_embedder_ranking():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("alpha", "Alpha", ("Alpha contains launch facts.",)),
			DocumentNode("beta", "Beta", ("Beta contains unrelated trivia.",)),
		]
	)
	query = "launch facts"
	alpha_text = "Alpha Alpha contains launch facts."
	beta_text = "Beta Beta contains unrelated trivia."
	selector = build_selector(
		"top_1_seed__sentence_transformer__hop_0__dense",
		sentence_transformer_embedder_factory=lambda _config: FakeEmbedder(
			{
				query: [1.0, 0.0],
				alpha_text: [0.9, 0.0],
				beta_text: [0.1, 0.8],
			}
		),
	)

	result = selector.select(
		graph,
		EvaluationCase(case_id="q-st-seed", query=query),
		EvaluationBudget(token_budget_tokens=128),  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	)

	assert result.selected_node_ids == ["alpha"]
	assert result.selector_metadata is not None
	assert result.selector_metadata.seed_strategy == "sentence_transformer"
	assert result.selector_metadata.seed_backend == "sentence_transformer"
	assert result.selector_metadata.seed_model == "multi-qa-MiniLM-L6-cos-v1"


def test_iterative_dense_selector_retrieves_with_expansion_context():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start", "Start", ("Start discusses the launch mission.",)),
			DocumentNode(
				"bridge", "Bridge", ("Bridge mentions Florida launch records.",)
			),
			DocumentNode(
				"goal", "Goal", ("Goal confirms the launch state is Florida.",)
			),
			DocumentNode("noise", "Noise", ("Noise is unrelated trivia.",)),
		]
	)
	query = "launch state"
	start_text = "Start Start discusses the launch mission."
	bridge_text = "Bridge Bridge mentions Florida launch records."
	goal_text = "Goal Goal confirms the launch state is Florida."
	noise_text = "Noise Noise is unrelated trivia."
	expansion_one = "launch state Start Start discusses the launch mission."
	expansion_two = "launch state Bridge Bridge mentions Florida launch records."

	selector = build_selector(
		"top_1_seed__sentence_transformer__hop_2__iterative_dense",
		sentence_transformer_embedder_factory=lambda _config: FakeEmbedder(
			{
				query: [1.0, 0.0, 0.0],
				expansion_one: [0.0, 1.0, 0.0],
				expansion_two: [0.0, 0.0, 1.0],
				start_text: [0.9, 0.1, 0.0],
				bridge_text: [0.1, 0.95, 0.0],
				goal_text: [0.1, 0.1, 0.95],
				noise_text: [0.0, 0.0, 0.1],
			}
		),
	)

	result = selector.select(
		graph,
		EvaluationCase(case_id="q-iterative", query=query),
		EvaluationBudget(token_budget_tokens=256),  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	)

	assert result.selected_node_ids == ["start", "bridge", "goal"]
	assert result.selector_metadata is not None
	assert result.selector_metadata.backend == "iterative_dense"
	assert result.stop_reason == "iterative_dense_retrieval"


def test_sentence_transformer_edge_scorer_records_future_edge_ids():
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"root",
				"Launch Root",
				("Launch Root offers multiple navigation paths.",),
			),
			DocumentNode(
				"bait", "Bait Page", ("Bait page looks relevant but dead ends.",)
			),
			DocumentNode(
				"bridge", "Bridge Page", ("Bridge page leads onward to the answer.",)
			),
			DocumentNode(
				"answer",
				"Answer Page",
				("Answer page contains the true launch harbor location.",),
			),
		]
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="bait",
			anchor_text="harbor location",
			sentence="generic bait sentence",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="bridge",
			anchor_text="plain note",
			sentence="harbor",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="bridge",
			target="answer",
			anchor_text="harbor location",
			sentence="harbor location",
			sent_idx=0,
		)
	)
	query = "launch navigation root"
	selector = build_selector(
		"top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2",
		sentence_transformer_embedder_factory=lambda _config: FakeEmbedder(
			{
				query: [1.0, 0.0],
				"harbor location generic bait sentence Bait Page Bait page looks relevant but dead ends.": [
					0.2,
					0.0,
				],
				"plain note harbor Bridge Page Bridge page leads onward to the answer.": [
					0.7,
					0.0,
				],
				"harbor location harbor location Answer Page Answer page contains the true launch harbor location.": [
					1.0,
					0.0,
				],
			}
		),
	)

	result = selector.select(
		graph,
		EvaluationCase(case_id="q-st-scorer", query=query),
		EvaluationBudget(token_budget_tokens=128),  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	)

	assert "bridge" in result.selected_node_ids
	assert result.selector_logs
	assert any(
		candidate.best_next_edge_id == "1-0"
		for log in result.selector_logs
		for candidate in log.candidates
	)


def test_overlap_profiles_change_selection_and_metadata():
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"zz_root", "Launch Site Root", ("Launch site root overview.",)
			),
			DocumentNode("anchor", "Anchor Evidence", ("Distractor details only.",)),
			DocumentNode("title", "Launch Site", ("Launch site evidence lives here.",)),
		]
	)
	graph.add_link(
		LinkContext(
			source="zz_root",
			target="anchor",
			anchor_text="launch site",
			sentence="miscellaneous filler",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="zz_root",
			target="title",
			anchor_text="miscellaneous filler",
			sentence="launch site",
			sent_idx=0,
		)
	)
	case = EvaluationCase(case_id="q-overlap-profile", query="launch site")
	budget = EvaluationBudget(token_budget_tokens=128)

	balanced = build_selector(
		"top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced"
	)
	title_aware = build_selector(
		"top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_title_aware"
	)

	balanced_result = balanced.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	title_aware_result = title_aware.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete

	assert balanced_result.selected_node_ids[:2] == ["zz_root", "anchor"]
	assert title_aware_result.selected_node_ids[:2] == ["zz_root", "title"]
	assert balanced_result.selector_metadata is not None
	assert balanced_result.selector_metadata.profile_name == "overlap_balanced"
	assert title_aware_result.selector_metadata is not None
	assert title_aware_result.selector_metadata.profile_name == "overlap_title_aware"


def test_sentence_transformer_profile_uses_configured_direct_weight_for_lookahead_two():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("root", "Root", ("Root context.",)),
			DocumentNode("direct", "Direct", ("Direct evidence.",)),
		]
	)
	link = LinkContext(
		source="root",
		target="direct",
		anchor_text="launch evidence",
		sentence="launch evidence",
		sent_idx=0,
	)
	scorer = SentenceTransformerStepScorer(
		embedder=FakeEmbedder(
			{
				"launch evidence": [1.0, 0.0],
				"launch evidence launch evidence Direct Direct evidence.": [1.0, 0.0],
			}
		),
		lookahead_steps=2,
		direct_weight=0.80,
		future_weight=0.10,
		novelty_weight=0.10,
		profile_name="st_direct_heavy",
	)

	cards = scorer.score_candidates(
		query="launch evidence",
		graph=graph,
		current_node_id="root",
		candidate_links=[link],
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)

	assert len(cards) == 1
	assert cards[0].subscores["future_potential"] == 0.0
	assert cards[0].total_score == pytest.approx(0.9)
	assert scorer.metadata.profile_name == "st_direct_heavy"


def test_mdr_light_differs_from_iterative_dense_for_multi_frontier_dense_hops():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("start_a", "Start Alpha", ("Start alpha context.",)),
			DocumentNode("start_b", "Start Beta", ("Start beta context.",)),
			DocumentNode("goal_a", "Goal Alpha", ("Goal alpha evidence.",)),
			DocumentNode("goal_b", "Goal Beta", ("Goal beta evidence.",)),
			DocumentNode("noise", "Noise", ("Noise distractor.",)),
		]
	)
	query = "launch answer"
	start_a_text = "Start Alpha Start alpha context."
	start_b_text = "Start Beta Start beta context."
	goal_a_text = "Goal Alpha Goal alpha evidence."
	goal_b_text = "Goal Beta Goal beta evidence."
	noise_text = "Noise Noise distractor."
	merged_query = (
		"launch answer Start Alpha Start alpha context. Start Beta Start beta context."
	)
	merged_query_reversed = (
		"launch answer Start Beta Start beta context. Start Alpha Start alpha context."
	)
	alpha_query = "launch answer Start Alpha Start alpha context."
	beta_query = "launch answer Start Beta Start beta context."

	def embedder(_config):
		return FakeEmbedder(
			{
				query: [1.0, 0.0, 0.0, 0.0],
				start_a_text: [0.9, 0.1, 0.0, 0.0],
				start_b_text: [0.9, 0.2, 0.0, 0.0],
				goal_a_text: [0.0, 1.0, 0.0, 0.0],
				goal_b_text: [0.0, 0.0, 1.0, 0.0],
				noise_text: [0.0, 0.0, 0.0, 1.0],
				merged_query: [0.0, 0.0, 0.0, 1.0],
				merged_query_reversed: [0.0, 0.0, 0.0, 1.0],
				alpha_query: [0.0, 1.0, 0.0, 0.0],
				beta_query: [0.0, 0.0, 1.0, 0.0],
			}
		)

	budget = EvaluationBudget(token_budget_tokens=256)
	case = EvaluationCase(case_id="q-mdr-light", query=query)

	iterative = build_selector(
		"top_2_seed__sentence_transformer__hop_1__iterative_dense",
		sentence_transformer_embedder_factory=embedder,
	)
	mdr_light = build_selector(
		"top_2_seed__sentence_transformer__hop_1__mdr_light",
		sentence_transformer_embedder_factory=embedder,
	)

	iterative_result = iterative.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	mdr_light_result = mdr_light.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete

	assert "noise" in iterative_result.selected_node_ids
	assert "goal_a" not in iterative_result.selected_node_ids
	assert "noise" not in mdr_light_result.selected_node_ids
	assert "goal_a" in mdr_light_result.selected_node_ids
	assert iterative_result.selector_metadata is not None
	assert iterative_result.selector_metadata.backend == "iterative_dense"
	assert mdr_light_result.selector_metadata is not None
	assert mdr_light_result.selector_metadata.backend == "mdr_light"
	assert mdr_light_result.stop_reason == "mdr_light_retrieval"


def test_budget_fill_variants_recover_small_nodes_and_stop_differently():
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"giant",
				"Launch City Harbor",
				("launch city harbor " + "filler " * 40,),
			),
			DocumentNode(
				"compact",
				"Launch City",
				("launch city",),
			),
			DocumentNode(
				"weak",
				"Launch",
				("launch",),
			),
			DocumentNode(
				"tiny",
				"Remote",
				("remote",),
			),
		]
	)
	case = EvaluationCase(case_id="budget-fill", query="launch city harbor")
	budget = EvaluationBudget(token_budget_tokens=6)

	canonical = build_selector("top_1_seed__lexical_overlap__hop_0__dense")
	always = build_selector(
		"top_1_seed__lexical_overlap__hop_0__dense__budget_fill_always"
	)
	score_floor = build_selector(
		"top_1_seed__lexical_overlap__hop_0__dense__budget_fill_score_floor"
	)
	relative_drop = build_selector(
		"top_1_seed__lexical_overlap__hop_0__dense__budget_fill_relative_drop"
	)

	canonical_result = canonical.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	always_result = always.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	score_floor_result = score_floor.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete
	relative_drop_result = relative_drop.select(graph, case, budget)  # ty: ignore[invalid-argument-type] # Protocol vs concrete

	assert canonical_result.selected_node_ids == []
	assert always_result.selected_node_ids == ["compact", "weak", "tiny"]
	assert score_floor_result.selected_node_ids == ["compact", "weak"]
	assert relative_drop_result.selected_node_ids == ["compact"]
	assert always_result.selector_metadata is not None
	assert always_result.selector_metadata.budget_fill_mode == "always"
	assert always_result.selector_metadata.budget_fill_pool_k == 64
	assert score_floor_result.selector_metadata is not None
	assert score_floor_result.selector_metadata.budget_fill_score_floor == 0.05
	assert relative_drop_result.selector_metadata is not None
	assert relative_drop_result.selector_metadata.budget_fill_relative_drop_ratio == 0.5
