from webwalker.eval import EvaluationBudget, EvaluationCase
from webwalker.graph import DocumentNode, LinkContext, LinkContextGraph
from webwalker.selector import (
    SelectorBudget,
    SelectionMode,
    SemanticAStarSelector,
    SemanticBeamSelector,
    SemanticPPRSelector,
    SemanticUCSSelector,
    build_selector,
    parse_selector_spec,
)


class FakeEmbedder:
    backend_name = "sentence_transformer"
    model_name = "multi-qa-MiniLM-L6-cos-v1"

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts):
        return [self.vectors[text] for text in texts]


def test_parse_selector_spec_handles_canonical_forms():
    dense = parse_selector_spec("top_1_seed__lexical_overlap__hop_0__dense")
    iterative = parse_selector_spec("top_1_seed__sentence_transformer__hop_2__iterative_dense")
    beam = parse_selector_spec("top_3_seed__sentence_transformer__hop_3__beam__link_context_llm__lookahead_2")

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


def test_parse_selector_spec_accepts_budget_fill_suffixes_and_rejects_diagnostics():
    filled = parse_selector_spec("top_1_seed__lexical_overlap__hop_0__dense__budget_fill_always")

    assert filled.canonical_name == "top_1_seed__lexical_overlap__hop_0__dense__budget_fill_always"
    assert filled.base_canonical_name == "top_1_seed__lexical_overlap__hop_0__dense"
    assert filled.budget_fill_mode == "always"
    assert filled.budget_fill_pool_k == 64

    try:
        parse_selector_spec("gold_support_context__budget_fill_always")
    except ValueError as exc:
        assert str(exc) == "Unknown selector: gold_support_context__budget_fill_always"
    else:
        raise AssertionError("expected diagnostic fill suffix to be rejected")


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
    assert all(link.source in set(selection.selected_node_ids) for link in selection.selected_links)
    assert all(link.target in set(selection.selected_node_ids) for link in selection.selected_links)
    assert selection.token_cost_estimate <= 256


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
        SelectorBudget(max_nodes=4, max_hops=2, max_tokens=256),
    )
    assert "bridge" in selection.selected_node_ids
    assert "florida" in selection.selected_node_ids


def test_astar_and_ucs_run_on_same_contract():
    graph = LinkContextGraph(
        documents=[
            DocumentNode("start", "Start", ("Start connects to two launch clues.",)),
            DocumentNode("a", "Route A", ("Route A eventually reaches the state clue.",)),
            DocumentNode("b", "Route B", ("Route B eventually reaches the state clue.",)),
            DocumentNode("goal", "Goal", ("Goal confirms the launch state.",)),
        ]
    )
    graph.add_link(LinkContext(source="start", target="a", anchor_text="launch", sentence="noise", sent_idx=0))
    graph.add_link(LinkContext(source="start", target="b", anchor_text="launch", sentence="launch", sent_idx=0))
    graph.add_link(LinkContext(source="a", target="goal", anchor_text="state", sentence="state proof", sent_idx=0))
    graph.add_link(LinkContext(source="b", target="goal", anchor_text="state", sentence="state proof", sent_idx=0))

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
            DocumentNode("relevant", "Relevant", ("Relevant page discusses the launch site.",)),
            DocumentNode("support", "Support", ("Support page mentions Florida.",)),
            DocumentNode("noise1", "Noise One", ("Noise one is unrelated.",)),
            DocumentNode("noise2", "Noise Two", ("Noise two is unrelated.",)),
        ]
    )
    graph.add_link(LinkContext(source="start", target="relevant", anchor_text="launch site", sentence="Relevant branch contains launch site evidence.", sent_idx=0))
    graph.add_link(LinkContext(source="relevant", target="support", anchor_text="Florida", sentence="The launch site is in Florida.", sent_idx=0))
    for noise in ("noise1", "noise2"):
        graph.add_link(LinkContext(source="start", target=noise, anchor_text="misc", sentence="unrelated", sent_idx=0))

    selection = SemanticPPRSelector().select(
        graph,
        "launch florida",
        ["start"],
        SelectorBudget(max_nodes=4, max_hops=3, max_tokens=256),
    )
    assert selection.selected_node_ids.index("relevant") < selection.selected_node_ids.index("noise1")
    assert selection.selected_node_ids.index("support") < selection.selected_node_ids.index("noise1")


def test_canonical_dense_selector_uses_seed_top_k(sample_graph):
    selector = build_selector("top_1_seed__lexical_overlap__hop_0__dense")
    result = selector.select(
        sample_graph,
        EvaluationCase(case_id="q1", query="Which city hosts the launch site?"),
        EvaluationBudget(token_budget_tokens=256),
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
        EvaluationBudget(token_budget_tokens=128),
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
            DocumentNode("bridge", "Bridge", ("Bridge mentions Florida launch records.",)),
            DocumentNode("goal", "Goal", ("Goal confirms the launch state is Florida.",)),
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
        EvaluationBudget(token_budget_tokens=256),
    )

    assert result.selected_node_ids == ["start", "bridge", "goal"]
    assert result.selector_metadata is not None
    assert result.selector_metadata.backend == "iterative_dense"
    assert result.stop_reason == "iterative_dense_retrieval"


def test_sentence_transformer_edge_scorer_records_future_edge_ids():
    graph = LinkContextGraph(
        documents=[
            DocumentNode("root", "Launch Root", ("Launch Root offers multiple navigation paths.",)),
            DocumentNode("bait", "Bait Page", ("Bait page looks relevant but dead ends.",)),
            DocumentNode("bridge", "Bridge Page", ("Bridge page leads onward to the answer.",)),
            DocumentNode("answer", "Answer Page", ("Answer page contains the true launch harbor location.",)),
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
                "harbor location generic bait sentence Bait Page Bait page looks relevant but dead ends.": [0.2, 0.0],
                "plain note harbor Bridge Page Bridge page leads onward to the answer.": [0.7, 0.0],
                "harbor location harbor location Answer Page Answer page contains the true launch harbor location.": [1.0, 0.0],
            }
        ),
    )

    result = selector.select(
        graph,
        EvaluationCase(case_id="q-st-scorer", query=query),
        EvaluationBudget(token_budget_tokens=128),
    )

    assert "bridge" in result.selected_node_ids
    assert result.selector_logs
    assert any(
        candidate.best_next_edge_id == "1-0"
        for log in result.selector_logs
        for candidate in log.candidates
    )


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
    always = build_selector("top_1_seed__lexical_overlap__hop_0__dense__budget_fill_always")
    score_floor = build_selector("top_1_seed__lexical_overlap__hop_0__dense__budget_fill_score_floor")
    relative_drop = build_selector("top_1_seed__lexical_overlap__hop_0__dense__budget_fill_relative_drop")

    canonical_result = canonical.select(graph, case, budget)
    always_result = always.select(graph, case, budget)
    score_floor_result = score_floor.select(graph, case, budget)
    relative_drop_result = relative_drop.select(graph, case, budget)

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
