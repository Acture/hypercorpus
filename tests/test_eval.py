import pytest

from webwalker.eval import (
    EvaluationCase,
    Evaluator,
    FullCorpusUpperBoundSelector,
    GoldSupportContextSelector,
    OracleSeedLinkContextOverlapSinglePathWalkSelector,
    RandomWalkSelector,
    SeedAnchorOverlapSinglePathWalkSelector,
    SeedAnchorOverlapTwoHopSinglePathWalkSelector,
    SeedPlusAnchorNeighborsSelector,
    SeedLinkContextOverlapSinglePathWalkSelector,
    SeedLinkContextOverlapTwoHopSinglePathWalkSelector,
    SeedPlusLinkContextNeighborsSelector,
    SeedPlusTopologyNeighborsSelector,
    SeedRerankSelector,
    SeedTitleAwareSinglePathWalkSelector,
    SelectionBudget,
    available_selector_names,
    select_selectors,
)
from webwalker.graph import DocumentNode, LinkContext, LinkContextGraph


class StaticStartPolicy:
    def __init__(self, node_ids: list[str]):
        self.node_ids = node_ids

    def select_start(self, _graph, _query) -> list[str]:
        return list(self.node_ids)


def test_available_selector_names_split_main_and_diagnostics():
    assert available_selector_names(include_diagnostics=False) == [
        "seed_rerank",
        "seed_plus_topology_neighbors",
        "seed_plus_anchor_neighbors",
        "seed_plus_link_context_neighbors",
        "seed__anchor_overlap__single_path_walk",
        "seed__link_context_overlap__single_path_walk",
        "seed__anchor_overlap__two_hop_single_path_walk",
        "seed__link_context_overlap__two_hop_single_path_walk",
    ]
    assert available_selector_names(include_diagnostics=True) == [
        "seed_rerank",
        "seed_plus_topology_neighbors",
        "seed_plus_anchor_neighbors",
        "seed_plus_link_context_neighbors",
        "seed__anchor_overlap__single_path_walk",
        "seed__link_context_overlap__single_path_walk",
        "seed__anchor_overlap__two_hop_single_path_walk",
        "seed__link_context_overlap__two_hop_single_path_walk",
        "seed__title_aware__single_path_walk",
        "oracle_seed__link_context_overlap__single_path_walk",
        "gold_support_context",
        "random__single_path_walk",
        "full_corpus_upper_bound",
        "seed__link_context_llm__single_path_walk",
        "seed__link_context_llm__two_hop_single_path_walk",
        "oracle_seed__link_context_llm__single_path_walk",
    ]


def test_select_selectors_rejects_legacy_ids():
    with pytest.raises(ValueError, match="Unknown selector: dense_topk"):
        select_selectors(["dense_topk"])
    with pytest.raises(ValueError, match="Unknown selector: adaptive_link_context_walk"):
        select_selectors(["adaptive_link_context_walk"])


def test_evaluator_runs_research_facing_selector_registry(sample_graph):
    case = EvaluationCase(
        case_id="launch-site",
        query="Which city hosts the launch site?",
        expected_answer="Cape Canaveral",
        gold_support_nodes=["mission", "cape"],
        gold_start_nodes=["mission"],
    )
    selector_names = [
        "seed_rerank",
        "seed_plus_topology_neighbors",
        "seed_plus_anchor_neighbors",
        "seed_plus_link_context_neighbors",
        "seed__anchor_overlap__single_path_walk",
        "seed__link_context_overlap__single_path_walk",
        "seed__anchor_overlap__two_hop_single_path_walk",
        "seed__link_context_overlap__two_hop_single_path_walk",
        "seed__title_aware__single_path_walk",
        "oracle_seed__link_context_overlap__single_path_walk",
        "gold_support_context",
        "full_corpus_upper_bound",
    ]
    evaluation = Evaluator(
        select_selectors(selector_names),
        budget=SelectionBudget(max_steps=3, top_k=2, token_budget_ratio=1.0),
        with_e2e=True,
    ).evaluate_case(sample_graph, case)
    results = {selection.selector_name: selection for selection in evaluation.selections}

    assert set(results) == set(selector_names)
    assert results["seed__link_context_overlap__single_path_walk"].metrics.support_recall == 1.0
    assert results["oracle_seed__link_context_overlap__single_path_walk"].metrics.start_hit is True
    assert results["gold_support_context"].metrics.support_precision == 1.0
    assert results["gold_support_context"].metrics.support_f1 == 1.0
    assert results["seed__title_aware__single_path_walk"].end_to_end is not None
    assert results["seed__title_aware__single_path_walk"].end_to_end.em == 1.0
    assert results["full_corpus_upper_bound"].metrics.compression_ratio == 1.0


def test_selection_budget_supports_absolute_tokens():
    budget = SelectionBudget(max_steps=3, top_k=2, token_budget_tokens=256, token_budget_ratio=None)

    assert budget.budget_mode == "tokens"
    assert budget.budget_value == 256
    assert budget.budget_label == "tokens-256"


def test_selection_budget_rejects_ambiguous_budget_sources():
    with pytest.raises(ValueError, match="exactly one"):
        SelectionBudget(max_steps=3, top_k=2, token_budget_tokens=256, token_budget_ratio=0.1)


def test_gold_support_context_selector_uses_gold_nodes(sample_graph):
    case = EvaluationCase(
        case_id="launch-site",
        query="Which city hosts the launch site?",
        gold_support_nodes=["mission", "cape"],
        gold_start_nodes=["mission"],
    )
    result = GoldSupportContextSelector().select(
        sample_graph,
        case,
        SelectionBudget(max_steps=3, top_k=2, token_budget_ratio=1.0),
    )

    assert result.corpus.node_ids == ["mission", "cape"]
    assert result.metrics.support_recall == 1.0
    assert result.metrics.support_precision == 1.0


def test_seed_plus_topology_neighbors_ignores_anchor_and_sentence_text():
    query = "Which harbor hosts the launch?"
    selector = SeedPlusTopologyNeighborsSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    graph_a = _build_ablation_graph(
        alpha_anchor="irrelevant mention",
        alpha_sentence="completely generic sentence",
        beta_anchor="launch harbor",
        beta_sentence="very relevant sentence",
    )
    graph_b = _build_ablation_graph(
        alpha_anchor="launch harbor",
        alpha_sentence="very relevant sentence",
        beta_anchor="irrelevant mention",
        beta_sentence="completely generic sentence",
    )

    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_ratio=1.0)
    result_a = selector.select(graph_a, EvaluationCase(case_id="a", query=query), budget)
    result_b = selector.select(graph_b, EvaluationCase(case_id="b", query=query), budget)

    assert result_a.corpus.node_ids[1] == "alpha"
    assert result_b.corpus.node_ids[1] == "alpha"


def test_seed_plus_anchor_neighbors_reacts_to_anchor_text_but_not_sentence_context():
    query = "Which harbor hosts the launch?"
    selector = SeedPlusAnchorNeighborsSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    graph_anchor_alpha = _build_ablation_graph(
        alpha_anchor="launch harbor",
        alpha_sentence="generic context",
        beta_anchor="plain note",
        beta_sentence="launch harbor appears only in the sentence",
    )
    graph_anchor_beta = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="launch harbor appears only in the sentence",
        beta_anchor="launch harbor",
        beta_sentence="generic context",
    )

    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_ratio=1.0)
    result_alpha = selector.select(graph_anchor_alpha, EvaluationCase(case_id="a", query=query), budget)
    result_beta = selector.select(graph_anchor_beta, EvaluationCase(case_id="b", query=query), budget)

    assert result_alpha.corpus.node_ids[1] == "alpha"
    assert result_beta.corpus.node_ids[1] == "beta"


def test_seed_plus_link_context_neighbors_reacts_to_sentence_context():
    query = "Which harbor hosts the launch?"
    selector = SeedPlusLinkContextNeighborsSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
        anchor_weight=0.0,
        sentence_weight=1.0,
    )
    graph_sentence_alpha = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="The launch harbor is in Alpha.",
        beta_anchor="plain note",
        beta_sentence="Generic Beta sentence.",
    )
    graph_sentence_beta = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="Generic Alpha sentence.",
        beta_anchor="plain note",
        beta_sentence="The launch harbor is in Beta.",
    )

    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_ratio=1.0)
    result_alpha = selector.select(graph_sentence_alpha, EvaluationCase(case_id="a", query=query), budget)
    result_beta = selector.select(graph_sentence_beta, EvaluationCase(case_id="b", query=query), budget)

    assert result_alpha.corpus.node_ids[1] == "alpha"
    assert result_beta.corpus.node_ids[1] == "beta"


def test_seed_anchor_overlap_single_path_walk_ignores_sentence_context():
    query = "Which harbor hosts the launch?"
    selector = SeedAnchorOverlapSinglePathWalkSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    graph_anchor_alpha = _build_ablation_graph(
        alpha_anchor="launch harbor",
        alpha_sentence="generic context",
        beta_anchor="plain note",
        beta_sentence="The launch harbor is in Beta.",
    )
    graph_anchor_beta = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="The launch harbor is in Alpha.",
        beta_anchor="launch harbor",
        beta_sentence="generic context",
    )

    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_ratio=1.0)
    result_alpha = selector.select(graph_anchor_alpha, EvaluationCase(case_id="a", query=query), budget)
    result_beta = selector.select(graph_anchor_beta, EvaluationCase(case_id="b", query=query), budget)

    assert result_alpha.corpus.node_ids[1] == "alpha"
    assert result_beta.corpus.node_ids[1] == "beta"


def test_seed_link_context_overlap_single_path_walk_reacts_to_sentence_context():
    query = "Which harbor hosts the launch?"
    selector = SeedLinkContextOverlapSinglePathWalkSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    graph_sentence_alpha = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="The launch harbor is in Alpha.",
        beta_anchor="plain note",
        beta_sentence="Generic Beta sentence.",
    )
    graph_sentence_beta = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="Generic Alpha sentence.",
        beta_anchor="plain note",
        beta_sentence="The launch harbor is in Beta.",
    )

    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_ratio=1.0)
    result_alpha = selector.select(graph_sentence_alpha, EvaluationCase(case_id="a", query=query), budget)
    result_beta = selector.select(graph_sentence_beta, EvaluationCase(case_id="b", query=query), budget)

    assert result_alpha.corpus.node_ids[1] == "alpha"
    assert result_beta.corpus.node_ids[1] == "beta"


def test_seed_link_context_overlap_two_hop_single_path_walk_prefers_bridge_node_over_greedy_decoy():
    query = "harbor location"
    graph = _build_bridge_graph(anchor_only=False)
    greedy = SeedLinkContextOverlapSinglePathWalkSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    lookahead = SeedLinkContextOverlapTwoHopSinglePathWalkSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    case = EvaluationCase(
        case_id="bridge-context",
        query=query,
        gold_support_nodes=["root", "bridge", "answer"],
        gold_start_nodes=["root"],
        gold_path_nodes=["root", "bridge", "answer"],
    )
    budget = SelectionBudget(max_steps=3, top_k=1, token_budget_ratio=1.0)

    greedy_result = greedy.select(graph, case, budget)
    lookahead_result = lookahead.select(graph, case, budget)

    assert greedy_result.corpus.node_ids == ["root", "bait"]
    assert lookahead_result.corpus.node_ids == ["root", "bridge", "answer"]
    assert greedy_result.metrics.path_hit is False
    assert lookahead_result.metrics.path_hit is True


def test_seed_anchor_overlap_two_hop_single_path_walk_prefers_bridge_node_over_greedy_decoy():
    query = "harbor location"
    graph = _build_bridge_graph(anchor_only=True)
    greedy = SeedAnchorOverlapSinglePathWalkSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    lookahead = SeedAnchorOverlapTwoHopSinglePathWalkSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    case = EvaluationCase(
        case_id="bridge-anchor",
        query=query,
        gold_support_nodes=["root", "bridge", "answer"],
        gold_start_nodes=["root"],
        gold_path_nodes=["root", "bridge", "answer"],
    )
    budget = SelectionBudget(max_steps=3, top_k=1, token_budget_ratio=1.0)

    greedy_result = greedy.select(graph, case, budget)
    lookahead_result = lookahead.select(graph, case, budget)

    assert greedy_result.corpus.node_ids == ["root", "bait"]
    assert lookahead_result.corpus.node_ids == ["root", "bridge", "answer"]
    assert greedy_result.metrics.path_hit is False
    assert lookahead_result.metrics.path_hit is True


def test_random_walk_is_deterministic_for_fixed_seed(cyclic_graph):
    selector = RandomWalkSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["a"]),
        seed=7,
    )
    case = EvaluationCase(case_id="loop", query="Which target holds the answer?")
    budget = SelectionBudget(max_steps=3, top_k=1, token_budget_ratio=1.0)

    first = selector.select(cyclic_graph, case, budget)
    second = selector.select(cyclic_graph, case, budget)

    assert first.corpus.node_ids == second.corpus.node_ids
    assert [step.node_id for step in first.trace] == [step.node_id for step in second.trace]


def test_budget_adherence_for_seed_expand_and_adaptive_walks(sample_graph):
    case = EvaluationCase(
        case_id="budget",
        query="Which city hosts the launch site?",
        gold_support_nodes=["mission", "cape"],
        gold_start_nodes=["mission"],
    )
    tight_budget = SelectionBudget(max_steps=2, top_k=2, token_budget_ratio=0.20)

    results = [
        SeedRerankSelector().select(sample_graph, case, tight_budget),
        SeedPlusLinkContextNeighborsSelector().select(sample_graph, case, tight_budget),
        SeedLinkContextOverlapSinglePathWalkSelector().select(sample_graph, case, tight_budget),
        SeedLinkContextOverlapTwoHopSinglePathWalkSelector().select(sample_graph, case, tight_budget),
    ]

    for result in results:
        assert result.metrics.budget_adherence is True
        assert result.metrics.selected_token_estimate <= result.metrics.budget_token_limit
        assert len(result.corpus.node_ids) <= tight_budget.max_steps


def test_full_corpus_upper_bound_ignores_small_budget_but_reports_non_adherence(sample_graph):
    selector = FullCorpusUpperBoundSelector()
    case = EvaluationCase(case_id="full", query="Which city hosts the launch site?")
    small_budget = SelectionBudget(max_steps=3, top_k=2, token_budget_ratio=0.10)
    full_budget = SelectionBudget(max_steps=3, top_k=2, token_budget_ratio=1.0)

    constrained = selector.select(sample_graph, case, small_budget)
    full = selector.select(sample_graph, case, full_budget)

    assert set(constrained.corpus.node_ids) == set(sample_graph.nodes)
    assert constrained.metrics.budget_adherence is False
    assert full.metrics.budget_adherence is True
    assert full.metrics.compression_ratio == 1.0


def _build_ablation_graph(
    *,
    alpha_anchor: str,
    alpha_sentence: str,
    beta_anchor: str,
    beta_sentence: str,
    titles: tuple[str, str] = ("Alpha Harbor", "Beta Ridge"),
) -> LinkContextGraph:
    graph = LinkContextGraph(
        documents=[
            DocumentNode("root", "Launch Root", ("Launch Root links outward.",)),
            DocumentNode("alpha", titles[0], ("Alpha supporting text.",)),
            DocumentNode("beta", titles[1], ("Beta supporting text.",)),
        ]
    )
    graph.add_link(
        LinkContext(
            source="root",
            target="alpha",
            anchor_text=alpha_anchor,
            sentence=alpha_sentence,
            sent_idx=0,
        )
    )
    graph.add_link(
        LinkContext(
            source="root",
            target="beta",
            anchor_text=beta_anchor,
            sentence=beta_sentence,
            sent_idx=0,
        )
    )
    return graph


def _build_bridge_graph(*, anchor_only: bool) -> LinkContextGraph:
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
            anchor_text="harbor location" if anchor_only else "harbor location",
            sentence="generic bait sentence" if anchor_only else "generic bait sentence",
            sent_idx=0,
        )
    )
    graph.add_link(
        LinkContext(
            source="root",
            target="bridge",
            anchor_text="harbor" if anchor_only else "plain note",
            sentence="generic bridge sentence" if anchor_only else "harbor",
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
    return graph
