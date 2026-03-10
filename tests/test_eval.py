from webwalker.eval import (
    DenseTopKSelector,
    EagerFullCorpusProxySelector,
    EvaluationCase,
    Evaluator,
    ExpandAnchorSelector,
    ExpandLinkContextSelector,
    ExpandTopologySelector,
    RandomWalkSelector,
    SelectionBudget,
    WebWalkerSelector,
    select_selectors,
)
from webwalker.graph import DocumentNode, LinkContext, LinkContextGraph


class StaticStartPolicy:
    def __init__(self, node_ids: list[str]):
        self.node_ids = node_ids

    def select_start(self, _graph, _query) -> list[str]:
        return list(self.node_ids)


def test_evaluator_runs_reviewer_facing_selector_registry(sample_graph):
    case = EvaluationCase(
        case_id="launch-site",
        query="Which city hosts the launch site?",
        expected_answer="Cape Canaveral",
        gold_support_nodes=["mission", "cape"],
        gold_start_nodes=["mission"],
    )
    evaluation = Evaluator(
        select_selectors(
            [
                "dense_topk",
                "expand_topology",
                "expand_anchor",
                "expand_link_context",
                "webwalker_selector",
                "oracle_start_webwalker",
                "eager_full_corpus_proxy",
            ],
            include_diagnostics=False,
        ),
        budget=SelectionBudget(max_steps=3, top_k=2, token_budget_ratio=1.0),
    ).evaluate_case(sample_graph, case)
    results = {selection.selector_name: selection for selection in evaluation.selections}

    assert set(results) == {
        "dense_topk",
        "expand_topology",
        "expand_anchor",
        "expand_link_context",
        "webwalker_selector",
        "oracle_start_webwalker",
        "eager_full_corpus_proxy",
    }
    assert results["webwalker_selector"].metrics.support_recall == 1.0
    assert results["oracle_start_webwalker"].metrics.start_hit is True
    assert results["webwalker_selector"].end_to_end is not None
    assert results["webwalker_selector"].end_to_end.em == 1.0
    assert results["eager_full_corpus_proxy"].metrics.compression_ratio == 1.0


def test_expand_topology_ignores_anchor_and_sentence_text():
    query = "Which harbor hosts the launch?"
    selector = ExpandTopologySelector(
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


def test_expand_anchor_reacts_to_anchor_text_but_not_sentence_context():
    query = "Which harbor hosts the launch?"
    selector = ExpandAnchorSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
    )
    graph_anchor_alpha = _build_ablation_graph(
        alpha_anchor="launch harbor",
        alpha_sentence="generic context",
        beta_anchor="plain note",
        beta_sentence="launch harbor appears only in the sentence",
        titles=("Node Alpha", "Node Beta"),
    )
    graph_anchor_beta = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="launch harbor appears only in the sentence",
        beta_anchor="launch harbor",
        beta_sentence="generic context",
        titles=("Node Alpha", "Node Beta"),
    )

    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_ratio=1.0)
    result_alpha = selector.select(graph_anchor_alpha, EvaluationCase(case_id="a", query=query), budget)
    result_beta = selector.select(graph_anchor_beta, EvaluationCase(case_id="b", query=query), budget)

    assert result_alpha.corpus.node_ids[1] == "alpha"
    assert result_beta.corpus.node_ids[1] == "beta"


def test_expand_link_context_reacts_to_sentence_context():
    query = "Which harbor hosts the launch?"
    selector = ExpandLinkContextSelector(
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
        anchor_weight=0.0,
        sentence_weight=1.0,
    )
    graph_sentence_alpha = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="The launch harbor is in Alpha.",
        beta_anchor="plain note",
        beta_sentence="Generic Beta sentence.",
        titles=("Node Alpha", "Node Beta"),
    )
    graph_sentence_beta = _build_ablation_graph(
        alpha_anchor="plain note",
        alpha_sentence="Generic Alpha sentence.",
        beta_anchor="plain note",
        beta_sentence="The launch harbor is in Beta.",
        titles=("Node Alpha", "Node Beta"),
    )

    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_ratio=1.0)
    result_alpha = selector.select(graph_sentence_alpha, EvaluationCase(case_id="a", query=query), budget)
    result_beta = selector.select(graph_sentence_beta, EvaluationCase(case_id="b", query=query), budget)

    assert result_alpha.corpus.node_ids[1] == "alpha"
    assert result_beta.corpus.node_ids[1] == "beta"


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


def test_budget_adherence_for_dense_expand_and_walker(sample_graph):
    case = EvaluationCase(
        case_id="budget",
        query="Which city hosts the launch site?",
        gold_support_nodes=["mission", "cape"],
        gold_start_nodes=["mission"],
    )
    tight_budget = SelectionBudget(max_steps=2, top_k=2, token_budget_ratio=0.20)

    dense = DenseTopKSelector().select(sample_graph, case, tight_budget)
    expand = ExpandLinkContextSelector().select(sample_graph, case, tight_budget)
    walker = WebWalkerSelector().select(sample_graph, case, tight_budget)

    assert dense.metrics.budget_adherence is True
    assert expand.metrics.budget_adherence is True
    assert walker.metrics.budget_adherence is True
    assert dense.metrics.selected_token_estimate <= dense.metrics.budget_token_limit
    assert expand.metrics.selected_token_estimate <= expand.metrics.budget_token_limit
    assert walker.metrics.selected_token_estimate <= walker.metrics.budget_token_limit
    assert len(walker.corpus.node_ids) <= tight_budget.max_steps


def test_eager_full_corpus_proxy_ignores_small_budget_but_reports_non_adherence(sample_graph):
    selector = EagerFullCorpusProxySelector()
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
