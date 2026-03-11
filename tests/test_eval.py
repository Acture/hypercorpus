import pytest

from webwalker.eval import (
    CaseEvaluation,
    EndToEndResult,
    EvaluationBudget,
    EvaluationCase,
    Evaluator,
    IncrementalExperimentAggregator,
    SelectedCorpus,
    SelectionMetrics,
    SelectionResult,
    summarize_evaluations,
)
from webwalker.selector import available_selector_names, parse_selector_spec, select_selectors
from webwalker.selector import SelectorMetadata, SelectorUsage


def test_available_selector_names_are_canonical_only():
    names = available_selector_names(include_diagnostics=False)

    assert "top_1_seed__lexical_overlap__hop_0__dense" in names
    assert "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2" in names
    assert "top_3_seed__lexical_overlap__hop_3__beam__link_context_llm__lookahead_2" in names
    assert "seed_rerank" not in names
    assert "seed__link_context_overlap__single_path_walk" not in names


def test_parse_selector_spec_rejects_legacy_names():
    with pytest.raises(ValueError, match="Unknown selector: seed_rerank"):
        parse_selector_spec("seed_rerank")
    with pytest.raises(ValueError, match="Unknown selector: adaptive_link_context_walk"):
        parse_selector_spec("adaptive_link_context_walk")


def test_evaluation_budget_supports_absolute_tokens():
    budget = EvaluationBudget(token_budget_tokens=256)

    assert budget.budget_mode == "tokens"
    assert budget.budget_value == 256
    assert budget.budget_label == "tokens-256"


def test_evaluation_budget_rejects_ambiguous_budget_sources():
    with pytest.raises(ValueError, match="exactly one"):
        EvaluationBudget(token_budget_tokens=256, token_budget_ratio=0.1)


def test_evaluator_runs_canonical_selectors(sample_graph):
    case = EvaluationCase(
        case_id="launch-site",
        query="Which city hosts the launch site?",
        expected_answer="Cape Canaveral",
        gold_support_nodes=["mission", "cape"],
        gold_start_nodes=["mission"],
        gold_path_nodes=["mission", "cape"],
    )
    selector_names = [
        "top_1_seed__lexical_overlap__hop_0__dense",
        "top_1_seed__lexical_overlap__hop_1__topology_neighbors",
        "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_overlap__lookahead_1",
        "gold_support_context",
        "full_corpus_upper_bound",
    ]

    evaluation = Evaluator(
        select_selectors(selector_names),
        budget=EvaluationBudget(token_budget_tokens=256),
        with_e2e=True,
    ).evaluate_case(sample_graph, case)

    results = {selection.selector_name: selection for selection in evaluation.selections}

    assert set(results) == set(selector_names)
    assert results["gold_support_context"].metrics.support_precision == 1.0
    assert results["gold_support_context"].metrics.support_f1 == 1.0
    assert results["top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_overlap__lookahead_1"].metrics.support_recall == 1.0
    assert results["top_1_seed__lexical_overlap__hop_0__dense"].metrics.selected_nodes_count == 1
    assert results["full_corpus_upper_bound"].metrics.compression_ratio == 1.0
    assert results["top_1_seed__lexical_overlap__hop_1__topology_neighbors"].end_to_end is not None
    assert results["top_1_seed__lexical_overlap__hop_1__topology_neighbors"].end_to_end.em == 1.0


def test_incremental_experiment_aggregator_matches_batch_summary():
    budget = EvaluationBudget(token_budget_tokens=128)
    case_one = EvaluationCase(case_id="q1", query="one", dataset_name="2wikimultihop")
    case_two = EvaluationCase(case_id="q2", query="two", dataset_name="2wikimultihop")

    shared_metadata = SelectorMetadata(
        scorer_kind="link_context_llm",
        backend="llm",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
    )
    other_metadata = SelectorMetadata(
        scorer_kind="dense",
        backend="baseline",
        provider="openai",
        model="gpt-4.1-mini",
    )

    shared_one = SelectionResult(
        selector_name="top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_llm__lookahead_1",
        budget=budget,
        corpus=SelectedCorpus(node_ids=["mission"], edge_contexts=[], token_estimate=32, root_node_ids=["mission"]),
        metrics=SelectionMetrics(
            budget_mode=budget.budget_mode,
            budget_value=budget.budget_value,
            budget_label=budget.budget_label,
            token_budget_ratio=budget.token_budget_ratio,
            token_budget_tokens=budget.token_budget_tokens,
            budget_token_limit=128,
            selection_runtime_s=0.2,
            selected_nodes_count=1,
            selected_token_estimate=32,
            compression_ratio=0.1,
            budget_adherence=True,
            start_hit=None,
            support_recall=0.5,
            support_precision=0.75,
            support_f1=0.6,
            path_hit=None,
        ),
        trace=[],
        selector_metadata=shared_metadata,
        selector_usage=SelectorUsage(
            runtime_s=1.5,
            llm_calls=2,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            step_count=4,
            fallback_steps=1,
            parse_failure_steps=0,
        ),
    )
    other = SelectionResult(
        selector_name="top_1_seed__lexical_overlap__hop_0__dense",
        budget=budget,
        corpus=SelectedCorpus(node_ids=["mission"], edge_contexts=[], token_estimate=24, root_node_ids=["mission"]),
        metrics=SelectionMetrics(
            budget_mode=budget.budget_mode,
            budget_value=budget.budget_value,
            budget_label=budget.budget_label,
            token_budget_ratio=budget.token_budget_ratio,
            token_budget_tokens=budget.token_budget_tokens,
            budget_token_limit=128,
            selection_runtime_s=0.05,
            selected_nodes_count=1,
            selected_token_estimate=24,
            compression_ratio=0.08,
            budget_adherence=True,
            start_hit=True,
            support_recall=0.25,
            support_precision=1.0,
            support_f1=0.4,
            path_hit=False,
        ),
        trace=[],
        selector_metadata=other_metadata,
        selector_usage=SelectorUsage(),
    )
    shared_two = SelectionResult(
        selector_name=shared_one.selector_name,
        budget=budget,
        corpus=SelectedCorpus(node_ids=["mission", "cape"], edge_contexts=[], token_estimate=48, root_node_ids=["mission"]),
        metrics=SelectionMetrics(
            budget_mode=budget.budget_mode,
            budget_value=budget.budget_value,
            budget_label=budget.budget_label,
            token_budget_ratio=budget.token_budget_ratio,
            token_budget_tokens=budget.token_budget_tokens,
            budget_token_limit=128,
            selection_runtime_s=0.3,
            selected_nodes_count=2,
            selected_token_estimate=48,
            compression_ratio=0.15,
            budget_adherence=False,
            start_hit=True,
            support_recall=1.0,
            support_precision=0.5,
            support_f1=2 / 3,
            path_hit=True,
        ),
        trace=[],
        end_to_end=EndToEndResult(
            mode="heuristic",
            model=None,
            answer="Cape Canaveral",
            confidence=1.0,
            evidence_count=1,
            em=1.0,
            f1=1.0,
            runtime_s=0.01,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        ),
        selector_metadata=shared_metadata,
        selector_usage=SelectorUsage(
            runtime_s=2.5,
            llm_calls=4,
            prompt_tokens=140,
            completion_tokens=30,
            total_tokens=170,
            step_count=5,
            fallback_steps=0,
            parse_failure_steps=1,
        ),
    )

    evaluations = [
        CaseEvaluation(case=case_one, selections=[shared_one, other]),
        CaseEvaluation(case=case_two, selections=[shared_two]),
    ]
    batch_summary = summarize_evaluations(evaluations, dataset_name="2wikimultihop")
    aggregator = IncrementalExperimentAggregator(dataset_name="2wikimultihop")
    for evaluation in evaluations:
        aggregator.add_case_evaluation(evaluation)
    incremental_summary = aggregator.to_summary()

    assert incremental_summary == batch_summary
    assert [(row.name, row.selector_provider, row.selector_model) for row in incremental_summary.selector_budgets] == [
        (shared_one.selector_name, "anthropic", "claude-haiku-4-5-20251001"),
        (other.selector_name, "openai", "gpt-4.1-mini"),
    ]
    llm_row = incremental_summary.selector_budgets[0]
    assert llm_row.avg_answer_em == 1.0
    assert llm_row.avg_answer_f1 == 1.0
    assert llm_row.avg_start_hit == 1.0
