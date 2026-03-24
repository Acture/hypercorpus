import pytest

from hypercorpus.eval import (
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
from hypercorpus.selector import available_selector_names, parse_selector_spec, selector_names_for_preset, select_selectors
from hypercorpus.selector import SelectorMetadata, SelectorUsage


def test_available_selector_names_are_canonical_only():
    names = available_selector_names(include_diagnostics=False)

    assert "top_1_seed__lexical_overlap__hop_0__dense" in names
    assert "top_1_seed__sentence_transformer__hop_2__iterative_dense" in names
    assert "top_1_seed__sentence_transformer__hop_2__mdr_light" in names
    assert "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2" in names
    assert "top_3_seed__lexical_overlap__hop_3__beam__link_context_llm__lookahead_2" in names
    assert "seed_rerank" not in names
    assert "seed__link_context_overlap__single_path_walk" not in names


def test_selector_names_for_paper_recommended_preset_include_expected_profiles_and_diagnostics():
    names = selector_names_for_preset("paper_recommended")

    assert names == [
        "top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_title_aware__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_1__profile_st_balanced__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm__lookahead_1__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop",
        "gold_support_context",
        "full_corpus_upper_bound",
    ]
    assert selector_names_for_preset("paper_recommended", include_diagnostics=False) == names[:-2]
    for selector_name in names[:-2]:
        assert parse_selector_spec(selector_name).canonical_name == selector_name


def test_selector_names_for_paper_recommended_local_preset_exclude_llm_and_keep_diagnostics():
    names = selector_names_for_preset("paper_recommended_local")

    assert names == [
        "top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_title_aware__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_1__profile_st_balanced__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop",
        "top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop",
        "gold_support_context",
        "full_corpus_upper_bound",
    ]
    assert all("link_context_llm" not in name for name in names)
    assert selector_names_for_preset("paper_recommended_local", include_diagnostics=False) == names[:-2]


def test_selector_names_for_branchy_profiles_preset_are_fixed_and_parseable():
    names = selector_names_for_preset("branchy_profiles")

    assert names == [
        "top_3_seed__sentence_transformer__hop_3__beam__link_context_overlap__lookahead_1__profile_overlap_balanced",
        "top_3_seed__sentence_transformer__hop_3__beam__link_context_overlap__lookahead_1__profile_overlap_title_aware",
        "top_3_seed__sentence_transformer__hop_3__beam__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
        "top_3_seed__sentence_transformer__hop_3__beam__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
        "top_3_seed__sentence_transformer__hop_3__astar__link_context_overlap__lookahead_1__profile_overlap_balanced",
        "top_3_seed__sentence_transformer__hop_3__astar__link_context_overlap__lookahead_1__profile_overlap_title_aware",
        "top_3_seed__sentence_transformer__hop_3__astar__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
        "top_3_seed__sentence_transformer__hop_3__astar__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
        "top_3_seed__sentence_transformer__hop_3__ucs__link_context_overlap__lookahead_1__profile_overlap_balanced",
        "top_3_seed__sentence_transformer__hop_3__ucs__link_context_overlap__lookahead_1__profile_overlap_title_aware",
        "top_3_seed__sentence_transformer__hop_3__ucs__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
        "top_3_seed__sentence_transformer__hop_3__ucs__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
        "top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_overlap__lookahead_1__profile_overlap_balanced",
        "top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_overlap__lookahead_1__profile_overlap_title_aware",
        "top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_sentence_transformer__lookahead_1__profile_st_balanced",
        "top_3_seed__sentence_transformer__hop_3__beam_ppr__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced",
        "top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy",
        "top_1_seed__sentence_transformer__hop_2__mdr_light",
        "top_1_seed__sentence_transformer__hop_0__dense",
        "gold_support_context",
        "full_corpus_upper_bound",
    ]
    assert selector_names_for_preset("branchy_profiles", include_diagnostics=False) == names[:-2]
    for selector_name in names[:-2]:
        assert parse_selector_spec(selector_name).canonical_name == selector_name


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
    assert results["gold_support_context"].metrics.support_set_em == 1.0
    assert results["top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_overlap__lookahead_1"].metrics.support_recall == 1.0
    assert results["top_1_seed__lexical_overlap__hop_0__dense"].metrics.selected_nodes_count == 1
    assert results["top_1_seed__lexical_overlap__hop_0__dense"].metrics.support_set_em == 0.0
    assert results["full_corpus_upper_bound"].metrics.compression_ratio == 1.0
    assert results["top_1_seed__lexical_overlap__hop_1__topology_neighbors"].end_to_end is not None
    assert results["top_1_seed__lexical_overlap__hop_1__topology_neighbors"].end_to_end.em == 1.0


def test_evaluation_case_infers_question_type_from_gold_fields():
    assert EvaluationCase(
        case_id="bridge",
        query="bridge question",
        gold_support_nodes=["a", "b"],
        gold_path_nodes=["a", "b"],
    ).question_type == "bridge"
    assert EvaluationCase(
        case_id="comparison",
        query="comparison question",
        gold_support_nodes=["a", "b"],
    ).question_type == "comparison"
    assert EvaluationCase(
        case_id="unknown",
        query="unknown question",
        gold_support_nodes=["a"],
    ).question_type == "unknown"


def test_select_selectors_prefers_explicit_names_over_preset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    selectors = select_selectors(
        names=["top_1_seed__lexical_overlap__hop_0__dense"],
        preset="paper_recommended",
        selector_provider="openai",
        selector_model="gpt-4.1-mini",
        selector_api_key_env="OPENAI_API_KEY",
    )

    assert [selector.name for selector in selectors] == ["top_1_seed__lexical_overlap__hop_0__dense"]


def test_select_selectors_accepts_paper_recommended_local_without_llm_config():
    selectors = select_selectors(preset="paper_recommended_local")

    assert [selector.name for selector in selectors] == selector_names_for_preset("paper_recommended_local")


def test_select_selectors_rejects_paper_recommended_without_llm_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing API key in environment variable OPENAI_API_KEY"):
        select_selectors(preset="paper_recommended")


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
            budget_utilization=0.25,
            empty_selection=False,
            start_hit=None,
            support_recall=0.5,
            support_precision=0.75,
            support_f1=0.6,
            support_f1_zero_on_empty=0.6,
            support_set_em=0.0,
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
            budget_utilization=24 / 128,
            empty_selection=False,
            start_hit=True,
            support_recall=0.25,
            support_precision=1.0,
            support_f1=0.4,
            support_f1_zero_on_empty=0.4,
            support_set_em=0.0,
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
            budget_utilization=48 / 128,
            empty_selection=False,
            start_hit=True,
            support_recall=1.0,
            support_precision=0.5,
            support_f1=2 / 3,
            support_f1_zero_on_empty=2 / 3,
            support_set_em=1.0,
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
    assert llm_row.avg_budget_utilization == pytest.approx((0.25 + (48 / 128)) / 2)
    assert llm_row.avg_empty_selection_rate == 0.0
    assert llm_row.avg_support_f1_zero_on_empty == pytest.approx((0.6 + (2 / 3)) / 2)
    assert llm_row.avg_support_set_em == 0.5


def test_empty_selection_metrics_are_visible_without_changing_raw_f1():
    budget = EvaluationBudget(token_budget_tokens=128)
    case = EvaluationCase(
        case_id="empty",
        query="launch site",
        dataset_name="2wikimultihop",
        gold_support_nodes=["mission"],
    )
    empty_result = SelectionResult(
        selector_name="top_1_seed__lexical_overlap__hop_0__dense",
        budget=budget,
        corpus=SelectedCorpus(node_ids=[], edge_contexts=[], token_estimate=0),
        metrics=SelectionMetrics(
            budget_mode=budget.budget_mode,
            budget_value=budget.budget_value,
            budget_label=budget.budget_label,
            token_budget_ratio=budget.token_budget_ratio,
            token_budget_tokens=budget.token_budget_tokens,
            budget_token_limit=128,
            selection_runtime_s=0.01,
            selected_nodes_count=0,
            selected_token_estimate=0,
            compression_ratio=0.0,
            budget_adherence=True,
            budget_utilization=0.0,
            empty_selection=True,
            start_hit=False,
            support_recall=0.0,
            support_precision=None,
            support_f1=None,
            support_f1_zero_on_empty=0.0,
            support_set_em=0.0,
            path_hit=False,
        ),
        trace=[],
        selector_metadata=SelectorMetadata(scorer_kind="baseline", backend="dense"),
        selector_usage=SelectorUsage(),
    )
    nonempty_result = SelectionResult(
        selector_name=empty_result.selector_name,
        budget=budget,
        corpus=SelectedCorpus(node_ids=["mission"], edge_contexts=[], token_estimate=64),
        metrics=SelectionMetrics(
            budget_mode=budget.budget_mode,
            budget_value=budget.budget_value,
            budget_label=budget.budget_label,
            token_budget_ratio=budget.token_budget_ratio,
            token_budget_tokens=budget.token_budget_tokens,
            budget_token_limit=128,
            selection_runtime_s=0.02,
            selected_nodes_count=1,
            selected_token_estimate=64,
            compression_ratio=0.1,
            budget_adherence=True,
            budget_utilization=0.5,
            empty_selection=False,
            start_hit=True,
            support_recall=1.0,
            support_precision=1.0,
            support_f1=1.0,
            support_f1_zero_on_empty=1.0,
            support_set_em=1.0,
            path_hit=True,
        ),
        trace=[],
        selector_metadata=SelectorMetadata(scorer_kind="baseline", backend="dense"),
        selector_usage=SelectorUsage(),
    )

    summary = summarize_evaluations(
        [
            CaseEvaluation(case=case, selections=[empty_result]),
            CaseEvaluation(
                case=EvaluationCase(
                    case_id="nonempty",
                    query="launch site",
                    dataset_name="2wikimultihop",
                    gold_support_nodes=["mission"],
                ),
                selections=[nonempty_result],
            ),
        ],
        dataset_name="2wikimultihop",
    )

    row = summary.selector_budgets[0]
    assert empty_result.metrics.support_precision is None
    assert empty_result.metrics.support_f1 is None
    assert empty_result.metrics.support_f1_zero_on_empty == 0.0
    assert empty_result.metrics.empty_selection is True
    assert empty_result.metrics.budget_utilization == 0.0
    assert row.avg_support_precision == 1.0
    assert row.avg_support_f1 == 1.0
    assert row.avg_support_f1_zero_on_empty == 0.5
    assert row.avg_support_set_em == 0.5
    assert row.avg_empty_selection_rate == 0.5
    assert row.avg_budget_utilization == 0.25
