"""Tests for the prefilter-only top-1 controller selector (Variant B).

This selector keeps the deterministic prefilter heuristics (bonus rescue,
generic-page penalty, answer-bearing pin, small-page bypass) but replaces
the LLM controller decision with an argmax over the visible candidate set
ranked by combined prefilter score. It is intended as a counterpart to the
``--prefilter-mode minimal`` LLM-controller ablation: together they isolate
"LLM judgment" from "deterministic curation" contributions.
"""

from __future__ import annotations

from typing import cast

from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.selector import RuntimeBudget, build_selector

PREFILTER_TOP1_NAME = (
	"top_1_seed__sentence_transformer__hop_adaptive__single_path_walk__"
	"link_context_prefilter_top1__lookahead_2"
)


def test_prefilter_top1_selector_runs_without_llm_calls(sample_graph) -> None:
	# The selector must build cleanly without an API key and without any
	# backend factory because it never calls the LLM.
	selector = build_selector(
		PREFILTER_TOP1_NAME,
		selector_provider="copilot",
		selector_model="gpt-4.1",
	)
	case = EvaluationCase(
		case_id="q-prefilter",
		query="Which city hosts the Moon Launch Program launch site?",
	)
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=512))

	result = selector.select(sample_graph, case, budget)

	# Walk should advance from the dense seed without calling any LLM.
	assert result.selector_usage is not None
	assert result.selector_usage.llm_calls == 0
	assert result.selector_usage.total_tokens in (None, 0)
	# Selector must produce a non-empty selection on the toy graph.
	assert result.selected_node_ids
	# Logs should be present and tagged with the prefilter-top1 backend.
	assert result.selector_logs
	for log in result.selector_logs:
		controller = getattr(log, "controller", None)
		if controller is None:
			continue
		# Decisions never involve an LLM call.
		assert controller.llm_calls == 0
		# Effective action is always choose_one or stop (no choose_two,
		# no LLM-driven backtrack).
		assert controller.effective_action in {"choose_one", "stop", "backtrack"}


def test_prefilter_top1_selector_metadata_is_distinct_from_llm_controller(
	sample_graph,
) -> None:
	selector = build_selector(
		PREFILTER_TOP1_NAME,
		selector_provider="copilot",
		selector_model="gpt-4.1",
	)
	case = EvaluationCase(case_id="q-meta", query="launch site city")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=512))
	result = selector.select(sample_graph, case, budget)
	assert result.selector_metadata is not None
	# The scorer kind must reflect the prefilter-top1 lineage so downstream
	# analyses can separate this ablation from the live LLM controller.
	assert (
		"prefilter_top1" in (result.selector_metadata.scorer_kind or "")
	)
