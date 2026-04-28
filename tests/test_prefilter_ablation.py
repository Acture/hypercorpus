"""Tests for the controller prefilter `minimal` ablation mode (Variant A).

These tests confirm that ``build_controller_exposure_plan`` honors
``prefilter_mode="minimal"`` by stripping out bonus rescue, answer-bearing
pinning, small-page bypass, and the lexical/semantic union, and only ranks
candidates by the embedding (or lexical fallback) similarity score.
"""

from __future__ import annotations

from hypercorpus.controller_exposure import build_controller_exposure_plan
from hypercorpus.graph import DocumentNode, LinkContext, LinkContextGraph
from hypercorpus.walker import StepScoreCard


def _card(edge_id: str, total: float) -> StepScoreCard:
	return StepScoreCard(
		edge_id=edge_id,
		total_score=total,
		subscores={
			"anchor_overlap": total,
			"sentence_overlap": total,
			"target_overlap": 0.0,
			"novelty": 1.0,
		},
		rationale=None,
		text=None,
		backend="overlap",
		provider=None,
		model=None,
		latency_s=0.0,
		prompt_tokens=None,
		completion_tokens=None,
		total_tokens=None,
		cache_hit=None,
		fallback_reason=None,
	)


def _make_graph_and_links() -> tuple[LinkContextGraph, list[LinkContext]]:
	# 6 valid + 1 dangling target. Lexical and semantic disagree, so the
	# full prefilter's hybrid union should differ from minimal's pure top-K.
	graph = LinkContextGraph(
		[
			DocumentNode("root", "Root", ("seed sentence about Alpha and Bravo.",)),
			DocumentNode("a", "Alpha", ("Alpha is a place.",)),
			DocumentNode("b", "Bravo", ("Bravo is a person.",)),
			DocumentNode("c", "Charlie", ("Charlie is unrelated.",)),
			DocumentNode("d", "Delta", ("Delta is unrelated.",)),
			DocumentNode("e", "Echo", ("Echo is unrelated.",)),
			DocumentNode("f", "Foxtrot", ("Foxtrot is unrelated.",)),
		]
	)
	candidate_links = [
		LinkContext(source="root", target="a", anchor_text="Alpha", sentence="s", sent_idx=0),
		LinkContext(source="root", target="b", anchor_text="Bravo", sentence="s", sent_idx=0),
		LinkContext(source="root", target="c", anchor_text="Charlie", sentence="s", sent_idx=0),
		LinkContext(source="root", target="d", anchor_text="Delta", sentence="s", sent_idx=0),
		LinkContext(source="root", target="e", anchor_text="Echo", sentence="s", sent_idx=0),
		LinkContext(source="root", target="f", anchor_text="Foxtrot", sentence="s", sent_idx=0),
		LinkContext(
			source="root",
			target="dangling-target-id",
			anchor_text="Ghost",
			sentence="s",
			sent_idx=0,
		),
	]
	return graph, candidate_links


def test_minimal_mode_uses_only_embedding_top_k() -> None:
	graph, candidate_links = _make_graph_and_links()
	# Lexical and semantic disagree to expose the difference between full
	# (lexical+semantic union, plus rescues/pins) and minimal (pure top-K).
	lexical_cards = [
		_card("0", 0.10),
		_card("1", 0.20),
		_card("2", 0.30),
		_card("3", 0.40),
		_card("4", 0.50),
		_card("5", 0.60),
		_card("dangling", 0.99),
	]
	semantic_cards = [
		_card("0", 0.95),
		_card("1", 0.90),
		_card("2", 0.05),
		_card("3", 0.05),
		_card("4", 0.05),
		_card("5", 0.05),
		_card("dangling", 0.99),
	]
	full = build_controller_exposure_plan(
		query="Alpha Bravo question",
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=lexical_cards,
		semantic_cards=semantic_cards,
		small_page_bypass_n=2,
		lexical_top_n=2,
		semantic_top_n=2,
		bonus_keep_n=2,
		visible_cap=4,
		prefilter_mode="full",
	)
	minimal = build_controller_exposure_plan(
		query="Alpha Bravo question",
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=lexical_cards,
		semantic_cards=semantic_cards,
		small_page_bypass_n=2,
		lexical_top_n=2,
		semantic_top_n=2,
		bonus_keep_n=2,
		visible_cap=4,
		prefilter_mode="minimal",
	)

	# Dangling targets are filtered in both modes (correctness, not heuristic).
	assert 6 not in minimal.visible_indices
	assert 6 not in full.visible_indices

	# Minimal mode never bypasses to "show all" or rescues anything.
	assert minimal.small_page_bypass is False
	assert minimal.bonus_rescued_edge_ids == []
	assert minimal.lexical_prefilter_edge_ids == []

	# Minimal preserves embedding ranking order (semantic descending).
	assert minimal.visible_indices == [0, 1, 2, 3]
	assert minimal.semantic_prefilter_edge_ids == ["0", "1", "2", "3"]

	# The two modes produce different visible candidate sets on this example
	# (full mode unions lexical top-2 [4,5] with semantic top-2 [0,1] plus
	# rescue/pin, whereas minimal returns the pure semantic top-4 [0,1,2,3]).
	assert set(full.visible_indices) != set(minimal.visible_indices)
	assert {4, 5} & set(full.visible_indices)
	assert not ({4, 5} & set(minimal.visible_indices))


def test_minimal_mode_skips_small_page_bypass() -> None:
	graph, candidate_links = _make_graph_and_links()
	cards = [_card(str(i), 0.1 * (i + 1)) for i in range(7)]
	# small_page_bypass_n is large enough that full mode would expose all
	# valid candidates without ranking; minimal mode must still cap to
	# visible_cap and rank.
	full = build_controller_exposure_plan(
		query="q",
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=cards,
		semantic_cards=None,
		small_page_bypass_n=20,
		lexical_top_n=2,
		semantic_top_n=2,
		bonus_keep_n=2,
		visible_cap=3,
		prefilter_mode="full",
	)
	minimal = build_controller_exposure_plan(
		query="q",
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=cards,
		semantic_cards=None,
		small_page_bypass_n=20,
		lexical_top_n=2,
		semantic_top_n=2,
		bonus_keep_n=2,
		visible_cap=3,
		prefilter_mode="minimal",
	)
	assert full.small_page_bypass is True
	assert len(full.visible_indices) == 6  # all valid (dangling 6 dropped)
	assert minimal.small_page_bypass is False
	assert len(minimal.visible_indices) == 3
	# When semantic is unavailable, minimal falls back to lexical ordering.
	assert minimal.visible_indices == [5, 4, 3]
	assert minimal.lexical_prefilter_edge_ids == ["5", "4", "3"]
	assert minimal.semantic_prefilter_edge_ids == []


def test_full_mode_default_is_unchanged() -> None:
	# Regression check: omitting prefilter_mode preserves prior behavior.
	graph, candidate_links = _make_graph_and_links()
	cards = [_card(str(i), 0.1 * (i + 1)) for i in range(7)]
	default = build_controller_exposure_plan(
		query="q",
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=cards,
		semantic_cards=None,
		small_page_bypass_n=2,
		lexical_top_n=2,
		semantic_top_n=2,
		bonus_keep_n=2,
		visible_cap=4,
	)
	explicit_full = build_controller_exposure_plan(
		query="q",
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=cards,
		semantic_cards=None,
		small_page_bypass_n=2,
		lexical_top_n=2,
		semantic_top_n=2,
		bonus_keep_n=2,
		visible_cap=4,
		prefilter_mode="full",
	)
	assert default.visible_indices == explicit_full.visible_indices
	assert default.bonus_rescued_edge_ids == explicit_full.bonus_rescued_edge_ids
	assert default.lexical_prefilter_edge_ids == explicit_full.lexical_prefilter_edge_ids
