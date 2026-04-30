import pytest

from hypercorpus.datasets import ShardedDocumentStore, prepare_normalized_graph_store
from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.selector import (
	available_selector_names,
	build_selector,
	parse_selector_spec,
)


BM25_SELECTOR = "top_1_seed__bm25__hop_0"
BM25_FILL_SELECTOR = "top_1_seed__bm25__hop_0__budget_fill_relative_drop"


@pytest.fixture
def bm25_iirc_store(iirc_files, tmp_path):
	questions_path, graph_path = iirc_files
	prepared = prepare_normalized_graph_store(
		tmp_path / "bm25-iirc-store",
		dataset_name="iirc",
		questions_source=questions_path,
		graph_source=graph_path,
		min_free_gib=0,
	)
	store = ShardedDocumentStore(prepared.root)
	try:
		yield store
	finally:
		store.close()


def test_bm25_selector_is_registered_and_builds():
	assert BM25_SELECTOR in available_selector_names(include_diagnostics=False)
	assert BM25_FILL_SELECTOR in available_selector_names(include_diagnostics=False)

	spec = parse_selector_spec(BM25_SELECTOR)
	assert spec.family == "baseline"
	assert spec.seed_strategy == "bm25"
	assert spec.seed_top_k == 1
	assert spec.hop_budget == 0
	assert spec.baseline == "bm25"

	selector = build_selector(BM25_SELECTOR)
	assert selector.__class__.__name__ == "CanonicalBM25Selector"


@pytest.mark.parametrize(
	("query", "expected_node_id"),
	[
		("southeastern United States", "Florida"),
		("launches from Cape Canaveral", "Moon Launch Program"),
		("Cape Canaveral is in", "Cape Canaveral"),
	],
)
def test_bm25_selector_returns_deterministic_seed(
	bm25_iirc_store, query: str, expected_node_id: str
):
	selector = build_selector(BM25_SELECTOR)
	result = selector.select(
		bm25_iirc_store,
		EvaluationCase(case_id=f"q-{expected_node_id}", query=query),
		EvaluationBudget(token_budget_tokens=128),
	)

	assert result.selected_node_ids == [expected_node_id]
	assert result.selector_metadata is not None
	assert result.selector_metadata.seed_strategy == "bm25"
	assert result.selector_metadata.seed_backend == "bm25"


def test_bm25_budget_fill_respects_budget(bm25_iirc_store):
	budget = EvaluationBudget(token_budget_tokens=8)
	selector = build_selector(BM25_FILL_SELECTOR)
	result = selector.select(
		bm25_iirc_store,
		EvaluationCase(case_id="q-fill", query="Cape Canaveral Florida launch"),
		budget,
	)

	assert result.selected_node_ids
	assert result.token_cost_estimate <= budget.token_budget_tokens
	assert result.selector_metadata is not None
	assert result.selector_metadata.budget_fill_mode == "relative_drop"
