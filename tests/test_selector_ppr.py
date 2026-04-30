import pytest

from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.graph import LinkContextGraph
from hypercorpus.selector import (
	available_selector_names,
	build_selector,
	parse_selector_spec,
)

SEMANTIC_PPR = "top_1_seed__sentence_transformer__hop_0__semantic_ppr"
SEMANTIC_PPR_FILL = (
	"top_1_seed__sentence_transformer__hop_0__semantic_ppr__budget_fill_relative_drop"
)


class FakeSemanticEmbedder:
	backend_name = "fake_sentence_transformer"
	model_name = "fake-semantic-ppr"

	def encode(self, texts):
		return [self._encode(text) for text in texts]

	def _encode(self, text: str) -> list[float]:
		lower = text.lower()
		if lower.startswith("who directed"):
			return [0.0, 1.0, 0.0]
		if lower.startswith("which state") or lower.startswith("where"):
			return [1.0, 0.0, 0.0]
		if lower.startswith("moon launch program"):
			return [0.95, 0.2, 0.0]
		if lower.startswith("cape canaveral"):
			return [0.9, 0.0, 0.2]
		if lower.startswith("alice johnson"):
			return [0.0, 1.0, 0.0]
		if lower.startswith("florida"):
			return [0.8, 0.0, 1.0]
		return [0.0, 0.0, 0.0]


def _fake_embedder_factory(_config):
	return FakeSemanticEmbedder()


def _within_two_hops(graph: LinkContextGraph, roots: list[str]) -> set[str]:
	seen = set(roots)
	frontier = set(roots)
	for _ in range(2):
		next_frontier: set[str] = set()
		for node_id in frontier:
			next_frontier.update(graph.neighbors(node_id))
		next_frontier.difference_update(seen)
		seen.update(next_frontier)
		frontier = next_frontier
	return seen


def test_semantic_ppr_selector_names_are_registered_and_parseable():
	names = available_selector_names(include_diagnostics=False)

	assert SEMANTIC_PPR in names
	assert SEMANTIC_PPR_FILL in names

	spec = parse_selector_spec(SEMANTIC_PPR)
	assert spec.family == "baseline"
	assert spec.baseline == "semantic_ppr"
	assert spec.seed_strategy == "sentence_transformer"
	assert spec.seed_top_k == 1
	assert spec.hop_budget == 0

	filled = parse_selector_spec(SEMANTIC_PPR_FILL)
	assert filled.baseline == "semantic_ppr"
	assert filled.budget_fill_mode == "relative_drop"

	with pytest.raises(ValueError, match="Unknown selector"):
		parse_selector_spec("top_1_seed__lexical_overlap__hop_0__semantic_ppr")


@pytest.mark.parametrize(
	"query",
	[
		"Which state contains Cape Canaveral?",
		"Who directed the Moon Launch Program?",
		"Where is Cape Canaveral located?",
	],
)
@pytest.mark.parametrize("selector_name", [SEMANTIC_PPR, SEMANTIC_PPR_FILL])
def test_semantic_ppr_selects_from_seed_or_local_neighborhood(
	sample_graph: LinkContextGraph,
	selector_name: str,
	query: str,
):
	selector = build_selector(
		selector_name,
		sentence_transformer_embedder_factory=_fake_embedder_factory,
	)
	case = EvaluationCase(
		case_id=f"ppr-{query}",
		query=query,
		gold_support_nodes=["mission", "cape", "florida"],
		gold_start_nodes=["mission"],
		dataset_name="synthetic",
	)
	selection = selector.select(
		sample_graph, case, EvaluationBudget(token_budget_tokens=512)
	)

	assert selection.selected_node_ids
	assert selection.selected_node_ids[0] in _within_two_hops(
		sample_graph, selection.root_node_ids
	)
	assert selection.selector_usage is not None
	assert 1 <= selection.selector_usage.ppr_iter_count <= 20
	assert (
		selection.selector_usage.ppr_final_delta <= 1e-6
		or selection.selector_usage.ppr_iter_count == 20
	)
	assert selection.selector_usage.ppr_wall_clock_s >= 0.0
	assert any(item.startswith("ppr_iter_count:") for item in selection.debug_trace)
