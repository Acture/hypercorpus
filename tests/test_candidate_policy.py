from hypercorpus.candidate import select_starting_candidate, select_starting_candidates
from hypercorpus.candidate.policy import MaxPhiOverAnchors, SelectByCosTopK


def test_max_phi_over_anchors_returns_ranked_list(sample_graph):
	selected = MaxPhiOverAnchors(k=2).select_start(sample_graph, "launch site")
	assert selected == ["cape", "mission"]


def test_select_by_cos_topk_returns_node_ids(sample_graph):
	selected = SelectByCosTopK(k=2).select_start(
		sample_graph,
		"Which city hosts the launch site?",
	)
	assert selected == ["mission", "cape"]


def test_select_starting_helpers_use_stable_contract(sample_graph):
	policy = MaxPhiOverAnchors(k=2)
	assert select_starting_candidates(sample_graph, "launch site", policy) == ["cape", "mission"]
	assert select_starting_candidate(sample_graph, "launch site", policy) == "cape"
