from webwalker.answering import Answerer
from webwalker.subgraph import SubgraphExtractor


def test_subgraph_extractor_keeps_query_relevant_context(sample_graph):
	subgraph = SubgraphExtractor().extract(
		"Which city hosts the launch site?",
		sample_graph,
		["mission", "cape"],
	)
	assert set(subgraph.node_ids) == {"mission", "cape"}
	assert all(snippet.node_id in {"mission", "cape"} for snippet in subgraph.snippets)
	assert [relation.target for relation in subgraph.relations] == ["cape"]


def test_answerer_prefers_relation_target_for_where_question(sample_graph):
	subgraph = SubgraphExtractor().extract(
		"Which city hosts the launch site?",
		sample_graph,
		["mission", "cape"],
	)
	answer = Answerer().answer("Which city hosts the launch site?", subgraph)
	assert answer.answer == "Cape Canaveral"
	assert answer.evidence
