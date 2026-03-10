from webwalker.graph import LinkContextGraph


def test_from_2wikimultihop_records_builds_semantic_links():
	records = [
		{
			"id": "mission",
			"title": "Moon Launch Program",
			"sentences": ["Moon Launch Program uses Cape Canaveral as its launch site."],
			"mentions": [
				{
					"id": 0,
					"start": 25,
					"end": 40,
					"ref_url": "Cape_Canaveral",
					"ref_ids": ["cape"],
					"sent_idx": 0,
				}
			],
		},
		{
			"id": "cape",
			"title": "Cape Canaveral",
			"sentences": ["Cape Canaveral is a city in Florida."],
			"mentions": [],
		},
	]

	graph = LinkContextGraph.from_2wikimultihop_records(records)
	assert graph.neighbors("mission") == ["cape"]
	link = graph.links_between("mission", "cape")[0]
	assert link.anchor_text == "Cape Canaveral"
	assert link.sentence == "Moon Launch Program uses Cape Canaveral as its launch site."
	assert graph.get_document("cape").title == "Cape Canaveral"


def test_induced_subgraph_keeps_only_selected_nodes(sample_graph):
	subgraph = sample_graph.induced_subgraph(["mission", "cape"])
	assert subgraph.nodes == ["mission", "cape"]
	assert subgraph.neighbors("mission") == ["cape"]
	assert subgraph.neighbors("cape") == []


def test_from_normalized_records_builds_graph():
	graph = LinkContextGraph.from_normalized_records(
		[
			{
				"node_id": "guide",
				"title": "Guide",
				"text": "Guide points to the API.",
				"links": [
					{
						"target": "api",
						"anchor_text": "API",
						"sentence": "Guide points to the API.",
						"sent_idx": 0,
					}
				],
			},
			{
				"node_id": "api",
				"title": "API Reference",
				"text": "API Reference explains TLS mode.",
			},
		],
		dataset_name="docs",
	)

	assert graph.neighbors("guide") == ["api"]
	assert graph.node_attr["guide"]["dataset"] == "docs"
	assert graph.links_between("guide", "api")[0].anchor_text == "API"
