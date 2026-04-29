from __future__ import annotations

from dataclasses import dataclass

from hypercorpus.graph import LinkContextGraph
from hypercorpus.text import approx_token_count, normalized_token_overlap


@dataclass(slots=True)
class EvidenceSnippet:
	node_id: str
	title: str
	text: str
	score: float


@dataclass(slots=True)
class ExtractedRelation:
	source: str
	target: str
	target_title: str
	anchor_text: str
	sentence: str
	score: float


@dataclass(slots=True)
class QuerySubgraph:
	query: str
	node_ids: list[str]
	snippets: list[EvidenceSnippet]
	relations: list[ExtractedRelation]
	token_cost_estimate: int


class SubgraphExtractor:
	def __init__(
		self,
		*,
		max_snippets_per_node: int = 2,
		max_relations: int = 8,
	):
		self.max_snippets_per_node = max_snippets_per_node
		self.max_relations = max_relations

	def extract(
		self,
		query: str,
		graph: LinkContextGraph,
		visited_nodes: list[str],
	) -> QuerySubgraph:
		node_ids = list(dict.fromkeys(visited_nodes))
		snippets: list[EvidenceSnippet] = []
		relations: list[ExtractedRelation] = []
		visited_set = set(node_ids)

		for node_id in node_ids:
			document = graph.get_document(node_id)
			if document is None:
				continue

			scored_sentences = [
				(normalized_token_overlap(query, sentence), sentence)
				for sentence in document.sentences
			]
			scored_sentences.sort(key=lambda item: item[0], reverse=True)
			for score, sentence in scored_sentences[: self.max_snippets_per_node]:
				if score <= 0 and snippets:
					continue
				snippets.append(
					EvidenceSnippet(
						node_id=node_id,
						title=document.title,
						text=sentence,
						score=score,
					)
				)

			for link in graph.links_from(node_id):
				target_title = graph.node_attr.get(link.target, {}).get(
					"title", link.target
				)
				anchor_score = normalized_token_overlap(query, link.anchor_text)
				sentence_score = normalized_token_overlap(query, link.sentence)
				target_score = normalized_token_overlap(query, str(target_title))
				if (
					link.target not in visited_set
					and max(anchor_score, target_score) <= 0
				):
					continue
				score = max(anchor_score, sentence_score, target_score)
				relations.append(
					ExtractedRelation(
						source=node_id,
						target=link.target,
						target_title=str(target_title),
						anchor_text=link.anchor_text,
						sentence=link.sentence,
						score=score,
					)
				)

		relations.sort(key=lambda item: item.score, reverse=True)
		relations = relations[: self.max_relations]
		token_cost_estimate = sum(
			approx_token_count(snippet.text) for snippet in snippets
		)
		token_cost_estimate += sum(
			approx_token_count(relation.sentence) for relation in relations
		)

		return QuerySubgraph(
			query=query,
			node_ids=node_ids,
			snippets=snippets,
			relations=relations,
			token_cost_estimate=token_cost_estimate,
		)


class FullDocumentExtractor:
	"""Builds a QuerySubgraph that exposes the full document text per visited node.

	Unlike `SubgraphExtractor`, which pre-filters sentences by lexical overlap with
	the query (capped at `max_snippets_per_node=2` by default), this extractor is
	meant for the answerer's evidence context: it emits every sentence of every
	visited document, in document order, without lexical filtering. Relations are
	intentionally omitted since the full sentence text already carries the link
	context.

	A global token cap (`max_input_tokens`, default 50_000) is enforced by
	dropping nodes from the tail of `visited_nodes` once the budget is exceeded.
	The first sentence that triggered the overflow is also dropped, ensuring the
	context never exceeds the cap.
	"""

	def __init__(
		self,
		*,
		max_input_tokens: int = 50_000,
	):
		self.max_input_tokens = max_input_tokens

	def extract(
		self,
		query: str,
		graph: LinkContextGraph,
		visited_nodes: list[str],
	) -> QuerySubgraph:
		node_ids = list(dict.fromkeys(visited_nodes))
		snippets: list[EvidenceSnippet] = []
		token_cost = 0

		for node_id in node_ids:
			document = graph.get_document(node_id)
			if document is None:
				continue
			for sentence in document.sentences:
				sentence_tokens = approx_token_count(sentence)
				if token_cost + sentence_tokens > self.max_input_tokens:
					return QuerySubgraph(
						query=query,
						node_ids=node_ids,
						snippets=snippets,
						relations=[],
						token_cost_estimate=token_cost,
					)
				snippets.append(
					EvidenceSnippet(
						node_id=node_id,
						title=document.title,
						text=sentence,
						score=1.0,
					)
				)
				token_cost += sentence_tokens

		return QuerySubgraph(
			query=query,
			node_ids=node_ids,
			snippets=snippets,
			relations=[],
			token_cost_estimate=token_cost,
		)
