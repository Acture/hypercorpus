from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

	Unlike `SubgraphExtractor`, which pre-filters sentences by lexical overlap
	with the query (capped at `max_snippets_per_node=2` by default), this
	extractor is meant for the answerer's evidence context: it emits one
	`EvidenceSnippet` per visited node, with `text` being the full document
	(all sentences joined by a space), in node-arrival order. Relations are
	intentionally omitted since the full sentence text already carries the
	link context.

	A real-token cap (`max_input_tokens`, default 55_000) is enforced against
	the actual rendered prompt size using `tiktoken` (`gpt-4`/cl100k by
	default). When the cap is exceeded, trailing nodes are dropped one at a
	time and the prompt is re-rendered until the count fits. This means the
	cap accounts for:

	  * subword tokenization (GPT-4 produces ~1.6×–2.2× more tokens than the
	    whitespace-based `approx_token_count` for English Wikipedia text);
	  * per-snippet renderer boilerplate (`[snippet] node | title\\nscore=...`);
	  * the system prompt + question + JSON wrapper overhead.

	The caller passes those overheads via `prompt_overhead_text` (a single
	concatenated string approximating the system prompt + question shell).
	If `tiktoken` is unavailable, the extractor falls back to
	`approx_token_count(rendered) * 2` as a safe over-estimate.
	"""

	def __init__(
		self,
		*,
		max_input_tokens: int = 55_000,
		tokenizer_model: str = "gpt-4",
	):
		self.max_input_tokens = max_input_tokens
		self.tokenizer_model = tokenizer_model
		self._encoder: Any = None

	def _count_tokens(self, text: str) -> int:
		if self._encoder is None:
			try:
				import tiktoken

				self._encoder = tiktoken.encoding_for_model(self.tokenizer_model)
			except Exception:  # noqa: BLE001 — tiktoken or model lookup failed
				self._encoder = False
		if self._encoder is False:
			# Conservative over-estimate when tiktoken is unavailable.
			return approx_token_count(text) * 2
		return len(self._encoder.encode(text))

	def extract(
		self,
		query: str,
		graph: LinkContextGraph,
		visited_nodes: list[str],
		*,
		prompt_overhead_text: str = "",
	) -> QuerySubgraph:
		node_ids = list(dict.fromkeys(visited_nodes))

		# Build one snippet per node containing the full document text. This
		# minimizes per-snippet renderer boilerplate compared to one snippet
		# per sentence (cuts ~8k tokens for 64-node dense+fill cases).
		all_snippets: list[EvidenceSnippet] = []
		for node_id in node_ids:
			document = graph.get_document(node_id)
			if document is None:
				continue
			text = " ".join(document.sentences)
			if not text.strip():
				continue
			all_snippets.append(
				EvidenceSnippet(
					node_id=node_id,
					title=document.title,
					text=text,
					score=1.0,
				)
			)

		# Iteratively drop trailing snippets until the rendered prompt fits.
		from hypercorpus.answering import _render_subgraph

		snippets = list(all_snippets)
		overhead_tokens = self._count_tokens(prompt_overhead_text) if prompt_overhead_text else 0
		while snippets:
			rendered = _render_subgraph(
				QuerySubgraph(
					query=query,
					node_ids=node_ids,
					snippets=snippets,
					relations=[],
					token_cost_estimate=0,
				)
			)
			total = self._count_tokens(rendered) + overhead_tokens
			if total <= self.max_input_tokens:
				break
			snippets.pop()  # drop trailing (lowest-priority) node

		token_cost_estimate = sum(approx_token_count(s.text) for s in snippets)
		return QuerySubgraph(
			query=query,
			node_ids=node_ids,
			snippets=snippets,
			relations=[],
			token_cost_estimate=token_cost_estimate,
		)
