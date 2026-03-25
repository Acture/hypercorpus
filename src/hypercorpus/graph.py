from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Sequence

from hypercorpus.candidate.protocol import EmbeddedGraphLike
from hypercorpus.text import normalized_token_overlap

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class DocumentNode:
	node_id: str
	title: str
	sentences: tuple[str, ...]
	metadata: dict[str, Any] = field(default_factory=dict)

	@property
	def text(self) -> str:
		return " ".join(self.sentences)


@dataclass(slots=True)
class LinkContext:
	source: str
	target: str
	anchor_text: str
	sentence: str
	sent_idx: int
	ref_id: str | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedLinkRecord:
	target: str
	anchor_text: str
	sentence: str
	sent_idx: int
	ref_id: str | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedDocumentRecord:
	node_id: str
	title: str
	sentences: tuple[str, ...]
	links: tuple[NormalizedLinkRecord, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)

	@property
	def text(self) -> str:
		return " ".join(self.sentences)


class LinkContextGraph(EmbeddedGraphLike[str]):
	"""Document graph with hyperlink context attached to each edge."""

	def __init__(
		self,
		documents: Iterable[DocumentNode] | None = None,
		links: Iterable[LinkContext] | None = None,
	):
		self.documents: dict[str, DocumentNode] = {}
		self.adj: dict[str, list[str]] = {}
		self.nodes: list[str] = []
		self.node_attr: dict[str, dict[str, Any]] = {}
		self._links_by_source: dict[str, list[LinkContext]] = defaultdict(list)
		self._links_by_edge: dict[tuple[str, str], list[LinkContext]] = defaultdict(
			list
		)

		for document in documents or ():
			self.add_document(document)
		for link in links or ():
			self.add_link(link)

	def add_document(self, document: DocumentNode) -> None:
		if document.node_id not in self.documents:
			self.nodes.append(document.node_id)
			self.adj[document.node_id] = []
		self.documents[document.node_id] = document
		self.node_attr[document.node_id] = {
			"title": document.title,
			"text": document.text,
			"sentences": list(document.sentences),
			**document.metadata,
		}

	def ensure_document(
		self,
		node_id: str,
		*,
		title: str | None = None,
		placeholder: bool = False,
	) -> None:
		if node_id in self.documents:
			return
		self.add_document(
			DocumentNode(
				node_id=node_id,
				title=title or node_id.replace("_", " "),
				sentences=(),
				metadata={"placeholder": placeholder},
			)
		)

	def add_link(self, link: LinkContext) -> None:
		self.ensure_document(link.source)
		self.ensure_document(
			link.target, title=link.target.replace("_", " "), placeholder=True
		)

		if link.target not in self.adj[link.source]:
			self.adj[link.source].append(link.target)
		self._links_by_source[link.source].append(link)
		self._links_by_edge[(link.source, link.target)].append(link)

	def get_document(self, node_id: str) -> DocumentNode | None:
		return self.documents.get(node_id)

	def neighbors(self, u: str) -> list[str]:
		return list(self.adj.get(u, ()))

	def links_from(self, source: str) -> list[LinkContext]:
		return list(self._links_by_source.get(source, ()))

	def links_between(self, source: str, target: str) -> list[LinkContext]:
		return list(self._links_by_edge.get((source, target), ()))

	def induced_subgraph(self, node_ids: Sequence[str]) -> "LinkContextGraph":
		selected = set(node_ids)
		subgraph = LinkContextGraph()
		for node_id in node_ids:
			document = self.get_document(node_id)
			if document is not None:
				subgraph.add_document(document)
		for source in node_ids:
			for link in self.links_from(source):
				if link.target in selected:
					subgraph.add_link(link)
		return subgraph

	def topk_similar(
		self, q: str, candidates: Sequence[str], k: int
	) -> list[tuple[str, float]]:
		scored: list[tuple[str, float]] = []
		for node_id in candidates:
			attr = self.node_attr.get(node_id, {})
			text = f"{attr.get('title', '')} {attr.get('text', '')}".strip()
			score = normalized_token_overlap(q, text)
			scored.append((node_id, score))
		scored.sort(key=lambda item: item[1], reverse=True)
		return scored[:k]

	@classmethod
	def from_normalized_records(
		cls,
		records: Iterable[NormalizedDocumentRecord | Mapping[str, Any]],
		*,
		dataset_name: str | None = None,
	) -> "LinkContextGraph":
		graph = cls()
		normalized_records = [
			_coerce_document_record(record, dataset_name=dataset_name)
			for record in records
		]

		for record in normalized_records:
			graph.add_document(
				DocumentNode(
					node_id=record.node_id,
					title=record.title,
					sentences=record.sentences,
					metadata=dict(record.metadata),
				)
			)

		for record in normalized_records:
			for link in record.links:
				graph.add_link(
					LinkContext(
						source=record.node_id,
						target=link.target,
						anchor_text=link.anchor_text,
						sentence=link.sentence,
						sent_idx=link.sent_idx,
						ref_id=link.ref_id,
						metadata=dict(link.metadata),
					)
				)

		return graph

	@classmethod
	def from_2wikimultihop_records(
		cls,
		records: Iterable[Mapping[str, Any]],
		*,
		id_field: str = "id",
		title_field: str = "title",
		sentences_field: str = "sentences",
		mentions_field: str = "mentions",
	) -> "LinkContextGraph":
		record_list = list(records)
		id_to_node: dict[str, str] = {}
		title_to_node: dict[str, str] = {}

		for record in record_list:
			node_id = str(record.get(id_field) or record.get(title_field))
			title = str(record.get(title_field) or node_id).replace("_", " ")
			id_to_node[str(node_id)] = node_id
			title_to_node[title] = node_id
			title_to_node[title.replace(" ", "_")] = node_id

		normalized_records: list[NormalizedDocumentRecord] = []
		for record in record_list:
			node_id = str(record.get(id_field) or record.get(title_field))
			title = str(record.get(title_field) or node_id).replace("_", " ")
			sentences = tuple(
				str(sentence) for sentence in record.get(sentences_field, ()) or ()
			)
			links: list[NormalizedLinkRecord] = []
			for mention in record.get(mentions_field, ()) or ():
				target = None
				for ref_id in mention.get("ref_ids") or ():
					target = id_to_node.get(str(ref_id))
					if target is not None:
						break

				ref_url = str(mention.get("ref_url") or "").strip()
				if target is None and ref_url:
					target = title_to_node.get(ref_url) or title_to_node.get(
						ref_url.replace("_", " ")
					)
				if target is None:
					target = ref_url or f"external:{mention.get('id', 'unknown')}"

				sent_idx = int(mention.get("sent_idx", 0))
				sentence = sentences[sent_idx] if 0 <= sent_idx < len(sentences) else ""
				start = int(mention.get("start", 0))
				end = int(mention.get("end", start))
				anchor_text = sentence[start:end].strip() or ref_url.replace("_", " ")

				links.append(
					NormalizedLinkRecord(
						target=str(target),
						anchor_text=anchor_text,
						sentence=sentence,
						sent_idx=sent_idx,
						ref_id=str((mention.get("ref_ids") or [None])[0])
						if mention.get("ref_ids")
						else None,
						metadata={"ref_url": ref_url},
					)
				)

			normalized_records.append(
				NormalizedDocumentRecord(
					node_id=node_id,
					title=title,
					sentences=sentences,
					links=tuple(links),
					metadata={"dataset": "2wikimultihop"},
				)
			)

		return cls.from_normalized_records(
			normalized_records, dataset_name="2wikimultihop"
		)


def _coerce_document_record(
	record: NormalizedDocumentRecord | Mapping[str, Any],
	*,
	dataset_name: str | None,
) -> NormalizedDocumentRecord:
	if isinstance(record, NormalizedDocumentRecord):
		metadata = dict(record.metadata)
		if dataset_name is not None and "dataset" not in metadata:
			metadata["dataset"] = dataset_name
		return NormalizedDocumentRecord(
			node_id=record.node_id,
			title=record.title,
			sentences=record.sentences,
			links=tuple(record.links),
			metadata=metadata,
		)

	node_id = str(record.get("node_id") or record.get("id") or record.get("title"))
	title = str(record.get("title") or node_id).replace("_", " ")
	sentences_raw = record.get("sentences")
	if sentences_raw:
		sentences = tuple(
			str(sentence).strip() for sentence in sentences_raw if str(sentence).strip()
		)
	else:
		text = str(record.get("text") or "").strip()
		sentences = _split_sentences(text)
	metadata = dict(record.get("metadata", {}) or {})
	if dataset_name is not None and "dataset" not in metadata:
		metadata["dataset"] = dataset_name
	links = tuple(_coerce_link_record(link) for link in record.get("links", ()) or ())
	return NormalizedDocumentRecord(
		node_id=node_id,
		title=title,
		sentences=sentences,
		links=links,
		metadata=metadata,
	)


def _coerce_link_record(
	record: NormalizedLinkRecord | Mapping[str, Any],
) -> NormalizedLinkRecord:
	if isinstance(record, NormalizedLinkRecord):
		return record
	return NormalizedLinkRecord(
		target=str(
			record.get("target")
			or record.get("target_id")
			or record.get("ref_url")
			or ""
		),
		anchor_text=str(record.get("anchor_text") or record.get("anchor") or ""),
		sentence=str(record.get("sentence") or record.get("context") or ""),
		sent_idx=int(record.get("sent_idx", 0)),
		ref_id=str(record["ref_id"]) if record.get("ref_id") is not None else None,
		metadata=dict(record.get("metadata", {}) or {}),
	)


def _split_sentences(text: str) -> tuple[str, ...]:
	normalized = " ".join(text.split())
	if not normalized:
		return ()
	return tuple(
		part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()
	)
