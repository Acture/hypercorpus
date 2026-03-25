from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Any
from urllib.parse import urlsplit

from hypercorpus.datasets.common import (
	BaseDatasetAdapter,
	coerce_question_type,
	dedupe_strings,
	load_json_records,
	pick_first,
)
from hypercorpus.eval import EvaluationCase
from hypercorpus.graph import LinkContextGraph

_BLOCK_TAGS = {"p", "li", "dd", "dt", "pre"}
_WHITESPACE_RE = re.compile(r"\s+")


def load_docs_graph(
	graph_source: str | Path,
	*,
	dataset_name: str = "docs",
) -> LinkContextGraph:
	resolved = Path(graph_source)
	if resolved.is_dir():
		records = _build_docs_records_from_html_dir(resolved, dataset_name=dataset_name)
		return LinkContextGraph.from_normalized_records(
			records, dataset_name=dataset_name
		)
	records = load_json_records(resolved)
	return LinkContextGraph.from_normalized_records(records, dataset_name=dataset_name)


def load_docs_questions(
	questions_path: str | Path,
	*,
	limit: int | None = None,
	dataset_name: str = "docs",
) -> list[EvaluationCase]:
	records = load_json_records(questions_path)
	if limit is not None:
		records = records[:limit]

	cases: list[EvaluationCase] = []
	for record in records:
		support_nodes = dedupe_strings(
			_coerce_node_refs(
				pick_first(
					record, "gold_support_nodes", "supporting_pages", "supporting_docs"
				)
			)
		)
		start_nodes = dedupe_strings(
			_coerce_node_refs(pick_first(record, "gold_start_nodes", "start_nodes"))
			or support_nodes
		)
		raw_path_nodes = _coerce_node_refs(
			pick_first(record, "gold_path_nodes", "path_nodes", "reasoning_path")
		)
		path_nodes = dedupe_strings(raw_path_nodes) if raw_path_nodes else None
		cases.append(
			EvaluationCase(
				case_id=str(pick_first(record, "case_id", "id", "qid") or ""),
				query=str(pick_first(record, "query", "question") or ""),
				expected_answer=_string_or_none(
					pick_first(record, "expected_answer", "answer")
				),
				dataset_name=dataset_name,
				gold_support_nodes=support_nodes,
				gold_start_nodes=start_nodes,
				gold_path_nodes=path_nodes,
				question_type=coerce_question_type(
					pick_first(record, "question_type", "type")
				),
			)
		)

	return cases


class DocumentationAdapter(BaseDatasetAdapter):
	def __init__(self, dataset_name: str = "docs"):
		self.dataset_name = dataset_name

	def load_graph(self, graph_source: str | Path) -> LinkContextGraph:
		return load_docs_graph(graph_source, dataset_name=self.dataset_name)

	def load_cases(
		self,
		questions_source: str | Path,
		*,
		limit: int | None = None,
	) -> list[EvaluationCase]:
		return load_docs_questions(
			questions_source, limit=limit, dataset_name=self.dataset_name
		)


@dataclass(slots=True)
class _CapturedAnchor:
	href: str
	text_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Block:
	tag: str
	text_parts: list[str] = field(default_factory=list)
	anchors: list[_CapturedAnchor] = field(default_factory=list)


@dataclass(slots=True)
class _ParsedPage:
	title: str
	sentences: list[str]
	links: list[dict[str, Any]]


class _DocsHTMLParser(HTMLParser):
	def __init__(self) -> None:
		super().__init__(convert_charrefs=True)
		self.title_parts: list[str] = []
		self.blocks: list[_Block] = []
		self._active_block: _Block | None = None
		self._active_anchor: _CapturedAnchor | None = None
		self._inside_title = False

	def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
		if tag == "title":
			self._inside_title = True
			return
		if tag in _BLOCK_TAGS and self._active_block is None:
			self._active_block = _Block(tag=tag)
			return
		if tag == "a" and self._active_block is not None:
			href = dict(attrs).get("href") or ""
			self._active_anchor = _CapturedAnchor(href=href)
			return
		if tag == "br" and self._active_block is not None:
			self._active_block.text_parts.append(" ")

	def handle_endtag(self, tag: str) -> None:
		if tag == "title":
			self._inside_title = False
			return
		if (
			tag == "a"
			and self._active_block is not None
			and self._active_anchor is not None
		):
			self._active_block.anchors.append(self._active_anchor)
			self._active_anchor = None
			return
		if self._active_block is not None and tag == self._active_block.tag:
			self.blocks.append(self._active_block)
			self._active_block = None

	def handle_data(self, data: str) -> None:
		if self._inside_title:
			self.title_parts.append(data)
		if self._active_block is not None:
			self._active_block.text_parts.append(data)
		if self._active_anchor is not None:
			self._active_anchor.text_parts.append(data)


def _build_docs_records_from_html_dir(
	root: Path, *, dataset_name: str
) -> list[dict[str, Any]]:
	html_paths = sorted(path for path in root.rglob("*.html") if path.is_file())
	aliases = _build_alias_map(root, html_paths)
	records: list[dict[str, Any]] = []

	for path in html_paths:
		node_id = _node_id_for_page(root, path)
		parsed = _parse_html_page(path)
		source_rel_path = path.relative_to(root).as_posix()
		links = []
		for raw_link in parsed.links:
			target = _resolve_target_node(
				source_rel_path=source_rel_path,
				href=str(raw_link["href"]),
				aliases=aliases,
			)
			if target is None:
				continue
			links.append(
				{
					"target": target,
					"anchor_text": str(raw_link["anchor_text"]),
					"sentence": str(raw_link["sentence"]),
					"sent_idx": int(raw_link["sent_idx"]),
					"metadata": {"href": str(raw_link["href"])},
				}
			)

		records.append(
			{
				"node_id": node_id,
				"title": parsed.title or node_id,
				"sentences": parsed.sentences,
				"links": links,
				"metadata": {
					"dataset": dataset_name,
					"url": source_rel_path,
					"source_path": str(path),
				},
			}
		)

	return records


def _parse_html_page(path: Path) -> _ParsedPage:
	parser = _DocsHTMLParser()
	parser.feed(path.read_text(encoding="utf-8"))

	sentences: list[str] = []
	links: list[dict[str, Any]] = []
	for sent_idx, block in enumerate(parser.blocks):
		sentence = _normalize_whitespace("".join(block.text_parts))
		if not sentence:
			continue
		effective_sent_idx = len(sentences)
		sentences.append(sentence)
		for anchor in block.anchors:
			anchor_text = _normalize_whitespace("".join(anchor.text_parts))
			if not anchor_text or not anchor.href:
				continue
			links.append(
				{
					"href": anchor.href,
					"anchor_text": anchor_text,
					"sentence": sentence,
					"sent_idx": effective_sent_idx,
				}
			)

	return _ParsedPage(
		title=_normalize_whitespace("".join(parser.title_parts)),
		sentences=sentences,
		links=links,
	)


def _build_alias_map(root: Path, html_paths: Iterable[Path]) -> dict[str, str]:
	aliases: dict[str, str] = {}
	for path in html_paths:
		rel = path.relative_to(root).as_posix()
		node_id = _node_id_for_page(root, path)
		aliases[rel] = node_id
		if rel.endswith(".html"):
			aliases[rel[:-5]] = node_id
		if rel == "index.html":
			aliases[""] = node_id
			aliases["index"] = node_id
		if rel.endswith("/index.html"):
			base = rel[: -len("/index.html")]
			aliases[base] = node_id
			aliases[f"{base}/"] = node_id
			aliases[f"{base}/index"] = node_id
	return aliases


def _resolve_target_node(
	*,
	source_rel_path: str,
	href: str,
	aliases: Mapping[str, str],
) -> str | None:
	parsed = urlsplit(href)
	if parsed.scheme or parsed.netloc:
		return None
	if not parsed.path:
		return None
	base_dir = PurePosixPath(source_rel_path).parent
	candidate = posixpath.normpath(str(base_dir / parsed.path)).lstrip("./")
	return aliases.get(candidate) or aliases.get(candidate.rstrip("/"))


def _node_id_for_page(root: Path, path: Path) -> str:
	rel = path.relative_to(root).as_posix()
	if rel == "index.html":
		return "index"
	if rel.endswith("/index.html"):
		return rel[: -len("/index.html")]
	if rel.endswith(".html"):
		return rel[:-5]
	return rel


def _coerce_node_refs(value: Any) -> list[str]:
	if value is None:
		return []
	if isinstance(value, str):
		stripped = value.strip()
		return [stripped] if stripped else []
	if isinstance(value, Mapping):
		ref = pick_first(value, "node_id", "title", "page", "id")
		if ref is not None and str(ref).strip():
			return [str(ref).strip()]
		return []
	refs: list[str] = []
	if isinstance(value, Iterable):
		for item in value:
			if isinstance(item, str):
				stripped = item.strip()
				if stripped:
					refs.append(stripped)
				continue
			if isinstance(item, Mapping):
				ref = pick_first(item, "node_id", "title", "page", "id")
				if ref is not None and str(ref).strip():
					refs.append(str(ref).strip())
	return refs


def _normalize_whitespace(value: str) -> str:
	return _WHITESPACE_RE.sub(" ", value).strip()


def _string_or_none(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text or None
