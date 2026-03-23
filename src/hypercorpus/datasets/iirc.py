from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from hypercorpus.datasets.common import (
    BaseDatasetAdapter,
    NormalizedDatasetLayout,
    dedupe_strings,
    load_json_records,
    pick_first,
    write_normalized_dataset,
)
from hypercorpus.eval import EvaluationCase
from hypercorpus.graph import LinkContextGraph

_WHITESPACE_RE = re.compile(r"\s+")


def load_iirc_graph(graph_records_path: str | Path) -> LinkContextGraph:
    records = load_json_records(graph_records_path)
    normalized = [_normalize_iirc_graph_record(record) for record in records]
    return LinkContextGraph.from_normalized_records(normalized, dataset_name="iirc")


def load_iirc_questions(
    questions_path: str | Path,
    *,
    limit: int | None = None,
) -> list[EvaluationCase]:
    records = load_json_records(questions_path)
    if limit is not None:
        records = records[:limit]

    cases: list[EvaluationCase] = []
    for record in records:
        support_nodes = dedupe_strings(_coerce_node_refs(pick_first(record, "gold_support_nodes", "supporting_docs", "supporting_pages")))
        start_nodes = dedupe_strings(
            _coerce_node_refs(pick_first(record, "gold_start_nodes", "start_nodes", "context_documents"))
            or support_nodes
        )
        raw_path_nodes = _coerce_node_refs(pick_first(record, "gold_path_nodes", "path_nodes", "reasoning_path"))
        path_nodes = dedupe_strings(raw_path_nodes) if raw_path_nodes else None
        cases.append(
            EvaluationCase(
                case_id=str(pick_first(record, "case_id", "id", "qid", "question_id") or ""),
                query=str(pick_first(record, "query", "question") or ""),
                expected_answer=_string_or_none(pick_first(record, "expected_answer", "answer")),
                dataset_name="iirc",
                gold_support_nodes=support_nodes,
                gold_start_nodes=start_nodes,
                gold_path_nodes=path_nodes,
            )
        )

    return cases


class IIRCAdapter(BaseDatasetAdapter):
    dataset_name = "iirc"

    def load_graph(self, graph_source: str | Path) -> LinkContextGraph:
        return load_iirc_graph(graph_source)

    def load_cases(
        self,
        questions_source: str | Path,
        *,
        limit: int | None = None,
    ) -> list[EvaluationCase]:
        return load_iirc_questions(questions_source, limit=limit)


def convert_iirc_raw_dataset(
    raw_source: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    source_manifest_path: str | Path | None = None,
) -> NormalizedDatasetLayout:
    question_files, context_source = _resolve_iirc_raw_sources(raw_source)
    title_aliases = _build_iirc_title_aliases(question_files)
    graph_records = _convert_iirc_graph_records(
        question_files=question_files,
        context_source=context_source,
        title_aliases=title_aliases,
    )
    question_splits: dict[str, list[EvaluationCase]] = {}
    for split_name, questions_path in question_files.items():
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        question_splits[split_name] = _convert_iirc_questions_payload(payload, title_aliases=title_aliases)
    return write_normalized_dataset(
        output_dir,
        dataset_name="iirc",
        question_splits=question_splits,
        graph_records=graph_records,
        source_manifest_path=source_manifest_path,
        overwrite=overwrite,
    )


def _normalize_iirc_graph_record(record: Mapping[str, Any]) -> dict[str, Any]:
    links = []
    for link in pick_first(record, "links", "mentions", "outbound_links") or ():
        if isinstance(link, str):
            target = link.strip()
            if not target:
                continue
            links.append(
                {
                    "target": target,
                    "anchor_text": target,
                    "sentence": "",
                    "sent_idx": 0,
                }
            )
            continue
        if not isinstance(link, Mapping):
            continue
        target = str(pick_first(link, "target", "target_id", "title", "page", "ref_url") or "").strip()
        if not target:
            continue
        links.append(
            {
                "target": target,
                "anchor_text": str(pick_first(link, "anchor_text", "anchor", "text") or target),
                "sentence": str(pick_first(link, "sentence", "context", "paragraph") or ""),
                "sent_idx": int(pick_first(link, "sent_idx", "sentence_index") or 0),
                "ref_id": _string_or_none(pick_first(link, "ref_id", "target_id")),
                "metadata": dict(link.get("metadata", {}) or {}),
            }
        )

    metadata = dict(record.get("metadata", {}) or {})
    metadata.setdefault("dataset", "iirc")
    return {
        "node_id": str(pick_first(record, "node_id", "id", "title")),
        "title": str(pick_first(record, "title", "node_id", "id")),
        "sentences": _coerce_sentences(record),
        "links": links,
        "metadata": metadata,
    }


def _normalize_iirc_whitespace(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _clean_iirc_title(value: Any) -> str | None:
    if value is None:
        return None
    text = _normalize_iirc_whitespace(unescape(str(value)))
    return text or None


def _register_iirc_title_alias(title_aliases: dict[str, str], raw_title: Any) -> str | None:
    cleaned = _clean_iirc_title(raw_title)
    if cleaned is None or cleaned.lower() == "main":
        return None
    normalized = cleaned.lower()
    return title_aliases.setdefault(normalized, cleaned)


def _resolve_iirc_title_alias(
    raw_title: Any,
    *,
    title_aliases: Mapping[str, str],
    source_title: str | None = None,
) -> str | None:
    cleaned = _clean_iirc_title(raw_title)
    if cleaned is None:
        return None
    if cleaned.lower() == "main":
        cleaned = _clean_iirc_title(source_title)
        if cleaned is None:
            return None
    return title_aliases.get(cleaned.lower(), cleaned)


class _IIRCContextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.anchors: list[dict[str, str]] = []
        self._active_href: str | None = None
        self._active_anchor_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._active_href = dict(attrs).get("href") or ""
            self._active_anchor_parts = []
        elif tag == "br":
            self.text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._active_href is None:
            return
        anchor_text = _normalize_iirc_whitespace("".join(self._active_anchor_parts))
        if anchor_text:
            self.anchors.append({"href": self._active_href, "anchor_text": anchor_text})
        self._active_href = None
        self._active_anchor_parts = []

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self._active_href is not None:
            self._active_anchor_parts.append(data)


def _coerce_sentences(record: Mapping[str, Any]) -> list[str]:
    raw_sentences = pick_first(record, "sentences", "paragraphs")
    if isinstance(raw_sentences, list):
        return [str(sentence).strip() for sentence in raw_sentences if str(sentence).strip()]
    text = str(pick_first(record, "text", "context", "paragraph") or "").strip()
    if not text:
        return []
    return [text]


def _coerce_node_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        ref = pick_first(value, "node_id", "title", "page", "doc", "id")
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
                ref = pick_first(item, "node_id", "title", "page", "doc", "id")
                if ref is not None and str(ref).strip():
                    refs.append(str(ref).strip())
    return refs


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_iirc_raw_sources(raw_source: str | Path) -> tuple[dict[str, Path], Path | None]:
    resolved = Path(raw_source)
    if resolved.is_file():
        return {_infer_split_name(resolved): resolved}, None
    candidates = sorted(path for path in resolved.rglob("*") if path.is_file() and path.suffix.lower() == ".json")
    if not candidates:
        raise ValueError(f"No IIRC raw JSON files found under {resolved}")
    context_source: Path | None = None
    question_files: dict[str, Path] = {}
    for candidate in candidates:
        lowered = candidate.name.lower()
        if "context_articles" in lowered or "context" in lowered:
            context_source = candidate
            continue
        split_name = _infer_split_name(candidate)
        question_files[split_name] = candidate
    if not question_files:
        raise ValueError(f"No IIRC question split files found under {resolved}")
    return question_files, context_source


def _infer_split_name(path: Path) -> str:
    stem = path.stem.lower()
    for split_name in ("train", "dev", "test"):
        if split_name in stem:
            return split_name
    return "dev"


def _build_iirc_title_aliases(question_files: Mapping[str, Path]) -> dict[str, str]:
    title_aliases: dict[str, str] = {}
    for questions_path in question_files.values():
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for passage_record in payload:
            if not isinstance(passage_record, Mapping):
                continue
            source_title = _register_iirc_title_alias(
                title_aliases,
                pick_first(passage_record, "title", "pid", "id", "node_id"),
            )
            for raw_link_target in _iter_iirc_passage_link_targets(passage_record):
                _register_iirc_title_alias(title_aliases, raw_link_target)
            for question in passage_record.get("questions", []) or []:
                if not isinstance(question, Mapping):
                    continue
                for raw_title in _iter_iirc_question_title_refs(question, source_title=source_title):
                    _register_iirc_title_alias(title_aliases, raw_title)
    return title_aliases


def _iter_iirc_passage_link_targets(record: Mapping[str, Any]) -> Iterable[str]:
    for link in record.get("links", []) or []:
        if isinstance(link, str):
            stripped = link.strip()
            if stripped:
                yield stripped
            continue
        if not isinstance(link, Mapping):
            continue
        target = pick_first(link, "target", "title", "page", "ref_url")
        if target is not None and str(target).strip():
            yield str(target).strip()


def _iter_iirc_question_title_refs(question: Mapping[str, Any], *, source_title: str | None) -> Iterable[str]:
    for question_link in question.get("question_links", []) or []:
        if isinstance(question_link, str) and question_link.strip():
            yield question_link.strip()
    for passage in question_context_passages(question, source_title=source_title):
        yield passage
    for passage in _answer_span_passages(question, source_title=source_title):
        yield passage


def _canonical_iirc_title_refs(
    refs: Iterable[str],
    *,
    title_aliases: Mapping[str, str],
    source_title: str | None = None,
) -> list[str]:
    canonical_refs: list[str] = []
    for ref in refs:
        canonical = _resolve_iirc_title_alias(ref, title_aliases=title_aliases, source_title=source_title)
        if canonical is not None:
            canonical_refs.append(canonical)
    return canonical_refs


def _convert_iirc_questions_payload(payload: Any, *, title_aliases: Mapping[str, str]) -> list[EvaluationCase]:
    if not isinstance(payload, list):
        raise ValueError("IIRC raw question payload must be a list of passage records")
    cases: list[EvaluationCase] = []
    for passage_record in payload:
        if not isinstance(passage_record, Mapping):
            continue
        source_title = _resolve_iirc_title_alias(
            pick_first(passage_record, "title", "pid", "id"),
            title_aliases=title_aliases,
        )
        for question in passage_record.get("questions", []) or []:
            if not isinstance(question, Mapping):
                continue
            question_links = _canonical_iirc_title_refs(
                [str(item).strip() for item in question.get("question_links", []) or [] if isinstance(item, str) and str(item).strip()],
                title_aliases=title_aliases,
                source_title=source_title,
            )
            answer_passages = _canonical_iirc_title_refs(
                _answer_span_passages(question, source_title=source_title),
                title_aliases=title_aliases,
                source_title=source_title,
            )
            context_passages = _canonical_iirc_title_refs(
                question_context_passages(question, source_title=source_title),
                title_aliases=title_aliases,
                source_title=source_title,
            )
            support_nodes = dedupe_strings(
                ([source_title] if source_title else [])
                + question_links
                + answer_passages
                + context_passages
            )
            path_nodes = dedupe_strings(([source_title] if source_title else []) + question_links)
            cases.append(
                EvaluationCase(
                    case_id=str(pick_first(question, "qid", "question_id", "id", "case_id") or ""),
                    query=str(pick_first(question, "question", "query") or ""),
                    expected_answer=_extract_iirc_answer_text(question.get("answer")),
                    dataset_name="iirc",
                    gold_support_nodes=support_nodes,
                    gold_start_nodes=[source_title] if source_title else None,
                    gold_path_nodes=path_nodes or None,
                )
            )
    return cases


def _convert_iirc_graph_records(
    *,
    question_files: dict[str, Path],
    context_source: Path | None,
    title_aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    graph_records: dict[str, dict[str, Any]] = {}
    for questions_path in question_files.values():
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for record in payload:
            if not isinstance(record, Mapping):
                continue
            normalized = _normalize_iirc_raw_passage_record(record, title_aliases=title_aliases)
            graph_records[normalized["node_id"]] = normalized
    if context_source is not None:
        context_payload = json.loads(context_source.read_text(encoding="utf-8"))
        for record in _iter_iirc_context_records(context_payload, title_aliases=title_aliases):
            normalized = _normalize_iirc_raw_passage_record(record, title_aliases=title_aliases)
            graph_records.setdefault(normalized["node_id"], normalized)
    return list(graph_records.values())


def _normalize_iirc_raw_passage_record(
    record: Mapping[str, Any],
    *,
    title_aliases: Mapping[str, str],
) -> dict[str, Any]:
    title = _resolve_iirc_title_alias(
        pick_first(record, "title", "pid", "id", "node_id"),
        title_aliases=title_aliases,
    ) or ""
    text = pick_first(record, "text", "context", "paragraph")
    sentences = [text] if isinstance(text, str) and text.strip() else _coerce_sentences(record)
    links = []
    for link in record.get("links", []) or []:
        if not isinstance(link, Mapping):
            continue
        target = _resolve_iirc_title_alias(
            pick_first(link, "target", "title", "page", "ref_url"),
            title_aliases=title_aliases,
        )
        if not target:
            continue
        sentence = sentences[0] if sentences else ""
        links.append(
            {
                "target": target,
                "anchor_text": target,
                "sentence": sentence,
                "sent_idx": 0,
                "metadata": {"indices": link.get("indices")},
            }
        )
    return {
        "node_id": title,
        "title": title,
        "sentences": [str(sentence).strip() for sentence in sentences if str(sentence).strip()],
        "links": links,
        "metadata": {"dataset": "iirc"},
    }


def _iter_iirc_context_records(
    payload: Any,
    *,
    title_aliases: Mapping[str, str],
) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, Mapping):
                merged = dict(value)
                merged.setdefault(
                    "title",
                    _resolve_iirc_title_alias(key, title_aliases=title_aliases) or str(key),
                )
                yield merged
            elif isinstance(value, str):
                yield _build_iirc_html_context_record(
                    raw_title=str(key),
                    html_text=value,
                    title_aliases=title_aliases,
                )


def _build_iirc_html_context_record(
    *,
    raw_title: str,
    html_text: str,
    title_aliases: Mapping[str, str],
) -> dict[str, Any]:
    title = _resolve_iirc_title_alias(raw_title, title_aliases=title_aliases) or (_clean_iirc_title(raw_title) or raw_title)
    sentences: list[str] = []
    links: list[dict[str, Any]] = []
    for paragraph in [segment for segment in html_text.splitlines() if segment.strip()]:
        parsed = _parse_iirc_context_paragraph(paragraph)
        sentence = parsed["sentence"]
        if not sentence:
            continue
        sent_idx = len(sentences)
        sentences.append(sentence)
        for anchor in parsed["anchors"]:
            target = _resolve_iirc_title_alias(
                _decode_iirc_href_target(anchor["href"]),
                title_aliases=title_aliases,
            )
            if not target:
                continue
            links.append(
                {
                    "target": target,
                    "anchor_text": anchor["anchor_text"],
                    "sentence": sentence,
                    "sent_idx": sent_idx,
                    "metadata": {"href": anchor["href"]},
                }
            )
    return {
        "title": title,
        "sentences": sentences,
        "links": links,
        "metadata": {"dataset": "iirc"},
    }


def _parse_iirc_context_paragraph(paragraph_html: str) -> dict[str, Any]:
    parser = _IIRCContextHTMLParser()
    parser.feed(paragraph_html)
    parser.close()
    return {
        "sentence": _normalize_iirc_whitespace("".join(parser.text_parts)),
        "anchors": parser.anchors,
    }


def _decode_iirc_href_target(href: str) -> str | None:
    cleaned_href = _clean_iirc_title(href)
    if cleaned_href is None:
        return None
    parsed = urlsplit(cleaned_href)
    raw_target = parsed.path or parsed.fragment or cleaned_href
    raw_target = raw_target.lstrip("/")
    if raw_target.startswith("wiki/"):
        raw_target = raw_target[len("wiki/") :]
    if "/" in raw_target:
        raw_target = raw_target.rsplit("/", 1)[-1]
    decoded = unquote(unescape(raw_target)).replace("_", " ")
    return _clean_iirc_title(decoded)


def question_context_passages(question: Mapping[str, Any], *, source_title: str | None = None) -> list[str]:
    passages: list[str] = []
    for item in question.get("context", []) or []:
        if not isinstance(item, Mapping):
            continue
        passage = pick_first(item, "passage", "title", "page", "doc")
        if passage is None or not str(passage).strip():
            continue
        raw_passage = str(passage).strip()
        if raw_passage.lower() == "main":
            if source_title:
                passages.append(source_title)
            continue
        passages.append(raw_passage)
    return passages


def _answer_span_passages(question: Mapping[str, Any], *, source_title: str | None = None) -> list[str]:
    answer = question.get("answer")
    if not isinstance(answer, Mapping):
        return []
    spans = answer.get("answer_spans", []) or []
    passages: list[str] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        passage = pick_first(span, "passage", "title", "page")
        if passage is None or not str(passage).strip():
            continue
        raw_passage = str(passage).strip()
        if raw_passage.lower() == "main":
            if source_title:
                passages.append(source_title)
            continue
        passages.append(raw_passage)
    return passages


def _extract_iirc_answer_text(answer: Any) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, str):
        return _string_or_none(answer)
    if isinstance(answer, Mapping):
        direct = pick_first(answer, "text", "value")
        if direct is not None:
            return _string_or_none(direct)
        spans = answer.get("answer_spans", []) or []
        if spans:
            texts = [str(span.get("text")).strip() for span in spans if isinstance(span, Mapping) and str(span.get("text", "")).strip()]
            if texts:
                return " ".join(texts)
        answer_type = pick_first(answer, "type")
        if answer_type is not None:
            return _string_or_none(answer_type)
    return _string_or_none(answer)
