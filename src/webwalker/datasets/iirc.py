from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

from webwalker.datasets.common import (
    BaseDatasetAdapter,
    NormalizedDatasetLayout,
    dedupe_strings,
    load_json_records,
    pick_first,
    write_normalized_dataset,
)
from webwalker.eval import EvaluationCase
from webwalker.graph import LinkContextGraph


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
    graph_records = _convert_iirc_graph_records(question_files=question_files, context_source=context_source)
    question_splits: dict[str, list[EvaluationCase]] = {}
    for split_name, questions_path in question_files.items():
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        question_splits[split_name] = _convert_iirc_questions_payload(payload)
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


def _convert_iirc_questions_payload(payload: Any) -> list[EvaluationCase]:
    if not isinstance(payload, list):
        raise ValueError("IIRC raw question payload must be a list of passage records")
    cases: list[EvaluationCase] = []
    for passage_record in payload:
        if not isinstance(passage_record, Mapping):
            continue
        source_title = str(pick_first(passage_record, "title", "pid", "id") or "").strip()
        for question in passage_record.get("questions", []) or []:
            if not isinstance(question, Mapping):
                continue
            support_nodes = dedupe_strings(
                [source_title]
                + _coerce_node_refs(question.get("question_links"))
                + _coerce_node_refs(_answer_span_passages(question))
                + _coerce_node_refs(question_context_passages(question))
            )
            path_nodes = dedupe_strings([source_title] + _coerce_node_refs(question.get("question_links")))
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
) -> list[dict[str, Any]]:
    graph_records: dict[str, dict[str, Any]] = {}
    for questions_path in question_files.values():
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for record in payload:
            if not isinstance(record, Mapping):
                continue
            normalized = _normalize_iirc_raw_passage_record(record)
            graph_records[normalized["node_id"]] = normalized
    if context_source is not None:
        context_payload = json.loads(context_source.read_text(encoding="utf-8"))
        for record in _iter_iirc_context_records(context_payload):
            normalized = _normalize_iirc_raw_passage_record(record)
            graph_records.setdefault(normalized["node_id"], normalized)
    return list(graph_records.values())


def _normalize_iirc_raw_passage_record(record: Mapping[str, Any]) -> dict[str, Any]:
    title = str(pick_first(record, "title", "pid", "id", "node_id") or "").strip()
    text = pick_first(record, "text", "context", "paragraph")
    sentences = [text] if isinstance(text, str) and text.strip() else _coerce_sentences(record)
    links = []
    for link in record.get("links", []) or []:
        if not isinstance(link, Mapping):
            continue
        target = str(pick_first(link, "target", "title", "page", "ref_url") or "").strip()
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


def _iter_iirc_context_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, Mapping):
                merged = dict(value)
                merged.setdefault("title", str(key))
                yield merged


def question_context_passages(question: Mapping[str, Any]) -> list[str]:
    passages: list[str] = []
    for item in question.get("context", []) or []:
        if not isinstance(item, Mapping):
            continue
        passage = pick_first(item, "passage", "title", "page", "doc")
        if passage is not None and str(passage).strip():
            passages.append(str(passage).strip())
    return passages


def _answer_span_passages(question: Mapping[str, Any]) -> list[str]:
    answer = question.get("answer")
    if not isinstance(answer, Mapping):
        return []
    spans = answer.get("answer_spans", []) or []
    passages: list[str] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        passage = pick_first(span, "passage", "title", "page")
        if passage is not None and str(passage).strip():
            passages.append(str(passage).strip())
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
