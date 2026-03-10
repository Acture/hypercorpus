from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from webwalker.datasets.common import BaseDatasetAdapter, dedupe_strings, load_json_records, pick_first
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
