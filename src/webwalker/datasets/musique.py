from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from webwalker.datasets.common import BaseDatasetAdapter, dedupe_strings, load_json_records, pick_first
from webwalker.eval import EvaluationCase
from webwalker.graph import LinkContextGraph


def load_musique_graph(graph_records_path: str | Path) -> LinkContextGraph:
    records = load_json_records(graph_records_path)
    normalized = [_normalize_musique_graph_record(record) for record in records]
    return LinkContextGraph.from_normalized_records(normalized, dataset_name="musique")


def load_musique_questions(
    questions_path: str | Path,
    *,
    limit: int | None = None,
) -> list[EvaluationCase]:
    records = load_json_records(questions_path)
    if limit is not None:
        records = records[:limit]

    cases: list[EvaluationCase] = []
    for record in records:
        support_nodes = dedupe_strings(
            _coerce_node_refs(
                pick_first(
                    record,
                    "gold_support_nodes",
                    "supporting_docs",
                    "supporting_pages",
                    "paragraph_support_idx",
                    "paragraphs",
                )
            )
        )
        start_nodes = dedupe_strings(
            _coerce_node_refs(pick_first(record, "gold_start_nodes", "start_nodes")) or support_nodes
        )
        raw_path_nodes = _coerce_node_refs(
            pick_first(record, "gold_path_nodes", "path_nodes", "reasoning_path", "question_decomposition")
        )
        path_nodes = dedupe_strings(raw_path_nodes) if raw_path_nodes else None
        cases.append(
            EvaluationCase(
                case_id=str(pick_first(record, "case_id", "id", "qid", "question_id") or ""),
                query=str(pick_first(record, "query", "question") or ""),
                expected_answer=_string_or_none(pick_first(record, "expected_answer", "answer", "answer_alias")),
                dataset_name="musique",
                gold_support_nodes=support_nodes,
                gold_start_nodes=start_nodes,
                gold_path_nodes=path_nodes,
            )
        )
    return cases


class MuSiQueAdapter(BaseDatasetAdapter):
    dataset_name = "musique"

    def load_graph(self, graph_source: str | Path) -> LinkContextGraph:
        return load_musique_graph(graph_source)

    def load_cases(
        self,
        questions_source: str | Path,
        *,
        limit: int | None = None,
    ) -> list[EvaluationCase]:
        return load_musique_questions(questions_source, limit=limit)


def _normalize_musique_graph_record(record: Mapping[str, Any]) -> dict[str, Any]:
    links = []
    for link in pick_first(record, "links", "mentions", "outbound_links") or ():
        if isinstance(link, str):
            target = _normalize_title(link)
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
        target = _normalize_title(pick_first(link, "target", "target_id", "title", "page", "ref_url"))
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
    metadata.setdefault("dataset", "musique")
    node_id = _normalize_title(pick_first(record, "node_id", "id", "title", "page"))
    return {
        "node_id": node_id,
        "title": str(pick_first(record, "title", "node_id", "id", "page") or node_id),
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
        normalized = _normalize_title(value)
        return [normalized] if normalized else []
    if isinstance(value, Mapping):
        ref = pick_first(value, "node_id", "title", "page", "doc", "id")
        normalized = _normalize_title(ref)
        return [normalized] if normalized else []
    refs: list[str] = []
    if isinstance(value, Iterable):
        for item in value:
            if isinstance(item, str):
                normalized = _normalize_title(item)
                if normalized:
                    refs.append(normalized)
                continue
            if isinstance(item, Mapping):
                ref = pick_first(item, "node_id", "title", "page", "doc", "id")
                normalized = _normalize_title(ref)
                if normalized:
                    refs.append(normalized)
    return refs


def _normalize_title(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("_", " ").strip()


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
