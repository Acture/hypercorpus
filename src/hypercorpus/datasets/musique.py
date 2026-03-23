from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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


def convert_musique_raw_dataset(
    raw_source: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    source_manifest_path: str | Path | None = None,
) -> NormalizedDatasetLayout:
    question_files = _resolve_question_files(raw_source)
    question_splits: dict[str, list[EvaluationCase]] = {}
    graph_records: list[dict[str, Any]] = []
    for split_name, questions_path in question_files.items():
        records = load_json_records(questions_path)
        split_cases: list[EvaluationCase] = []
        for record in records:
            split_cases.append(_convert_musique_question_record(record))
            graph_records.extend(_convert_musique_graph_records(record))
        question_splits[split_name] = split_cases
    return write_normalized_dataset(
        output_dir,
        dataset_name="musique",
        question_splits=question_splits,
        graph_records=graph_records,
        source_manifest_path=source_manifest_path,
        overwrite=overwrite,
    )


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


def _resolve_question_files(raw_source: str | Path) -> dict[str, Path]:
    resolved = Path(raw_source)
    if resolved.is_file():
        return {_infer_split_name(resolved): resolved}
    candidates = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    if not candidates:
        raise ValueError(f"No MuSiQue raw question files found under {resolved}")
    return {_infer_split_name(path): path for path in candidates}


def _infer_split_name(path: Path) -> str:
    stem = path.stem.lower()
    for split_name in ("train", "dev", "test"):
        if split_name in stem:
            return split_name
    return "dev"


def _convert_musique_question_record(record: Mapping[str, Any]) -> EvaluationCase:
    case_id = str(pick_first(record, "case_id", "id", "qid", "question_id") or "")
    node_ids_by_index, node_ids_by_title = _musique_node_maps(record, case_id=case_id)
    support_nodes = dedupe_strings(
        _support_node_ids_from_record(record, node_ids_by_index=node_ids_by_index, node_ids_by_title=node_ids_by_title)
    )
    path_nodes = dedupe_strings(_path_node_ids_from_record(record, node_ids_by_index=node_ids_by_index, node_ids_by_title=node_ids_by_title))
    start_nodes = path_nodes[:1] or support_nodes[:1] or list(node_ids_by_index.values())[:1]
    return EvaluationCase(
        case_id=case_id,
        query=str(pick_first(record, "query", "question") or ""),
        expected_answer=_string_or_none(pick_first(record, "expected_answer", "answer", "answer_alias")),
        dataset_name="musique",
        gold_support_nodes=support_nodes,
        gold_start_nodes=start_nodes,
        gold_path_nodes=path_nodes or None,
    )


def _convert_musique_graph_records(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    case_id = str(pick_first(record, "case_id", "id", "qid", "question_id") or "")
    graph_records: list[dict[str, Any]] = []
    for index, paragraph in enumerate(_raw_musique_paragraphs(record)):
        title = _normalize_title(pick_first(paragraph, "title", "page", "node_id", "id")) or f"paragraph-{index}"
        text = str(pick_first(paragraph, "paragraph_text", "text", "context", "paragraph") or "").strip()
        graph_records.append(
            {
                "node_id": _musique_node_id(case_id, index=index, title=title),
                "title": title,
                "sentences": [text] if text else [],
                "links": _normalize_paragraph_links(paragraph),
                "metadata": {
                    "dataset": "musique",
                    "case_id": case_id,
                    "paragraph_index": index,
                },
            }
        )
    return graph_records


def _raw_musique_paragraphs(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    paragraphs = pick_first(record, "paragraphs", "contexts", "context_paragraphs")
    if isinstance(paragraphs, list):
        return [item for item in paragraphs if isinstance(item, Mapping)]
    return []


def _musique_node_maps(record: Mapping[str, Any], *, case_id: str) -> tuple[dict[int, str], dict[str, list[str]]]:
    node_ids_by_index: dict[int, str] = {}
    node_ids_by_title: dict[str, list[str]] = {}
    for index, paragraph in enumerate(_raw_musique_paragraphs(record)):
        title = _normalize_title(pick_first(paragraph, "title", "page", "node_id", "id")) or f"paragraph-{index}"
        node_id = _musique_node_id(case_id, index=index, title=title)
        node_ids_by_index[index] = node_id
        node_ids_by_title.setdefault(title, []).append(node_id)
    return node_ids_by_index, node_ids_by_title


def _support_node_ids_from_record(
    record: Mapping[str, Any],
    *,
    node_ids_by_index: dict[int, str],
    node_ids_by_title: dict[str, list[str]],
) -> list[str]:
    support_indices = pick_first(record, "paragraph_support_idx", "supporting_paragraph_indices")
    if isinstance(support_indices, list):
        return [node_ids_by_index[index] for index in support_indices if index in node_ids_by_index]
    supporting_from_flags = [
        node_ids_by_index[index]
        for index, paragraph in enumerate(_raw_musique_paragraphs(record))
        if paragraph.get("is_supporting") and index in node_ids_by_index
    ]
    if supporting_from_flags:
        return supporting_from_flags
    explicit_titles = _coerce_node_refs(
        pick_first(record, "gold_support_nodes", "supporting_pages", "supporting_docs", "paragraphs")
    )
    return _node_ids_from_titles(explicit_titles, node_ids_by_title=node_ids_by_title)


def _path_node_ids_from_record(
    record: Mapping[str, Any],
    *,
    node_ids_by_index: dict[int, str],
    node_ids_by_title: dict[str, list[str]],
) -> list[str]:
    decomposition = pick_first(record, "question_decomposition", "reasoning_path", "gold_path_nodes")
    resolved: list[str] = []
    if isinstance(decomposition, list):
        for item in decomposition:
            if isinstance(item, Mapping):
                paragraph_idx = pick_first(item, "paragraph_idx", "paragraph_support_idx", "paragraph_index")
                if isinstance(paragraph_idx, int) and paragraph_idx in node_ids_by_index:
                    resolved.append(node_ids_by_index[paragraph_idx])
                    continue
                titles = _coerce_node_refs(pick_first(item, "title", "page", "node_id", "supporting_title"))
                resolved.extend(_node_ids_from_titles(titles, node_ids_by_title=node_ids_by_title))
                continue
            if isinstance(item, int) and item in node_ids_by_index:
                resolved.append(node_ids_by_index[item])
                continue
            resolved.extend(_node_ids_from_titles(_coerce_node_refs(item), node_ids_by_title=node_ids_by_title))
    return resolved


def _node_ids_from_titles(titles: list[str], *, node_ids_by_title: dict[str, list[str]]) -> list[str]:
    resolved: list[str] = []
    for title in titles:
        resolved.extend(node_ids_by_title.get(_normalize_title(title), []))
    return resolved


def _normalize_paragraph_links(paragraph: Mapping[str, Any]) -> list[dict[str, Any]]:
    links = []
    for link in pick_first(paragraph, "links", "mentions", "outbound_links") or ():
        if isinstance(link, str):
            target = _normalize_title(link)
            if not target:
                continue
            links.append({"target": target, "anchor_text": target, "sentence": "", "sent_idx": 0})
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
    return links


def _musique_node_id(case_id: str, *, index: int, title: str) -> str:
    return f"{case_id}::p{index}::{_normalize_title(title)}"
