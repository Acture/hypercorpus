from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from hypercorpus.datasets.common import (
	BaseDatasetAdapter,
	NormalizedDatasetLayout,
	coerce_question_type,
	dedupe_strings,
	load_json_records,
	pick_first,
	write_normalized_dataset,
)
from hypercorpus.eval import EvaluationCase
from hypercorpus.graph import LinkContextGraph
from hypercorpus.datasets.store import iter_normalized_graph_records

HotpotVariant = Literal["distractor", "fullwiki"]


def load_hotpotqa_graph(
	graph_records_path: str | Path,
	*,
	variant: HotpotVariant = "fullwiki",
) -> LinkContextGraph:
	if variant == "distractor":
		records = load_json_records(graph_records_path)
		normalized = build_hotpotqa_distractor_records(records)
		return LinkContextGraph.from_normalized_records(
			normalized, dataset_name="hotpotqa-distractor"
		)
	records = load_json_records(graph_records_path)
	normalized = [
		_normalize_hotpotqa_graph_record(record, dataset_name="hotpotqa-fullwiki")
		for record in records
	]
	return LinkContextGraph.from_normalized_records(
		normalized, dataset_name="hotpotqa-fullwiki"
	)


def load_hotpotqa_questions(
	questions_path: str | Path,
	*,
	limit: int | None = None,
	variant: HotpotVariant = "fullwiki",
) -> list[EvaluationCase]:
	records = load_json_records(questions_path)
	if limit is not None:
		records = records[:limit]

	cases: list[EvaluationCase] = []
	dataset_name = (
		"hotpotqa-distractor" if variant == "distractor" else "hotpotqa-fullwiki"
	)
	for record in records:
		case_id = str(pick_first(record, "_id", "id", "question_id", "case_id") or "")
		support_nodes = dedupe_strings(
			_support_nodes_from_record(record, case_id=case_id, variant=variant)
		)
		cases.append(
			EvaluationCase(
				case_id=case_id,
				query=str(pick_first(record, "query", "question") or ""),
				expected_answer=_string_or_none(
					pick_first(record, "expected_answer", "answer")
				),
				dataset_name=dataset_name,
				gold_support_nodes=support_nodes,
				gold_start_nodes=list(support_nodes),
				gold_path_nodes=list(support_nodes) if support_nodes else None,
				question_type=coerce_question_type(
					pick_first(record, "question_type", "type")
				),
			)
		)
	return cases


class HotpotQAAdapter(BaseDatasetAdapter):
	def __init__(self, variant: HotpotVariant = "fullwiki"):
		self.variant = variant
		self.dataset_name = (
			"hotpotqa-distractor" if variant == "distractor" else "hotpotqa-fullwiki"
		)

	def load_graph(self, graph_source: str | Path) -> LinkContextGraph:
		return load_hotpotqa_graph(graph_source, variant=self.variant)

	def load_cases(
		self,
		questions_source: str | Path,
		*,
		limit: int | None = None,
	) -> list[EvaluationCase]:
		return load_hotpotqa_questions(
			questions_source, limit=limit, variant=self.variant
		)


def convert_hotpotqa_raw_dataset(
	raw_source: str | Path,
	output_dir: str | Path,
	*,
	variant: HotpotVariant = "distractor",
	graph_source: str | Path | None = None,
	overwrite: bool = False,
	source_manifest_path: str | Path | None = None,
) -> NormalizedDatasetLayout:
	raw_questions = _resolve_question_files(raw_source, variant=variant)
	question_splits: dict[str, list[EvaluationCase]] = {}
	distractor_graph_records: list[dict[str, Any]] = []
	for split_name, questions_path in raw_questions.items():
		records = load_json_records(questions_path)
		question_splits[split_name] = load_hotpotqa_questions(
			questions_path, variant=variant
		)
		if variant == "distractor":
			distractor_graph_records.extend(build_hotpotqa_distractor_records(records))

	if variant == "fullwiki":
		if graph_source is None:
			raise ValueError("HotpotQA fullwiki conversion requires --graph-source")
		graph_records = list(iter_normalized_graph_records(graph_source))
	else:
		graph_records = distractor_graph_records
	return write_normalized_dataset(
		output_dir,
		dataset_name="hotpotqa-distractor"
		if variant == "distractor"
		else "hotpotqa-fullwiki",
		variant=variant,
		question_splits=question_splits,
		graph_records=graph_records,
		source_manifest_path=source_manifest_path,
		overwrite=overwrite,
	)


def build_hotpotqa_distractor_records(
	records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
	normalized: list[dict[str, Any]] = []
	for record in records:
		case_id = str(pick_first(record, "_id", "id", "question_id", "case_id") or "")
		for title, sentences in pick_first(record, "context", "paragraphs") or ():
			node_id = _case_node_id(case_id, title)
			normalized.append(
				{
					"node_id": node_id,
					"title": _normalize_title(title),
					"sentences": [
						str(sentence).strip()
						for sentence in sentences
						if str(sentence).strip()
					],
					"links": [],
					"metadata": {
						"dataset": "hotpotqa-distractor",
						"case_id": case_id,
					},
				}
			)
	return normalized


def build_hotpotqa_distractor_graph_for_case(
	record: Mapping[str, Any],
) -> LinkContextGraph:
	return LinkContextGraph.from_normalized_records(
		build_hotpotqa_distractor_records([record]),
		dataset_name="hotpotqa-distractor",
	)


def _support_nodes_from_record(
	record: Mapping[str, Any],
	*,
	case_id: str,
	variant: HotpotVariant,
) -> list[str]:
	explicit = _coerce_node_refs(
		pick_first(record, "gold_support_nodes", "supporting_pages", "supporting_docs")
	)
	if explicit:
		if variant == "distractor":
			if all("::" in node_id for node_id in explicit):
				return explicit
			return [_case_node_id(case_id, node_id) for node_id in explicit]
		return explicit

	supporting_facts = pick_first(record, "supporting_facts") or ()
	titles = [
		_normalize_title(title)
		for title, _sent_idx in supporting_facts
		if _normalize_title(title)
	]
	if variant == "distractor":
		return [_case_node_id(case_id, title) for title in titles]
	return titles


def _normalize_hotpotqa_graph_record(
	record: Mapping[str, Any], *, dataset_name: str
) -> dict[str, Any]:
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
		target = _normalize_title(
			pick_first(link, "target", "target_id", "title", "page", "ref_url")
		)
		if not target:
			continue
		links.append(
			{
				"target": target,
				"anchor_text": str(
					pick_first(link, "anchor_text", "anchor", "text") or target
				),
				"sentence": str(
					pick_first(link, "sentence", "context", "paragraph") or ""
				),
				"sent_idx": int(pick_first(link, "sent_idx", "sentence_index") or 0),
				"ref_id": _string_or_none(pick_first(link, "ref_id", "target_id")),
				"metadata": dict(link.get("metadata", {}) or {}),
			}
		)

	metadata = dict(record.get("metadata", {}) or {})
	metadata.setdefault("dataset", dataset_name)
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
		return [
			str(sentence).strip() for sentence in raw_sentences if str(sentence).strip()
		]
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


def _case_node_id(case_id: str, title: Any) -> str:
	return f"{case_id}::{_normalize_title(title)}"


def _normalize_title(value: Any) -> str:
	if value is None:
		return ""
	return str(value).replace("_", " ").strip()


def _string_or_none(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _resolve_question_files(
	raw_source: str | Path, *, variant: HotpotVariant
) -> dict[str, Path]:
	resolved = Path(raw_source)
	if resolved.is_file():
		return {_infer_split_name(resolved): resolved}
	candidates = sorted(
		path
		for path in resolved.rglob("*")
		if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
	)
	variant_token = "distractor" if variant == "distractor" else "fullwiki"
	filtered = [
		path
		for path in candidates
		if variant_token in path.stem.lower() or "train_v1.1" in path.stem.lower()
	]
	if filtered:
		candidates = filtered
	if not candidates:
		raise ValueError(f"No HotpotQA raw question files found under {resolved}")
	return {_infer_split_name(path): path for path in candidates}


def _infer_split_name(path: Path) -> str:
	stem = path.stem.lower()
	for split_name in ("train", "dev", "test"):
		if split_name in stem:
			return split_name
	return "dev"
