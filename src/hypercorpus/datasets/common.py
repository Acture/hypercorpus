from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from hypercorpus.eval import EvaluationCase, QuestionType
from hypercorpus.graph import LinkContextGraph


@dataclass(slots=True)
class PreparedDataset:
	dataset_name: str
	graph: LinkContextGraph
	cases: list[EvaluationCase]
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedDatasetLayout:
	output_dir: Path
	dataset_name: str
	variant: str | None = None
	question_paths: dict[str, Path] = field(default_factory=dict)
	graph_path: Path | None = None
	manifest_path: Path | None = None


class DatasetAdapter(Protocol):
	dataset_name: str

	def load_graph(self, graph_source: str | Path) -> LinkContextGraph: ...

	def load_cases(
		self,
		questions_source: str | Path,
		*,
		limit: int | None = None,
	) -> list[EvaluationCase]: ...

	def load_dataset(
		self,
		*,
		graph_source: str | Path,
		questions_source: str | Path,
		limit: int | None = None,
	) -> PreparedDataset: ...


class BaseDatasetAdapter:
	dataset_name = "dataset"

	def load_graph(self, graph_source: str | Path) -> LinkContextGraph:
		raise NotImplementedError

	def load_cases(
		self,
		questions_source: str | Path,
		*,
		limit: int | None = None,
	) -> list[EvaluationCase]:
		raise NotImplementedError

	def load_dataset(
		self,
		*,
		graph_source: str | Path,
		questions_source: str | Path,
		limit: int | None = None,
	) -> PreparedDataset:
		return PreparedDataset(
			dataset_name=self.dataset_name,
			graph=self.load_graph(graph_source),
			cases=self.load_cases(questions_source, limit=limit),
		)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
	resolved = Path(path)
	if resolved.suffix.lower() == ".jsonl":
		records: list[dict[str, Any]] = []
		with resolved.open("r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if line:
					records.append(json.loads(line))
		return records

	payload = json.loads(resolved.read_text(encoding="utf-8"))
	if isinstance(payload, list):
		return [dict(item) for item in payload]
	if isinstance(payload, dict):
		for key in ("records", "documents", "pages", "questions", "items"):
			value = payload.get(key)
			if isinstance(value, list):
				return [dict(item) for item in value]
	raise ValueError(
		f"Unsupported JSON payload in {resolved}: expected a list of records"
	)


def dedupe_strings(values: Iterable[str]) -> list[str]:
	return list(dict.fromkeys(value for value in values if value))


def pick_first(record: Mapping[str, Any], *keys: str) -> Any:
	for key in keys:
		if key in record and record[key] is not None:
			return record[key]
	return None


def coerce_question_type(value: Any) -> QuestionType | None:
	if value is None:
		return None
	normalized = str(value).strip().lower()
	if normalized == "bridge":
		return "bridge"
	if normalized == "comparison":
		return "comparison"
	if normalized == "unknown":
		return "unknown"
	return None


def evaluation_case_to_record(case: EvaluationCase) -> dict[str, Any]:
	return {
		"case_id": case.case_id,
		"query": case.query,
		"expected_answer": case.expected_answer,
		"dataset_name": case.dataset_name,
		"gold_support_nodes": list(case.gold_support_nodes),
		"gold_start_nodes": list(case.gold_start_nodes)
		if case.gold_start_nodes is not None
		else None,
		"gold_path_nodes": list(case.gold_path_nodes)
		if case.gold_path_nodes is not None
		else None,
		"question_type": case.question_type,
	}


def write_normalized_dataset(
	output_dir: str | Path,
	*,
	dataset_name: str,
	question_splits: Mapping[str, Sequence[EvaluationCase]],
	graph_records: Iterable[Mapping[str, Any]],
	variant: str | None = None,
	source_manifest_path: str | Path | None = None,
	overwrite: bool = False,
) -> NormalizedDatasetLayout:
	resolved_output = Path(output_dir)
	questions_dir = resolved_output / "questions"
	graph_dir = resolved_output / "graph"
	manifest_path = resolved_output / "conversion-manifest.json"
	graph_path = graph_dir / "normalized.jsonl"

	if resolved_output.exists() and overwrite:
		for path in (manifest_path, graph_path):
			path.unlink(missing_ok=True)
		if questions_dir.exists():
			for item in questions_dir.glob("*.json"):
				item.unlink(missing_ok=True)
	questions_dir.mkdir(parents=True, exist_ok=True)
	graph_dir.mkdir(parents=True, exist_ok=True)

	question_paths: dict[str, Path] = {}
	for split_name, cases in question_splits.items():
		destination = questions_dir / f"{split_name}.json"
		payload = [evaluation_case_to_record(case) for case in cases]
		destination.write_text(
			json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
		)
		question_paths[split_name] = destination

	graph_records_list = [dict(record) for record in graph_records]
	graph_path.write_text(
		"\n".join(
			json.dumps(record, ensure_ascii=False) for record in graph_records_list
		)
		+ ("\n" if graph_records_list else ""),
		encoding="utf-8",
	)
	manifest_payload = {
		"dataset_name": dataset_name,
		"variant": variant,
		"questions": {
			split: str(path.relative_to(resolved_output))
			for split, path in sorted(question_paths.items())
		},
		"graph": str(graph_path.relative_to(resolved_output)),
		"source_manifest_path": str(Path(source_manifest_path))
		if source_manifest_path is not None
		else None,
		"graph_record_count": len(graph_records_list),
		"question_count": sum(len(cases) for cases in question_splits.values()),
	}
	manifest_path.write_text(
		json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
	)
	return NormalizedDatasetLayout(
		output_dir=resolved_output,
		dataset_name=dataset_name,
		variant=variant,
		question_paths=question_paths,
		graph_path=graph_path,
		manifest_path=manifest_path,
	)
