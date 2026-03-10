from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from webwalker.eval import EvaluationCase
from webwalker.graph import LinkContextGraph


@dataclass(slots=True)
class PreparedDataset:
    dataset_name: str
    graph: LinkContextGraph
    cases: list[EvaluationCase]
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    dataset_name: str

    def load_graph(self, graph_source: str | Path) -> LinkContextGraph:
        ...

    def load_cases(
        self,
        questions_source: str | Path,
        *,
        limit: int | None = None,
    ) -> list[EvaluationCase]:
        ...

    def load_dataset(
        self,
        *,
        graph_source: str | Path,
        questions_source: str | Path,
        limit: int | None = None,
    ) -> PreparedDataset:
        ...


class BaseDatasetAdapter:
    dataset_name = "dataset"

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
    raise ValueError(f"Unsupported JSON payload in {resolved}: expected a list of records")


def dedupe_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def pick_first(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None
