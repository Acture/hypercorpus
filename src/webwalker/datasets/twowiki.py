from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from webwalker.eval import EvaluationCase
from webwalker.graph import LinkContextGraph


def load_2wiki_graph(graph_records_path: str | Path) -> LinkContextGraph:
	records_path = Path(graph_records_path)
	raw_records = list(_iter_jsonl(records_path))
	id_to_title = {
		str(record.get("id")): _normalize_title(record.get("title", ""))
		for record in raw_records
	}
	normalized_records: list[dict[str, Any]] = []

	for record in raw_records:
		title = _normalize_title(record.get("title", ""))
		mentions = []
		for mention in record.get("mentions", ()) or ():
			mentions.append(
				{
					**mention,
					"ref_url": _normalize_title(mention.get("ref_url", "")),
					"ref_ids": [
						id_to_title.get(str(ref_id), _normalize_title(ref_id))
						for ref_id in (mention.get("ref_ids") or ())
					],
				}
			)
		normalized_records.append(
			{
				"node_id": title,
				"title": title,
				"sentences": [str(sentence) for sentence in record.get("sentences", ()) or ()],
				"mentions": mentions,
			}
		)

	return LinkContextGraph.from_2wikimultihop_records(
		normalized_records,
		id_field="node_id",
		title_field="title",
		sentences_field="sentences",
		mentions_field="mentions",
	)


def load_2wiki_questions(
	questions_path: str | Path,
	*,
	limit: int | None = None,
) -> list[EvaluationCase]:
	path = Path(questions_path)
	records = json.loads(path.read_text(encoding="utf-8"))
	if limit is not None:
		records = records[:limit]

	cases: list[EvaluationCase] = []
	for record in records:
		gold_support_nodes = _dedupe(
			[
				_normalize_title(title)
				for title, _sent_idx in record.get("supporting_facts", ()) or ()
			]
		)
		cases.append(
			EvaluationCase(
				case_id=str(record.get("_id") or record.get("id")),
				query=str(record.get("question", "")),
				expected_answer=record.get("answer"),
				dataset_name="2wikimultihop",
				gold_support_nodes=gold_support_nodes,
				gold_start_nodes=list(gold_support_nodes),
			)
		)

	return cases


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def _normalize_title(value: Any) -> str:
	return str(value).replace("_", " ").strip()


def _dedupe(values: list[str]) -> list[str]:
	return list(dict.fromkeys(values))
