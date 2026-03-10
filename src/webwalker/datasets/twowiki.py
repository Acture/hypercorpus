from __future__ import annotations

import gzip
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TextIO

from webwalker.logging import create_progress, should_render_progress
from webwalker.eval import EvaluationCase
from webwalker.graph import LinkContextGraph

TWOWIKI_DEFAULT_ROOT = Path("dataset/2wikimultihop")
TWOWIKI_DEFAULT_QUESTIONS_DIR = TWOWIKI_DEFAULT_ROOT / "data_ids_april7"
TWOWIKI_DEFAULT_GRAPH_PATH = TWOWIKI_DEFAULT_ROOT / "para_with_hyperlink.jsonl"

logger = logging.getLogger(__name__)


def load_2wiki_graph(graph_records_path: str | Path) -> LinkContextGraph:
    records_path = Path(graph_records_path)
    id_to_title = {
        str(record.get("id")): normalize_2wiki_title(record.get("title", ""))
        for record in _iter_records_with_optional_progress(
            records_path,
            description="index 2wiki titles",
        )
    }
    normalized_records = [
        normalize_2wiki_graph_record(record, id_to_title=id_to_title)
        for record in _iter_records_with_optional_progress(
            records_path,
            description="normalize 2wiki graph",
        )
    ]
    logger.info("Building LinkContextGraph from %s normalized 2Wiki records", len(normalized_records))

    graph = LinkContextGraph.from_2wikimultihop_records(
        normalized_records,
        id_field="node_id",
        title_field="title",
        sentences_field="sentences",
        mentions_field="mentions",
    )
    logger.info("Loaded 2Wiki graph from %s with %s nodes", records_path, len(graph.nodes))
    return graph


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
                normalize_2wiki_title(title)
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


def iter_2wiki_graph_records(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = Path(path)
    suffixes = [suffix.lower() for suffix in resolved.suffixes]
    if suffixes[-2:] == [".jsonl", ".gz"]:
        with gzip.open(resolved, "rt", encoding="utf-8") as handle:
            yield from _iter_jsonl_lines(handle)
        return
    if suffixes and suffixes[-1] == ".jsonl":
        with resolved.open("r", encoding="utf-8") as handle:
            yield from _iter_jsonl_lines(handle)
        return
    if suffixes and suffixes[-1] == ".zip":
        with zipfile.ZipFile(resolved) as archive:
            member_name = _find_zip_member(archive, "para_with_hyperlink.jsonl")
            if member_name is None:
                raise FileNotFoundError(f"Archive {resolved} does not contain para_with_hyperlink.jsonl")
            with archive.open(member_name, "r") as handle:
                text_handle = _binary_to_text(handle)
                yield from _iter_jsonl_lines(text_handle)
        return
    raise ValueError(f"Unsupported 2Wiki graph source: {resolved}")


def normalize_2wiki_graph_record(
    record: Mapping[str, Any],
    *,
    id_to_title: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    title = normalize_2wiki_title(record.get("title", ""))
    normalized_mentions: list[dict[str, Any]] = []
    title_lookup = id_to_title or {}
    for mention in record.get("mentions", ()) or ():
        normalized_mentions.append(
            {
                **mention,
                "ref_url": normalize_2wiki_title(mention.get("ref_url", "")),
                "ref_ids": [
                    title_lookup.get(str(ref_id), normalize_2wiki_title(ref_id))
                    for ref_id in (mention.get("ref_ids") or ())
                ],
            }
        )
    return {
        "node_id": title,
        "title": title,
        "sentences": [str(sentence) for sentence in record.get("sentences", ()) or ()],
        "mentions": normalized_mentions,
    }


def normalize_2wiki_title(value: Any) -> str:
    return str(value).replace("_", " ").strip()


def iter_2wiki_question_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_2wiki_raw_paths(root: str | Path = TWOWIKI_DEFAULT_ROOT) -> tuple[Path, Path]:
    resolved = Path(root)
    return resolved / "data_ids_april7" / "dev.json", resolved / "para_with_hyperlink.jsonl"


def _iter_jsonl_lines(handle: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _binary_to_text(handle: Any) -> TextIO:
    import io

    return io.TextIOWrapper(handle, encoding="utf-8")


def _find_zip_member(archive: zipfile.ZipFile, basename: str) -> str | None:
    for name in archive.namelist():
        if Path(name).name == basename:
            return name
    return None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _iter_records_with_optional_progress(
    path: Path,
    *,
    description: str,
    update_every: int = 10_000,
) -> Iterator[dict[str, Any]]:
    logger.info("%s from %s", description.capitalize(), path)
    if not should_render_progress():
        count = 0
        for count, record in enumerate(iter_2wiki_graph_records(path), start=1):
            yield record
        logger.info("%s complete (%s records)", description.capitalize(), count)
        return

    count = 0
    with create_progress(transient=True) as progress:
        task_id = progress.add_task(description, total=None)
        for count, record in enumerate(iter_2wiki_graph_records(path), start=1):
            if count == 1 or count % update_every == 0:
                progress.update(task_id, description=f"{description} [{count:,} records]")
            yield record
        progress.update(task_id, description=f"{description} [{count:,} records]")
    logger.info("%s complete (%s records)", description.capitalize(), count)
