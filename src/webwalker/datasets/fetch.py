from __future__ import annotations

import json
import logging
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from webwalker.logging import copy_stream_with_progress

TWOWIKI_QUESTIONS_URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
TWOWIKI_GRAPH_URL = "https://www.dropbox.com/s/wlhw26kik59wbh8/para_with_hyperlink.zip?dl=1"
TWOWIKI_GRAPH_BASENAME = "para_with_hyperlink.jsonl"
TWOWIKI_SPLITS = ("train", "dev", "test")

DownloadFn = Callable[[str, Path], None]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TwoWikiDatasetLayout:
    output_dir: Path
    question_paths: dict[str, Path] = field(default_factory=dict)
    graph_path: Path | None = None
    archive_paths: dict[str, Path] = field(default_factory=dict)
    source: str = "remote"


def fetch_2wiki_dataset(
    output_dir: str | Path,
    *,
    split: str = "dev",
    include_questions: bool = True,
    include_graph: bool = False,
    keep_archives: bool = False,
    overwrite: bool = False,
    questions_url: str | None = None,
    graph_url: str | None = None,
    downloader: DownloadFn = None,
) -> TwoWikiDatasetLayout:
    if not include_questions and not include_graph:
        raise ValueError("At least one of include_questions or include_graph must be enabled.")

    logger.info(
        "Preparing 2Wiki dataset under %s (split=%s, questions=%s, graph=%s, keep_archives=%s, overwrite=%s)",
        output_dir,
        split,
        include_questions,
        include_graph,
        keep_archives,
        overwrite,
    )
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    download_impl = downloader or _download_url
    selected_splits = _resolve_splits(split)
    resolved_questions_url = questions_url or TWOWIKI_QUESTIONS_URL
    resolved_graph_url = graph_url or TWOWIKI_GRAPH_URL
    layout = TwoWikiDatasetLayout(output_dir=resolved_output)

    if include_questions:
        questions_dir = resolved_output / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)
        required = {name: questions_dir / f"{name}.json" for name in selected_splits}
        missing = [name for name, path in required.items() if overwrite or not path.exists()]
        if missing:
            logger.info("Fetching 2Wiki questions for splits: %s", ", ".join(missing))
            archive_path = _prepare_archive(
                resolved_output,
                archive_name="2wiki-questions.zip",
                source_url=resolved_questions_url,
                keep_archives=keep_archives,
                overwrite=overwrite,
                downloader=download_impl,
            )
            if keep_archives:
                layout.archive_paths["questions"] = archive_path
            for name, destination in required.items():
                _extract_zip_member(
                    archive_path,
                    basename=f"{name}.json",
                    destination=destination,
                    overwrite=overwrite,
                )
            _cleanup_archive(archive_path, keep_archives=keep_archives)
        layout.question_paths = required

    if include_graph:
        graph_dir = resolved_output / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        destination = graph_dir / TWOWIKI_GRAPH_BASENAME
        if overwrite or not destination.exists():
            logger.info("Fetching shared 2Wiki graph archive into %s", destination)
            archive_path = _prepare_archive(
                resolved_output,
                archive_name="2wiki-graph.zip",
                source_url=resolved_graph_url,
                keep_archives=keep_archives,
                overwrite=overwrite,
                downloader=download_impl,
            )
            if keep_archives:
                layout.archive_paths["graph"] = archive_path
            _extract_zip_member(
                archive_path,
                basename=TWOWIKI_GRAPH_BASENAME,
                destination=destination,
                overwrite=overwrite,
            )
            _cleanup_archive(archive_path, keep_archives=keep_archives)
        layout.graph_path = destination

    logger.info(
        "Prepared 2Wiki dataset under %s (question_files=%s, graph=%s)",
        resolved_output,
        len(layout.question_paths),
        layout.graph_path is not None,
    )
    return layout


def write_2wiki_sample_dataset(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> TwoWikiDatasetLayout:
    logger.info("Writing bundled 2Wiki sample dataset into %s", output_dir)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    questions_dir = resolved_output / "questions"
    graph_dir = resolved_output / "graph"
    questions_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    questions_path = questions_dir / "dev.json"
    graph_path = graph_dir / TWOWIKI_GRAPH_BASENAME

    if overwrite or not questions_path.exists():
        questions_path.write_text(
            json.dumps(_sample_questions(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if overwrite or not graph_path.exists():
        graph_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in _sample_graph_records()) + "\n",
            encoding="utf-8",
        )

    return TwoWikiDatasetLayout(
        output_dir=resolved_output,
        question_paths={"dev": questions_path},
        graph_path=graph_path,
        source="sample",
    )


def _prepare_archive(
    output_dir: Path,
    *,
    archive_name: str,
    source_url: str,
    keep_archives: bool,
    overwrite: bool,
    downloader: DownloadFn,
) -> Path:
    if keep_archives:
        archive_dir = output_dir / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / archive_name
        if overwrite or not archive_path.exists():
            logger.info("Downloading archive %s from %s", archive_name, source_url)
            downloader(source_url, archive_path)
        return archive_path

    with tempfile.NamedTemporaryFile(prefix="webwalker-", suffix=f"-{archive_name}", delete=False) as handle:
        archive_path = Path(handle.name)
    logger.info("Downloading temporary archive %s from %s", archive_name, source_url)
    downloader(source_url, archive_path)
    return archive_path


def _cleanup_archive(archive_path: Path, *, keep_archives: bool) -> None:
    if keep_archives:
        return
    archive_path.unlink(missing_ok=True)


def _extract_zip_member(
    archive_path: Path,
    *,
    basename: str,
    destination: Path,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        return

    with zipfile.ZipFile(archive_path) as archive:
        member_name = _find_member_by_basename(archive.namelist(), basename)
        if member_name is None:
            raise FileNotFoundError(f"Archive {archive_path} does not contain {basename}")
        info = archive.getinfo(member_name)
        logger.info("Extracting %s from %s to %s", basename, archive_path, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member_name) as src, destination.open("wb") as dst:
            copy_stream_with_progress(
                src,
                dst,
                description=f"extract {basename}",
                total=info.file_size,
            )


def _find_member_by_basename(names: list[str], basename: str) -> str | None:
    for name in names:
        if Path(name).name == basename:
            return name
    return None


def _resolve_splits(split: str) -> tuple[str, ...]:
    normalized = split.strip().lower()
    if normalized == "all":
        return TWOWIKI_SPLITS
    if normalized not in TWOWIKI_SPLITS:
        valid = ", ".join([*TWOWIKI_SPLITS, "all"])
        raise ValueError(f"Unsupported 2Wiki split '{split}'. Expected one of: {valid}")
    return (normalized,)


def _download_url(source_url: str, destination: Path) -> None:
    normalized_url = _normalize_dropbox_url(source_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(normalized_url)
    request_or_url: str | urllib.request.Request
    expected_bytes: int | None = None
    if parsed.scheme == "file":
        request_or_url = normalized_url
        file_path = Path(parsed.path)
        if file_path.exists():
            expected_bytes = file_path.stat().st_size
    else:
        request_or_url = urllib.request.Request(
            normalized_url,
            headers={"User-Agent": "webwalker/0.1 (+https://github.com/Acture/webwalker)"},
        )
    with urllib.request.urlopen(request_or_url) as response, destination.open("wb") as handle:
        content_length = response.headers.get("Content-Length")
        if content_length:
            expected_bytes = int(content_length)
        logger.info("Downloading %s to %s", normalized_url, destination)
        copy_stream_with_progress(
            response,
            handle,
            description=f"download {destination.name}",
            total=expected_bytes,
        )


def _normalize_dropbox_url(source_url: str) -> str:
    parsed = urllib.parse.urlparse(source_url)
    if "dropbox.com" not in parsed.netloc:
        return source_url

    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query["dl"] = ["1"]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def _sample_graph_records() -> list[dict[str, object]]:
    return [
        {
            "id": "100",
            "title": "Moon Launch Program",
            "sentences": [
                "Moon Launch Program uses Cape Canaveral as its launch site.",
                "The program was directed by Alice Johnson.",
            ],
            "mentions": [
                {
                    "id": 0,
                    "start": 25,
                    "end": 40,
                    "ref_url": "Cape_Canaveral",
                    "ref_ids": ["200"],
                    "sent_idx": 0,
                },
                {
                    "id": 1,
                    "start": 28,
                    "end": 41,
                    "ref_url": "Alice_Johnson",
                    "ref_ids": ["300"],
                    "sent_idx": 1,
                },
            ],
        },
        {
            "id": "200",
            "title": "Cape Canaveral",
            "sentences": ["Cape Canaveral is a city in Florida."],
            "mentions": [
                {
                    "id": 0,
                    "start": 29,
                    "end": 36,
                    "ref_url": "Florida",
                    "ref_ids": ["400"],
                    "sent_idx": 0,
                },
            ],
        },
        {
            "id": "300",
            "title": "Alice Johnson",
            "sentences": ["Alice Johnson directed the Moon Launch Program in 1969."],
            "mentions": [],
        },
        {
            "id": "400",
            "title": "Florida",
            "sentences": ["Florida is a state in the southeastern United States."],
            "mentions": [],
        },
    ]


def _sample_questions() -> list[dict[str, object]]:
    return [
        {
            "_id": "q1",
            "question": "Which city hosts the launch site?",
            "answer": "Cape Canaveral",
            "supporting_facts": [
                ["Moon Launch Program", 0],
                ["Cape Canaveral", 0],
            ],
        },
        {
            "_id": "q2",
            "question": "Who directed the Moon Launch Program?",
            "answer": "Alice Johnson",
            "supporting_facts": [
                ["Moon Launch Program", 1],
                ["Alice Johnson", 0],
            ],
        },
    ]
