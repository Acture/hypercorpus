from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from webwalker.datasets import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MIN_FREE_GIB,
    TWOWIKI_GRAPH_URL,
    TWOWIKI_QUESTIONS_URL,
    TWOWIKI_SPLITS,
    ensure_min_free_space,
    estimate_prepare_bytes,
    fetch_2wiki_dataset,
    inspect_2wiki_store,
    prepare_2wiki_store,
    probe_source_size,
    write_2wiki_sample_dataset,
)

datasets_app = typer.Typer(
    name="webwalker datasets",
    help="dataset fetchers and store utilities",
    add_completion=False,
)


@datasets_app.command("fetch-2wiki")
def fetch_2wiki(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for extracted files"),
    split: str = typer.Option("dev", "--split", help=f"Question split to fetch: {', '.join([*TWOWIKI_SPLITS, 'all'])}"),
    questions: bool = typer.Option(True, "--questions/--no-questions", help="Fetch question JSON files"),
    graph: bool = typer.Option(
        False,
        "--graph/--no-graph",
        help="Fetch para_with_hyperlink graph records. This is the large shared artifact.",
    ),
    keep_archives: bool = typer.Option(
        False,
        "--keep-archives/--no-keep-archives",
        help="Keep downloaded zip archives under output-dir/archives",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing extracted files"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
    yes: bool = typer.Option(False, "--yes", help="Skip interactive confirmation before downloading the large graph"),
) -> None:
    console = Console()
    estimated_bytes = 0
    if questions:
        estimated_bytes += probe_source_size(TWOWIKI_QUESTIONS_URL) or 0
    if graph:
        estimated_bytes += probe_source_size(TWOWIKI_GRAPH_URL) or 0
    ensure_min_free_space(output_dir, min_free_gib=min_free_gib, expected_new_bytes=estimated_bytes or None)
    if graph and not yes:
        graph_size = probe_source_size(TWOWIKI_GRAPH_URL)
        size_text = _format_size(graph_size) if graph_size is not None else "unknown size"
        if not typer.confirm(f"Download the shared 2Wiki graph archive ({size_text}) into {output_dir}?"):
            raise typer.Abort()

    try:
        layout = fetch_2wiki_dataset(
            output_dir,
            split=split,
            include_questions=questions,
            include_graph=graph,
            keep_archives=keep_archives,
            overwrite=overwrite,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_dataset_layout(console, layout)
    if layout.graph_path is None:
        console.print("graph not downloaded: add --graph when you are ready for the large shared file.")
    console.print("for a zero-download smoke dataset: webwalker-cli datasets write-2wiki-sample --output-dir /tmp/webwalker-2wiki-sample")


@datasets_app.command("write-2wiki-sample")
def write_2wiki_sample(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for the sample dataset"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing sample files"),
) -> None:
    console = Console()
    layout = write_2wiki_sample_dataset(output_dir, overwrite=overwrite)
    _print_dataset_layout(console, layout)
    console.print("sample source: synthetic smoke dataset bundled with the repo")


@datasets_app.command("prepare-2wiki-store")
def prepare_2wiki_store_cli(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for the prepared sharded store"),
    questions_source: str | None = typer.Option(None, "--questions-source", help="Local path or URL to 2Wiki questions json/zip/dir"),
    graph_source: str | None = typer.Option(None, "--graph-source", help="Local path or URL to 2Wiki graph jsonl/zip"),
    keep_raw: bool = typer.Option(False, "--keep-raw/--no-keep-raw", help="Keep downloaded raw graph artifacts under the store root"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing prepared store"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
) -> None:
    console = Console()
    graph_size = probe_source_size(graph_source or TWOWIKI_GRAPH_URL)
    ensure_min_free_space(output_dir, min_free_gib=min_free_gib, expected_new_bytes=estimate_prepare_bytes(graph_size))
    prepared = prepare_2wiki_store(
        output_dir,
        questions_source=questions_source,
        graph_source=graph_source,
        keep_raw=keep_raw,
        overwrite=overwrite,
        min_free_gib=min_free_gib,
    )
    console.print(f"store root -> {prepared.root}")
    console.print(f"manifest.json -> {prepared.manifest_path}")
    console.print(f"catalog.sqlite -> {prepared.catalog_path}")
    console.print(f"shards -> {prepared.manifest.shard_count}")
    for split_name, path in sorted(prepared.questions_paths.items()):
        console.print(f"questions[{split_name}] -> {path}")
    console.print("run:")
    console.print(
        "uv run webwalker-cli experiments run-2wiki-store "
        f"--store {prepared.root} "
        "--exp-name pilot-2wiki "
        "--split dev "
        "--chunk-size 100 "
        "--chunk-index 0"
    )


@datasets_app.command("inspect-2wiki-store")
def inspect_2wiki_store_cli(
    store: str | None = typer.Option(None, "--store", help="Optional prepared store path or s3:// URI"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir", file_okay=False, help="Local cache directory for prepared stores"),
    raw_root: Path | None = typer.Option(None, "--raw-root", file_okay=False, help="Optional raw 2Wiki root to inspect"),
) -> None:
    console = Console()
    inspection = inspect_2wiki_store(store_uri=store, cache_dir=cache_dir, raw_root=raw_root)
    console.print(f"cache_dir -> {inspection.cache_dir} ({_format_size(inspection.cache_size_bytes)})")
    console.print(f"free_space -> {_format_size(inspection.free_space_bytes)}")
    console.print(f"recommended_action -> {inspection.recommended_action}")
    if inspection.raw_questions_path is not None:
        console.print(f"raw_questions -> {inspection.raw_questions_path}")
    if inspection.raw_graph_path is not None:
        console.print(f"raw_graph -> {inspection.raw_graph_path} ({_format_size(inspection.raw_graph_size_bytes)})")
    if inspection.store_manifest_path is not None:
        console.print(f"store_manifest -> {inspection.store_manifest_path}")
    if inspection.remote_questions_size_bytes is not None:
        console.print(f"remote_questions_zip -> {_format_size(inspection.remote_questions_size_bytes)}")
    if inspection.remote_graph_size_bytes is not None:
        console.print(f"remote_graph_zip -> {_format_size(inspection.remote_graph_size_bytes)}")


def _print_dataset_layout(console: Console, layout) -> None:
    console.print(f"dataset root -> {layout.output_dir}")
    for split, path in sorted(layout.question_paths.items()):
        console.print(f"questions[{split}] -> {path} ({_format_path_size(path)})")
    if layout.graph_path is not None:
        console.print(f"graph -> {layout.graph_path} ({_format_path_size(layout.graph_path)})")
    if layout.archive_paths:
        for name, path in sorted(layout.archive_paths.items()):
            console.print(f"archive[{name}] -> {path} ({_format_path_size(path)})")

    default_question = layout.question_paths.get("dev")
    if default_question is None and layout.question_paths:
        default_question = next(iter(layout.question_paths.values()))
    if default_question is not None and layout.graph_path is not None:
        console.print("run:")
        console.print(
            "uv run webwalker-cli experiments run-2wiki "
            f"--questions {default_question} "
            f"--graph-records {layout.graph_path} "
            "--output /tmp/webwalker-2wiki-run"
        )


def _format_path_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    return _format_size(path.stat().st_size)


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"
