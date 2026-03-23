from __future__ import annotations

import json
from pathlib import Path
import urllib.parse

import typer
from rich.console import Console

from hypercorpus.datasets import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MIN_FREE_GIB,
    HOTPOTQA_DISTRACTOR_SPLIT_URLS,
    HOTPOTQA_FULLWIKI_SPLIT_URLS,
    IIRC_ARCHIVE_URL,
    MUSIQUE_ANSWERABLE_SPLIT_URLS,
    MUSIQUE_FULL_SPLIT_URLS,
    TWOWIKI_GRAPH_URL,
    TWOWIKI_QUESTIONS_URL,
    TWOWIKI_SPLITS,
    convert_hotpotqa_raw_dataset,
    convert_iirc_raw_dataset,
    convert_musique_raw_dataset,
    ensure_min_free_space,
    estimate_prepare_bytes,
    fetch_hotpotqa_dataset,
    fetch_iirc_dataset,
    fetch_musique_dataset,
    fetch_2wiki_dataset,
    inspect_2wiki_store,
    inspect_prepared_store,
    prepare_normalized_graph_store,
    prepare_2wiki_store,
    probe_source_size,
    write_2wiki_sample_dataset,
)

datasets_app = typer.Typer(
    name="hypercorpus datasets",
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
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_dataset_layout(console, layout)
    if layout.graph_path is None:
        console.print("graph not downloaded: add --graph when you are ready for the large shared file.")
    console.print("for a zero-download smoke dataset: hypercorpus-cli datasets write-2wiki-sample --output-dir /tmp/hypercorpus-2wiki-sample")


@datasets_app.command("write-2wiki-sample")
def write_2wiki_sample(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for the sample dataset"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing sample files"),
) -> None:
    console = Console()
    layout = write_2wiki_sample_dataset(output_dir, overwrite=overwrite)
    _print_dataset_layout(console, layout)
    console.print("sample source: synthetic smoke dataset bundled with the repo")


@datasets_app.command("fetch-iirc")
def fetch_iirc(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for downloaded raw artifacts"),
    archive_url: str = typer.Option(IIRC_ARCHIVE_URL, "--archive-url", help="Override URL for the IIRC train/dev archive"),
    context_source: str | None = typer.Option(None, "--context-source", help="Override the built-in IIRC context source with a local path or URL to context_articles.json or context_articles.tar.gz"),
    question_only: bool = typer.Option(False, "--question-only", help="Skip downloading the built-in IIRC context corpus and allow a partial fetch"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing downloaded/extracted files"),
) -> None:
    console = Console()
    try:
        layout = fetch_iirc_dataset(
            output_dir,
            archive_url=archive_url,
            context_source=context_source,
            require_context=not question_only,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_raw_layout(console, layout)
    if question_only:
        console.print("warning: fetched IIRC raw artifacts do not guarantee a complete context graph; convert/store results may be partial.")
    console.print("next:")
    console.print(f"uv run hypercorpus-cli datasets convert-iirc-raw --raw-dir {layout.raw_dir / 'iirc'} --output-dir {output_dir / 'normalized'}")


@datasets_app.command("fetch-musique")
def fetch_musique(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for downloaded raw artifacts"),
    split: str = typer.Option("dev", "--split", help="Question split to fetch: train, dev, test, or all"),
    subset: str = typer.Option("full", "--subset", help="MuSiQue subset: full or ans"),
    train_url: str | None = typer.Option(None, "--train-url", help="Override the train questions URL"),
    dev_url: str | None = typer.Option(None, "--dev-url", help="Override the dev questions URL"),
    test_url: str | None = typer.Option(None, "--test-url", help="Override the test questions URL"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing downloaded files"),
) -> None:
    console = Console()
    split_urls = {
        "train": train_url,
        "dev": dev_url,
        "test": test_url,
    }
    default_urls = MUSIQUE_FULL_SPLIT_URLS if subset == "full" else MUSIQUE_ANSWERABLE_SPLIT_URLS
    resolved_urls = {name: override or default_urls[name] for name, override in split_urls.items()}
    layout = fetch_musique_dataset(
        output_dir,
        split=split,
        subset=subset,
        split_urls=resolved_urls,
        overwrite=overwrite,
    )
    _print_raw_layout(console, layout)
    console.print("next:")
    console.print(f"uv run hypercorpus-cli datasets convert-musique-raw --raw-dir {layout.raw_dir / 'musique' / subset} --output-dir {output_dir / 'normalized'}")


@datasets_app.command("fetch-hotpotqa")
def fetch_hotpotqa(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for downloaded raw artifacts"),
    variant: str = typer.Option("distractor", "--variant", help="HotpotQA variant: distractor or fullwiki"),
    split: str = typer.Option("dev", "--split", help="Question split to fetch: train, dev, or all"),
    train_url: str | None = typer.Option(None, "--train-url", help="Override the train questions URL"),
    dev_url: str | None = typer.Option(None, "--dev-url", help="Override the dev questions URL"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing downloaded files"),
) -> None:
    console = Console()
    defaults = HOTPOTQA_DISTRACTOR_SPLIT_URLS if variant == "distractor" else HOTPOTQA_FULLWIKI_SPLIT_URLS
    resolved_urls = {
        "train": train_url or defaults.get("train"),
        "dev": dev_url or defaults.get("dev"),
    }
    layout = fetch_hotpotqa_dataset(
        output_dir,
        variant=variant,
        split=split,
        split_urls={name: url for name, url in resolved_urls.items() if url is not None},
        overwrite=overwrite,
    )
    _print_raw_layout(console, layout)
    console.print("next:")
    follow_up = (
        f"uv run hypercorpus-cli datasets convert-hotpotqa-raw --raw-dir {layout.raw_dir / 'hotpotqa' / variant} "
        f"--output-dir {output_dir / 'normalized'} --variant {variant}"
    )
    if variant == "fullwiki":
        follow_up += " --graph-source /path/to/fullwiki-normalized-graph"
    console.print(follow_up)


@datasets_app.command("convert-iirc-raw")
def convert_iirc_raw(
    raw_dir: Path = typer.Option(..., "--raw-dir", exists=True, file_okay=True, dir_okay=True, help="Path to fetched IIRC raw artifacts"),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for normalized graph/questions"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing normalized outputs"),
) -> None:
    console = Console()
    layout = convert_iirc_raw_dataset(
        raw_dir,
        output_dir,
        overwrite=overwrite,
        source_manifest_path=_source_manifest_for(raw_dir),
    )
    _print_normalized_layout(console, layout)


@datasets_app.command("convert-musique-raw")
def convert_musique_raw(
    raw_dir: Path = typer.Option(..., "--raw-dir", exists=True, file_okay=True, dir_okay=True, help="Path to fetched MuSiQue raw artifacts"),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for normalized graph/questions"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing normalized outputs"),
) -> None:
    console = Console()
    layout = convert_musique_raw_dataset(
        raw_dir,
        output_dir,
        overwrite=overwrite,
        source_manifest_path=_source_manifest_for(raw_dir),
    )
    _print_normalized_layout(console, layout)


@datasets_app.command("convert-hotpotqa-raw")
def convert_hotpotqa_raw(
    raw_dir: Path = typer.Option(..., "--raw-dir", exists=True, file_okay=True, dir_okay=True, help="Path to fetched HotpotQA raw artifacts"),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for normalized graph/questions"),
    variant: str = typer.Option("distractor", "--variant", help="HotpotQA variant: distractor or fullwiki"),
    graph_source: str | None = typer.Option(None, "--graph-source", help="Required for fullwiki conversion; path or URL to a normalized graph bundle"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing normalized outputs"),
) -> None:
    console = Console()
    layout = convert_hotpotqa_raw_dataset(
        raw_dir,
        output_dir,
        variant=variant,
        graph_source=graph_source,
        overwrite=overwrite,
        source_manifest_path=_source_manifest_for(raw_dir),
    )
    _print_normalized_layout(console, layout)


@datasets_app.command("prepare-iirc-store-from-raw")
def prepare_iirc_store_from_raw(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory containing raw/ normalized/ store/"),
    archive_url: str = typer.Option(IIRC_ARCHIVE_URL, "--archive-url", help="Override URL for the IIRC train/dev archive"),
    context_source: str | None = typer.Option(None, "--context-source", help="Override the built-in IIRC context source with a local path or URL to context_articles.json or context_articles.tar.gz"),
    question_only: bool = typer.Option(False, "--question-only", help="Skip downloading the built-in IIRC context corpus and allow a partial fetch"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing outputs under output-dir"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
) -> None:
    console = Console()
    raw_root = output_dir / "raw-fetch"
    normalized_root = output_dir / "normalized"
    store_root = output_dir / "store"
    try:
        layout = fetch_iirc_dataset(
            raw_root,
            archive_url=archive_url,
            context_source=context_source,
            require_context=not question_only,
            overwrite=overwrite,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    normalized = convert_iirc_raw_dataset(
        layout.raw_dir / "iirc",
        normalized_root,
        overwrite=overwrite,
        source_manifest_path=layout.source_manifest_path,
    )
    _prepare_generic_store(
        console=console,
        dataset_name="iirc",
        output_dir=store_root,
        questions_source=str(normalized.output_dir / "questions"),
        graph_source=str(normalized.graph_path),
        keep_raw=False,
        overwrite=overwrite,
        min_free_gib=min_free_gib,
    )


@datasets_app.command("prepare-musique-store-from-raw")
def prepare_musique_store_from_raw(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory containing raw/ normalized/ store/"),
    split: str = typer.Option("dev", "--split", help="Question split to fetch: train, dev, test, or all"),
    subset: str = typer.Option("full", "--subset", help="MuSiQue subset: full or ans"),
    train_url: str | None = typer.Option(None, "--train-url", help="Override the train questions URL"),
    dev_url: str | None = typer.Option(None, "--dev-url", help="Override the dev questions URL"),
    test_url: str | None = typer.Option(None, "--test-url", help="Override the test questions URL"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing outputs under output-dir"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
) -> None:
    console = Console()
    raw_root = output_dir / "raw-fetch"
    normalized_root = output_dir / "normalized"
    store_root = output_dir / "store"
    default_urls = MUSIQUE_FULL_SPLIT_URLS if subset == "full" else MUSIQUE_ANSWERABLE_SPLIT_URLS
    resolved_urls = {
        "train": train_url or default_urls["train"],
        "dev": dev_url or default_urls["dev"],
        "test": test_url or default_urls["test"],
    }
    layout = fetch_musique_dataset(raw_root, split=split, subset=subset, split_urls=resolved_urls, overwrite=overwrite)
    normalized = convert_musique_raw_dataset(
        layout.raw_dir / "musique" / subset,
        normalized_root,
        overwrite=overwrite,
        source_manifest_path=layout.source_manifest_path,
    )
    _prepare_generic_store(
        console=console,
        dataset_name="musique",
        output_dir=store_root,
        questions_source=str(normalized.output_dir / "questions"),
        graph_source=str(normalized.graph_path),
        keep_raw=False,
        overwrite=overwrite,
        min_free_gib=min_free_gib,
    )


@datasets_app.command("prepare-hotpotqa-store-from-raw")
def prepare_hotpotqa_store_from_raw(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory containing raw/ normalized/ store/"),
    variant: str = typer.Option("distractor", "--variant", help="HotpotQA variant: distractor or fullwiki"),
    split: str = typer.Option("dev", "--split", help="Question split to fetch: train, dev, or all"),
    train_url: str | None = typer.Option(None, "--train-url", help="Override the train questions URL"),
    dev_url: str | None = typer.Option(None, "--dev-url", help="Override the dev questions URL"),
    graph_source: str | None = typer.Option(None, "--graph-source", help="Required for fullwiki conversion; path or URL to a normalized graph bundle"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing outputs under output-dir"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
) -> None:
    console = Console()
    if variant == "fullwiki" and graph_source is None:
        raise typer.BadParameter("HotpotQA fullwiki store preparation requires --graph-source.")
    raw_root = output_dir / "raw-fetch"
    normalized_root = output_dir / "normalized"
    store_root = output_dir / "store"
    defaults = HOTPOTQA_DISTRACTOR_SPLIT_URLS if variant == "distractor" else HOTPOTQA_FULLWIKI_SPLIT_URLS
    resolved_urls = {
        "train": train_url or defaults.get("train"),
        "dev": dev_url or defaults.get("dev"),
    }
    layout = fetch_hotpotqa_dataset(
        raw_root,
        variant=variant,
        split=split,
        split_urls={name: url for name, url in resolved_urls.items() if url is not None},
        overwrite=overwrite,
    )
    normalized = convert_hotpotqa_raw_dataset(
        layout.raw_dir / "hotpotqa" / variant,
        normalized_root,
        variant=variant,
        graph_source=graph_source,
        overwrite=overwrite,
        source_manifest_path=layout.source_manifest_path,
    )
    dataset_name = "hotpotqa-distractor" if variant == "distractor" else "hotpotqa-fullwiki"
    _prepare_generic_store(
        console=console,
        dataset_name=dataset_name,
        output_dir=store_root,
        questions_source=str(normalized.output_dir / "questions"),
        graph_source=str(normalized.graph_path),
        keep_raw=False,
        overwrite=overwrite,
        min_free_gib=min_free_gib,
    )


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
    try:
        prepared = prepare_2wiki_store(
            output_dir,
            questions_source=questions_source,
            graph_source=graph_source,
            keep_raw=keep_raw,
            overwrite=overwrite,
            min_free_gib=min_free_gib,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"store root -> {prepared.root}")
    console.print(f"manifest.json -> {prepared.manifest_path}")
    console.print(f"catalog.sqlite -> {prepared.catalog_path}")
    console.print(f"shards -> {prepared.manifest.shard_count}")
    for split_name, path in sorted(prepared.questions_paths.items()):
        console.print(f"questions[{split_name}] -> {path}")
    console.print("run:")
    console.print(
        "uv run hypercorpus-cli experiments run-2wiki-store "
        f"--store {prepared.root} "
        "--exp-name pilot-2wiki "
        "--split dev "
        "--chunk-size 100 "
        "--chunk-index 0"
    )


def _prepare_generic_store(
    *,
    console: Console,
    dataset_name: str,
    output_dir: Path,
    questions_source: str,
    graph_source: str,
    keep_raw: bool,
    overwrite: bool,
    min_free_gib: float,
) -> None:
    graph_size = probe_source_size(graph_source)
    ensure_min_free_space(output_dir, min_free_gib=min_free_gib, expected_new_bytes=estimate_prepare_bytes(graph_size))
    try:
        prepared = prepare_normalized_graph_store(
            output_dir,
            dataset_name=dataset_name,
            questions_source=questions_source,
            graph_source=graph_source,
            keep_raw=keep_raw,
            overwrite=overwrite,
            min_free_gib=min_free_gib,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"store root -> {prepared.root}")
    console.print(f"manifest.json -> {prepared.manifest_path}")
    console.print(f"catalog.sqlite -> {prepared.catalog_path}")
    console.print(f"shards -> {prepared.manifest.shard_count}")
    for split_name, path in sorted(prepared.questions_paths.items()):
        console.print(f"questions[{split_name}] -> {path}")


def _print_raw_layout(console: Console, layout) -> None:
    console.print(f"raw root -> {layout.raw_dir}")
    for name, path in sorted(layout.artifact_paths.items()):
        console.print(f"artifact[{name}] -> {path} ({_format_path_size(path)})")
    if layout.source_manifest_path is not None:
        console.print(f"source-manifest.json -> {layout.source_manifest_path}")


def _print_normalized_layout(console: Console, layout) -> None:
    console.print(f"normalized root -> {layout.output_dir}")
    for split_name, path in sorted(layout.question_paths.items()):
        console.print(f"questions[{split_name}] -> {path} ({_format_path_size(path)})")
    if layout.graph_path is not None:
        console.print(f"graph -> {layout.graph_path} ({_format_path_size(layout.graph_path)})")
    if layout.manifest_path is not None:
        console.print(f"conversion-manifest.json -> {layout.manifest_path}")
    follow_up = _normalized_follow_up_command(layout)
    if follow_up is not None:
        label, command = follow_up
        console.print(label)
        console.print(command)


def _source_manifest_for(raw_dir: Path) -> Path | None:
    if raw_dir.is_file():
        parent = raw_dir.parent
    else:
        parent = raw_dir
    for candidate in [parent / "source-manifest.json", parent.parent / "source-manifest.json", parent.parent.parent / "source-manifest.json"]:
        if candidate.exists():
            return candidate
    return None


def _normalized_follow_up_command(layout) -> tuple[str, str] | None:
    if layout.graph_path is None:
        return None
    questions_path = layout.output_dir / "questions"
    store_root = layout.output_dir.parent / "store"
    if layout.dataset_name == "iirc":
        return (
            "prepare store:",
            "uv run hypercorpus-cli datasets prepare-iirc-store "
            f"--output-dir {store_root} "
            f"--questions-source {questions_path} "
            f"--graph-source {layout.graph_path}",
        )
    if layout.dataset_name == "musique":
        return (
            "prepare store:",
            "uv run hypercorpus-cli datasets prepare-musique-store "
            f"--output-dir {store_root} "
            f"--questions-source {questions_path} "
            f"--graph-source {layout.graph_path}",
        )
    if layout.dataset_name == "hotpotqa-fullwiki":
        return (
            "prepare store:",
            "uv run hypercorpus-cli datasets prepare-hotpotqa-store "
            f"--output-dir {store_root} "
            f"--questions-source {questions_path} "
            f"--graph-source {layout.graph_path}",
        )
    if layout.dataset_name == "hotpotqa-distractor":
        default_question = layout.question_paths.get("dev")
        if default_question is None and layout.question_paths:
            default_question = next(iter(layout.question_paths.values()))
        if default_question is not None:
            return (
                "run:",
                "uv run hypercorpus-cli experiments run-hotpotqa "
                f"--questions {default_question} "
                f"--graph-records {layout.graph_path} "
                "--output /tmp/hypercorpus-hotpotqa-run "
                "--variant distractor",
            )
    return None


def _normalized_manifest_dataset_name(source: str) -> tuple[str | None, Path | None]:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme and parsed.scheme != "file":
        return None, None
    if parsed.scheme == "file":
        resolved = Path(urllib.parse.unquote(parsed.path))
    else:
        resolved = Path(source)
    if not resolved.exists():
        return None, None
    candidate_roots = [resolved]
    if resolved.is_file():
        candidate_roots.extend(resolved.parents[:2])
    else:
        candidate_roots.extend(resolved.parents[:1])
    for root in candidate_roots:
        manifest_path = root / "conversion-manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_name = payload.get("dataset_name")
        if dataset_name is None:
            continue
        return str(dataset_name), manifest_path
    return None, None


def _resolved_local_path(source: str) -> Path | None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        resolved = Path(urllib.parse.unquote(parsed.path))
    else:
        resolved = Path(source)
    return resolved if resolved.exists() else None


def _infer_dataset_name_from_questions_source(source: str) -> str | None:
    resolved = _resolved_local_path(source)
    if resolved is None:
        return None
    candidate = resolved
    if resolved.is_dir():
        json_paths = sorted(resolved.glob("*.json"))
        if not json_paths:
            return None
        candidate = json_paths[0]
    if candidate.suffix.lower() not in {".json", ".jsonl"}:
        return None
    if candidate.suffix.lower() == ".jsonl":
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            dataset_name = payload.get("dataset_name")
            return str(dataset_name) if dataset_name is not None else None
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        dataset_name = payload[0].get("dataset_name")
        return str(dataset_name) if dataset_name is not None else None
    if isinstance(payload, dict):
        for key in ("questions", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                dataset_name = value[0].get("dataset_name")
                return str(dataset_name) if dataset_name is not None else None
    return None


def _infer_dataset_name_from_graph_source(source: str) -> str | None:
    resolved = _resolved_local_path(source)
    if resolved is None or not resolved.is_file():
        return None
    if resolved.suffix.lower() != ".jsonl":
        return None
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                dataset_name = payload.get("dataset_name")
                if dataset_name is not None:
                    return str(dataset_name)
                metadata = payload.get("metadata")
                if isinstance(metadata, dict) and metadata.get("dataset") is not None:
                    return str(metadata["dataset"])
            return None
    return None


def _store_command_for_dataset(dataset_name: str) -> str | None:
    if dataset_name == "iirc":
        return "prepare-iirc-store"
    if dataset_name == "musique":
        return "prepare-musique-store"
    if dataset_name == "hotpotqa-fullwiki":
        return "prepare-hotpotqa-store"
    return None


def _validate_normalized_inputs_match_dataset(
    *,
    dataset_name: str,
    questions_source: str,
    graph_source: str,
) -> None:
    seen_mismatches: list[tuple[str, Path]] = []
    for source in (questions_source, graph_source):
        source_dataset, manifest_path = _normalized_manifest_dataset_name(source)
        if source_dataset is None or manifest_path is None or source_dataset == dataset_name:
            continue
        seen_mismatches.append((source_dataset, manifest_path))
    if seen_mismatches:
        source_dataset, manifest_path = seen_mismatches[0]
        expected_command = _store_command_for_dataset(source_dataset)
        message = (
            f"Normalized inputs look like dataset '{source_dataset}' based on {manifest_path}, "
            f"but this command prepares a '{dataset_name}' store."
        )
        if expected_command is not None:
            message += f" Use '{expected_command}' instead."
        raise typer.BadParameter(message)

    inferred_names = {
        name
        for name in (
            _infer_dataset_name_from_questions_source(questions_source),
            _infer_dataset_name_from_graph_source(graph_source),
        )
        if name is not None
    }
    inferred_names.discard(dataset_name)
    if inferred_names:
        source_dataset = sorted(inferred_names)[0]
        expected_command = _store_command_for_dataset(source_dataset)
        message = (
            f"Normalized inputs look like dataset '{source_dataset}', "
            f"but this command prepares a '{dataset_name}' store."
        )
        if expected_command is not None:
            message += f" Use '{expected_command}' instead."
        raise typer.BadParameter(message)


@datasets_app.command("prepare-iirc-store")
def prepare_iirc_store_cli(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for the prepared sharded store"),
    questions_source: str = typer.Option(..., "--questions-source", help="Local path or URL to IIRC questions JSON/JSONL"),
    graph_source: str = typer.Option(..., "--graph-source", help="Local path or URL to normalized IIRC graph JSON/JSONL"),
    keep_raw: bool = typer.Option(False, "--keep-raw/--no-keep-raw", help="Keep downloaded raw graph artifacts under the store root"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing prepared store"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
) -> None:
    console = Console()
    _validate_normalized_inputs_match_dataset(
        dataset_name="iirc",
        questions_source=questions_source,
        graph_source=graph_source,
    )
    _prepare_generic_store(
        console=console,
        dataset_name="iirc",
        output_dir=output_dir,
        questions_source=questions_source,
        graph_source=graph_source,
        keep_raw=keep_raw,
        overwrite=overwrite,
        min_free_gib=min_free_gib,
    )
    console.print("run:")
    console.print(
        "uv run hypercorpus-cli experiments run-iirc-store "
        f"--store {output_dir} "
        "--exp-name pilot-iirc "
        "--split dev "
        "--chunk-size 100 "
        "--chunk-index 0"
    )


@datasets_app.command("prepare-musique-store")
def prepare_musique_store_cli(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for the prepared sharded store"),
    questions_source: str = typer.Option(..., "--questions-source", help="Local path or URL to MuSiQue questions JSON/JSONL"),
    graph_source: str = typer.Option(..., "--graph-source", help="Local path or URL to normalized MuSiQue graph JSON/JSONL"),
    keep_raw: bool = typer.Option(False, "--keep-raw/--no-keep-raw", help="Keep downloaded raw graph artifacts under the store root"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing prepared store"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
) -> None:
    console = Console()
    _validate_normalized_inputs_match_dataset(
        dataset_name="musique",
        questions_source=questions_source,
        graph_source=graph_source,
    )
    _prepare_generic_store(
        console=console,
        dataset_name="musique",
        output_dir=output_dir,
        questions_source=questions_source,
        graph_source=graph_source,
        keep_raw=keep_raw,
        overwrite=overwrite,
        min_free_gib=min_free_gib,
    )
    console.print("run:")
    console.print(
        "uv run hypercorpus-cli experiments run-musique-store "
        f"--store {output_dir} "
        "--exp-name pilot-musique "
        "--split dev "
        "--chunk-size 100 "
        "--chunk-index 0"
    )


@datasets_app.command("prepare-hotpotqa-store")
def prepare_hotpotqa_store_cli(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for the prepared sharded store"),
    questions_source: str = typer.Option(..., "--questions-source", help="Local path or URL to HotpotQA fullwiki questions JSON/JSONL"),
    graph_source: str = typer.Option(..., "--graph-source", help="Local path or URL to normalized HotpotQA fullwiki graph JSON/JSONL"),
    variant: str = typer.Option("fullwiki", "--variant", help="HotpotQA store variant; only fullwiki is supported for stores"),
    keep_raw: bool = typer.Option(False, "--keep-raw/--no-keep-raw", help="Keep downloaded raw graph artifacts under the store root"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite any existing prepared store"),
    min_free_gib: float = typer.Option(DEFAULT_MIN_FREE_GIB, "--min-free-gib", min=1.0, help="Minimum free space to preserve"),
) -> None:
    if variant != "fullwiki":
        raise typer.BadParameter("HotpotQA distractor runs are direct-only; use --variant fullwiki for prepared stores.")
    console = Console()
    _validate_normalized_inputs_match_dataset(
        dataset_name="hotpotqa-fullwiki",
        questions_source=questions_source,
        graph_source=graph_source,
    )
    _prepare_generic_store(
        console=console,
        dataset_name="hotpotqa-fullwiki",
        output_dir=output_dir,
        questions_source=questions_source,
        graph_source=graph_source,
        keep_raw=keep_raw,
        overwrite=overwrite,
        min_free_gib=min_free_gib,
    )
    console.print("run:")
    console.print(
        "uv run hypercorpus-cli experiments run-hotpotqa-store "
        f"--store {output_dir} "
        "--exp-name pilot-hotpotqa "
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


@datasets_app.command("inspect-store")
def inspect_store_cli(
    store: str = typer.Argument(..., help="Path to a prepared store directory (must contain manifest.json)"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir", file_okay=False, help="Local cache directory for prepared stores"),
) -> None:
    """Inspect any prepared dataset store (IIRC, MuSiQue, HotpotQA, 2Wiki)."""
    console = Console()
    inspection = inspect_prepared_store(store_uri=store, cache_dir=cache_dir)
    if inspection.manifest is None:
        console.print(f"[red]No manifest.json found at {store}[/red]")
        raise typer.Exit(1)
    m = inspection.manifest
    console.print(f"dataset          {m.dataset_name}")
    console.print(f"version          {m.version}")
    console.print(f"documents        {m.total_document_count:,}")
    console.print(f"tokens (est.)    {m.total_token_estimate:,}")
    console.print(f"shards           {m.shard_count}")
    total_compressed = sum(s.compressed_bytes for s in m.shards)
    total_uncompressed = sum(s.uncompressed_bytes for s in m.shards)
    console.print(f"size (compressed)    {_format_size(total_compressed)}")
    console.print(f"size (uncompressed)  {_format_size(total_uncompressed)}")
    console.print(f"questions        {', '.join(m.questions_files)}")
    console.print(f"graph_source     {m.graph_source}")
    console.print(f"questions_source {m.questions_source}")
    console.print(f"store_manifest   {inspection.store_manifest_path}")
    console.print(f"cache_dir        {inspection.cache_dir} ({_format_size(inspection.cache_size_bytes)})")
    console.print(f"free_space       {_format_size(inspection.free_space_bytes)}")


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
            "uv run hypercorpus-cli experiments run-2wiki "
            f"--questions {default_question} "
            f"--graph-records {layout.graph_path} "
            "--output /tmp/hypercorpus-2wiki-run"
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
