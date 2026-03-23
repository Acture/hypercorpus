from __future__ import annotations

from pathlib import Path

import typer

from hypercorpus.baselines import (
    build_mdr_index,
    export_iirc_store_to_mdr,
    train_mdr_model,
)

baselines_app = typer.Typer(
    name="hypercorpus baselines",
    help="published baseline wrappers",
    add_completion=False,
)


@baselines_app.command("export-mdr-iirc")
def export_mdr_iirc(
    store: str = typer.Option(..., "--store", help="Path or s3:// URI to a prepared IIRC store"),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for exported MDR files"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", file_okay=False, help="Local cache directory for remote stores"),
) -> None:
    manifest = export_iirc_store_to_mdr(
        store_uri=store,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    typer.echo(f"corpus.jsonl -> {manifest.corpus_path}")
    typer.echo(f"train.jsonl -> {manifest.train_path}")
    typer.echo(f"val.jsonl -> {manifest.val_path}")
    if manifest.dev_eval_path is not None:
        typer.echo(f"dev_eval.jsonl -> {manifest.dev_eval_path}")
    typer.echo(f"mdr_export_manifest.json -> {manifest.export_manifest_path}")


@baselines_app.command(
    "train-mdr",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_mdr(
    ctx: typer.Context,
    export_manifest: Path = typer.Option(..., "--export-manifest", exists=True, dir_okay=False, help="Export manifest produced by export-mdr-iirc"),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Wrapper output directory for official MDR training"),
    init_checkpoint: Path | None = typer.Option(None, "--init-checkpoint", exists=True, dir_okay=False, help="Public checkpoint used to initialize MDR"),
    mdr_home: Path | None = typer.Option(None, "--mdr-home", file_okay=False, help="Path to the pinned official MDR checkout. Defaults to ./baselines/mdr when present."),
    model_name: str = typer.Option("roberta-base", "--model-name", help="Official MDR model name"),
    shared_encoder: bool = typer.Option(False, "--shared-encoder/--no-shared-encoder", help="Forward the shared-encoder flag to official MDR"),
) -> None:
    manifest = train_mdr_model(
        export_manifest_path=export_manifest,
        output_dir=output_dir,
        mdr_home=mdr_home,
        init_checkpoint=init_checkpoint,
        model_name=model_name,
        shared_encoder=shared_encoder,
        extra_args=list(ctx.args),
    )
    typer.echo(f"checkpoint -> {manifest.checkpoint_path}")
    typer.echo(f"mdr_train_manifest.json -> {manifest.train_manifest_path}")


@baselines_app.command(
    "build-mdr-index",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def build_index(
    ctx: typer.Context,
    export_manifest: Path = typer.Option(..., "--export-manifest", exists=True, dir_okay=False, help="Export manifest produced by export-mdr-iirc"),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Directory for encoded corpus artifacts"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", exists=True, dir_okay=False, help="Trained or public MDR checkpoint"),
    train_manifest: Path | None = typer.Option(None, "--train-manifest", exists=True, dir_okay=False, help="Optional training manifest from train-mdr"),
    mdr_home: Path | None = typer.Option(None, "--mdr-home", file_okay=False, help="Path to the pinned official MDR checkout. Defaults to ./baselines/mdr when present."),
    model_name: str | None = typer.Option(None, "--model-name", help="Override the model name used for corpus encoding"),
    shared_encoder: bool | None = typer.Option(None, "--shared-encoder/--no-shared-encoder", help="Override the shared-encoder flag used for corpus encoding"),
    max_q_len: int = typer.Option(70, "--max-q-len", min=1, help="Question encoder max length"),
    max_c_len: int = typer.Option(300, "--max-c-len", min=1, help="Corpus encoder max length"),
    max_q_sp_len: int = typer.Option(350, "--max-q-sp-len", min=1, help="Second-hop query max length"),
    beam_size: int = typer.Option(5, "--beam-size", min=1, help="Default first-hop beam size recorded in the artifact manifest"),
    topk_paths: int = typer.Option(10, "--topk-paths", min=1, help="Default number of 2-hop paths recorded in the artifact manifest"),
) -> None:
    if checkpoint is None and train_manifest is None:
        raise typer.BadParameter("Specify --checkpoint or --train-manifest.")

    manifest = build_mdr_index(
        export_manifest_path=export_manifest,
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        train_manifest_path=train_manifest,
        mdr_home=mdr_home,
        model_name=model_name,
        shared_encoder=shared_encoder,
        max_q_len=max_q_len,
        max_c_len=max_c_len,
        max_q_sp_len=max_q_sp_len,
        beam_size=beam_size,
        topk_paths=topk_paths,
        extra_args=list(ctx.args),
    )
    typer.echo(f"corpus_vectors.npy -> {manifest.corpus_embeddings_path}")
    typer.echo(f"id2doc.json -> {manifest.id2doc_path}")
    if manifest.faiss_index_path is not None:
        typer.echo(f"corpus_vectors.faiss -> {manifest.faiss_index_path}")
    typer.echo(f"mdr_artifact_manifest.json -> {manifest.artifact_manifest_path}")
