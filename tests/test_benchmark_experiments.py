import json

from hypercorpus.experiments import (
    merge_hotpotqa_results,
    merge_iirc_results,
    merge_musique_results,
    run_hotpotqa_experiment,
    run_hotpotqa_store_experiment,
    run_iirc_experiment,
    run_iirc_store_experiment,
    run_musique_experiment,
    run_musique_store_experiment,
)


CANONICAL_DENSE = "top_1_seed__lexical_overlap__hop_0__dense"


def test_run_iirc_store_experiment_matches_direct_dataset_name(iirc_files, prepared_iirc_store, tmp_path):
    questions_path, graph_path = iirc_files
    direct_output = tmp_path / "iirc-direct"

    direct_evaluations, direct_summary = run_iirc_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=direct_output,
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )
    store_evaluations, store_summary, chunk_dir = run_iirc_store_experiment(
        store_uri=prepared_iirc_store.root,
        output_root=tmp_path / "runs",
        exp_name="iirc",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(direct_evaluations) == len(store_evaluations) == 2
    assert direct_summary.dataset_name == store_summary.dataset_name == "iirc"
    assert chunk_dir.exists()


def test_run_musique_direct_and_store(musique_files, prepared_musique_store, tmp_path):
    questions_path, graph_path = musique_files

    evaluations, summary = run_musique_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=tmp_path / "musique-direct",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )
    store_evaluations, store_summary, chunk_dir = run_musique_store_experiment(
        store_uri=prepared_musique_store.root,
        output_root=tmp_path / "runs",
        exp_name="musique",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations) == len(store_evaluations) == 1
    assert summary.dataset_name == store_summary.dataset_name == "musique"
    assert (chunk_dir / "results.jsonl").exists()


def test_run_hotpotqa_distractor_direct(hotpotqa_distractor_file, tmp_path):
    evaluations, summary = run_hotpotqa_experiment(
        questions_path=hotpotqa_distractor_file,
        output_dir=tmp_path / "hotpot-distractor",
        variant="distractor",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations) == 1
    assert summary.dataset_name == "hotpotqa-distractor"
    record = json.loads((tmp_path / "hotpot-distractor" / "results.jsonl").read_text(encoding="utf-8").strip())
    assert record["dataset_name"] == "hotpotqa-distractor"


def test_run_hotpotqa_fullwiki_direct_and_store(hotpotqa_fullwiki_files, prepared_hotpotqa_store, tmp_path):
    questions_path, graph_path = hotpotqa_fullwiki_files

    evaluations, summary = run_hotpotqa_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=tmp_path / "hotpot-fullwiki",
        variant="fullwiki",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )
    store_evaluations, store_summary, chunk_dir = run_hotpotqa_store_experiment(
        store_uri=prepared_hotpotqa_store.root,
        output_root=tmp_path / "runs",
        exp_name="hotpot",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations) == len(store_evaluations) == 1
    assert summary.dataset_name == store_summary.dataset_name == "hotpotqa-fullwiki"
    assert chunk_dir.exists()


def test_merge_store_results_for_new_datasets(prepared_iirc_store, prepared_musique_store, prepared_hotpotqa_store, tmp_path):
    _iirc, _summary_iirc, _chunk_iirc = run_iirc_store_experiment(
        store_uri=prepared_iirc_store.root,
        output_root=tmp_path / "runs",
        exp_name="iirc",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )
    _musique, _summary_musique, _chunk_musique = run_musique_store_experiment(
        store_uri=prepared_musique_store.root,
        output_root=tmp_path / "runs",
        exp_name="musique",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )
    _hotpot, _summary_hotpot, _chunk_hotpot = run_hotpotqa_store_experiment(
        store_uri=prepared_hotpotqa_store.root,
        output_root=tmp_path / "runs",
        exp_name="hotpot",
        selector_names=[CANONICAL_DENSE],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    iirc_summary, iirc_missing = merge_iirc_results(run_dir=tmp_path / "runs" / "iirc")
    musique_summary, musique_missing = merge_musique_results(run_dir=tmp_path / "runs" / "musique")
    hotpot_summary, hotpot_missing = merge_hotpotqa_results(run_dir=tmp_path / "runs" / "hotpot")

    assert iirc_summary.dataset_name == "iirc"
    assert musique_summary.dataset_name == "musique"
    assert hotpot_summary.dataset_name == "hotpotqa-fullwiki"
    assert iirc_missing == []
    assert musique_missing == []
    assert hotpot_missing == []

