import csv
import json
import pytest

from webwalker.experiments import (
    _summarize_result_records,
    merge_2wiki_results,
    parse_budget_ratios,
    parse_token_budgets,
    parse_selector_names,
    run_docs_experiment,
    run_iirc_experiment,
    run_2wiki_experiment,
    run_2wiki_store_experiment,
)


def test_parse_selector_names_handles_empty_values():
    assert parse_selector_names(None) is None
    assert parse_selector_names("") is None
    assert parse_selector_names("seed__link_context_overlap__single_path_walk,seed_rerank") == [
        "seed__link_context_overlap__single_path_walk",
        "seed_rerank",
    ]


def test_parse_budget_ratios_handles_empty_values():
    assert parse_budget_ratios(None) is None
    assert parse_budget_ratios("") is None
    assert parse_budget_ratios("0.05,1.0") == [0.05, 1.0]


def test_parse_token_budgets_handles_empty_values():
    assert parse_token_budgets(None) is None
    assert parse_token_budgets("") is None
    assert parse_token_budgets("128,256") == [128, 256]


def test_run_2wiki_experiment_rejects_conflicting_budget_inputs(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files

    with pytest.raises(ValueError, match="either token_budgets or budget_ratios"):
        run_2wiki_experiment(
            questions_path=questions_path,
            graph_records_path=graph_path,
            output_dir=tmp_path / "conflicting-budgets",
            limit=1,
            selector_names=["seed_rerank"],
            token_budgets=[128],
            budget_ratios=[0.1],
            with_e2e=False,
            export_graphrag_inputs=False,
        )


def test_run_2wiki_experiment_fails_fast_for_missing_llm_key(two_wiki_files, tmp_path, monkeypatch):
    questions_path, graph_path = two_wiki_files
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        run_2wiki_experiment(
            questions_path=questions_path,
            graph_records_path=graph_path,
            output_dir=tmp_path / "missing-key",
            limit=1,
            selector_names=["seed_rerank"],
            token_budgets=[128],
            with_e2e=True,
            answerer_mode="llm_fixed",
            export_graphrag_inputs=False,
        )


def test_run_2wiki_experiment_rejects_legacy_selector_ids(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files

    with pytest.raises(ValueError, match="Unknown selector: dense_topk"):
        run_2wiki_experiment(
            questions_path=questions_path,
            graph_records_path=graph_path,
            output_dir=tmp_path / "legacy-selector",
            limit=1,
            selector_names=["dense_topk"],
            budget_ratios=[0.10],
            with_e2e=False,
            export_graphrag_inputs=False,
        )


def test_run_2wiki_experiment_writes_selector_budget_outputs(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "out"

    evaluations, summary = run_2wiki_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=output_dir,
        limit=1,
        selector_names=[
            "seed__link_context_overlap__single_path_walk",
            "oracle_seed__link_context_overlap__single_path_walk",
            "gold_support_context",
            "full_corpus_upper_bound",
        ],
        token_budgets=[128, 256],
        seed=11,
        max_steps=3,
        top_k=2,
        with_e2e=False,
    )

    assert len(evaluations) == 1
    assert summary.total_cases == 1
    assert [(row.name, row.budget_label) for row in summary.selector_budgets] == [
        ("seed__link_context_overlap__single_path_walk", "tokens-128"),
        ("oracle_seed__link_context_overlap__single_path_walk", "tokens-128"),
        ("gold_support_context", "tokens-128"),
        ("full_corpus_upper_bound", "tokens-128"),
        ("seed__link_context_overlap__single_path_walk", "tokens-256"),
        ("oracle_seed__link_context_overlap__single_path_walk", "tokens-256"),
        ("gold_support_context", "tokens-256"),
        ("full_corpus_upper_bound", "tokens-256"),
    ]

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    first_record = json.loads(lines[0])
    assert first_record["case_id"] == "q1"
    assert first_record["selector"] == "seed__link_context_overlap__single_path_walk"
    assert first_record["budget_mode"] == "tokens"
    assert first_record["budget_value"] == 128
    assert first_record["budget_label"] == "tokens-128"
    assert "selection" in first_record
    assert "budget" in first_record["selection"]
    assert first_record["selection"]["graphrag_input_path"] is not None

    summary_record = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_record["total_cases"] == 1
    eager_full = [
        row
        for row in summary_record["selector_budgets"]
        if row["name"] == "full_corpus_upper_bound" and row["budget_label"] == "tokens-256"
    ][0]
    assert eager_full["avg_compression_ratio"] == 1.0


def test_run_2wiki_experiment_exports_graphrag_csv(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "csv-out"

    evaluations, _summary = run_2wiki_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=output_dir,
        limit=1,
        selector_names=["full_corpus_upper_bound"],
        budget_ratios=[1.0],
    )

    selection = evaluations[0].selections[0]
    assert selection.graphrag_input_path is not None
    export_path = output_dir / selection.graphrag_input_path
    assert export_path.exists()

    with export_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert set(rows[0].keys()) == {"id", "title", "text", "url"}
    assert {row["title"] for row in rows} == {
        "Moon Launch Program",
        "Cape Canaveral",
        "Alice Johnson",
        "Florida",
    }


def test_run_2wiki_experiment_can_disable_e2e_and_export(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "no-e2e"

    evaluations, summary = run_2wiki_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=output_dir,
        limit=1,
        selector_names=["seed_rerank"],
        token_budgets=[128],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations[0].selections) == 1
    assert evaluations[0].selections[0].end_to_end is None
    assert evaluations[0].selections[0].graphrag_input_path is None
    assert summary.selector_budgets[0].avg_answer_em is None
    assert not (output_dir / "graphrag_inputs").exists()

    first_record = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8").strip())
    assert first_record["end_to_end"] is None
    assert first_record["selection"]["graphrag_input_path"] is None


def test_run_2wiki_store_experiment_writes_chunk_outputs(prepared_two_wiki_store, tmp_path):
    evaluations, summary, chunk_dir = run_2wiki_store_experiment(
        store_uri=prepared_two_wiki_store.root,
        output_root=tmp_path / "runs",
        exp_name="pilot",
        chunk_size=1,
        chunk_index=0,
        selector_names=["seed__link_context_overlap__single_path_walk", "gold_support_context", "full_corpus_upper_bound"],
        token_budgets=[128, 256],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations) == 1
    assert summary.total_cases == 1
    assert chunk_dir == tmp_path / "runs" / "pilot" / "chunks" / "chunk-00000"
    assert (chunk_dir / "results.jsonl").exists()
    assert (chunk_dir / "summary.json").exists()
    assert (chunk_dir / "chunk.json").exists()
    chunk_meta = json.loads((chunk_dir / "chunk.json").read_text(encoding="utf-8"))
    assert chunk_meta["token_budgets"] == [128, 256]
    assert chunk_meta["budget_ratios"] is None


def test_merge_2wiki_results_rebuilds_summary_and_checks_missing_chunks(prepared_two_wiki_store, tmp_path):
    run_2wiki_store_experiment(
        store_uri=prepared_two_wiki_store.root,
        output_root=tmp_path / "runs",
        exp_name="pilot",
        chunk_size=1,
        chunk_index=0,
        selector_names=["seed_rerank"],
        budget_ratios=[0.10],
        with_e2e=False,
        export_graphrag_inputs=False,
    )
    run_2wiki_store_experiment(
        store_uri=prepared_two_wiki_store.root,
        output_root=tmp_path / "runs",
        exp_name="pilot",
        chunk_size=1,
        chunk_index=1,
        selector_names=["seed_rerank"],
        budget_ratios=[0.10],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    summary, missing = merge_2wiki_results(run_dir=tmp_path / "runs" / "pilot")

    assert summary.total_cases == 2
    assert missing == []
    assert (tmp_path / "runs" / "pilot" / "results.jsonl").exists()
    assert (tmp_path / "runs" / "pilot" / "summary.json").exists()


def test_run_2wiki_store_experiment_emits_progress_logs(prepared_two_wiki_store, tmp_path, caplog):
    caplog.set_level("INFO")

    run_2wiki_store_experiment(
        store_uri=prepared_two_wiki_store.root,
        output_root=tmp_path / "runs",
        exp_name="pilot",
        chunk_size=1,
        chunk_index=0,
        selector_names=["seed_rerank"],
        budget_ratios=[0.10],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Running store-backed 2Wiki experiment" in message for message in messages)
    assert any("Completed store-backed 2Wiki chunk" in message for message in messages)


def test_summarize_result_records_separates_selector_model_groups():
    base_record = {
        "dataset_name": "2wikimultihop",
        "case_id": "q1",
        "selector": "seed__link_context_llm__single_path_walk",
        "budget_mode": "tokens",
        "budget_value": 128,
        "budget_label": "tokens-128",
        "token_budget_tokens": 128,
        "token_budget_ratio": None,
        "selection": {
            "metrics": {
                "start_hit": True,
                "support_recall": 1.0,
                "support_precision": 1.0,
                "support_f1": 1.0,
                "path_hit": True,
                "selected_nodes_count": 2,
                "selected_token_estimate": 64,
                "compression_ratio": 0.1,
                "budget_adherence": True,
                "selection_runtime_s": 0.01,
            },
            "selector_usage": {
                "runtime_s": 0.3,
                "llm_calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cache_hits": 0,
            },
        },
        "end_to_end": None,
    }
    summary = _summarize_result_records(
        [
            {
                **base_record,
                "selector_provider": "openai",
                "selector_model": "gpt-small",
            },
            {
                **base_record,
                "case_id": "q2",
                "selector_provider": "openai",
                "selector_model": "gpt-large",
            },
        ]
    )

    assert [(row.selector_provider, row.selector_model) for row in summary.selector_budgets] == [
        ("openai", "gpt-small"),
        ("openai", "gpt-large"),
    ]


def test_run_iirc_experiment_handles_missing_path_supervision(iirc_files, tmp_path):
    questions_path, graph_path = iirc_files
    output_dir = tmp_path / "iirc-out"

    evaluations, summary = run_iirc_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=output_dir,
        selector_names=["seed_rerank"],
        budget_ratios=[0.10],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations) == 2
    assert summary.dataset_name == "iirc"
    assert summary.total_cases == 2
    assert summary.selector_budgets[0].avg_path_hit is not None
    lines = (output_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    first_record = json.loads(lines[0])
    assert first_record["dataset_name"] == "iirc"
    assert first_record["selection"]["metrics"]["path_hit"] is None


def test_run_docs_experiment_accepts_html_root(docs_files, tmp_path):
    questions_path, docs_root = docs_files
    output_dir = tmp_path / "docs-out"

    evaluations, summary = run_docs_experiment(
        questions_path=questions_path,
        docs_source=docs_root,
        output_dir=output_dir,
        dataset_name="python_docs",
        selector_names=["seed_rerank"],
        budget_ratios=[0.10],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations) == 1
    assert summary.dataset_name == "python_docs"
    assert (output_dir / "results.jsonl").exists()
    assert json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8").strip())["dataset_name"] == "python_docs"
