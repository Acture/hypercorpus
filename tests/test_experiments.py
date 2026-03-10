import csv
import json

from webwalker.experiments import (
    merge_2wiki_results,
    parse_budget_ratios,
    parse_selector_names,
    run_2wiki_experiment,
    run_2wiki_store_experiment,
)


def test_parse_selector_names_handles_empty_values():
    assert parse_selector_names(None) is None
    assert parse_selector_names("") is None
    assert parse_selector_names("webwalker_selector,dense_topk") == ["webwalker_selector", "dense_topk"]


def test_parse_budget_ratios_handles_empty_values():
    assert parse_budget_ratios(None) is None
    assert parse_budget_ratios("") is None
    assert parse_budget_ratios("0.05,1.0") == [0.05, 1.0]


def test_run_2wiki_experiment_writes_selector_budget_outputs(two_wiki_files, tmp_path):
    questions_path, graph_path = two_wiki_files
    output_dir = tmp_path / "out"

    evaluations, summary = run_2wiki_experiment(
        questions_path=questions_path,
        graph_records_path=graph_path,
        output_dir=output_dir,
        limit=1,
        selector_names=["webwalker_selector", "oracle_start_webwalker", "eager_full_corpus_proxy"],
        budget_ratios=[0.10, 1.0],
        seed=11,
        max_steps=3,
        top_k=2,
    )

    assert len(evaluations) == 1
    assert summary.total_cases == 1
    assert [(row.name, row.token_budget_ratio) for row in summary.selector_budgets] == [
        ("webwalker_selector", 0.10),
        ("oracle_start_webwalker", 0.10),
        ("eager_full_corpus_proxy", 0.10),
        ("webwalker_selector", 1.0),
        ("oracle_start_webwalker", 1.0),
        ("eager_full_corpus_proxy", 1.0),
    ]

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    first_record = json.loads(lines[0])
    assert first_record["case_id"] == "q1"
    assert first_record["selector"] == "webwalker_selector"
    assert first_record["token_budget_ratio"] == 0.10
    assert "selection" in first_record
    assert "budget" in first_record["selection"]
    assert first_record["selection"]["graphrag_input_path"] is not None

    summary_record = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_record["total_cases"] == 1
    eager_full = [
        row
        for row in summary_record["selector_budgets"]
        if row["name"] == "eager_full_corpus_proxy" and row["token_budget_ratio"] == 1.0
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
        selector_names=["eager_full_corpus_proxy"],
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
        selector_names=["dense_topk"],
        budget_ratios=[0.10],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations[0].selections) == 1
    assert evaluations[0].selections[0].end_to_end is None
    assert evaluations[0].selections[0].graphrag_input_path is None
    assert summary.selector_budgets[0].avg_e2e_em is None
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
        selector_names=["webwalker_selector", "eager_full_corpus_proxy"],
        budget_ratios=[0.10, 1.0],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    assert len(evaluations) == 1
    assert summary.total_cases == 1
    assert chunk_dir == tmp_path / "runs" / "pilot" / "chunks" / "chunk-00000"
    assert (chunk_dir / "results.jsonl").exists()
    assert (chunk_dir / "summary.json").exists()
    assert (chunk_dir / "chunk.json").exists()


def test_merge_2wiki_results_rebuilds_summary_and_checks_missing_chunks(prepared_two_wiki_store, tmp_path):
    run_2wiki_store_experiment(
        store_uri=prepared_two_wiki_store.root,
        output_root=tmp_path / "runs",
        exp_name="pilot",
        chunk_size=1,
        chunk_index=0,
        selector_names=["dense_topk"],
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
        selector_names=["dense_topk"],
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
        selector_names=["dense_topk"],
        budget_ratios=[0.10],
        with_e2e=False,
        export_graphrag_inputs=False,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Running store-backed 2Wiki experiment" in message for message in messages)
    assert any("Completed store-backed 2Wiki chunk" in message for message in messages)
