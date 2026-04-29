"""Tests for the `reanswer-iirc` CLI / `run_reanswer` pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hypercorpus.answering import AnswerWithEvidence, SupportsAnswer
from hypercorpus.reanswer import (
	ReanswerInputRow,
	iter_input_rows,
	outcome_to_record,
	reanswer_row,
	run_reanswer,
	summarize_outcomes,
)
from hypercorpus.subgraph import SubgraphExtractor
from hypercorpus_cli.experiments import experiments_app


def _make_record(
	*,
	case_id: str,
	selector: str,
	budget_label: str,
	node_ids: list[str],
	expected_answer: str | None,
	gold_support: list[str] | None = None,
) -> dict[str, object]:
	return {
		"dataset_name": "iirc",
		"case_id": case_id,
		"query": "Which city hosts the launch site?",
		"expected_answer": expected_answer,
		"gold_support_nodes": gold_support or ["mission", "cape"],
		"gold_start_nodes": ["mission"],
		"gold_path_nodes": None,
		"question_type": "bridge",
		"selector": selector,
		"budget_label": budget_label,
		"selection": {
			"corpus": {"node_ids": list(node_ids), "token_estimate": 42},
			"metrics": {"support_f1": 0.75},
			"budget": {"budget_label": budget_label},
		},
	}


def _write_run(tmp_path: Path, run_name: str, records: list[dict[str, object]]) -> Path:
	run_dir = tmp_path / run_name
	chunk_dir = run_dir / "chunks" / "chunk-00000"
	chunk_dir.mkdir(parents=True)
	results_path = chunk_dir / "results.jsonl"
	with results_path.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record) + "\n")
	return run_dir


class _FakeAnswerer(SupportsAnswer):
	def __init__(self, answer_text: str = "Cape Canaveral"):
		self.answer_text = answer_text
		self.calls = 0

	def answer(self, query, subgraph):  # type: ignore[override]
		self.calls += 1
		return AnswerWithEvidence(
			query=query,
			answer=self.answer_text,
			confidence=1.0,
			evidence=[],
			mode="llm_fixed",
			model="fake-model",
			runtime_s=0.01,
			prompt_tokens=10,
			completion_tokens=2,
			total_tokens=12,
		)


def test_iter_input_rows_filters_selectors_and_budgets(tmp_path: Path) -> None:
	run_dir = _write_run(
		tmp_path,
		"run-a",
		[
			_make_record(
				case_id="q1",
				selector="dense",
				budget_label="ratio-0.0100",
				node_ids=["mission", "cape"],
				expected_answer="Cape Canaveral",
			),
			_make_record(
				case_id="q2",
				selector="ctrl",
				budget_label="ratio-0.0100",
				node_ids=["mission"],
				expected_answer="Cape Canaveral",
			),
			_make_record(
				case_id="q3",
				selector="dense",
				budget_label="ratio-0.0500",
				node_ids=["mission"],
				expected_answer="Cape Canaveral",
			),
		],
	)
	rows = list(
		iter_input_rows(
			[run_dir],
			selector_filter=["dense"],
			budget_label_filter=["ratio-0.0100"],
		)
	)
	assert len(rows) == 1
	assert rows[0].case_id == "q1"
	assert rows[0].selected_node_ids == ["mission", "cape"]


def test_reanswer_row_computes_f1_and_em(sample_graph) -> None:
	row = ReanswerInputRow(
		source_run="run-a",
		dataset_name="iirc",
		case_id="q1",
		query="Which city hosts the launch site?",
		expected_answer="Cape Canaveral",
		gold_support_nodes=["mission", "cape"],
		gold_start_nodes=["mission"],
		gold_path_nodes=None,
		question_type="bridge",
		selector="dense",
		budget_label="ratio-0.0100",
		selected_node_ids=["mission", "cape"],
		selected_token_estimate=42,
		support_f1=0.75,
	)
	outcome = reanswer_row(
		row,
		graph=sample_graph,
		answerer=_FakeAnswerer("Cape Canaveral"),
		extractor=SubgraphExtractor(),
	)
	assert outcome.answer == "Cape Canaveral"
	assert outcome.answer_em == 1.0
	assert outcome.answer_f1 == 1.0
	record = outcome_to_record(outcome)
	assert record["end_to_end"]["f1"] == 1.0
	assert record["selector"] == "dense"


def test_run_reanswer_writes_outputs_and_summary(
	tmp_path: Path, sample_graph, monkeypatch
) -> None:
	run_dir = _write_run(
		tmp_path,
		"run-a",
		[
			_make_record(
				case_id="q1",
				selector="dense",
				budget_label="ratio-0.0100",
				node_ids=["mission", "cape"],
				expected_answer="Cape Canaveral",
			),
			_make_record(
				case_id="q2",
				selector="dense",
				budget_label="ratio-0.0100",
				node_ids=["mission"],
				expected_answer="Cape Canaveral",
			),
		],
	)
	fake_answerer = _FakeAnswerer("Cape Canaveral")
	monkeypatch.setattr(
		"hypercorpus.reanswer.build_answerer",
		lambda **_kwargs: fake_answerer,
	)
	output_dir = run_reanswer(
		input_runs=[run_dir],
		store_uri="ignored",
		output_root=tmp_path / "runs",
		exp_name="iirc-reanswer-test",
		answerer_mode="llm_fixed",
		answer_provider="copilot",
		answer_model="gpt-4.1",
		store_factory=lambda _uri: sample_graph,
	)
	results_path = output_dir / "results.jsonl"
	summary_path = output_dir / "summary.json"
	assert results_path.is_file()
	rows = [json.loads(line) for line in results_path.read_text().splitlines()]
	assert len(rows) == 2
	assert all(row["end_to_end"]["f1"] == 1.0 for row in rows)
	summary = json.loads(summary_path.read_text())
	assert summary["total_outcomes"] == 2
	assert summary["groups"][0]["avg_answer_f1"] == 1.0


def test_summarize_outcomes_handles_mixed_groups(sample_graph) -> None:
	rows = [
		ReanswerInputRow(
			source_run="r",
			dataset_name="iirc",
			case_id=f"q{i}",
			query="Which city hosts the launch site?",
			expected_answer="Cape Canaveral",
			gold_support_nodes=["mission", "cape"],
			gold_start_nodes=["mission"],
			gold_path_nodes=None,
			question_type="bridge",
			selector=selector,
			budget_label="ratio-0.0100",
			selected_node_ids=["mission", "cape"],
			selected_token_estimate=42,
			support_f1=None,
		)
		for i, selector in enumerate(["dense", "dense", "ctrl"])
	]
	answers = ["Cape Canaveral", "Wrong", "Cape Canaveral"]
	outcomes = [
		reanswer_row(
			row,
			graph=sample_graph,
			answerer=_FakeAnswerer(answers[i]),
			extractor=SubgraphExtractor(),
		)
		for i, row in enumerate(rows)
	]
	summary = summarize_outcomes(outcomes)
	groups = {g["selector"]: g for g in summary["groups"]}
	assert groups["dense"]["num_cases"] == 2
	assert groups["dense"]["avg_answer_em"] == 0.5
	assert groups["ctrl"]["avg_answer_em"] == 1.0


def test_cli_reanswer_iirc_smoke(tmp_path: Path, sample_graph, monkeypatch) -> None:
	run_dir = _write_run(
		tmp_path,
		"run-a",
		[
			_make_record(
				case_id="q1",
				selector="dense",
				budget_label="ratio-0.0100",
				node_ids=["mission", "cape"],
				expected_answer="Cape Canaveral",
			),
		],
	)
	monkeypatch.setattr(
		"hypercorpus.reanswer.build_answerer",
		lambda **_kwargs: _FakeAnswerer("Cape Canaveral"),
	)
	monkeypatch.setattr(
		"hypercorpus.reanswer.ShardedDocumentStore",
		lambda *args, **kwargs: sample_graph,
	)
	output_root = tmp_path / "runs"
	runner = CliRunner()
	result = runner.invoke(
		experiments_app,
		[
			"reanswer-iirc",
			"--input-runs",
			str(run_dir),
			"--store",
			"ignored",
			"--exp-name",
			"iirc-reanswer-cli-test",
			"--output-root",
			str(output_root),
			"--max-cases",
			"1",
		],
	)
	assert result.exit_code == 0, result.output
	assert (output_root / "iirc-reanswer-cli-test" / "results.jsonl").is_file()
