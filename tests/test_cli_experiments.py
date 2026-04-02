from rich.console import Console
from typer.testing import CliRunner

from hypercorpus.eval import ExperimentSummary, SelectorBudgetSummary
from hypercorpus.logging import (
	DashboardLogBuffer,
	DashboardLogEntry,
	DashboardProgressState,
)
from hypercorpus_cli import app
from hypercorpus_cli.experiments import (
	_ExperimentDashboardRenderable,
	_LiveDashboardState,
	_print_summary,
	_run_with_optional_dashboard,
)

CANONICAL_DENSE = "top_1_seed__lexical_overlap__hop_0__dense"
CANONICAL_OVERLAP = "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_overlap__lookahead_1"
CANONICAL_LLM = "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_llm__lookahead_1"


def _make_selector_budget_summary(
	*,
	name: str,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	budget_label: str = "tokens-128",
	avg_support_precision: float | None = 0.6,
	avg_support_f1: float | None = 0.545,
	avg_support_f1_zero_on_empty: float | None = 0.5,
	avg_budget_utilization: float = 0.55,
	avg_empty_selection_rate: float = 0.1,
	avg_selector_total_tokens: float | None = None,
	avg_selector_runtime_s: float | None = None,
	avg_selector_llm_calls: float | None = None,
	avg_selector_fallback_rate: float | None = None,
	avg_selector_parse_failure_rate: float | None = None,
) -> SelectorBudgetSummary:
	return SelectorBudgetSummary(
		name=name,
		selector_provider=selector_provider,
		selector_model=selector_model,
		budget_mode="tokens",
		budget_value=128,
		budget_label=budget_label,
		token_budget_ratio=None,
		token_budget_tokens=128,
		num_cases=1,
		avg_start_hit=None,
		avg_support_recall=0.5,
		avg_support_precision=avg_support_precision,
		avg_support_f1=avg_support_f1,
		avg_path_hit=None,
		avg_selected_nodes=2.0,
		avg_selected_token_estimate=70.0,
		avg_compression_ratio=0.1,
		avg_budget_adherence=1.0,
		avg_budget_utilization=avg_budget_utilization,
		avg_empty_selection_rate=avg_empty_selection_rate,
		avg_selection_runtime_s=0.01,
		avg_support_f1_zero_on_empty=avg_support_f1_zero_on_empty,
		avg_selector_prompt_tokens=None,
		avg_selector_completion_tokens=None,
		avg_selector_total_tokens=avg_selector_total_tokens,
		avg_selector_runtime_s=avg_selector_runtime_s,
		avg_selector_llm_calls=avg_selector_llm_calls,
		avg_selector_fallback_rate=avg_selector_fallback_rate,
		avg_selector_parse_failure_rate=avg_selector_parse_failure_rate,
		avg_answer_em=None,
		avg_answer_f1=None,
	)


def test_run_2wiki_cli_smoke(two_wiki_files, tmp_path):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-out"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--limit",
			"1",
			"--selectors",
			f"{CANONICAL_OVERLAP},full_corpus_upper_bound",
			"--token-budgets",
			"128,256",
			"--no-e2e",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "2wikimultihop summary" in result.stdout
	assert "selector" in result.stdout
	assert "budget" in result.stdout
	assert (output_dir / "results.jsonl").exists()
	assert (output_dir / "selector_logs.jsonl").exists()
	assert (output_dir / "summary.json").exists()


def test_run_2wiki_cli_supports_no_e2e_and_no_export(two_wiki_files, tmp_path):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-no-e2e"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--limit",
			"1",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "2wikimultihop summary" in result.stdout
	assert not (output_dir / "graphrag_inputs").exists()


def test_run_2wiki_cli_forwards_selector_preset_when_selectors_are_omitted(
	two_wiki_files, tmp_path, monkeypatch
):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-preset"
	runner = CliRunner()
	captured: dict[str, object] = {}

	def _fake_run_2wiki_experiment(**kwargs):
		captured.update(kwargs)
		return [], ExperimentSummary(
			dataset_name="2wikimultihop", total_cases=0, selector_budgets=[]
		)

	monkeypatch.setattr(
		"hypercorpus_cli.experiments.run_2wiki_experiment", _fake_run_2wiki_experiment
	)

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--selector-preset",
			"paper_recommended_local",
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert captured["selector_names"] is None
	assert captured["selector_preset"] == "paper_recommended_local"


def test_run_2wiki_cli_forwards_study_preset_and_case_ids_file(
	two_wiki_files, tmp_path, monkeypatch
):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-study"
	case_ids_path = tmp_path / "case-ids.txt"
	case_ids_path.write_text("q1\n", encoding="utf-8")
	runner = CliRunner()
	captured: dict[str, object] = {}

	def _fake_run_2wiki_experiment(**kwargs):
		captured.update(kwargs)
		return [], ExperimentSummary(
			dataset_name="2wikimultihop", total_cases=0, selector_budgets=[]
		)

	monkeypatch.setattr(
		"hypercorpus_cli.experiments.run_2wiki_experiment", _fake_run_2wiki_experiment
	)

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--study-preset",
			"baseline_retest_local",
			"--case-ids-file",
			str(case_ids_path),
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert captured["study_preset"] == "baseline_retest_local"
	assert captured["case_ids_file"] == case_ids_path


def test_run_2wiki_cli_forwards_selector_openai_api_mode(
	two_wiki_files, tmp_path, monkeypatch
):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-openai-api-mode"
	runner = CliRunner()
	captured: dict[str, object] = {}

	def _fake_run_2wiki_experiment(**kwargs):
		captured.update(kwargs)
		return [], ExperimentSummary(
			dataset_name="2wikimultihop", total_cases=0, selector_budgets=[]
		)

	monkeypatch.setattr(
		"hypercorpus_cli.experiments.run_2wiki_experiment", _fake_run_2wiki_experiment
	)

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--selectors",
			CANONICAL_DENSE,
			"--selector-provider",
			"openai",
			"--selector-openai-api-mode",
			"responses",
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert captured["selector_openai_api_mode"] == "responses"


def test_run_2wiki_cli_help_mentions_new_presets():
	runner = CliRunner()

	result = runner.invoke(
		app, ["experiments", "run-2wiki", "--help"], env={"COLUMNS": "240"}
	)

	assert result.exit_code == 0, result.stdout
	assert "paper_recommended_local" in result.stdout
	assert "branchy_profiles" in result.stdout
	assert "single_path_edge_ablation_local" in result.stdout
	assert "baseline_retest_local" in result.stdout
	assert "--study-preset" in result.stdout
	assert "--case-ids-file" in result.stdout
	assert "--answer-provider" in result.stdout
	assert "configured LLM backend defaults" in result.stdout
	assert "avoids LLM selector" in result.stdout


def test_run_2wiki_cli_rejects_paper_recommended_without_llm_config(
	two_wiki_files, tmp_path, monkeypatch
):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-paper-recommended"
	runner = CliRunner()
	monkeypatch.delenv("GITHUB_TOKEN", raising=False)
	monkeypatch.chdir(tmp_path)

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--selector-preset",
			"paper_recommended",
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code != 0
	assert isinstance(result.exception, ValueError)
	assert (
		str(result.exception) == "Missing API key in environment variable GITHUB_TOKEN"
	)


def test_run_2wiki_cli_passes_explicit_selectors_alongside_selector_preset(
	two_wiki_files, tmp_path, monkeypatch
):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-preset-explicit"
	runner = CliRunner()
	captured: dict[str, object] = {}

	def _fake_run_2wiki_experiment(**kwargs):
		captured.update(kwargs)
		return [], ExperimentSummary(
			dataset_name="2wikimultihop", total_cases=0, selector_budgets=[]
		)

	monkeypatch.setattr(
		"hypercorpus_cli.experiments.run_2wiki_experiment", _fake_run_2wiki_experiment
	)

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--selectors",
			CANONICAL_DENSE,
			"--selector-preset",
			"paper_recommended",
			"--answer-provider",
			"openai",
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert captured["selector_names"] == [CANONICAL_DENSE]
	assert captured["selector_preset"] == "paper_recommended"
	assert captured["answer_provider"] == "openai"


def test_run_2wiki_cli_rejects_legacy_selector_ids(two_wiki_files, tmp_path):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-legacy-selector"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--limit",
			"1",
			"--selectors",
			"seed_rerank",
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code != 0
	assert isinstance(result.exception, ValueError)
	assert str(result.exception) == "Unknown selector: seed_rerank"


def test_run_2wiki_cli_rejects_conflicting_budget_flags(two_wiki_files, tmp_path):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "cli-conflicting-budgets"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--limit",
			"1",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--budget-ratios",
			"0.10",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code != 0
	assert isinstance(result.exception, ValueError)
	assert (
		str(result.exception)
		== "Specify either --token-budgets or --budget-ratios, not both."
	)


def test_run_2wiki_store_cli_smoke(prepared_two_wiki_store, tmp_path):
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki-store",
			"--store",
			str(prepared_two_wiki_store.root),
			"--exp-name",
			"pilot",
			"--output-root",
			str(tmp_path / "runs"),
			"--chunk-size",
			"1",
			"--chunk-index",
			"0",
			"--selectors",
			f"{CANONICAL_OVERLAP},full_corpus_upper_bound",
			"--token-budgets",
			"128,256",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "chunk_dir" in result.stdout
	assert (
		tmp_path / "runs" / "pilot" / "chunks" / "chunk-00000" / "results.jsonl"
	).exists()


def test_run_iirc_cli_smoke(iirc_files, tmp_path):
	questions_path, graph_path = iirc_files
	output_dir = tmp_path / "iirc-cli-out"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-iirc",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "iirc summary" in result.stdout


def test_run_docs_cli_smoke(docs_files, tmp_path):
	questions_path, docs_root = docs_files
	output_dir = tmp_path / "docs-cli-out"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-docs",
			"--questions",
			str(questions_path),
			"--docs-source",
			str(docs_root),
			"--output",
			str(output_dir),
			"--dataset-name",
			"python_docs",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "python_docs summary" in result.stdout


def test_merge_2wiki_results_cli_reports_missing_chunks(
	prepared_two_wiki_store, tmp_path
):
	runner = CliRunner()
	run_root = tmp_path / "runs"
	run_dir = run_root / "pilot"

	run_result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki-store",
			"--store",
			str(prepared_two_wiki_store.root),
			"--exp-name",
			"pilot",
			"--output-root",
			str(run_root),
			"--chunk-size",
			"1",
			"--chunk-index",
			"0",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)
	assert run_result.exit_code == 0, run_result.stdout

	merge_result = runner.invoke(
		app,
		[
			"experiments",
			"merge-2wiki-results",
			"--run-dir",
			str(run_dir),
		],
	)

	assert merge_result.exit_code == 0, merge_result.stdout
	assert "missing_chunks -> [1]" in merge_result.stdout
	assert "merged summary_rows.csv" in merge_result.stdout
	assert "merged study_comparison_rows.csv" in merge_result.stdout
	assert "merged subset_comparison_rows.csv" in merge_result.stdout
	assert "merged run_manifest.json" in merge_result.stdout
	assert "merged evaluated_case_ids.txt" in merge_result.stdout


def test_export_summary_report_cli_writes_csv(two_wiki_files, tmp_path):
	questions_path, graph_path = two_wiki_files
	output_dir = tmp_path / "report-cli-out"
	runner = CliRunner()

	run_result = runner.invoke(
		app,
		[
			"experiments",
			"run-2wiki",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--limit",
			"1",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)
	assert run_result.exit_code == 0, run_result.stdout

	report_path = tmp_path / "exports" / "custom-summary.csv"
	export_result = runner.invoke(
		app,
		[
			"experiments",
			"export-summary-report",
			"--summary",
			str(output_dir / "summary.json"),
			"--output",
			str(report_path),
		],
	)

	assert export_result.exit_code == 0, export_result.stdout
	assert "summary_rows.csv ->" in export_result.stdout
	assert "study_comparison_rows.csv ->" in export_result.stdout
	assert "subset_comparison_rows.csv ->" in export_result.stdout
	assert report_path.name in export_result.stdout
	assert report_path.exists()
	default_comparison_path = report_path.parent / "study_comparison_rows.csv"
	default_subset_path = report_path.parent / "subset_comparison_rows.csv"
	assert default_comparison_path.name in export_result.stdout
	assert default_comparison_path.exists()
	assert default_subset_path.name in export_result.stdout
	assert default_subset_path.exists()

	comparison_path = tmp_path / "exports-2" / "custom-comparison.csv"
	export_with_custom_comparison = runner.invoke(
		app,
		[
			"experiments",
			"export-summary-report",
			"--summary",
			str(output_dir / "summary.json"),
			"--output",
			str(report_path),
			"--comparison-output",
			str(comparison_path),
		],
	)

	assert export_with_custom_comparison.exit_code == 0, (
		export_with_custom_comparison.stdout
	)
	assert comparison_path.name in export_with_custom_comparison.stdout
	assert comparison_path.exists()


def test_print_summary_renders_main_table_only_for_non_llm_rows():
	summary = ExperimentSummary(
		dataset_name="2wikimultihop",
		total_cases=1,
		selector_budgets=[_make_selector_budget_summary(name=CANONICAL_DENSE)],
	)
	console = Console(record=True, width=220)

	_print_summary(console, summary)

	output = console.export_text()
	assert "2wikimultihop summary" in output
	assert "recall" in output
	assert "prec_nonempty" in output
	assert "f1_nonempty" in output
	assert "f1_all" in output
	assert "sel_mass" in output
	assert "util" in output
	assert "empty" in output
	assert "ans_em" not in output
	assert "ans_f1" not in output
	assert "selector legend" not in output
	assert "selector health" not in output
	assert "2wikimultihop selector health" not in output
	assert "columns:" not in output


def test_print_summary_renders_selector_health_table_for_llm_rows():
	summary = ExperimentSummary(
		dataset_name="2wikimultihop",
		total_cases=1,
		selector_budgets=[
			_make_selector_budget_summary(
				name=CANONICAL_LLM,
				selector_provider="anthropic",
				selector_model="claude-haiku-4-5-20251001",
				avg_selector_total_tokens=1736.75,
				avg_selector_runtime_s=5.173,
				avg_selector_llm_calls=1.45,
				avg_selector_fallback_rate=0.0,
				avg_selector_parse_failure_rate=0.0,
			),
		],
	)
	console = Console(record=True, width=220)

	_print_summary(console, summary)

	output = console.export_text()
	assert "2wikimultihop summary" in output
	assert "2wikimultihop selector health" in output
	assert "llm_toks" in output
	assert "sel_calls" in output
	assert "fallback" in output
	assert "parse_fail" in output
	assert "top_1_seed__lexical_overlap__hop_2__single_path_walk__" in output
	assert "@anthropic" in output
	assert "claude-haiku-4-5-20251001" in output
	assert "columns:" not in output


def test_live_dashboard_renderable_includes_status_summary_health_and_logs():
	summary = ExperimentSummary(
		dataset_name="2wikimultihop",
		total_cases=2,
		selector_budgets=[
			_make_selector_budget_summary(
				name=CANONICAL_LLM,
				selector_provider="anthropic",
				selector_model="claude-haiku-4-5-20251001",
				avg_selector_total_tokens=1736.75,
				avg_selector_runtime_s=5.173,
				avg_selector_llm_calls=1.45,
				avg_selector_fallback_rate=0.0,
				avg_selector_parse_failure_rate=0.0,
			),
		],
	)
	state = _LiveDashboardState(
		command_label="run-2wiki-store",
		split="dev",
		dataset_name="2wikimultihop",
		phase="evaluating",
		total_cases=2,
		completed_cases=1,
		total_selections=8,
		completed_selections=3,
		current_case_id="q2",
		current_query="Which city hosts the launch site?",
		current_selector_name=CANONICAL_LLM,
		current_budget_label="tokens-256",
		case_total_selectors=4,
		case_completed_selectors=1,
		case_total_selections=8,
		case_completed_selections=3,
		summary=summary,
	)
	progress_state = DashboardProgressState()
	task_id = progress_state.add_task(
		"evaluate store-backed 2wiki cases", total=2, detail="q2"
	)
	progress_state.update(task_id, completed=1)
	log_buffer = DashboardLogBuffer()
	log_buffer.append(
		DashboardLogEntry(
			rendered="[12:00:00] INFO hypercorpus.experiments: evaluating q2",
			logger_name="hypercorpus.experiments",
			level_name="INFO",
			levelno=20,
		)
	)
	log_buffer.append(
		DashboardLogEntry(
			rendered="[12:00:01] WARNING urllib3.connectionpool: retrying request",
			logger_name="urllib3.connectionpool",
			level_name="WARNING",
			levelno=30,
		)
	)

	console = Console(record=True, width=220)
	console.print(
		_ExperimentDashboardRenderable(
			state=state,
			progress_state=progress_state,
			log_buffer=log_buffer,
		)
	)

	output = console.export_text()
	assert "2wikimultihop live status" in output
	assert "run-2wiki-store" in output
	assert "2wikimultihop [dev]" in output
	assert "selections" in output
	assert "3/8" in output
	assert "case selectors" in output
	assert "1/4" in output
	assert "case selections" in output
	assert "evaluate store-backed 2wiki cases" in output
	assert "tokens-256" in output
	assert "2wikimultihop summary" in output
	assert "2wikimultihop selector health" in output
	assert "log tail" in output
	assert "evaluating q2" in output
	assert "retrying request" in output


def test_run_with_optional_dashboard_keeps_final_panel(monkeypatch):
	captured: dict[str, object] = {}

	class FakeLive:
		def __init__(self, renderable, *, console, refresh_per_second, transient):
			captured["renderable"] = renderable
			captured["console"] = console
			captured["refresh_per_second"] = refresh_per_second
			captured["transient"] = transient

		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, tb):
			return False

	monkeypatch.setattr("hypercorpus_cli.experiments.Live", FakeLive)

	console = Console(force_terminal=True, record=True, width=120)

	result = _run_with_optional_dashboard(
		console=console,
		command_label="run-2wiki-store",
		split="dev",
		runner=lambda observer: "ok",
	)

	assert result == "ok"
	assert captured["transient"] is False
