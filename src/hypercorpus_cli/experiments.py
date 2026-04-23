from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
import time

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from rich.console import Group

from hypercorpus.experiments import (
	ExperimentProgressUpdate,
	ExperimentSummary,
	study_preset_choices_help,
	budget_ratio_choices_help,
	merge_hotpotqa_results,
	merge_iirc_results,
	merge_musique_results,
	merge_2wiki_results,
	parse_budget_ratios,
	parse_token_budgets,
	parse_selector_names,
	run_docs_experiment,
	run_hotpotqa_experiment,
	run_iirc_experiment,
	run_iirc_store_experiment,
	run_musique_experiment,
	run_musique_store_experiment,
	run_hotpotqa_store_experiment,
	run_2wiki_experiment,
	run_2wiki_store_experiment,
	selector_choices_help,
	selector_preset_choices_help,
	token_budget_choices_help,
)
from hypercorpus.baselines import EXTERNAL_MDR_SELECTOR_NAME
from hypercorpus.logging import (
	DashboardLogBuffer,
	DashboardProgressState,
	dashboard_session,
)
from hypercorpus.reports import export_report_bundle_from_file

experiments_app = typer.Typer(
	name="hypercorpus experiments",
	help="hypercorpus experiment runners",
	add_completion=False,
)

_SELECTOR_PRESET_HELP = (
	f"Selector preset override when --selectors is omitted. Choices: {selector_preset_choices_help()}; "
	"paper_recommended uses the configured LLM backend defaults for its LLM selector; "
	"paper_recommended_local avoids LLM selector config. --selectors takes precedence."
)

_STUDY_PRESET_HELP = (
	f"Higher-level local study preset. Choices: {study_preset_choices_help()}; "
	"--selectors, --selector-preset, --token-budgets, and --budget-ratios override study defaults."
)


@dataclass(slots=True)
class _LiveDashboardState:
	command_label: str
	split: str | None = None
	started_at: float = field(default_factory=time.monotonic)
	dataset_name: str = "experiment"
	phase: str = "loading"
	total_cases: int | None = None
	completed_cases: int = 0
	total_selections: int | None = None
	completed_selections: int | None = None
	current_case_id: str | None = None
	current_query: str | None = None
	current_selector_name: str | None = None
	current_budget_label: str | None = None
	case_total_selectors: int | None = None
	case_completed_selectors: int | None = None
	case_total_selections: int | None = None
	case_completed_selections: int | None = None
	summary: ExperimentSummary | None = None

	def apply(self, update: ExperimentProgressUpdate) -> None:
		self.dataset_name = update.dataset_name
		self.phase = update.phase
		self.total_cases = update.total_cases
		self.completed_cases = update.completed_cases
		self.total_selections = update.total_selections
		self.completed_selections = update.completed_selections
		self.current_case_id = update.current_case_id
		self.current_query = update.current_query
		self.current_selector_name = update.current_selector_name
		self.current_budget_label = update.current_budget_label
		self.case_total_selectors = update.case_total_selectors
		self.case_completed_selectors = update.case_completed_selectors
		self.case_total_selections = update.case_total_selections
		self.case_completed_selections = update.case_completed_selections
		if update.summary is not None:
			self.summary = update.summary

	@property
	def elapsed_s(self) -> float:
		return max(0.0, time.monotonic() - self.started_at)

	@property
	def throughput(self) -> float:
		"""Return throughput in selections/s (falls back to cases/s)."""
		if self.elapsed_s <= 0:
			return 0.0
		if self.completed_selections is not None and self.completed_selections > 0:
			return self.completed_selections / self.elapsed_s
		return self.completed_cases / self.elapsed_s

	@property
	def eta_s(self) -> float | None:
		"""Estimated seconds remaining based on selection throughput."""
		if (
			self.total_selections is None
			or self.completed_selections is None
			or self.completed_selections <= 0
		):
			return None
		remaining = self.total_selections - self.completed_selections
		rate = self.completed_selections / self.elapsed_s
		if rate <= 0:
			return None
		return remaining / rate


class _ExperimentDashboardRenderable:
	def __init__(
		self,
		*,
		state: _LiveDashboardState,
		progress_state: DashboardProgressState,
		log_buffer: DashboardLogBuffer,
	) -> None:
		self.state = state
		self.progress_state = progress_state
		self.log_buffer = log_buffer

	def __rich_console__(self, console, options):
		from rich.console import RenderableType

		renderables: list[RenderableType] = [
			_build_dashboard_status_panel(self.state, self.progress_state),
			_build_summary_renderable(
				self.state.summary, title=f"{self.state.dataset_name} summary"
			),
		]
		health = _build_selector_health_renderable(
			self.state.summary,
			title=f"{self.state.dataset_name} selector health",
		)
		if health is not None:
			renderables.append(health)
		renderables.append(_build_log_panel(self.log_buffer))
		yield Group(*renderables)


def _resolve_budget_options(
	*, token_budgets: str | None, budget_ratios: str | None
) -> tuple[list[int] | None, list[float] | None]:
	if token_budgets is not None and budget_ratios is not None:
		raise ValueError("Specify either --token-budgets or --budget-ratios, not both.")
	return parse_token_budgets(token_budgets), parse_budget_ratios(budget_ratios)


def _run_with_optional_dashboard(
	*,
	console: Console,
	command_label: str,
	split: str | None,
	runner,
):
	if not console.is_terminal:
		return runner(None)

	state = _LiveDashboardState(command_label=command_label, split=split)
	progress_state = DashboardProgressState()
	log_buffer = DashboardLogBuffer()
	renderable = _ExperimentDashboardRenderable(
		state=state,
		progress_state=progress_state,
		log_buffer=log_buffer,
	)

	def _observer(update: ExperimentProgressUpdate) -> None:
		state.apply(update)

	with dashboard_session(log_buffer=log_buffer, progress_state=progress_state):
		with Live(renderable, console=console, refresh_per_second=4, transient=False):
			return runner(_observer)


def _print_direct_outputs(
	*, console: Console, output: Path, export_graphrag_inputs: bool
) -> None:
	console.print(f"results.jsonl -> {output / 'results.jsonl'}")
	console.print(f"selector_logs.jsonl -> {output / 'selector_logs.jsonl'}")
	console.print(f"summary.json -> {output / 'summary.json'}")
	console.print(f"summary_rows.csv -> {output / 'summary_rows.csv'}")
	console.print(
		f"study_comparison_rows.csv -> {output / 'study_comparison_rows.csv'}"
	)
	console.print(
		f"subset_comparison_rows.csv -> {output / 'subset_comparison_rows.csv'}"
	)
	console.print(f"run_manifest.json -> {output / 'run_manifest.json'}")
	console.print(f"evaluated_case_ids.txt -> {output / 'evaluated_case_ids.txt'}")
	if export_graphrag_inputs:
		console.print(f"graphrag_inputs -> {output / 'graphrag_inputs'}")


def _print_store_outputs(
	*, console: Console, chunk_dir: Path, export_graphrag_inputs: bool
) -> None:
	console.print(f"chunk_dir -> {chunk_dir}")
	console.print(f"results.jsonl -> {chunk_dir / 'results.jsonl'}")
	console.print(f"selector_logs.jsonl -> {chunk_dir / 'selector_logs.jsonl'}")
	console.print(f"summary.json -> {chunk_dir / 'summary.json'}")
	console.print(f"summary_rows.csv -> {chunk_dir / 'summary_rows.csv'}")
	console.print(
		f"study_comparison_rows.csv -> {chunk_dir / 'study_comparison_rows.csv'}"
	)
	console.print(
		f"subset_comparison_rows.csv -> {chunk_dir / 'subset_comparison_rows.csv'}"
	)
	console.print(f"run_manifest.json -> {chunk_dir / 'run_manifest.json'}")
	console.print(f"evaluated_case_ids.txt -> {chunk_dir / 'evaluated_case_ids.txt'}")
	if export_graphrag_inputs:
		console.print(f"graphrag_inputs -> {chunk_dir / 'graphrag_inputs'}")


@experiments_app.command("run-2wiki")
def run_2wiki(
	questions: Path = typer.Option(
		...,
		"--questions",
		exists=True,
		dir_okay=False,
		help="Path to 2Wiki questions JSON",
	),
	graph_records: Path = typer.Option(
		...,
		"--graph-records",
		exists=True,
		dir_okay=False,
		help="Path to 2Wiki paragraph hyperlink JSONL",
	),
	output: Path = typer.Option(
		..., "--output", file_okay=False, help="Output directory"
	),
	limit: int | None = typer.Option(
		None, "--limit", min=1, help="Optional number of questions to evaluate"
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
	)
	_, summary = _run_with_optional_dashboard(
		console=console,
		command_label="run-2wiki",
		split=None,
		runner=lambda progress_observer: run_2wiki_experiment(
			questions_path=questions,
			graph_records_path=graph_records,
			output_dir=output,
			limit=limit,
			case_ids_file=case_ids_file,
			selector_names=parse_selector_names(selectors),
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
		),
	)
	_print_summary(console, summary)
	_print_direct_outputs(
		console=console, output=output, export_graphrag_inputs=export_graphrag_inputs
	)


@experiments_app.command("run-iirc")
def run_iirc(
	questions: Path = typer.Option(
		...,
		"--questions",
		exists=True,
		dir_okay=False,
		help="Path to IIRC-style questions JSON/JSONL",
	),
	graph_records: Path = typer.Option(
		...,
		"--graph-records",
		exists=True,
		dir_okay=False,
		help="Path to normalized IIRC graph JSON/JSONL",
	),
	output: Path = typer.Option(
		..., "--output", file_okay=False, help="Output directory"
	),
	limit: int | None = typer.Option(
		None, "--limit", min=1, help="Optional number of questions to evaluate"
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
	)
	_, summary = _run_with_optional_dashboard(
		console=console,
		command_label="run-iirc",
		split=None,
		runner=lambda progress_observer: run_iirc_experiment(
			questions_path=questions,
			graph_records_path=graph_records,
			output_dir=output,
			limit=limit,
			case_ids_file=case_ids_file,
			selector_names=parse_selector_names(selectors),
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
		),
	)
	_print_summary(console, summary)
	_print_direct_outputs(
		console=console, output=output, export_graphrag_inputs=export_graphrag_inputs
	)


@experiments_app.command("run-musique")
def run_musique(
	questions: Path = typer.Option(
		...,
		"--questions",
		exists=True,
		dir_okay=False,
		help="Path to MuSiQue questions JSON/JSONL",
	),
	graph_records: Path = typer.Option(
		...,
		"--graph-records",
		exists=True,
		dir_okay=False,
		help="Path to normalized MuSiQue graph JSON/JSONL",
	),
	output: Path = typer.Option(
		..., "--output", file_okay=False, help="Output directory"
	),
	limit: int | None = typer.Option(
		None, "--limit", min=1, help="Optional number of questions to evaluate"
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
	)
	_, summary = _run_with_optional_dashboard(
		console=console,
		command_label="run-musique",
		split=None,
		runner=lambda progress_observer: run_musique_experiment(
			questions_path=questions,
			graph_records_path=graph_records,
			output_dir=output,
			limit=limit,
			case_ids_file=case_ids_file,
			selector_names=parse_selector_names(selectors),
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
		),
	)
	_print_summary(console, summary)
	_print_direct_outputs(
		console=console, output=output, export_graphrag_inputs=export_graphrag_inputs
	)


@experiments_app.command("run-hotpotqa")
def run_hotpotqa(
	questions: Path = typer.Option(
		...,
		"--questions",
		exists=True,
		dir_okay=False,
		help="Path to HotpotQA questions JSON/JSONL",
	),
	output: Path = typer.Option(
		..., "--output", file_okay=False, help="Output directory"
	),
	variant: str = typer.Option(
		..., "--variant", help="HotpotQA variant: distractor or fullwiki"
	),
	graph_records: Path | None = typer.Option(
		None,
		"--graph-records",
		exists=True,
		dir_okay=False,
		help="Path to normalized HotpotQA graph JSON/JSONL for fullwiki runs",
	),
	limit: int | None = typer.Option(
		None, "--limit", min=1, help="Optional number of questions to evaluate"
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
) -> None:
	if variant not in {"distractor", "fullwiki"}:
		raise typer.BadParameter("--variant must be one of: distractor, fullwiki")
	if variant == "fullwiki" and graph_records is None:
		raise typer.BadParameter("--graph-records is required when --variant fullwiki")

	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
	)
	_, summary = _run_with_optional_dashboard(
		console=console,
		command_label="run-hotpotqa",
		split=None,
		runner=lambda progress_observer: run_hotpotqa_experiment(
			questions_path=questions,
			output_dir=output,
			variant=cast("Literal['distractor', 'fullwiki']", variant),
			graph_records_path=graph_records,
			limit=limit,
			case_ids_file=case_ids_file,
			selector_names=parse_selector_names(selectors),
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
		),
	)
	_print_summary(console, summary)
	_print_direct_outputs(
		console=console, output=output, export_graphrag_inputs=export_graphrag_inputs
	)


@experiments_app.command("run-docs")
def run_docs(
	questions: Path = typer.Option(
		...,
		"--questions",
		exists=True,
		dir_okay=False,
		help="Path to documentation QA questions JSON/JSONL",
	),
	docs_source: Path = typer.Option(
		...,
		"--docs-source",
		exists=True,
		help="Documentation HTML root or normalized graph JSON/JSONL",
	),
	output: Path = typer.Option(
		..., "--output", file_okay=False, help="Output directory"
	),
	dataset_name: str = typer.Option(
		"docs", "--dataset-name", help="Dataset label for the experiment summary"
	),
	limit: int | None = typer.Option(
		None, "--limit", min=1, help="Optional number of questions to evaluate"
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
	)
	_, summary = _run_with_optional_dashboard(
		console=console,
		command_label="run-docs",
		split=None,
		runner=lambda progress_observer: run_docs_experiment(
			questions_path=questions,
			docs_source=docs_source,
			output_dir=output,
			dataset_name=dataset_name,
			limit=limit,
			case_ids_file=case_ids_file,
			selector_names=parse_selector_names(selectors),
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
		),
	)
	_print_summary(console, summary)
	_print_direct_outputs(
		console=console, output=output, export_graphrag_inputs=export_graphrag_inputs
	)


@experiments_app.command("run-2wiki-store")
def run_2wiki_store(
	store: str = typer.Option(
		..., "--store", help="Path or s3:// URI to a prepared 2Wiki store"
	),
	exp_name: str = typer.Option(
		..., "--exp-name", help="Experiment name under the output root"
	),
	output_root: Path = typer.Option(
		Path("runs"),
		"--output-root",
		file_okay=False,
		help="Root directory for chunk outputs",
	),
	split: str = typer.Option(
		"dev", "--split", help="Question split inside the prepared store"
	),
	cache_dir: Path | None = typer.Option(
		None,
		"--cache-dir",
		file_okay=False,
		help="Local cache directory for remote stores",
	),
	limit: int | None = typer.Option(
		None,
		"--limit",
		min=1,
		help="Optional total number of questions to consider before slicing",
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	case_start: int = typer.Option(
		0, "--case-start", min=0, help="Start offset after split selection"
	),
	case_limit: int | None = typer.Option(
		None, "--case-limit", min=1, help="Maximum number of questions for this run"
	),
	chunk_size: int | None = typer.Option(
		None, "--chunk-size", min=1, help="Questions per chunk"
	),
	chunk_index: int | None = typer.Option(
		None, "--chunk-index", min=0, help="Chunk index to run"
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
	link_context_mask: str | None = typer.Option(
		None,
		"--link-context-mask",
		help="Ablation mask mode: none, mask_anchor, mask_sentence, or mask_both",
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
	)
	_evaluations, summary, chunk_dir = _run_with_optional_dashboard(
		console=console,
		command_label="run-2wiki-store",
		split=split,
		runner=lambda progress_observer: run_2wiki_store_experiment(
			store_uri=store,
			output_root=output_root,
			exp_name=exp_name,
			split=split,
			cache_dir=cache_dir,
			limit=limit,
			case_ids_file=case_ids_file,
			case_start=case_start,
			case_limit=case_limit,
			chunk_size=chunk_size,
			chunk_index=chunk_index,
			selector_names=parse_selector_names(selectors),
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
			link_context_mask=link_context_mask,
		),
	)
	_print_summary(console, summary)
	_print_store_outputs(
		console=console,
		chunk_dir=chunk_dir,
		export_graphrag_inputs=export_graphrag_inputs,
	)


def _run_store_command(
	*,
	console: Console,
	command_label: str,
	split: str,
	runner,
	export_graphrag_inputs: bool,
) -> None:
	_evaluations, summary, chunk_dir = _run_with_optional_dashboard(
		console=console,
		command_label=command_label,
		split=split,
		runner=runner,
	)
	_print_summary(console, summary)
	_print_store_outputs(
		console=console,
		chunk_dir=chunk_dir,
		export_graphrag_inputs=export_graphrag_inputs,
	)


def _validate_external_mdr_requirements(
	*,
	selector_names: list[str] | None,
	mdr_artifact_manifest: Path | None,
) -> None:
	if selector_names is None:
		return
	if EXTERNAL_MDR_SELECTOR_NAME in selector_names and mdr_artifact_manifest is None:
		raise typer.BadParameter(
			f"{EXTERNAL_MDR_SELECTOR_NAME} requires --mdr-artifact-manifest.",
			param_hint="--mdr-artifact-manifest",
		)


@experiments_app.command("run-iirc-store")
def run_iirc_store(
	store: str = typer.Option(
		..., "--store", help="Path or s3:// URI to a prepared IIRC store"
	),
	exp_name: str = typer.Option(
		..., "--exp-name", help="Experiment name under the output root"
	),
	output_root: Path = typer.Option(
		Path("runs"),
		"--output-root",
		file_okay=False,
		help="Root directory for chunk outputs",
	),
	split: str = typer.Option(
		"dev", "--split", help="Question split inside the prepared store"
	),
	cache_dir: Path | None = typer.Option(
		None,
		"--cache-dir",
		file_okay=False,
		help="Local cache directory for remote stores",
	),
	limit: int | None = typer.Option(
		None,
		"--limit",
		min=1,
		help="Optional total number of questions to consider before slicing",
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	case_start: int = typer.Option(
		0, "--case-start", min=0, help="Start offset after split selection"
	),
	case_limit: int | None = typer.Option(
		None, "--case-limit", min=1, help="Maximum number of questions for this run"
	),
	chunk_size: int | None = typer.Option(
		None, "--chunk-size", min=1, help="Questions per chunk"
	),
	chunk_index: int | None = typer.Option(
		None, "--chunk-index", min=0, help="Chunk index to run"
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	mdr_home: Path | None = typer.Option(
		None,
		"--mdr-home",
		file_okay=False,
		help="Path to the pinned official MDR checkout. Defaults to ./baselines/mdr when present.",
	),
	mdr_artifact_manifest: Path | None = typer.Option(
		None,
		"--mdr-artifact-manifest",
		exists=True,
		dir_okay=False,
		help="Artifact manifest produced by `hypercorpus-cli baselines build-mdr-index`",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
	link_context_mask: str | None = typer.Option(
		None,
		"--link-context-mask",
		help="Ablation mask mode: none, mask_anchor, mask_sentence, or mask_both",
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets, budget_ratios=budget_ratios
	)
	resolved_selector_names = parse_selector_names(selectors)
	_validate_external_mdr_requirements(
		selector_names=resolved_selector_names,
		mdr_artifact_manifest=mdr_artifact_manifest,
	)
	_run_store_command(
		console=console,
		command_label="run-iirc-store",
		split=split,
		export_graphrag_inputs=export_graphrag_inputs,
		runner=lambda progress_observer: run_iirc_store_experiment(
			store_uri=store,
			output_root=output_root,
			exp_name=exp_name,
			split=split,
			cache_dir=cache_dir,
			limit=limit,
			case_ids_file=case_ids_file,
			case_start=case_start,
			case_limit=case_limit,
			chunk_size=chunk_size,
			chunk_index=chunk_index,
			selector_names=resolved_selector_names,
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			mdr_home=mdr_home,
			mdr_artifact_manifest=mdr_artifact_manifest,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
			link_context_mask=link_context_mask,
		),
	)


@experiments_app.command("run-musique-store")
def run_musique_store(
	store: str = typer.Option(
		..., "--store", help="Path or s3:// URI to a prepared MuSiQue store"
	),
	exp_name: str = typer.Option(
		..., "--exp-name", help="Experiment name under the output root"
	),
	output_root: Path = typer.Option(
		Path("runs"),
		"--output-root",
		file_okay=False,
		help="Root directory for chunk outputs",
	),
	split: str = typer.Option(
		"dev", "--split", help="Question split inside the prepared store"
	),
	cache_dir: Path | None = typer.Option(
		None,
		"--cache-dir",
		file_okay=False,
		help="Local cache directory for remote stores",
	),
	limit: int | None = typer.Option(
		None,
		"--limit",
		min=1,
		help="Optional total number of questions to consider before slicing",
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	case_start: int = typer.Option(
		0, "--case-start", min=0, help="Start offset after split selection"
	),
	case_limit: int | None = typer.Option(
		None, "--case-limit", min=1, help="Maximum number of questions for this run"
	),
	chunk_size: int | None = typer.Option(
		None, "--chunk-size", min=1, help="Questions per chunk"
	),
	chunk_index: int | None = typer.Option(
		None, "--chunk-index", min=0, help="Chunk index to run"
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	mdr_home: Path | None = typer.Option(
		None,
		"--mdr-home",
		file_okay=False,
		help="Path to the pinned official MDR checkout. Defaults to ./baselines/mdr when present.",
	),
	mdr_artifact_manifest: Path | None = typer.Option(
		None,
		"--mdr-artifact-manifest",
		exists=True,
		dir_okay=False,
		help="Artifact manifest produced by `hypercorpus-cli baselines build-mdr-index`",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
	link_context_mask: str | None = typer.Option(
		None,
		"--link-context-mask",
		help="Ablation mask mode: none, mask_anchor, mask_sentence, or mask_both",
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets, budget_ratios=budget_ratios
	)
	resolved_selector_names = parse_selector_names(selectors)
	_validate_external_mdr_requirements(
		selector_names=resolved_selector_names,
		mdr_artifact_manifest=mdr_artifact_manifest,
	)
	_run_store_command(
		console=console,
		command_label="run-musique-store",
		split=split,
		export_graphrag_inputs=export_graphrag_inputs,
		runner=lambda progress_observer: run_musique_store_experiment(
			store_uri=store,
			output_root=output_root,
			exp_name=exp_name,
			split=split,
			cache_dir=cache_dir,
			limit=limit,
			case_ids_file=case_ids_file,
			case_start=case_start,
			case_limit=case_limit,
			chunk_size=chunk_size,
			chunk_index=chunk_index,
			selector_names=resolved_selector_names,
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			mdr_home=mdr_home,
			mdr_artifact_manifest=mdr_artifact_manifest,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
			link_context_mask=link_context_mask,
		),
	)


@experiments_app.command("run-hotpotqa-store")
def run_hotpotqa_store(
	store: str = typer.Option(
		..., "--store", help="Path or s3:// URI to a prepared HotpotQA fullwiki store"
	),
	exp_name: str = typer.Option(
		..., "--exp-name", help="Experiment name under the output root"
	),
	output_root: Path = typer.Option(
		Path("runs"),
		"--output-root",
		file_okay=False,
		help="Root directory for chunk outputs",
	),
	split: str = typer.Option(
		"dev", "--split", help="Question split inside the prepared store"
	),
	cache_dir: Path | None = typer.Option(
		None,
		"--cache-dir",
		file_okay=False,
		help="Local cache directory for remote stores",
	),
	limit: int | None = typer.Option(
		None,
		"--limit",
		min=1,
		help="Optional total number of questions to consider before slicing",
	),
	case_ids_file: Path | None = typer.Option(
		None,
		"--case-ids-file",
		exists=True,
		dir_okay=False,
		help="Optional newline-delimited case_id file for reproducible samples",
	),
	case_start: int = typer.Option(
		0, "--case-start", min=0, help="Start offset after split selection"
	),
	case_limit: int | None = typer.Option(
		None, "--case-limit", min=1, help="Maximum number of questions for this run"
	),
	chunk_size: int | None = typer.Option(
		None, "--chunk-size", min=1, help="Questions per chunk"
	),
	chunk_index: int | None = typer.Option(
		None, "--chunk-index", min=0, help="Chunk index to run"
	),
	selectors: str | None = typer.Option(
		None,
		"--selectors",
		help=f"Comma-separated selector names. Choices: {selector_choices_help()}",
	),
	selector_preset: str | None = typer.Option(
		None, "--selector-preset", help=_SELECTOR_PRESET_HELP
	),
	study_preset: str | None = typer.Option(
		None, "--study-preset", help=_STUDY_PRESET_HELP
	),
	token_budgets: str | None = typer.Option(
		None,
		"--token-budgets",
		help=f"Comma-separated token budgets. Default: {token_budget_choices_help()}",
	),
	budget_ratios: str | None = typer.Option(
		None,
		"--budget-ratios",
		help=f"Comma-separated token budget ratios. Default: {budget_ratio_choices_help()}",
	),
	selector_provider: str | None = typer.Option(
		None,
		"--selector-provider",
		help="Selector LLM provider: copilot, openai, anthropic, or gemini",
	),
	selector_model: str | None = typer.Option(
		None, "--selector-model", help="Selector LLM model name"
	),
	selector_api_key_env: str | None = typer.Option(
		None,
		"--selector-api-key-env",
		help="Env var containing the selector LLM API key",
	),
	selector_base_url: str | None = typer.Option(
		None,
		"--selector-base-url",
		help="Optional selector base URL override for OpenAI-compatible providers",
	),
	selector_openai_api_mode: str | None = typer.Option(
		None,
		"--selector-openai-api-mode",
		help="OpenAI selector transport: chat_completions, responses, azure_foundry_chat_completions, or github_models_chat_completions",
	),
	selector_cache_path: Path | None = typer.Option(
		None,
		"--selector-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for selector LLM outputs",
	),
	mdr_home: Path | None = typer.Option(
		None,
		"--mdr-home",
		file_okay=False,
		help="Path to the pinned official MDR checkout. Defaults to ./baselines/mdr when present.",
	),
	mdr_artifact_manifest: Path | None = typer.Option(
		None,
		"--mdr-artifact-manifest",
		exists=True,
		dir_okay=False,
		help="Artifact manifest produced by `hypercorpus-cli baselines build-mdr-index`",
	),
	sentence_transformer_model: str | None = typer.Option(
		None,
		"--sentence-transformer-model",
		help="Local sentence-transformer model name for seed/scorer retrieval",
	),
	sentence_transformer_cache_path: Path | None = typer.Option(
		None,
		"--sentence-transformer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional SQLite cache path for sentence-transformer embeddings",
	),
	sentence_transformer_device: str | None = typer.Option(
		None,
		"--sentence-transformer-device",
		help="Optional sentence-transformer device override, for example cpu or mps",
	),
	with_e2e: bool = typer.Option(
		False, "--with-e2e/--no-e2e", help="Attach secondary end-to-end QA metrics"
	),
	answerer: str = typer.Option(
		"heuristic", "--answerer", help="Answerer mode: heuristic or llm_fixed"
	),
	answer_provider: str | None = typer.Option(
		None,
		"--answer-provider",
		help="Fixed reader provider: copilot or openai",
	),
	answer_model: str | None = typer.Option(
		None, "--answer-model", help="Fixed reader model name"
	),
	answer_api_key_env: str | None = typer.Option(
		None,
		"--answer-api-key-env",
		help="Env var containing the reader API key",
	),
	answer_base_url: str | None = typer.Option(
		None, "--answer-base-url", help="Optional reader base URL override"
	),
	answer_cache_path: Path | None = typer.Option(
		None,
		"--answer-cache-path",
		file_okay=True,
		dir_okay=False,
		help="Optional JSONL cache path for fixed reader outputs",
	),
	export_graphrag_inputs: bool = typer.Option(
		True,
		"--export-graphrag-inputs/--no-export-graphrag-inputs",
		help="Write GraphRAG-compatible CSV slices for each case/selector/budget",
	),
	resume: bool = typer.Option(
		False, "--resume", help="Resume an interrupted run from existing checkpoints"
	),
	restart: bool = typer.Option(
		False, "--restart", help="Discard existing run artifacts and start over"
	),
	link_context_mask: str | None = typer.Option(
		None,
		"--link-context-mask",
		help="Ablation mask mode: none, mask_anchor, mask_sentence, or mask_both",
	),
) -> None:
	console = Console()
	resolved_token_budgets, resolved_budget_ratios = _resolve_budget_options(
		token_budgets=token_budgets, budget_ratios=budget_ratios
	)
	resolved_selector_names = parse_selector_names(selectors)
	_validate_external_mdr_requirements(
		selector_names=resolved_selector_names,
		mdr_artifact_manifest=mdr_artifact_manifest,
	)
	_run_store_command(
		console=console,
		command_label="run-hotpotqa-store",
		split=split,
		export_graphrag_inputs=export_graphrag_inputs,
		runner=lambda progress_observer: run_hotpotqa_store_experiment(
			store_uri=store,
			output_root=output_root,
			exp_name=exp_name,
			split=split,
			cache_dir=cache_dir,
			limit=limit,
			case_ids_file=case_ids_file,
			case_start=case_start,
			case_limit=case_limit,
			chunk_size=chunk_size,
			chunk_index=chunk_index,
			selector_names=resolved_selector_names,
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=resolved_token_budgets,
			budget_ratios=resolved_budget_ratios,
			selector_provider=selector_provider,
			selector_model=selector_model,
			selector_api_key_env=selector_api_key_env,
			selector_base_url=selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode,
			selector_cache_path=selector_cache_path,
			mdr_home=mdr_home,
			mdr_artifact_manifest=mdr_artifact_manifest,
			sentence_transformer_model=sentence_transformer_model,
			sentence_transformer_cache_path=sentence_transformer_cache_path,
			sentence_transformer_device=sentence_transformer_device,
			with_e2e=with_e2e,
			answerer_mode=answerer,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
			link_context_mask=link_context_mask,
		),
	)


@experiments_app.command("merge-2wiki-results")
def merge_2wiki_store_results(
	run_dir: Path = typer.Option(
		..., "--run-dir", file_okay=False, help="Run directory containing chunks/"
	),
	output_dir: Path | None = typer.Option(
		None,
		"--output-dir",
		file_okay=False,
		help="Optional output directory for merged files",
	),
) -> None:
	console = Console()
	summary, missing_chunks = merge_2wiki_results(
		run_dir=run_dir, output_dir=output_dir
	)
	merged_dir = output_dir or run_dir
	_print_summary(console, summary)
	console.print(f"merged results.jsonl -> {merged_dir / 'results.jsonl'}")
	console.print(f"merged summary.json -> {merged_dir / 'summary.json'}")
	console.print(f"merged summary_rows.csv -> {merged_dir / 'summary_rows.csv'}")
	console.print(
		f"merged study_comparison_rows.csv -> {merged_dir / 'study_comparison_rows.csv'}"
	)
	console.print(
		f"merged subset_comparison_rows.csv -> {merged_dir / 'subset_comparison_rows.csv'}"
	)
	console.print(f"merged run_manifest.json -> {merged_dir / 'run_manifest.json'}")
	console.print(
		f"merged evaluated_case_ids.txt -> {merged_dir / 'evaluated_case_ids.txt'}"
	)
	console.print(f"missing_chunks -> {missing_chunks if missing_chunks else '[]'}")


@experiments_app.command("merge-iirc-results")
def merge_iirc_store_results(
	run_dir: Path = typer.Option(
		..., "--run-dir", file_okay=False, help="Run directory containing chunks/"
	),
	output_dir: Path | None = typer.Option(
		None,
		"--output-dir",
		file_okay=False,
		help="Optional output directory for merged files",
	),
) -> None:
	console = Console()
	summary, missing_chunks = merge_iirc_results(run_dir=run_dir, output_dir=output_dir)
	merged_dir = output_dir or run_dir
	_print_summary(console, summary)
	console.print(f"merged results.jsonl -> {merged_dir / 'results.jsonl'}")
	console.print(f"merged summary.json -> {merged_dir / 'summary.json'}")
	console.print(f"merged summary_rows.csv -> {merged_dir / 'summary_rows.csv'}")
	console.print(
		f"merged study_comparison_rows.csv -> {merged_dir / 'study_comparison_rows.csv'}"
	)
	console.print(
		f"merged subset_comparison_rows.csv -> {merged_dir / 'subset_comparison_rows.csv'}"
	)
	console.print(f"merged run_manifest.json -> {merged_dir / 'run_manifest.json'}")
	console.print(
		f"merged evaluated_case_ids.txt -> {merged_dir / 'evaluated_case_ids.txt'}"
	)
	console.print(f"missing_chunks -> {missing_chunks if missing_chunks else '[]'}")


@experiments_app.command("merge-musique-results")
def merge_musique_store_results(
	run_dir: Path = typer.Option(
		..., "--run-dir", file_okay=False, help="Run directory containing chunks/"
	),
	output_dir: Path | None = typer.Option(
		None,
		"--output-dir",
		file_okay=False,
		help="Optional output directory for merged files",
	),
) -> None:
	console = Console()
	summary, missing_chunks = merge_musique_results(
		run_dir=run_dir, output_dir=output_dir
	)
	merged_dir = output_dir or run_dir
	_print_summary(console, summary)
	console.print(f"merged results.jsonl -> {merged_dir / 'results.jsonl'}")
	console.print(f"merged summary.json -> {merged_dir / 'summary.json'}")
	console.print(f"merged summary_rows.csv -> {merged_dir / 'summary_rows.csv'}")
	console.print(
		f"merged study_comparison_rows.csv -> {merged_dir / 'study_comparison_rows.csv'}"
	)
	console.print(
		f"merged subset_comparison_rows.csv -> {merged_dir / 'subset_comparison_rows.csv'}"
	)
	console.print(f"merged run_manifest.json -> {merged_dir / 'run_manifest.json'}")
	console.print(
		f"merged evaluated_case_ids.txt -> {merged_dir / 'evaluated_case_ids.txt'}"
	)
	console.print(f"missing_chunks -> {missing_chunks if missing_chunks else '[]'}")


@experiments_app.command("merge-hotpotqa-results")
def merge_hotpotqa_store_results(
	run_dir: Path = typer.Option(
		..., "--run-dir", file_okay=False, help="Run directory containing chunks/"
	),
	output_dir: Path | None = typer.Option(
		None,
		"--output-dir",
		file_okay=False,
		help="Optional output directory for merged files",
	),
) -> None:
	console = Console()
	summary, missing_chunks = merge_hotpotqa_results(
		run_dir=run_dir, output_dir=output_dir
	)
	merged_dir = output_dir or run_dir
	_print_summary(console, summary)
	console.print(f"merged results.jsonl -> {merged_dir / 'results.jsonl'}")
	console.print(f"merged summary.json -> {merged_dir / 'summary.json'}")
	console.print(f"merged summary_rows.csv -> {merged_dir / 'summary_rows.csv'}")
	console.print(
		f"merged study_comparison_rows.csv -> {merged_dir / 'study_comparison_rows.csv'}"
	)
	console.print(
		f"merged subset_comparison_rows.csv -> {merged_dir / 'subset_comparison_rows.csv'}"
	)
	console.print(f"merged run_manifest.json -> {merged_dir / 'run_manifest.json'}")
	console.print(
		f"merged evaluated_case_ids.txt -> {merged_dir / 'evaluated_case_ids.txt'}"
	)
	console.print(f"missing_chunks -> {missing_chunks if missing_chunks else '[]'}")


@experiments_app.command("export-summary-report")
def export_summary_report_cli(
	summary: Path = typer.Option(
		...,
		"--summary",
		exists=True,
		dir_okay=False,
		help="Path to an experiment summary.json",
	),
	output: Path | None = typer.Option(
		None, "--output", dir_okay=False, help="Optional output CSV path"
	),
	comparison_output: Path | None = typer.Option(
		None,
		"--comparison-output",
		dir_okay=False,
		help="Optional output CSV path for study_comparison_rows.csv",
	),
	subset_output: Path | None = typer.Option(
		None,
		"--subset-output",
		dir_okay=False,
		help="Optional output CSV path for subset_comparison_rows.csv",
	),
) -> None:
	console = Console()
	bundle = export_report_bundle_from_file(
		summary,
		summary_rows_output_path=output,
		study_comparison_output_path=comparison_output,
		subset_comparison_output_path=subset_output,
	)
	console.print(f"summary_rows.csv -> {bundle.summary_rows_path}")
	console.print(f"study_comparison_rows.csv -> {bundle.study_comparison_rows_path}")
	console.print(f"subset_comparison_rows.csv -> {bundle.subset_comparison_rows_path}")


def _format_metric(value: float | None) -> str:
	if value is None:
		return "-"
	return f"{value:.3f}"


def _selector_label(row) -> str:
	if row.selector_provider and row.selector_model:
		return f"{row.name}@{row.selector_provider}:{row.selector_model}"
	return row.name


def _has_selector_health(summary) -> bool:
	if summary is None:
		return False
	return any(
		(row.avg_selector_total_tokens or 0) > 0
		or (row.avg_selector_llm_calls or 0) > 0
		or (row.avg_selector_fallback_rate or 0) > 0
		or (row.avg_selector_parse_failure_rate or 0) > 0
		for row in summary.selector_budgets
	)


def _build_main_summary_table(summary, *, title: str | None = None) -> Table:
	main_table = Table(title=title or f"{summary.dataset_name} summary")
	main_table.add_column("selector", overflow="fold", max_width=72)
	main_table.add_column("budget", justify="right", no_wrap=True)
	main_table.add_column("recall", justify="right", no_wrap=True)
	main_table.add_column("prec_nonempty", justify="right", no_wrap=True)
	main_table.add_column("f1_nonempty", justify="right", no_wrap=True)
	main_table.add_column("f1_all", justify="right", no_wrap=True)
	main_table.add_column("sel_mass", justify="right", no_wrap=True)
	main_table.add_column("util", justify="right", no_wrap=True)
	main_table.add_column("empty", justify="right", no_wrap=True)
	main_table.add_column("adhere", justify="right", no_wrap=True)
	main_table.add_column("rt_s", justify="right", no_wrap=True)

	for row in summary.selector_budgets:
		cells = [
			_selector_label(row),
			row.budget_label,
			_format_metric(row.avg_support_recall),
			_format_metric(row.avg_support_precision),
			_format_metric(row.avg_support_f1),
			_format_metric(row.avg_support_f1_zero_on_empty),
			_format_metric(row.avg_selected_corpus_mass),
			_format_metric(row.avg_budget_utilization),
			_format_metric(row.avg_empty_selection_rate),
			_format_metric(row.avg_budget_adherence),
			_format_metric(row.avg_selection_runtime_s),
		]
		main_table.add_row(*cells)

	return main_table


def _build_selector_health_table(summary, *, title: str | None = None) -> Table:
	health_table = Table(title=title or f"{summary.dataset_name} selector health")
	health_table.add_column("selector", overflow="fold", max_width=88)
	health_table.add_column("budget", justify="right", no_wrap=True)
	health_table.add_column("llm_toks", justify="right", no_wrap=True)
	health_table.add_column("sel_calls", justify="right", no_wrap=True)
	health_table.add_column("sel_rt_s", justify="right", no_wrap=True)
	health_table.add_column("fallback", justify="right", no_wrap=True)
	health_table.add_column("parse_fail", justify="right", no_wrap=True)

	for row in summary.selector_budgets:
		health_table.add_row(
			_selector_label(row),
			row.budget_label,
			_format_metric(row.avg_selector_total_tokens),
			_format_metric(row.avg_selector_llm_calls),
			_format_metric(row.avg_selector_runtime_s),
			_format_metric(row.avg_selector_fallback_rate),
			_format_metric(row.avg_selector_parse_failure_rate),
		)

	return health_table


def _build_summary_renderable(summary, *, title: str) -> Panel | Table:
	if summary is None or not summary.selector_budgets:
		return Panel("Waiting for first completed case...", title=title)
	return _build_main_summary_table(summary, title=title)


def _build_selector_health_renderable(summary, *, title: str) -> Panel | Table | None:
	if summary is None or not _has_selector_health(summary):
		return None
	return _build_selector_health_table(summary, title=title)


def _build_dashboard_status_panel(
	state: _LiveDashboardState,
	progress_state: DashboardProgressState,
) -> Panel:
	def _plain(value: str) -> Text:
		return Text(value)

	grid = Table.grid(expand=True)
	grid.add_column(style="cyan", ratio=1)
	grid.add_column(ratio=5)
	cases_value = (
		f"{state.completed_cases}/{state.total_cases}"
		if state.total_cases is not None
		else str(state.completed_cases)
	)
	grid.add_row("command", _plain(state.command_label))
	dataset_value = (
		state.dataset_name
		if state.split is None
		else f"{state.dataset_name} [{state.split}]"
	)
	grid.add_row("dataset", _plain(dataset_value))
	grid.add_row("phase", _plain(state.phase))
	grid.add_row("cases", _plain(cases_value))
	if state.total_selections is not None and state.completed_selections is not None:
		grid.add_row(
			"selections",
			_plain(f"{state.completed_selections}/{state.total_selections}"),
		)
	grid.add_row("elapsed", _plain(f"{state.elapsed_s:.1f}s"))
	tp = state.throughput
	unit = (
		"sel/s"
		if state.completed_selections and state.completed_selections > 0
		else "case/s"
	)
	if tp >= 0.01:
		tp_str = f"{tp:.2f} {unit}"
	elif tp > 0:
		min_unit = "sel/min" if unit == "sel/s" else "case/min"
		tp_str = f"{tp * 60:.1f} {min_unit}"
	else:
		tp_str = f"-- {unit}"
	grid.add_row("throughput", _plain(tp_str))
	eta = state.eta_s
	if eta is not None:
		if eta >= 3600:
			eta_str = f"{eta / 3600:.1f}h"
		elif eta >= 60:
			eta_str = f"{eta / 60:.1f}min"
		else:
			eta_str = f"{eta:.0f}s"
		grid.add_row("eta", _plain(eta_str))
	if state.current_case_id is not None:
		grid.add_row("current_case", _plain(state.current_case_id))
	if state.current_query:
		grid.add_row("query", _plain(state.current_query))
	if (
		state.case_total_selectors is not None
		and state.case_completed_selectors is not None
	):
		grid.add_row(
			"case selectors",
			_build_progress_cell(
				completed=state.case_completed_selectors,
				total=state.case_total_selectors,
				detail=state.current_selector_name,
			),
		)
	if (
		state.case_total_selections is not None
		and state.case_completed_selections is not None
	):
		selection_detail = None
		if (
			state.current_selector_name is not None
			and state.current_budget_label is not None
		):
			selection_detail = (
				f"{state.current_selector_name} @ {state.current_budget_label}"
			)
		elif state.current_selector_name is not None:
			selection_detail = state.current_selector_name
		elif state.current_budget_label is not None:
			selection_detail = state.current_budget_label
		grid.add_row(
			"case selections",
			_build_progress_cell(
				completed=state.case_completed_selections,
				total=state.case_total_selections,
				detail=selection_detail,
			),
		)

	task = progress_state.latest_task()
	if task is not None:
		if task.total is not None and task.total > 0:
			task_value = f"{task.description} ({task.completed:,.0f}/{task.total:,.0f}, {task.completed / task.total:.0%})"
		else:
			task_value = f"{task.description} ({task.completed:,.0f})"
		if task.detail:
			task_value = f"{task_value} [{task.detail}]"
		grid.add_row("task", _plain(task_value))

	return Panel(grid, title=f"{state.dataset_name} live status")


def _build_progress_cell(*, completed: int, total: int, detail: str | None) -> Table:
	normalized_total = max(total, 1)
	bar = ProgressBar(
		total=normalized_total, completed=min(max(completed, 0), normalized_total)
	)
	detail_table = Table.grid(expand=True)
	detail_table.add_column(ratio=5)
	detail_table.add_column(justify="right", no_wrap=True)
	detail_table.add_row(bar, Text(f"{completed}/{total}"))
	if detail:
		detail_table.add_row(Text(detail, overflow="ellipsis"), Text(""))
	return detail_table


def _build_log_panel(log_buffer: DashboardLogBuffer, *, max_lines: int = 12) -> Panel:
	entries = log_buffer.tail(limit=max_lines)
	if not entries:
		return Panel("No logs yet.", title="log tail")
	lines = "\n".join(entry.rendered for entry in entries)
	return Panel(Text(lines), title="log tail")


def _print_summary(console: Console, summary) -> None:
	console.print(_build_main_summary_table(summary))

	if not _has_selector_health(summary):
		return

	console.print(_build_selector_health_table(summary))
