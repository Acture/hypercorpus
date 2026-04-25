from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
import logging
import os
import signal
import shutil
from time import monotonic
import tomllib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence, cast

from hypercorpus.answering import (
	Answerer,
	AnswererProvider,
	LLMAnswerer,
	LLMAnswererConfig,
	SupportsAnswer,
)
from hypercorpus.copilot import is_copilot_model_name
from hypercorpus.datasets.common import (
	DatasetAdapter,
	coerce_question_type,
	load_json_records,
)
from hypercorpus.datasets.docs import DocumentationAdapter
from hypercorpus.datasets.hotpotqa import (
	HotpotQAAdapter,
	build_hotpotqa_distractor_graph_for_case,
	load_hotpotqa_questions,
)
from hypercorpus.datasets.iirc import IIRCAdapter
from hypercorpus.datasets.musique import MuSiQueAdapter
from hypercorpus.datasets.store import ShardedDocumentStore
from hypercorpus.datasets.twowiki import TwoWikiAdapter
from hypercorpus.graph import LinkContextMaskMode
from hypercorpus.eval import (
	DEFAULT_BUDGET_RATIOS,
	DEFAULT_TOKEN_BUDGETS,
	CaseEvaluation,
	EndToEndResult,
	EvaluationBudget,
	EvaluationCase,
	Evaluator,
	ExperimentSummary,
	SelectedCorpus,
	SelectedEdgeContext,
	SelectionMetrics,
	SelectionResult,
	SelectorBudgetSummary,
	_run_end_to_end,
	_selection_result_from_raw,
)
from hypercorpus.logging import create_progress, should_render_progress
from hypercorpus.resume import (
	CheckpointStore,
	HardStopRequested,
	InterruptController,
	RUN_STATE_VERSION,
	RunState,
	RunStatus,
	SelectionCheckpointBundle,
	SelectionPlanItem,
	SelectionResumeState,
	SelectionStage,
	StopRequested,
	atomic_write_text,
	build_config_fingerprint,
	build_selection_key,
)
from hypercorpus.reports import (
	export_study_comparison_report,
	export_subset_comparison_report,
	export_summary_report,
)
from hypercorpus.selector_llm import SelectorLLMConfig
from hypercorpus.selector import (
	BudgetFillSelector,
	CanonicalConstrainedMultipathSelector,
	CanonicalIterativeDenseSelector,
	CanonicalMDRLightSelector,
	CanonicalSearchSelector,
	CanonicalSinglePathSelector,
	CorpusSelectionResult,
	SelectorUsage,
	available_selector_names,
	available_selector_presets,
	corpus_selection_result_from_dict,
	selection_trace_step_from_dict,
	selector_metadata_from_dict,
	selector_usage_from_dict,
	select_selectors,
)
from hypercorpus.subgraph import SubgraphExtractor
from hypercorpus.walker import walk_step_log_from_dict

logger = logging.getLogger(__name__)

_SELECTION_HEARTBEAT_INTERVAL_ENV = "HYPERCORPUS_SELECTION_HEARTBEAT_INTERVAL_S"
_SELECTION_STAGE_TIMEOUT_ENV = "HYPERCORPUS_SELECTION_STAGE_TIMEOUT_S"
_BUDGET_FILL_TIMEOUT_ENV = "HYPERCORPUS_BUDGET_FILL_TIMEOUT_S"
_DEFAULT_SELECTION_HEARTBEAT_INTERVAL_S = 30.0
_DEFAULT_SELECTION_STAGE_TIMEOUT_S = 1800.0
_DEFAULT_BUDGET_FILL_TIMEOUT_S = 600.0


ExperimentPhase = Literal[
	"loading",
	"initializing",
	"evaluating",
	"exporting",
	"finalizing",
	"completed",
	"interrupted",
	"failed",
]
StudyPresetName = Literal[
	"single_path_edge_ablation_local",
	"baseline_retest_local",
	"branchy_profiles_384_512",
	"iirc_selector_main",
]


@dataclass(slots=True)
class ExperimentProgressUpdate:
	dataset_name: str
	phase: ExperimentPhase
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


ExperimentProgressObserver = Callable[[ExperimentProgressUpdate], None]


@dataclass(frozen=True, slots=True)
class StudyPresetSpec:
	name: str
	description: str
	selector_preset: str | None = None
	selector_names: tuple[str, ...] | None = None
	token_budgets: tuple[int, ...] | None = None
	budget_ratios: tuple[float, ...] | None = None
	include_diagnostics: bool = True
	control_selector_name: str | None = None


@dataclass(frozen=True, slots=True)
class _SelectionProgressSnapshot:
	total_selections: int
	completed_selections: int
	current_selector_name: str | None
	current_budget_label: str | None
	case_total_selectors: int | None
	case_completed_selectors: int | None
	case_total_selections: int | None
	case_completed_selections: int | None


@dataclass(frozen=True, slots=True)
class _ResolvedExperimentConfig:
	selector_names: list[str] | None
	selector_preset: str
	include_diagnostics: bool
	token_budgets: list[int] | None
	budget_ratios: list[float] | None
	study_preset: str | None
	control_selector_name: str | None


class SelectionStageTimeout(RuntimeError):
	"""Raised when a single selection stage exceeds its allowed wall clock budget."""


@dataclass(frozen=True, slots=True)
class _SelectionExecutionConfig:
	heartbeat_interval_s: float | None
	selection_stage_timeout_s: float | None
	budget_fill_timeout_s: float | None


def _utc_now_iso() -> str:
	return datetime.now(UTC).isoformat(timespec="seconds")


def _read_optional_positive_float_env(
	name: str, *, default: float | None
) -> float | None:
	raw = os.environ.get(name)
	if raw is None:
		return default
	text = raw.strip()
	if not text:
		return default
	value = float(text)
	if value <= 0:
		return None
	return value


def _load_selection_execution_config() -> _SelectionExecutionConfig:
	return _SelectionExecutionConfig(
		heartbeat_interval_s=_read_optional_positive_float_env(
			_SELECTION_HEARTBEAT_INTERVAL_ENV,
			default=_DEFAULT_SELECTION_HEARTBEAT_INTERVAL_S,
		),
		selection_stage_timeout_s=_read_optional_positive_float_env(
			_SELECTION_STAGE_TIMEOUT_ENV,
			default=_DEFAULT_SELECTION_STAGE_TIMEOUT_S,
		),
		budget_fill_timeout_s=_read_optional_positive_float_env(
			_BUDGET_FILL_TIMEOUT_ENV,
			default=_DEFAULT_BUDGET_FILL_TIMEOUT_S,
		),
	)


def _timeout_for_stage(
	config: _SelectionExecutionConfig, stage: SelectionStage
) -> float | None:
	if stage == SelectionStage.BUDGET_FILL:
		return config.budget_fill_timeout_s
	return config.selection_stage_timeout_s


@contextmanager
def _selection_stage_timeout(
	*,
	item: SelectionPlanItem,
	stage: SelectionStage,
	timeout_s: float | None,
) -> Iterator[None]:
	if timeout_s is None:
		yield
		return

	def _raise_timeout(_signum: int, _frame: Any) -> None:
		raise SelectionStageTimeout(
			"Selection stage timed out after "
			f"{timeout_s:.1f}s: case={item.case_id} "
			f"selector={item.selector_name} budget={item.budget_label} "
			f"stage={stage.value}"
		)

	previous_handler = signal.getsignal(signal.SIGALRM)
	previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)
	signal.signal(signal.SIGALRM, _raise_timeout)
	signal.setitimer(signal.ITIMER_REAL, timeout_s)
	try:
		yield
	finally:
		signal.setitimer(signal.ITIMER_REAL, 0.0)
		signal.signal(signal.SIGALRM, previous_handler)
		if previous_delay > 0 or previous_interval > 0:
			signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)


class _SelectionExecutionMonitor:
	def __init__(
		self,
		*,
		item: SelectionPlanItem,
		checkpoint_store: CheckpointStore,
		run_state: RunState,
		heartbeat_interval_s: float | None,
	):
		self._item = item
		self._checkpoint_store = checkpoint_store
		self._run_state = run_state
		self._heartbeat_interval_s = heartbeat_interval_s
		self._stage_start_monotonic: float | None = None
		self._last_heartbeat_monotonic: float | None = None

	def start_stage(self, stage: SelectionStage, *, detail: str | None = None) -> None:
		now = _utc_now_iso()
		now_monotonic = monotonic()
		self._stage_start_monotonic = now_monotonic
		self._last_heartbeat_monotonic = now_monotonic
		self._run_state.current_stage = stage.value
		self._run_state.stage_started_at = now
		self._run_state.last_heartbeat_at = now
		self._checkpoint_store.save_run_state(self._run_state)
		logger.info(
			"Selection stage start case=%s selector=%s budget=%s stage=%s%s",
			self._item.case_id,
			self._item.selector_name,
			self._item.budget_label,
			stage.value,
			f" detail={detail}" if detail is not None else "",
		)

	def heartbeat(self, *, detail: str | None = None, force: bool = False) -> None:
		now_monotonic = monotonic()
		if (
			not force
			and self._heartbeat_interval_s is not None
			and self._last_heartbeat_monotonic is not None
			and now_monotonic - self._last_heartbeat_monotonic
			< self._heartbeat_interval_s
		):
			return
		self._last_heartbeat_monotonic = now_monotonic
		self._run_state.last_heartbeat_at = _utc_now_iso()
		self._checkpoint_store.save_run_state(self._run_state)
		elapsed_s = self.current_stage_elapsed_s()
		logger.info(
			"Selection heartbeat case=%s selector=%s budget=%s stage=%s elapsed_s=%.1f%s",
			self._item.case_id,
			self._item.selector_name,
			self._item.budget_label,
			self._run_state.current_stage,
			elapsed_s,
			f" detail={detail}" if detail is not None else "",
		)

	def finish_stage(self) -> None:
		if self._run_state.current_stage is None:
			return
		logger.info(
			"Selection stage done case=%s selector=%s budget=%s stage=%s elapsed_s=%.1f",
			self._item.case_id,
			self._item.selector_name,
			self._item.budget_label,
			self._run_state.current_stage,
			self.current_stage_elapsed_s(),
		)

	def current_stage_elapsed_s(self) -> float:
		if self._stage_start_monotonic is None:
			return 0.0
		return monotonic() - self._stage_start_monotonic

	@contextmanager
	def stage(
		self,
		stage: SelectionStage,
		*,
		timeout_s: float | None,
		detail: str | None = None,
	) -> Iterator[None]:
		self.start_stage(stage, detail=detail)
		try:
			with _selection_stage_timeout(
				item=self._item,
				stage=stage,
				timeout_s=timeout_s,
			):
				yield
		finally:
			self.finish_stage()


_EXPERIMENT_DEFAULTS_FILENAME = ".hypercorpus-experiments.toml"
_EXPERIMENT_DEFAULTS_FILE_ENV = "HYPERCORPUS_EXPERIMENT_DEFAULTS_FILE"
_SELECTOR_PROVIDER_ENV = "HYPERCORPUS_SELECTOR_PROVIDER"
_SELECTOR_MODEL_ENV = "HYPERCORPUS_SELECTOR_MODEL"
_SELECTOR_API_KEY_ENV = "HYPERCORPUS_SELECTOR_API_KEY_ENV"
_SELECTOR_BASE_URL_ENV = "HYPERCORPUS_SELECTOR_BASE_URL"
_SELECTOR_OPENAI_API_MODE_ENV = "HYPERCORPUS_SELECTOR_OPENAI_API_MODE"
_ANSWER_PROVIDER_ENV = "HYPERCORPUS_ANSWER_PROVIDER"
_ANSWER_MODEL_ENV = "HYPERCORPUS_ANSWER_MODEL"
_ANSWER_API_KEY_ENV = "HYPERCORPUS_ANSWER_API_KEY_ENV"
_ANSWER_BASE_URL_ENV = "HYPERCORPUS_ANSWER_BASE_URL"
_SENTENCE_TRANSFORMER_MODEL_ENV = "HYPERCORPUS_SENTENCE_TRANSFORMER_MODEL"
_SENTENCE_TRANSFORMER_DEVICE_ENV = "HYPERCORPUS_SENTENCE_TRANSFORMER_DEVICE"


@dataclass(frozen=True, slots=True)
class _ExperimentRuntimeDefaults:
	selector_provider: str | None = None
	selector_model: str | None = None
	selector_api_key_env: str | None = None
	selector_base_url: str | None = None
	selector_openai_api_mode: str | None = None
	answer_provider: str | None = None
	answer_model: str | None = None
	answer_api_key_env: str | None = None
	answer_base_url: str | None = None
	sentence_transformer_model: str | None = None
	sentence_transformer_device: str | None = None
	cross_encoder_model: str | None = None


_SINGLE_PATH_EDGE_ABLATION_SELECTORS: tuple[str, ...] = (
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__anchor_overlap__lookahead_1__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_anchor_heavy__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_title_aware__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_1__profile_st_balanced__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_balanced__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_direct_heavy__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop",
	"gold_support_context",
	"full_corpus_upper_bound",
)

_BASELINE_RETEST_SELECTORS: tuple[str, ...] = (
	"top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
	"top_3_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__iterative_dense__budget_fill_relative_drop",
	"top_3_seed__sentence_transformer__hop_2__iterative_dense__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop",
	"top_3_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop",
	"gold_support_context",
	"full_corpus_upper_bound",
)

_IIRC_SELECTOR_MAIN_SELECTORS: tuple[str, ...] = (
	"top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop",
	"top_1_seed__sentence_transformer__hop_adaptive__constrained_multipath__link_context_llm_controller__lookahead_2",
	"gold_support_context",
	"full_corpus_upper_bound",
)

_STUDY_PRESETS: tuple[StudyPresetSpec, ...] = (
	StudyPresetSpec(
		name="single_path_edge_ablation_local",
		description="Local-only single-path scorer ablation.",
		selector_names=_SINGLE_PATH_EDGE_ABLATION_SELECTORS,
		token_budgets=(128, 256),
		include_diagnostics=True,
		control_selector_name="top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced__budget_fill_relative_drop",
	),
	StudyPresetSpec(
		name="baseline_retest_local",
		description="Local-only dense, iterative_dense, and mdr_light baseline retest.",
		selector_names=_BASELINE_RETEST_SELECTORS,
		token_budgets=(128, 256, 384, 512),
		include_diagnostics=True,
		control_selector_name="top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
	),
	StudyPresetSpec(
		name="branchy_profiles_384_512",
		description="Branchy profile sweep at wider token budgets.",
		selector_preset="branchy_profiles",
		token_budgets=(384, 512),
		include_diagnostics=True,
		control_selector_name="top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_overlap__lookahead_1__profile_overlap_balanced",
	),
	StudyPresetSpec(
		name="iirc_selector_main",
		description="Canonical full-IIRC selector-only comparison on ratio-controlled corpus-mass budgets.",
		selector_names=_IIRC_SELECTOR_MAIN_SELECTORS,
		budget_ratios=(0.01, 0.02, 0.05, 0.10, 1.0),
		include_diagnostics=True,
		control_selector_name="top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
	),
)


def _discover_experiment_defaults_path() -> Path | None:
	explicit = os.environ.get(_EXPERIMENT_DEFAULTS_FILE_ENV)
	if explicit:
		path = Path(explicit).expanduser()
		if not path.exists():
			raise ValueError(
				f"{_EXPERIMENT_DEFAULTS_FILE_ENV} points at missing file: {path}"
			)
		return path
	for root in (Path.cwd(), *Path.cwd().parents):
		candidate = root / _EXPERIMENT_DEFAULTS_FILENAME
		if candidate.exists():
			return candidate
	return None


def _read_optional_string(
	table: dict[str, Any], key: str, *, path: Path, section: str
) -> str | None:
	value = table.get(key)
	if value is None:
		return None
	if not isinstance(value, str):
		raise ValueError(f"{path}: [{section}].{key} must be a string when provided.")
	text = value.strip()
	return text or None


def _load_experiment_defaults_file(path: Path) -> _ExperimentRuntimeDefaults:
	with path.open("rb") as handle:
		payload = tomllib.load(handle)
	selector = payload.get("selector", {})
	answerer = payload.get("answerer", {})
	sentence_transformer = payload.get("sentence_transformer", {})
	if not isinstance(selector, dict):
		raise ValueError(f"{path}: [selector] must be a table.")
	if not isinstance(answerer, dict):
		raise ValueError(f"{path}: [answerer] must be a table.")
	if not isinstance(sentence_transformer, dict):
		raise ValueError(f"{path}: [sentence_transformer] must be a table.")
	return _ExperimentRuntimeDefaults(
		selector_provider=_read_optional_string(
			selector, "provider", path=path, section="selector"
		),
		selector_model=_read_optional_string(
			selector, "model", path=path, section="selector"
		),
		selector_api_key_env=_read_optional_string(
			selector, "api_key_env", path=path, section="selector"
		),
		selector_base_url=_read_optional_string(
			selector, "base_url", path=path, section="selector"
		),
		selector_openai_api_mode=_read_optional_string(
			selector, "openai_api_mode", path=path, section="selector"
		),
		answer_provider=_read_optional_string(
			answerer, "provider", path=path, section="answerer"
		),
		answer_model=_read_optional_string(
			answerer, "model", path=path, section="answerer"
		),
		answer_api_key_env=_read_optional_string(
			answerer, "api_key_env", path=path, section="answerer"
		),
		answer_base_url=_read_optional_string(
			answerer, "base_url", path=path, section="answerer"
		),
		sentence_transformer_model=_read_optional_string(
			sentence_transformer,
			"model",
			path=path,
			section="sentence_transformer",
		),
		sentence_transformer_device=_read_optional_string(
			sentence_transformer,
			"device",
			path=path,
			section="sentence_transformer",
		),
	)


def _load_environment_runtime_defaults() -> _ExperimentRuntimeDefaults:
	return _ExperimentRuntimeDefaults(
		selector_provider=os.environ.get(_SELECTOR_PROVIDER_ENV) or None,
		selector_model=os.environ.get(_SELECTOR_MODEL_ENV) or None,
		selector_api_key_env=os.environ.get(_SELECTOR_API_KEY_ENV) or None,
		selector_base_url=os.environ.get(_SELECTOR_BASE_URL_ENV) or None,
		selector_openai_api_mode=os.environ.get(_SELECTOR_OPENAI_API_MODE_ENV) or None,
		answer_provider=os.environ.get(_ANSWER_PROVIDER_ENV) or None,
		answer_model=os.environ.get(_ANSWER_MODEL_ENV) or None,
		answer_api_key_env=os.environ.get(_ANSWER_API_KEY_ENV) or None,
		answer_base_url=os.environ.get(_ANSWER_BASE_URL_ENV) or None,
		sentence_transformer_model=os.environ.get(_SENTENCE_TRANSFORMER_MODEL_ENV)
		or None,
		sentence_transformer_device=os.environ.get(_SENTENCE_TRANSFORMER_DEVICE_ENV)
		or None,
	)


def _merge_runtime_defaults(
	*defaults: _ExperimentRuntimeDefaults,
) -> _ExperimentRuntimeDefaults:
	merged = _ExperimentRuntimeDefaults()
	for defaults_item in defaults:
		merged = _ExperimentRuntimeDefaults(
			selector_provider=defaults_item.selector_provider
			or merged.selector_provider,
			selector_model=defaults_item.selector_model or merged.selector_model,
			selector_api_key_env=defaults_item.selector_api_key_env
			or merged.selector_api_key_env,
			selector_base_url=defaults_item.selector_base_url
			or merged.selector_base_url,
			selector_openai_api_mode=defaults_item.selector_openai_api_mode
			or merged.selector_openai_api_mode,
			answer_provider=defaults_item.answer_provider or merged.answer_provider,
			answer_model=defaults_item.answer_model or merged.answer_model,
			answer_api_key_env=defaults_item.answer_api_key_env
			or merged.answer_api_key_env,
			answer_base_url=defaults_item.answer_base_url or merged.answer_base_url,
			sentence_transformer_model=defaults_item.sentence_transformer_model
			or merged.sentence_transformer_model,
			sentence_transformer_device=defaults_item.sentence_transformer_device
			or merged.sentence_transformer_device,
		)
	return merged


def _provider_matches_default_source(
	*, resolved_provider: str, source_provider: str | None
) -> bool:
	return source_provider is None or source_provider == resolved_provider


def _selector_value_is_compatible(
	*, field_name: str, provider: str, value: str
) -> bool:
	if provider != "copilot":
		return True
	if field_name == "selector_model":
		return is_copilot_model_name(value)
	if field_name in {
		"selector_api_key_env",
		"selector_base_url",
		"selector_openai_api_mode",
	}:
		return False
	return True


def _answer_value_is_compatible(*, field_name: str, provider: str, value: str) -> bool:
	if provider != "copilot":
		return True
	if field_name == "answer_model":
		return is_copilot_model_name(value)
	if field_name in {"answer_api_key_env", "answer_base_url"}:
		return False
	return True


def _resolve_provider_bound_default(
	*,
	explicit_value: str | None,
	field_name: str,
	provider: str,
	defaults_sources: Sequence[tuple[str | None, _ExperimentRuntimeDefaults]],
	validator: Callable[[str], bool] | None = None,
) -> str | None:
	if explicit_value is not None:
		return explicit_value
	for source_provider, source_defaults in defaults_sources:
		if not _provider_matches_default_source(
			resolved_provider=provider,
			source_provider=source_provider,
		):
			continue
		value = cast(str | None, getattr(source_defaults, field_name))
		if value is None:
			continue
		if validator is not None and not validator(value):
			continue
		return value
	return None


def _resolve_runtime_defaults(
	*,
	selector_provider: str | None,
	selector_model: str | None,
	selector_api_key_env: str | None,
	selector_base_url: str | None,
	selector_openai_api_mode: str | None,
	answer_provider: str | None,
	answer_model: str | None,
	answer_api_key_env: str | None,
	answer_base_url: str | None,
	sentence_transformer_model: str | None,
	sentence_transformer_device: str | None,
	cross_encoder_model: str | None = None,
) -> _ExperimentRuntimeDefaults:
	file_defaults = _ExperimentRuntimeDefaults()
	defaults_path = _discover_experiment_defaults_path()
	if defaults_path is not None:
		file_defaults = _load_experiment_defaults_file(defaults_path)
	env_defaults = _load_environment_runtime_defaults()
	resolved_defaults = _merge_runtime_defaults(file_defaults, env_defaults)
	selector_defaults_sources = (
		(env_defaults.selector_provider, env_defaults),
		(file_defaults.selector_provider, file_defaults),
	)
	answer_defaults_sources = (
		(env_defaults.answer_provider, env_defaults),
		(file_defaults.answer_provider, file_defaults),
	)
	resolved_selector_provider = (
		selector_provider
		or resolved_defaults.selector_provider
		or _infer_default_selector_provider(
			selector_api_key_env=selector_api_key_env
			or resolved_defaults.selector_api_key_env,
			selector_base_url=selector_base_url or resolved_defaults.selector_base_url,
			selector_openai_api_mode=selector_openai_api_mode
			or resolved_defaults.selector_openai_api_mode,
		)
	)
	resolved_selector_config = SelectorLLMConfig(
		provider=cast(Any, resolved_selector_provider),
		model=_resolve_provider_bound_default(
			explicit_value=selector_model,
			field_name="selector_model",
			provider=resolved_selector_provider,
			defaults_sources=selector_defaults_sources,
			validator=lambda value: _selector_value_is_compatible(
				field_name="selector_model",
				provider=resolved_selector_provider,
				value=value,
			),
		),
		api_key_env=_resolve_provider_bound_default(
			explicit_value=selector_api_key_env,
			field_name="selector_api_key_env",
			provider=resolved_selector_provider,
			defaults_sources=selector_defaults_sources,
			validator=lambda value: _selector_value_is_compatible(
				field_name="selector_api_key_env",
				provider=resolved_selector_provider,
				value=value,
			),
		),
		base_url=_resolve_provider_bound_default(
			explicit_value=selector_base_url,
			field_name="selector_base_url",
			provider=resolved_selector_provider,
			defaults_sources=selector_defaults_sources,
			validator=lambda value: _selector_value_is_compatible(
				field_name="selector_base_url",
				provider=resolved_selector_provider,
				value=value,
			),
		),
		openai_api_mode=cast(
			Any,
			_resolve_provider_bound_default(
				explicit_value=selector_openai_api_mode,
				field_name="selector_openai_api_mode",
				provider=resolved_selector_provider,
				defaults_sources=selector_defaults_sources,
				validator=lambda value: _selector_value_is_compatible(
					field_name="selector_openai_api_mode",
					provider=resolved_selector_provider,
					value=value,
				),
			)
			or (
				None if resolved_selector_provider == "copilot" else "chat_completions"
			),
		),
	)
	resolved_selector_openai_api_mode = resolved_selector_config.openai_api_mode
	resolved_answer_provider = cast(
		AnswererProvider,
		answer_provider or resolved_defaults.answer_provider or "copilot",
	)
	resolved_answer_config = LLMAnswererConfig(
		provider=resolved_answer_provider,
		model=_resolve_provider_bound_default(
			explicit_value=answer_model,
			field_name="answer_model",
			provider=resolved_answer_provider,
			defaults_sources=answer_defaults_sources,
			validator=lambda value: _answer_value_is_compatible(
				field_name="answer_model",
				provider=resolved_answer_provider,
				value=value,
			),
		),
		api_key_env=_resolve_provider_bound_default(
			explicit_value=answer_api_key_env,
			field_name="answer_api_key_env",
			provider=resolved_answer_provider,
			defaults_sources=answer_defaults_sources,
			validator=lambda value: _answer_value_is_compatible(
				field_name="answer_api_key_env",
				provider=resolved_answer_provider,
				value=value,
			),
		),
		base_url=_resolve_provider_bound_default(
			explicit_value=answer_base_url,
			field_name="answer_base_url",
			provider=resolved_answer_provider,
			defaults_sources=answer_defaults_sources,
			validator=lambda value: _answer_value_is_compatible(
				field_name="answer_base_url",
				provider=resolved_answer_provider,
				value=value,
			),
		),
	)
	return _ExperimentRuntimeDefaults(
		selector_provider=resolved_selector_config.provider,
		selector_model=resolved_selector_config.model,
		selector_api_key_env=resolved_selector_config.api_key_env,
		selector_base_url=resolved_selector_config.base_url,
		selector_openai_api_mode=resolved_selector_openai_api_mode,
		answer_provider=resolved_answer_config.provider,
		answer_model=resolved_answer_config.model,
		answer_api_key_env=resolved_answer_config.api_key_env,
		answer_base_url=resolved_answer_config.base_url,
		sentence_transformer_model=sentence_transformer_model
		or resolved_defaults.sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device
		or resolved_defaults.sentence_transformer_device,
		cross_encoder_model=cross_encoder_model
		or resolved_defaults.cross_encoder_model,
	)


def _infer_default_selector_provider(
	*,
	selector_api_key_env: str | None,
	selector_base_url: str | None,
	selector_openai_api_mode: str | None,
) -> str:
	if selector_openai_api_mode is not None:
		return "openai"
	if selector_base_url is not None:
		return "openai"
	if selector_api_key_env is not None:
		return "openai"
	return "copilot"


def run_dataset_experiment(
	*,
	adapter: DatasetAdapter,
	questions_path: str | Path,
	graph_source: str | Path,
	output_dir: str | Path,
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	cross_encoder_model: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	dataset_label = getattr(adapter, "dataset_name", "dataset")
	_notify_progress(
		progress_observer,
		dataset_name=dataset_label,
		phase="loading",
		total_cases=None,
		completed_cases=0,
	)
	logger.info("Loading %s graph from %s", dataset_label, graph_source)
	graph = adapter.load_graph(graph_source)
	logger.info("Loading %s questions from %s", dataset_label, questions_path)
	cases = adapter.load_cases(
		questions_path, limit=None if case_ids_file is not None else limit
	)
	selected_cases = _resolve_case_selection(
		cases, limit=limit, case_ids_file=case_ids_file
	)
	runtime_defaults = _resolve_runtime_defaults(
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device,
		cross_encoder_model=cross_encoder_model,
	)
	selector_provider = runtime_defaults.selector_provider
	selector_model = runtime_defaults.selector_model
	selector_api_key_env = runtime_defaults.selector_api_key_env
	selector_base_url = runtime_defaults.selector_base_url
	selector_openai_api_mode = runtime_defaults.selector_openai_api_mode
	assert selector_provider is not None
	answer_provider = runtime_defaults.answer_provider
	answer_model = runtime_defaults.answer_model
	answer_api_key_env = runtime_defaults.answer_api_key_env
	answer_base_url = runtime_defaults.answer_base_url
	sentence_transformer_model = runtime_defaults.sentence_transformer_model
	sentence_transformer_device = runtime_defaults.sentence_transformer_device
	cross_encoder_model = runtime_defaults.cross_encoder_model
	resolved = _resolve_experiment_config(
		selector_names=selector_names,
		selector_preset=selector_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
		study_preset=study_preset,
	)
	selectors = select_selectors(
		resolved.selector_names,
		preset=resolved.selector_preset,
		include_diagnostics=resolved.include_diagnostics,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		selector_cache_path=str(selector_cache_path)
		if selector_cache_path is not None
		else None,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_cache_path=sentence_transformer_cache_path,
		sentence_transformer_device=sentence_transformer_device,
		cross_encoder_model=cross_encoder_model,
	)
	budgets = _resolve_budgets(
		token_budgets=resolved.token_budgets,
		budget_ratios=resolved.budget_ratios,
	)
	total_selection_count = len(selected_cases) * len(selectors) * len(budgets)
	logger.info(
		"Running %s experiment (cases=%s, selectors=%s, budgets=%s, with_e2e=%s, answerer=%s, export_graphrag_inputs=%s)",
		dataset_label,
		len(selected_cases),
		[selector.name for selector in selectors],
		[budget.budget_label for budget in budgets],
		with_e2e,
		answerer_mode,
		export_graphrag_inputs,
	)
	evaluations, summary = _run_loaded_experiment(
		graph=graph,
		selected_cases=selected_cases,
		selectors=selectors,
		budgets=budgets,
		output_dir=Path(output_dir),
		dataset_name=dataset_label,
		study_preset=resolved.study_preset,
		selector_preset=resolved.selector_preset,
		token_budgets=resolved.token_budgets,
		budget_ratios=resolved.budget_ratios,
		case_ids_file=case_ids_file,
		control_selector_name=resolved.control_selector_name,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		mdr_home=None,
		mdr_artifact_manifest=None,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device,
		with_e2e=with_e2e,
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
		export_graphrag_inputs=export_graphrag_inputs,
		progress_observer=progress_observer,
		resume=resume,
		restart=restart,
	)
	_notify_progress(
		progress_observer,
		dataset_name=dataset_label,
		phase="completed",
		total_cases=len(selected_cases),
		completed_cases=len(evaluations),
		total_selections=total_selection_count,
		completed_selections=total_selection_count,
		summary=summary,
	)
	logger.info(
		"Completed %s experiment; results written to %s", dataset_label, output_dir
	)

	return evaluations, summary


def run_2wiki_experiment(
	*,
	questions_path: str | Path,
	graph_records_path: str | Path,
	output_dir: str | Path,
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	return run_dataset_experiment(
		adapter=TwoWikiAdapter(),
		questions_path=questions_path,
		graph_source=graph_records_path,
		output_dir=output_dir,
		limit=limit,
		case_ids_file=case_ids_file,
		selector_names=selector_names,
		selector_preset=selector_preset,
		study_preset=study_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
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
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
		export_graphrag_inputs=export_graphrag_inputs,
		progress_observer=progress_observer,
		resume=resume,
		restart=restart,
	)


def run_iirc_experiment(
	*,
	questions_path: str | Path,
	graph_records_path: str | Path,
	output_dir: str | Path,
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	cross_encoder_model: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	return run_dataset_experiment(
		adapter=IIRCAdapter(),
		questions_path=questions_path,
		graph_source=graph_records_path,
		output_dir=output_dir,
		limit=limit,
		case_ids_file=case_ids_file,
		selector_names=selector_names,
		selector_preset=selector_preset,
		study_preset=study_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		selector_cache_path=selector_cache_path,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_cache_path=sentence_transformer_cache_path,
		sentence_transformer_device=sentence_transformer_device,
		cross_encoder_model=cross_encoder_model,
		with_e2e=with_e2e,
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
		export_graphrag_inputs=export_graphrag_inputs,
		progress_observer=progress_observer,
		resume=resume,
		restart=restart,
	)


def run_hotpotqa_experiment(
	*,
	questions_path: str | Path,
	output_dir: str | Path,
	variant: Literal["distractor", "fullwiki"],
	graph_records_path: str | Path | None = None,
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	if variant == "fullwiki":
		if graph_records_path is None:
			raise ValueError(
				"graph_records_path is required for hotpotqa fullwiki runs"
			)
		return run_dataset_experiment(
			adapter=HotpotQAAdapter(variant="fullwiki"),
			questions_path=questions_path,
			graph_source=graph_records_path,
			output_dir=output_dir,
			limit=limit,
			case_ids_file=case_ids_file,
			selector_names=selector_names,
			selector_preset=selector_preset,
			study_preset=study_preset,
			token_budgets=token_budgets,
			budget_ratios=budget_ratios,
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
			answerer_mode=answerer_mode,
			answer_provider=answer_provider,
			answer_model=answer_model,
			answer_api_key_env=answer_api_key_env,
			answer_base_url=answer_base_url,
			answer_cache_path=answer_cache_path,
			export_graphrag_inputs=export_graphrag_inputs,
			progress_observer=progress_observer,
			resume=resume,
			restart=restart,
		)

	dataset_label = "hotpotqa-distractor"
	_notify_progress(
		progress_observer,
		dataset_name=dataset_label,
		phase="loading",
		total_cases=None,
		completed_cases=0,
	)
	effective_limit = None if case_ids_file is not None else limit
	records = load_json_records(questions_path)
	cases = load_hotpotqa_questions(
		questions_path,
		limit=effective_limit,
		variant="distractor",
	)
	if effective_limit is not None:
		records = records[:effective_limit]
	paired_cases = list(zip(records, cases, strict=True))
	if case_ids_file is not None and limit is not None:
		raise ValueError("Specify either limit or case_ids_file, not both.")
	if case_ids_file is None:
		if limit is not None:
			paired_cases = paired_cases[:limit]
	else:
		selected_cases = _select_cases_by_id(
			[case for _record, case in paired_cases],
			_load_case_ids(case_ids_file),
			case_ids_file=case_ids_file,
		)
		paired_by_id = {case.case_id: (record, case) for record, case in paired_cases}
		paired_cases = [paired_by_id[case.case_id] for case in selected_cases]
	runtime_defaults = _resolve_runtime_defaults(
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device,
	)
	selector_provider = runtime_defaults.selector_provider
	selector_model = runtime_defaults.selector_model
	selector_api_key_env = runtime_defaults.selector_api_key_env
	selector_base_url = runtime_defaults.selector_base_url
	selector_openai_api_mode = runtime_defaults.selector_openai_api_mode
	assert selector_provider is not None
	answer_provider = runtime_defaults.answer_provider
	answer_model = runtime_defaults.answer_model
	answer_api_key_env = runtime_defaults.answer_api_key_env
	answer_base_url = runtime_defaults.answer_base_url
	sentence_transformer_model = runtime_defaults.sentence_transformer_model
	sentence_transformer_device = runtime_defaults.sentence_transformer_device
	resolved = _resolve_experiment_config(
		selector_names=selector_names,
		selector_preset=selector_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
		study_preset=study_preset,
	)
	selectors = select_selectors(
		resolved.selector_names,
		preset=resolved.selector_preset,
		include_diagnostics=resolved.include_diagnostics,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		selector_cache_path=str(selector_cache_path)
		if selector_cache_path is not None
		else None,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_cache_path=sentence_transformer_cache_path,
		sentence_transformer_device=sentence_transformer_device,
	)
	budgets = _resolve_budgets(
		token_budgets=resolved.token_budgets, budget_ratios=resolved.budget_ratios
	)
	total = len(paired_cases)
	logger.info(
		"Running hotpotqa distractor experiment (cases=%s, selectors=%s, budgets=%s, with_e2e=%s, answerer=%s, export_graphrag_inputs=%s)",
		total,
		[selector.name for selector in selectors],
		[budget.budget_label for budget in budgets],
		with_e2e,
		answerer_mode,
		export_graphrag_inputs,
	)
	record_by_case_id = {case.case_id: record for record, case in paired_cases}
	selected_cases = [case for _record, case in paired_cases]
	evaluations, summary = _run_loaded_experiment(
		graph=None,
		graph_for_case=lambda case: build_hotpotqa_distractor_graph_for_case(
			record_by_case_id[case.case_id]
		),
		selected_cases=selected_cases,
		selectors=selectors,
		budgets=budgets,
		output_dir=Path(output_dir),
		dataset_name=dataset_label,
		study_preset=resolved.study_preset,
		selector_preset=resolved.selector_preset,
		token_budgets=resolved.token_budgets,
		budget_ratios=resolved.budget_ratios,
		case_ids_file=case_ids_file,
		control_selector_name=resolved.control_selector_name,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		mdr_home=None,
		mdr_artifact_manifest=None,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device,
		with_e2e=with_e2e,
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
		export_graphrag_inputs=export_graphrag_inputs,
		progress_observer=progress_observer,
		variant="distractor",
		resume=resume,
		restart=restart,
	)
	_notify_progress(
		progress_observer,
		dataset_name=dataset_label,
		phase="completed",
		total_cases=total,
		completed_cases=len(evaluations),
		summary=summary,
	)
	return evaluations, summary


def run_musique_experiment(
	*,
	questions_path: str | Path,
	graph_records_path: str | Path,
	output_dir: str | Path,
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	return run_dataset_experiment(
		adapter=MuSiQueAdapter(),
		questions_path=questions_path,
		graph_source=graph_records_path,
		output_dir=output_dir,
		limit=limit,
		case_ids_file=case_ids_file,
		selector_names=selector_names,
		selector_preset=selector_preset,
		study_preset=study_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
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
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
		export_graphrag_inputs=export_graphrag_inputs,
		progress_observer=progress_observer,
		resume=resume,
		restart=restart,
	)


def run_docs_experiment(
	*,
	questions_path: str | Path,
	docs_source: str | Path,
	output_dir: str | Path,
	dataset_name: str = "docs",
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	return run_dataset_experiment(
		adapter=DocumentationAdapter(dataset_name=dataset_name),
		questions_path=questions_path,
		graph_source=docs_source,
		output_dir=output_dir,
		limit=limit,
		case_ids_file=case_ids_file,
		selector_names=selector_names,
		selector_preset=selector_preset,
		study_preset=study_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
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
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
		export_graphrag_inputs=export_graphrag_inputs,
		progress_observer=progress_observer,
		resume=resume,
		restart=restart,
	)


def run_store_experiment(
	*,
	store_uri: str | Path,
	output_root: str | Path,
	exp_name: str,
	dataset_label: str | None = None,
	split: str = "dev",
	cache_dir: str | Path | None = None,
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	case_start: int = 0,
	case_limit: int | None = None,
	chunk_size: int | None = None,
	chunk_index: int | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	mdr_home: str | Path | None = None,
	mdr_artifact_manifest: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
	link_context_mask: str | None = None,
	budget_fill_ratio: float | None = None,
	budget_fill_pool_k: int | None = None,
	walk_score_threshold: float | None = None,
	cross_encoder_model: str | None = None,
	seed_top_k: int | None = None,
) -> tuple[list[CaseEvaluation], ExperimentSummary, Path]:
	mask_mode = (
		LinkContextMaskMode(link_context_mask)
		if link_context_mask
		else LinkContextMaskMode.NONE
	)
	store = ShardedDocumentStore(store_uri, cache_dir=cache_dir, mask_mode=mask_mode)
	resolved_dataset_name = dataset_label or store.manifest.dataset_name
	_notify_progress(
		progress_observer,
		dataset_name=resolved_dataset_name,
		phase="loading",
		total_cases=None,
		completed_cases=0,
	)
	logger.info("Loading %s split questions from sharded store", split)
	cases = store.load_questions(split)
	if case_ids_file is not None and limit is not None:
		raise ValueError("Specify either limit or case_ids_file, not both.")
	if case_ids_file is not None:
		cases = _select_cases_by_id(
			cases, _load_case_ids(case_ids_file), case_ids_file=case_ids_file
		)
	selected_cases, chunk_meta = _slice_cases(
		cases,
		split=split,
		limit=limit,
		case_start=case_start,
		case_limit=case_limit,
		chunk_size=chunk_size,
		chunk_index=chunk_index,
	)
	runtime_defaults = _resolve_runtime_defaults(
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device,
	)
	selector_provider = runtime_defaults.selector_provider
	selector_model = runtime_defaults.selector_model
	selector_api_key_env = runtime_defaults.selector_api_key_env
	selector_base_url = runtime_defaults.selector_base_url
	selector_openai_api_mode = runtime_defaults.selector_openai_api_mode
	assert selector_provider is not None
	answer_provider = runtime_defaults.answer_provider
	answer_model = runtime_defaults.answer_model
	answer_api_key_env = runtime_defaults.answer_api_key_env
	answer_base_url = runtime_defaults.answer_base_url
	sentence_transformer_model = runtime_defaults.sentence_transformer_model
	sentence_transformer_device = runtime_defaults.sentence_transformer_device
	resolved = _resolve_experiment_config(
		selector_names=selector_names,
		selector_preset=selector_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
		study_preset=study_preset,
	)
	selectors = select_selectors(
		resolved.selector_names,
		preset=resolved.selector_preset,
		include_diagnostics=resolved.include_diagnostics,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		selector_cache_path=str(selector_cache_path)
		if selector_cache_path is not None
		else None,
		mdr_home=mdr_home,
		mdr_artifact_manifest=mdr_artifact_manifest,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_cache_path=sentence_transformer_cache_path,
		sentence_transformer_device=sentence_transformer_device,
		budget_fill_ratio=budget_fill_ratio,
		budget_fill_pool_k=budget_fill_pool_k,
		walk_score_threshold=walk_score_threshold,
		cross_encoder_model=cross_encoder_model,
		seed_top_k=seed_top_k,
	)
	budgets = _resolve_budgets(		token_budgets=resolved.token_budgets,
		budget_ratios=resolved.budget_ratios,
	)
	total_selection_count = len(selected_cases) * len(selectors) * len(budgets)
	logger.info(
		"Running store-backed %s experiment (split=%s, selected_cases=%s, selectors=%s, budgets=%s, with_e2e=%s, answerer=%s, export_graphrag_inputs=%s)",
		resolved_dataset_name,
		split,
		len(selected_cases),
		[selector.name for selector in selectors],
		[budget.budget_label for budget in budgets],
		with_e2e,
		answerer_mode,
		export_graphrag_inputs,
	)

	chunk_dir = _chunk_output_dir(
		output_root=Path(output_root), exp_name=exp_name, chunk_meta=chunk_meta
	)
	evaluations, summary = _run_loaded_experiment(
		graph=store,
		selected_cases=selected_cases,
		selectors=selectors,
		budgets=budgets,
		output_dir=chunk_dir,
		dataset_name=resolved_dataset_name,
		study_preset=resolved.study_preset,
		selector_preset=resolved.selector_preset,
		token_budgets=resolved.token_budgets,
		budget_ratios=resolved.budget_ratios,
		case_ids_file=case_ids_file,
		control_selector_name=resolved.control_selector_name,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		mdr_home=mdr_home,
		mdr_artifact_manifest=mdr_artifact_manifest,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device,
		with_e2e=with_e2e,
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
		export_graphrag_inputs=export_graphrag_inputs,
		progress_observer=progress_observer,
		split=split,
		chunk_meta=chunk_meta,
		store_uri=store_uri,
		resume=resume,
		restart=restart,
	)
	_notify_progress(
		progress_observer,
		dataset_name=resolved_dataset_name,
		phase="completed",
		total_cases=len(selected_cases),
		completed_cases=len(evaluations),
		total_selections=total_selection_count,
		completed_selections=total_selection_count,
		summary=summary,
	)
	logger.info(
		"Completed store-backed %s chunk; results written to %s",
		resolved_dataset_name,
		chunk_dir,
	)
	return evaluations, summary, chunk_dir


def run_2wiki_store_experiment(
	*,
	store_uri: str | Path,
	output_root: str | Path,
	exp_name: str,
	split: str = "dev",
	cache_dir: str | Path | None = None,
	limit: int | None = None,
	case_ids_file: str | Path | None = None,
	case_start: int = 0,
	case_limit: int | None = None,
	chunk_size: int | None = None,
	chunk_index: int | None = None,
	selector_names: Sequence[str] | None = None,
	selector_preset: str | None = None,
	study_preset: str | None = None,
	token_budgets: Sequence[int] | None = None,
	budget_ratios: Sequence[float] | None = None,
	selector_provider: str | None = None,
	selector_model: str | None = None,
	selector_api_key_env: str | None = None,
	selector_base_url: str | None = None,
	selector_openai_api_mode: str | None = None,
	selector_cache_path: str | Path | None = None,
	mdr_home: str | Path | None = None,
	mdr_artifact_manifest: str | Path | None = None,
	sentence_transformer_model: str | None = None,
	sentence_transformer_cache_path: str | Path | None = None,
	sentence_transformer_device: str | None = None,
	with_e2e: bool = False,
	answerer_mode: str = "heuristic",
	answer_provider: str | None = None,
	answer_model: str | None = None,
	answer_api_key_env: str | None = None,
	answer_base_url: str | None = None,
	answer_cache_path: str | Path | None = None,
	export_graphrag_inputs: bool = True,
	progress_observer: ExperimentProgressObserver | None = None,
	resume: bool = False,
	restart: bool = False,
	link_context_mask: str | None = None,
	budget_fill_ratio: float | None = None,
	budget_fill_pool_k: int | None = None,
	walk_score_threshold: float | None = None,
	seed_top_k: int | None = None,
) -> tuple[list[CaseEvaluation], ExperimentSummary, Path]:
	return run_store_experiment(
		store_uri=store_uri,
		output_root=output_root,
		exp_name=exp_name,
		dataset_label="2wikimultihop",
		split=split,
		cache_dir=cache_dir,
		limit=limit,
		case_ids_file=case_ids_file,
		case_start=case_start,
		case_limit=case_limit,
		chunk_size=chunk_size,
		chunk_index=chunk_index,
		selector_names=selector_names,
		selector_preset=selector_preset,
		study_preset=study_preset,
		token_budgets=token_budgets,
		budget_ratios=budget_ratios,
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
		answerer_mode=answerer_mode,
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
		budget_fill_ratio=budget_fill_ratio,
		budget_fill_pool_k=budget_fill_pool_k,
		walk_score_threshold=walk_score_threshold,
		seed_top_k=seed_top_k,
	)


def run_iirc_store_experiment(
	**kwargs,
) -> tuple[list[CaseEvaluation], ExperimentSummary, Path]:
	return run_store_experiment(dataset_label="iirc", **kwargs)


def run_musique_store_experiment(
	**kwargs,
) -> tuple[list[CaseEvaluation], ExperimentSummary, Path]:
	return run_store_experiment(dataset_label="musique", **kwargs)


def run_hotpotqa_store_experiment(
	**kwargs,
) -> tuple[list[CaseEvaluation], ExperimentSummary, Path]:
	return run_store_experiment(dataset_label="hotpotqa-fullwiki", **kwargs)


def merge_store_results(
	*,
	run_dir: str | Path,
	output_dir: str | Path | None = None,
) -> tuple[ExperimentSummary, list[int]]:
	root = Path(run_dir)
	chunk_root = root / "chunks"
	chunk_dirs = sorted(path for path in chunk_root.iterdir() if path.is_dir())
	logger.info("Merging chunk results from %s (%s chunks)", root, len(chunk_dirs))
	records: list[dict[str, Any]] = []
	selector_log_records: list[dict[str, Any]] = []
	chunk_indices: list[int] = []
	total_cases: int | None = None
	chunk_size: int | None = None
	chunk_metas: list[dict[str, Any]] = []

	for chunk_dir in _iterate_with_optional_progress(
		chunk_dirs, description="merge chunk results"
	):
		chunk_meta_path = chunk_dir / "chunk.json"
		if chunk_meta_path.exists():
			chunk_meta = json.loads(chunk_meta_path.read_text(encoding="utf-8"))
			chunk_metas.append(chunk_meta)
			if chunk_meta.get("chunk_index") is not None:
				chunk_indices.append(int(chunk_meta["chunk_index"]))
			if chunk_meta.get("total_cases") is not None:
				total_cases = int(chunk_meta["total_cases"])
			if chunk_meta.get("chunk_size") is not None:
				chunk_size = int(chunk_meta["chunk_size"])
		results_path = chunk_dir / "results.jsonl"
		if not results_path.exists():
			continue
		with results_path.open("r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if line:
					records.append(json.loads(line))
		selector_logs_path = chunk_dir / "selector_logs.jsonl"
		if selector_logs_path.exists():
			with selector_logs_path.open("r", encoding="utf-8") as handle:
				for line in handle:
					line = line.strip()
					if line:
						selector_log_records.append(json.loads(line))

	summary = _summarize_result_records(records)
	merged_dir = Path(output_dir) if output_dir is not None else root
	merged_dir.mkdir(parents=True, exist_ok=True)
	merged_results = merged_dir / "results.jsonl"
	merged_selector_logs = merged_dir / "selector_logs.jsonl"
	with merged_results.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")
	with merged_selector_logs.open("w", encoding="utf-8") as handle:
		for record in selector_log_records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")
	merged_study_preset = _consistent_chunk_value(chunk_metas, "study_preset")
	missing_chunks = _missing_chunk_indices(
		chunk_indices, total_cases=total_cases, chunk_size=chunk_size
	)
	merged_control_selector = (
		resolve_study_preset(merged_study_preset).control_selector_name
		if merged_study_preset is not None
		else None
	)
	_write_summary_file(
		summary_path=merged_dir / "summary.json",
		summary=summary,
		study_preset=merged_study_preset,
		control_selector_name=merged_control_selector,
		results_path=merged_results,
	)
	_write_evaluated_case_ids(merged_dir, _ordered_case_ids_from_records(records))
	_write_run_manifest(
		merged_dir,
		dataset_name=summary.dataset_name,
		split=_consistent_chunk_value(chunk_metas, "split"),
		study_preset=merged_study_preset,
		selector_preset=_consistent_chunk_value(chunk_metas, "selector_preset"),
		resolved_selectors=_merged_resolved_selectors(chunk_metas, records),
		resolved_token_budgets=_consistent_chunk_value(chunk_metas, "token_budgets"),
		resolved_budget_ratios=_consistent_chunk_value(chunk_metas, "budget_ratios"),
		case_ids_file=_consistent_chunk_value(chunk_metas, "case_ids_file"),
		total_selected_cases=summary.total_cases,
		control_selector_name=merged_control_selector,
		missing_chunks=missing_chunks,
	)
	logger.info(
		"Merged %s result records into %s (missing_chunks=%s)",
		len(records),
		merged_dir,
		missing_chunks,
	)
	return summary, missing_chunks


def merge_2wiki_results(
	*,
	run_dir: str | Path,
	output_dir: str | Path | None = None,
) -> tuple[ExperimentSummary, list[int]]:
	return merge_store_results(run_dir=run_dir, output_dir=output_dir)


def merge_iirc_results(
	*,
	run_dir: str | Path,
	output_dir: str | Path | None = None,
) -> tuple[ExperimentSummary, list[int]]:
	return merge_store_results(run_dir=run_dir, output_dir=output_dir)


def merge_musique_results(
	*,
	run_dir: str | Path,
	output_dir: str | Path | None = None,
) -> tuple[ExperimentSummary, list[int]]:
	return merge_store_results(run_dir=run_dir, output_dir=output_dir)


def merge_hotpotqa_results(
	*,
	run_dir: str | Path,
	output_dir: str | Path | None = None,
) -> tuple[ExperimentSummary, list[int]]:
	return merge_store_results(run_dir=run_dir, output_dir=output_dir)


def parse_selector_names(value: str | None) -> list[str] | None:
	if value is None:
		return None
	names = [name.strip() for name in value.split(",") if name.strip()]
	if not names:
		return None
	return names


def parse_budget_ratios(value: str | None) -> list[float] | None:
	if value is None:
		return None
	ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
	if not ratios:
		return None
	for ratio in ratios:
		if ratio <= 0 or ratio > 1:
			raise ValueError(f"Budget ratio must be in (0, 1], got {ratio}")
	return ratios


def parse_token_budgets(value: str | None) -> list[int] | None:
	if value is None:
		return None
	budgets = [int(part.strip()) for part in value.split(",") if part.strip()]
	if not budgets:
		return None
	for budget in budgets:
		if budget <= 0:
			raise ValueError(f"Token budget must be positive, got {budget}")
	return budgets


def available_study_presets() -> list[str]:
	return [preset.name for preset in _STUDY_PRESETS]


def resolve_study_preset(name: StudyPresetName | str) -> StudyPresetSpec:
	for preset in _STUDY_PRESETS:
		if preset.name == name:
			return preset
	raise ValueError(f"Unknown study preset: {name}")


def selector_choices_help(*, include_diagnostics: bool = True) -> str:
	return ",".join(available_selector_names(include_diagnostics=include_diagnostics))


def selector_preset_choices_help() -> str:
	return ",".join(available_selector_presets())


def study_preset_choices_help() -> str:
	return ",".join(available_study_presets())


def budget_ratio_choices_help() -> str:
	return ",".join(f"{ratio:.2f}" for ratio in DEFAULT_BUDGET_RATIOS)


def token_budget_choices_help() -> str:
	return ",".join(str(budget) for budget in DEFAULT_TOKEN_BUDGETS)


def store_budget_ratio_choices_help() -> str:
	return budget_ratio_choices_help()


def store_token_budget_choices_help() -> str:
	return token_budget_choices_help()


def _resolve_experiment_config(
	*,
	selector_names: Sequence[str] | None,
	selector_preset: str | None,
	token_budgets: Sequence[int] | None,
	budget_ratios: Sequence[float] | None,
	study_preset: str | None,
) -> _ResolvedExperimentConfig:
	study = resolve_study_preset(study_preset) if study_preset is not None else None
	resolved_selector_names: list[str] | None = (
		list(selector_names) if selector_names is not None else None
	)
	resolved_selector_preset = selector_preset or "full"
	include_diagnostics = selector_names is not None

	if (
		resolved_selector_names is None
		and selector_preset is None
		and study is not None
	):
		if study.selector_names is not None:
			resolved_selector_names = list(study.selector_names)
			resolved_selector_preset = study.selector_preset or "full"
		elif study.selector_preset is not None:
			resolved_selector_preset = study.selector_preset
			include_diagnostics = study.include_diagnostics

	resolved_token_budgets = list(token_budgets) if token_budgets is not None else None
	resolved_budget_ratios = list(budget_ratios) if budget_ratios is not None else None
	if (
		resolved_token_budgets is None
		and resolved_budget_ratios is None
		and study is not None
		and study.token_budgets is not None
	):
		resolved_token_budgets = list(study.token_budgets)
	if (
		resolved_token_budgets is None
		and resolved_budget_ratios is None
		and study is not None
		and study.budget_ratios is not None
	):
		resolved_budget_ratios = list(study.budget_ratios)

	return _ResolvedExperimentConfig(
		selector_names=resolved_selector_names,
		selector_preset=resolved_selector_preset,
		include_diagnostics=include_diagnostics,
		token_budgets=resolved_token_budgets,
		budget_ratios=resolved_budget_ratios,
		study_preset=study.name if study is not None else None,
		control_selector_name=study.control_selector_name
		if study is not None
		else None,
	)


def _load_case_ids(case_ids_file: str | Path) -> list[str]:
	path = Path(case_ids_file)
	seen: set[str] = set()
	case_ids: list[str] = []
	for line_number, raw_line in enumerate(
		path.read_text(encoding="utf-8").splitlines(), start=1
	):
		case_id = raw_line.strip()
		if not case_id:
			continue
		if case_id in seen:
			raise ValueError(
				f"Duplicate case_id '{case_id}' in {path} at line {line_number}."
			)
		seen.add(case_id)
		case_ids.append(case_id)
	return case_ids


def _select_cases_by_id(
	cases: Sequence[Any], case_ids: Sequence[str], *, case_ids_file: str | Path
) -> list[Any]:
	indexed: dict[str, Any] = {}
	for case in cases:
		case_id = str(case.case_id)
		if case_id in indexed:
			raise ValueError(f"Duplicate case_id '{case_id}' found in loaded cases.")
		indexed[case_id] = case
	missing = [case_id for case_id in case_ids if case_id not in indexed]
	if missing:
		preview = ", ".join(missing[:5])
		raise ValueError(f"Unknown case_ids in {case_ids_file}: {preview}")
	return [indexed[case_id] for case_id in case_ids]


def _resolve_case_selection(
	cases: Sequence[Any],
	*,
	limit: int | None,
	case_ids_file: str | Path | None,
) -> list[Any]:
	if case_ids_file is not None and limit is not None:
		raise ValueError("Specify either limit or case_ids_file, not both.")
	if case_ids_file is None:
		return list(cases[:limit]) if limit is not None else list(cases)
	case_ids = _load_case_ids(case_ids_file)
	return _select_cases_by_id(cases, case_ids, case_ids_file=case_ids_file)


def _write_evaluated_case_ids(output_dir: Path, cases: Sequence[Any]) -> None:
	lines = [_case_id_value(case) for case in cases]
	content = "\n".join(lines)
	if content:
		content += "\n"
	(output_dir / "evaluated_case_ids.txt").write_text(content, encoding="utf-8")


def _write_run_manifest(
	output_dir: Path,
	*,
	dataset_name: str,
	study_preset: str | None,
	selector_preset: str | None,
	resolved_selectors: Sequence[str],
	resolved_token_budgets: Sequence[int] | None,
	resolved_budget_ratios: Sequence[float] | None,
	case_ids_file: str | Path | None,
	total_selected_cases: int,
	split: str | None = None,
	variant: str | None = None,
	control_selector_name: str | None = None,
	missing_chunks: Sequence[int] | None = None,
	extra: dict[str, Any] | None = None,
) -> None:
	payload = {
		"dataset_name": dataset_name,
		"split": split,
		"variant": variant,
		"study_preset": study_preset,
		"selector_preset": selector_preset,
		"resolved_selectors": list(resolved_selectors),
		"resolved_token_budgets": list(resolved_token_budgets)
		if resolved_token_budgets is not None
		else None,
		"resolved_budget_ratios": list(resolved_budget_ratios)
		if resolved_budget_ratios is not None
		else None,
		"case_ids_file": str(case_ids_file) if case_ids_file is not None else None,
		"total_selected_cases": total_selected_cases,
		"control_selector_name": control_selector_name,
		"missing_chunks": list(missing_chunks) if missing_chunks is not None else None,
	}
	if extra:
		payload.update(extra)
	(output_dir / "run_manifest.json").write_text(
		json.dumps(payload, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)


def _ordered_case_ids_from_records(records: Sequence[dict[str, Any]]) -> list[str]:
	ordered_case_ids: list[str] = []
	seen: set[str] = set()
	for record in records:
		case_id = str(record["case_id"])
		if case_id in seen:
			continue
		seen.add(case_id)
		ordered_case_ids.append(case_id)
	return ordered_case_ids


def _case_id_value(case: Any) -> str:
	return str(case if isinstance(case, str) else case.case_id)


def _selection_family(selector: Any) -> str:
	if isinstance(selector, BudgetFillSelector):
		return "budget_fill"
	if isinstance(selector, CanonicalConstrainedMultipathSelector):
		return "controller_multipath"
	if isinstance(selector, CanonicalSinglePathSelector):
		return "dynamic_walk"
	if isinstance(selector, CanonicalSearchSelector):
		return "path_search"
	if isinstance(
		selector, (CanonicalIterativeDenseSelector, CanonicalMDRLightSelector)
	):
		return "iterative_dense"
	return "stateless"


def _build_selection_plan(
	*,
	cases: Sequence[Any],
	budgets: Sequence[EvaluationBudget],
	selectors: Sequence[Any],
) -> list[SelectionPlanItem]:
	plan: list[SelectionPlanItem] = []
	for case in cases:
		case_id = str(case.case_id)
		for budget in budgets:
			for selector in selectors:
				plan.append(
					SelectionPlanItem(
						selection_key=build_selection_key(
							case_id=case_id,
							budget_label=budget.budget_label,
							selector_name=selector.name,
						),
						case_id=case_id,
						budget_label=budget.budget_label,
						selector_name=selector.name,
						selector_family=_selection_family(selector),
					)
				)
	return plan


def _selection_keys_by_case(
	plan_items: Sequence[SelectionPlanItem],
) -> dict[str, list[str]]:
	grouped: dict[str, list[str]] = {}
	for item in plan_items:
		grouped.setdefault(item.case_id, []).append(item.selection_key)
	return grouped


def _selection_keys_by_case_selector(
	plan_items: Sequence[SelectionPlanItem],
) -> dict[str, dict[str, list[str]]]:
	grouped: dict[str, dict[str, list[str]]] = {}
	for item in plan_items:
		case_group = grouped.setdefault(item.case_id, {})
		case_group.setdefault(item.selector_name, []).append(item.selection_key)
	return grouped


def _selection_progress_snapshot(
	*,
	plan_items: Sequence[SelectionPlanItem],
	completed_selection_keys: set[str],
	selection_keys_by_case: dict[str, list[str]],
	selection_keys_by_case_selector: dict[str, dict[str, list[str]]],
	current_case_id: str | None,
	current_selector_name: str | None = None,
	current_budget_label: str | None = None,
) -> _SelectionProgressSnapshot:
	case_total_selectors: int | None = None
	case_completed_selectors: int | None = None
	case_total_selections: int | None = None
	case_completed_selections: int | None = None
	if current_case_id is not None:
		case_selection_keys = selection_keys_by_case.get(current_case_id, [])
		case_total_selections = len(case_selection_keys)
		case_completed_selections = sum(
			1 for key in case_selection_keys if key in completed_selection_keys
		)
		selector_groups = selection_keys_by_case_selector.get(current_case_id, {})
		case_total_selectors = len(selector_groups)
		case_completed_selectors = sum(
			1
			for selector_keys in selector_groups.values()
			if selector_keys
			and all(key in completed_selection_keys for key in selector_keys)
		)
	return _SelectionProgressSnapshot(
		total_selections=len(plan_items),
		completed_selections=len(completed_selection_keys),
		current_selector_name=current_selector_name,
		current_budget_label=current_budget_label,
		case_total_selectors=case_total_selectors,
		case_completed_selectors=case_completed_selectors,
		case_total_selections=case_total_selections,
		case_completed_selections=case_completed_selections,
	)


def _ordered_completed_selection_keys(
	plan_items: Sequence[SelectionPlanItem],
	completed_selection_keys: set[str],
) -> list[str]:
	return [
		item.selection_key
		for item in plan_items
		if item.selection_key in completed_selection_keys
	]


def _completed_case_ids(
	*,
	planned_case_ids: Sequence[str],
	selection_keys_by_case: dict[str, list[str]],
	completed_selection_keys: set[str],
) -> list[str]:
	completed: list[str] = []
	for case_id in planned_case_ids:
		planned_keys = selection_keys_by_case.get(case_id, [])
		if planned_keys and all(
			key in completed_selection_keys for key in planned_keys
		):
			completed.append(case_id)
	return completed


def _selection_stage_payload(
	case: EvaluationCase, selection: SelectionResult
) -> dict[str, Any]:
	evaluation = CaseEvaluation(case=case, selections=[selection])
	return {
		"selection_record": _selection_record(evaluation, selection),
		"selector_log_records": _selector_log_records(evaluation, selection),
	}


def _case_from_selection_record(record: dict[str, Any]) -> EvaluationCase:
	return EvaluationCase(
		case_id=str(record["case_id"]),
		query=str(record["query"]),
		expected_answer=None
		if record.get("expected_answer") is None
		else str(record["expected_answer"]),
		dataset_name=str(record["dataset_name"]),
		gold_support_nodes=[
			str(node_id) for node_id in record.get("gold_support_nodes", [])
		],
		gold_start_nodes=[
			str(node_id) for node_id in record.get("gold_start_nodes", [])
		],
		gold_path_nodes=(
			None
			if record.get("gold_path_nodes") is None
			else [str(node_id) for node_id in record.get("gold_path_nodes", [])]
		),
		question_type=coerce_question_type(record.get("question_type")),
	)


def _selection_result_from_record(
	record: dict[str, Any],
	selector_log_records: Sequence[dict[str, Any]],
) -> SelectionResult:
	selection_payload = dict(record["selection"])
	budget_payload = dict(selection_payload["budget"])
	if budget_payload.get("token_budget_tokens") is not None:
		budget = EvaluationBudget(
			token_budget_tokens=int(budget_payload["token_budget_tokens"])
		)
	else:
		budget = EvaluationBudget(
			token_budget_ratio=float(budget_payload["token_budget_ratio"])
		)
	corpus_payload = dict(selection_payload["corpus"])
	metrics_payload = dict(selection_payload["metrics"])
	return SelectionResult(
		selector_name=str(record["selector"]),
		budget=budget,
		corpus=SelectedCorpus(
			node_ids=[str(node_id) for node_id in corpus_payload.get("node_ids", [])],
			edge_contexts=[
				SelectedEdgeContext(
					source=str(edge["source"]),
					target=str(edge["target"]),
					anchor_text=str(edge["anchor_text"]),
					sentence=str(edge["sentence"]),
					score=float(edge["score"]),
				)
				for edge in corpus_payload.get("edge_contexts", [])
			],
			token_estimate=int(corpus_payload.get("token_estimate", 0)),
			root_node_ids=[
				str(node_id) for node_id in corpus_payload.get("root_node_ids", [])
			],
		),
		metrics=SelectionMetrics(
			budget_mode=cast(
				Literal["tokens", "ratio"], str(metrics_payload["budget_mode"])
			),
			budget_value=metrics_payload["budget_value"],
			budget_label=str(metrics_payload["budget_label"]),
			token_budget_ratio=(
				None
				if metrics_payload.get("token_budget_ratio") is None
				else float(metrics_payload["token_budget_ratio"])
			),
			token_budget_tokens=(
				None
				if metrics_payload.get("token_budget_tokens") is None
				else int(metrics_payload["token_budget_tokens"])
			),
			budget_token_limit=int(metrics_payload["budget_token_limit"]),
			selection_runtime_s=float(metrics_payload["selection_runtime_s"]),
			selected_nodes_count=int(metrics_payload["selected_nodes_count"]),
			selected_token_estimate=int(metrics_payload["selected_token_estimate"]),
			compression_ratio=float(metrics_payload["compression_ratio"]),
			budget_adherence=bool(metrics_payload["budget_adherence"]),
			budget_utilization=float(metrics_payload["budget_utilization"]),
			empty_selection=bool(metrics_payload["empty_selection"]),
			start_hit=metrics_payload.get("start_hit"),
			support_recall=metrics_payload.get("support_recall"),
			support_precision=metrics_payload.get("support_precision"),
			support_f1=metrics_payload.get("support_f1"),
			support_f1_zero_on_empty=metrics_payload.get("support_f1_zero_on_empty"),
			support_set_em=metrics_payload.get("support_set_em"),
			path_hit=metrics_payload.get("path_hit"),
		),
		trace=[
			selection_trace_step_from_dict(step)
			for step in selection_payload.get("trace", [])
		],
		end_to_end=(
			None
			if record.get("end_to_end") is None
			else EndToEndResult(
				mode=str(record["end_to_end"]["mode"]),
				model=None
				if record["end_to_end"].get("model") is None
				else str(record["end_to_end"]["model"]),
				answer=str(record["end_to_end"]["answer"]),
				confidence=float(record["end_to_end"]["confidence"]),
				evidence_count=int(record["end_to_end"]["evidence_count"]),
				em=record["end_to_end"].get("em"),
				f1=record["end_to_end"].get("f1"),
				runtime_s=float(record["end_to_end"]["runtime_s"]),
				prompt_tokens=(
					None
					if record["end_to_end"].get("prompt_tokens") is None
					else int(record["end_to_end"]["prompt_tokens"])
				),
				completion_tokens=(
					None
					if record["end_to_end"].get("completion_tokens") is None
					else int(record["end_to_end"]["completion_tokens"])
				),
				total_tokens=(
					None
					if record["end_to_end"].get("total_tokens") is None
					else int(record["end_to_end"]["total_tokens"])
				),
			)
		),
		stop_reason=None
		if selection_payload.get("stop_reason") is None
		else str(selection_payload["stop_reason"]),
		graphrag_input_path=(
			None
			if selection_payload.get("graphrag_input_path") is None
			else str(selection_payload["graphrag_input_path"])
		),
		selector_metadata=selector_metadata_from_dict(
			selection_payload.get("selector_metadata")
		),
		selector_usage=selector_usage_from_dict(selection_payload.get("selector_usage"))
		or SelectorUsage(),
		selector_logs=[
			walk_step_log_from_dict(record["log"]) for record in selector_log_records
		],
	)


def _selection_from_stage_payload(
	payload: dict[str, Any],
) -> tuple[EvaluationCase, SelectionResult]:
	record = dict(payload["selection_record"])
	case = _case_from_selection_record(record)
	selection = _selection_result_from_record(
		record, payload.get("selector_log_records", [])
	)
	return case, selection


def _bundle_from_selection(
	case: EvaluationCase, selection: SelectionResult
) -> SelectionCheckpointBundle:
	payload = _selection_stage_payload(case, selection)
	record = dict(payload["selection_record"])
	return SelectionCheckpointBundle(
		selection_key=build_selection_key(
			case_id=case.case_id,
			budget_label=selection.budget.budget_label,
			selector_name=selection.selector_name,
		),
		case_id=case.case_id,
		budget_label=selection.budget.budget_label,
		selector_name=selection.selector_name,
		selection_record=record,
		selector_log_records=[dict(item) for item in payload["selector_log_records"]],
	)


def _build_timeout_result_bundle(
	*,
	item: SelectionPlanItem,
	case: EvaluationCase,
	budget: EvaluationBudget,
	timeout_exc: SelectionStageTimeout,
	checkpoint_store: CheckpointStore,
) -> SelectionCheckpointBundle:
	"""Build a minimal checkpoint bundle for a timed-out selection and persist it."""
	empty_selection = SelectionResult(
		selector_name=item.selector_name,
		budget=budget,
		corpus=SelectedCorpus(
			node_ids=[],
			edge_contexts=[],
			token_estimate=0,
			root_node_ids=[],
		),
		metrics=SelectionMetrics(
			budget_mode=budget.budget_mode,
			budget_value=budget.budget_value,
			budget_label=budget.budget_label,
			token_budget_ratio=budget.token_budget_ratio,
			token_budget_tokens=budget.token_budget_tokens,
			budget_token_limit=0,
			selection_runtime_s=0.0,
			selected_nodes_count=0,
			selected_token_estimate=0,
			compression_ratio=0.0,
			budget_adherence=True,
			budget_utilization=0.0,
			empty_selection=True,
			support_f1=0.0,
			support_f1_zero_on_empty=0.0,
			support_precision=0.0,
			support_recall=0.0,
			support_set_em=0.0,
		),
		trace=[],
		stop_reason="timeout",
		selector_usage=SelectorUsage(),
	)
	bundle = _bundle_from_selection(case, empty_selection)
	checkpoint_store.save_selection_checkpoint(bundle)
	checkpoint_store.remove_resume_state(item.selection_key)
	return bundle


def _evaluations_from_checkpoint_bundles(
	*,
	plan_items: Sequence[SelectionPlanItem],
	bundles_by_key: dict[str, SelectionCheckpointBundle],
	completed_case_ids: Sequence[str],
) -> list[CaseEvaluation]:
	items_by_case: dict[str, list[SelectionPlanItem]] = {}
	for item in plan_items:
		items_by_case.setdefault(item.case_id, []).append(item)

	evaluations: list[CaseEvaluation] = []
	for case_id in completed_case_ids:
		case_items = items_by_case.get(case_id, [])
		selections: list[SelectionResult] = []
		case: EvaluationCase | None = None
		for item in case_items:
			bundle = bundles_by_key[item.selection_key]
			case = _case_from_selection_record(bundle.selection_record)
			selections.append(
				_selection_result_from_record(
					bundle.selection_record, bundle.selector_log_records
				)
			)
		if case is not None:
			evaluations.append(CaseEvaluation(case=case, selections=selections))
	return evaluations


def _rebuild_public_outputs(
	*,
	output_dir: Path,
	plan_items: Sequence[SelectionPlanItem],
	planned_case_ids: Sequence[str],
	store: CheckpointStore,
	study_preset: str | None,
	control_selector_name: str | None,
	dataset_name: str,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	completed_selection_keys = set(store.list_selection_keys())
	bundles_by_key = {
		selection_key: store.load_selection_checkpoint(selection_key)
		for selection_key in completed_selection_keys
	}
	bundles_by_key = {
		key: bundle for key, bundle in bundles_by_key.items() if bundle is not None
	}
	completed_selection_keys = set(bundles_by_key)
	case_keys = _selection_keys_by_case(plan_items)
	completed_case_ids = _completed_case_ids(
		planned_case_ids=planned_case_ids,
		selection_keys_by_case=case_keys,
		completed_selection_keys=completed_selection_keys,
	)
	evaluations = _evaluations_from_checkpoint_bundles(
		plan_items=plan_items,
		bundles_by_key=bundles_by_key,
		completed_case_ids=completed_case_ids,
	)
	results_records: list[dict[str, Any]] = []
	selector_log_records: list[dict[str, Any]] = []
	for evaluation in evaluations:
		for selection in evaluation.selections:
			results_records.append(_selection_record(evaluation, selection))
			selector_log_records.extend(_selector_log_records(evaluation, selection))

	atomic_write_text(
		output_dir / "results.jsonl",
		"".join(
			json.dumps(record, ensure_ascii=False) + "\n" for record in results_records
		),
	)
	atomic_write_text(
		output_dir / "selector_logs.jsonl",
		"".join(
			json.dumps(record, ensure_ascii=False) + "\n"
			for record in selector_log_records
		),
	)
	_write_evaluated_case_ids(output_dir, completed_case_ids)

	if results_records:
		summary = _summarize_result_records(results_records)
		_write_summary_file(
			summary_path=output_dir / "summary.json",
			summary=summary,
			study_preset=study_preset,
			control_selector_name=control_selector_name,
			results_path=output_dir / "results.jsonl",
		)
	else:
		for path in (
			output_dir / "summary.json",
			output_dir / "summary_rows.csv",
			output_dir / "study_comparison_rows.csv",
			output_dir / "subset_comparison_rows.csv",
		):
			path.unlink(missing_ok=True)
		summary = ExperimentSummary(
			dataset_name=dataset_name, total_cases=0, selector_budgets=[]
		)
	return evaluations, summary


def _has_run_artifacts(output_dir: Path) -> bool:
	return any(
		path.exists()
		for path in (
			output_dir / "run_manifest.json",
			output_dir / "run_state.json",
			output_dir / "results.jsonl",
			output_dir / "selector_logs.jsonl",
			output_dir / "summary.json",
			output_dir / "evaluated_case_ids.txt",
			output_dir / "_checkpoints",
			output_dir / "chunk.json",
		)
	)


def _clear_run_artifacts(output_dir: Path, checkpoint_store: CheckpointStore) -> None:
	checkpoint_store.clear_artifacts()
	for path in (
		output_dir / "results.jsonl",
		output_dir / "selector_logs.jsonl",
		output_dir / "summary.json",
		output_dir / "summary_rows.csv",
		output_dir / "study_comparison_rows.csv",
		output_dir / "subset_comparison_rows.csv",
		output_dir / "run_manifest.json",
		output_dir / "evaluated_case_ids.txt",
		output_dir / "chunk.json",
	):
		path.unlink(missing_ok=True)
	shutil.rmtree(output_dir / "graphrag_inputs", ignore_errors=True)


def _build_runtime_config_payload(
	*,
	dataset_name: str,
	split: str | None,
	variant: str | None,
	planned_case_ids: Sequence[str],
	plan_items: Sequence[SelectionPlanItem],
	selectors: Sequence[Any],
	budgets: Sequence[EvaluationBudget],
	selector_provider: str,
	selector_model: str | None,
	selector_api_key_env: str | None,
	selector_base_url: str | None,
	selector_openai_api_mode: str | None,
	mdr_home: str | Path | None,
	mdr_artifact_manifest: str | Path | None,
	sentence_transformer_model: str | None,
	sentence_transformer_device: str | None,
	with_e2e: bool,
	answerer_mode: str,
	answer_provider: str | None,
	answer_model: str | None,
	answer_api_key_env: str | None,
	answer_base_url: str | None,
	export_graphrag_inputs: bool,
) -> dict[str, Any]:
	return {
		"dataset_name": dataset_name,
		"split": split,
		"variant": variant,
		"planned_case_ids": list(planned_case_ids),
		"planned_selection_keys": [item.selection_key for item in plan_items],
		"selectors": [selector.name for selector in selectors],
		"selector_families": {
			selector.name: _selection_family(selector) for selector in selectors
		},
		"budget_labels": [budget.budget_label for budget in budgets],
		"selector_provider": selector_provider,
		"selector_model": selector_model,
		"selector_api_key_env": selector_api_key_env,
		"selector_base_url": selector_base_url,
		"selector_openai_api_mode": selector_openai_api_mode,
		"mdr_home": str(mdr_home) if mdr_home is not None else None,
		"mdr_artifact_manifest": str(mdr_artifact_manifest)
		if mdr_artifact_manifest is not None
		else None,
		"sentence_transformer_model": sentence_transformer_model,
		"sentence_transformer_device": sentence_transformer_device,
		"with_e2e": with_e2e,
		"answerer_mode": answerer_mode,
		"answer_provider": answer_provider
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"answer_model": answer_model
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"answer_api_key_env": answer_api_key_env
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"answer_base_url": answer_base_url
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"export_graphrag_inputs": export_graphrag_inputs,
	}


def _validate_resume_configuration(
	*,
	manifest_path: Path,
	expected_manifest_payload: dict[str, Any],
	run_state: RunState,
	config_fingerprint: str,
) -> None:
	existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	keys_to_compare = (
		"dataset_name",
		"split",
		"variant",
		"study_preset",
		"selector_preset",
		"resolved_selectors",
		"resolved_token_budgets",
		"resolved_budget_ratios",
		"case_ids_file",
		"total_selected_cases",
		"control_selector_name",
		"selector_provider",
		"selector_model",
		"selector_api_key_env",
		"selector_base_url",
		"selector_openai_api_mode",
		"mdr_home",
		"mdr_artifact_manifest",
		"sentence_transformer_model",
		"sentence_transformer_device",
		"with_e2e",
		"answerer_mode",
		"answer_provider",
		"answer_model",
		"answer_api_key_env",
		"answer_base_url",
		"export_graphrag_inputs",
		"planned_case_ids",
		"planned_selection_keys",
	)
	mismatches = [
		key
		for key in keys_to_compare
		if existing_manifest.get(key) != expected_manifest_payload.get(key)
	]
	if run_state.config_fingerprint != config_fingerprint:
		mismatches.append("config_fingerprint")
	if run_state.planned_case_ids != list(
		expected_manifest_payload["planned_case_ids"]
	):
		mismatches.append("planned_case_ids")
	if run_state.planned_selection_keys != list(
		expected_manifest_payload["planned_selection_keys"]
	):
		mismatches.append("planned_selection_keys")
	if mismatches:
		fields = ", ".join(dict.fromkeys(mismatches))
		raise ValueError(
			f"Cannot resume because the experiment configuration changed: {fields}"
		)


def _save_selection_resume_state(
	*,
	checkpoint_store: CheckpointStore,
	item: SelectionPlanItem,
	stage: SelectionStage,
	payload: dict[str, Any],
) -> None:
	checkpoint_store.save_resume_state(
		SelectionResumeState(
			selection_key=item.selection_key,
			case_id=item.case_id,
			budget_label=item.budget_label,
			selector_name=item.selector_name,
			family=item.selector_family,
			stage=stage,
			payload=payload,
		)
	)


def _selection_checkpoint_detail(payload: dict[str, Any]) -> str | None:
	detail_parts: list[str] = []
	if "candidate_index" in payload:
		detail_parts.append(f"candidate_index={int(payload['candidate_index'])}")
	if "selected_node_ids" in payload:
		detail_parts.append(
			f"selected_nodes={len(payload.get('selected_node_ids', []))}"
		)
	if "token_cost_estimate" in payload:
		detail_parts.append(
			f"token_cost_estimate={int(payload['token_cost_estimate'])}"
		)
	if "trace" in payload:
		detail_parts.append(f"trace_len={len(payload.get('trace', []))}")
	if "base_resume_state" in payload:
		detail_parts.append("substage=base_selector")
	return " ".join(detail_parts) or None


def _execute_selector_body_raw(
	*,
	selector: Any,
	graph: Any,
	case: EvaluationCase,
	budget: EvaluationBudget,
	family: str,
	resume_payload: dict[str, Any] | None,
	checkpoint_callback: Callable[[dict[str, Any]], None] | None,
	stop_callback: Callable[[], None] | None,
) -> CorpusSelectionResult:
	if family == "dynamic_walk" and isinstance(selector, CanonicalSinglePathSelector):
		return selector.select(
			graph,
			case,
			budget,
			resume_state=resume_payload,
			checkpoint_callback=checkpoint_callback,
			stop_callback=stop_callback,
		)
	if family == "controller_multipath" and isinstance(
		selector, CanonicalConstrainedMultipathSelector
	):
		return selector.select(
			graph,
			case,
			budget,
			resume_state=resume_payload,
			checkpoint_callback=checkpoint_callback,
			stop_callback=stop_callback,
		)
	if family == "path_search" and isinstance(selector, CanonicalSearchSelector):
		return selector.select(
			graph,
			case,
			budget,
			resume_state=resume_payload,
			checkpoint_callback=checkpoint_callback,
			stop_callback=stop_callback,
		)
	if family == "iterative_dense" and isinstance(
		selector, (CanonicalIterativeDenseSelector, CanonicalMDRLightSelector)
	):
		return selector.select(
			graph,
			case,
			budget,
			resume_state=resume_payload,
			checkpoint_callback=checkpoint_callback,
			stop_callback=stop_callback,
		)
	return selector.select(graph, case, budget)


def _execute_selector_body(
	*,
	item: SelectionPlanItem,
	selector: Any,
	graph: Any,
	case: EvaluationCase,
	budget: EvaluationBudget,
	checkpoint_store: CheckpointStore,
	interrupt_controller: InterruptController,
	resume_state: SelectionResumeState | None,
	run_state: RunState,
	execution_config: _SelectionExecutionConfig,
	base_result_cache: dict[tuple[str, str], CorpusSelectionResult] | None = None,
) -> CorpusSelectionResult:
	monitor = _SelectionExecutionMonitor(
		item=item,
		checkpoint_store=checkpoint_store,
		run_state=run_state,
		heartbeat_interval_s=execution_config.heartbeat_interval_s,
	)
	if isinstance(selector, BudgetFillSelector):
		base_resume_payload = None
		fill_resume_payload = None
		if (
			resume_state is not None
			and resume_state.stage == SelectionStage.BASE_SELECTOR
		):
			base_resume_payload = dict(
				resume_state.payload.get("base_resume_state", {})
			)
		elif (
			resume_state is not None
			and resume_state.stage == SelectionStage.BUDGET_FILL
		):
			fill_resume_payload = dict(resume_state.payload)

		base_selector = selector.base_selector
		base_cache_key = (item.case_id, base_selector.name)

		def _base_checkpoint(payload: dict[str, Any]) -> None:
			_save_selection_resume_state(
				checkpoint_store=checkpoint_store,
				item=item,
				stage=SelectionStage.BASE_SELECTOR,
				payload={
					"base_family": _selection_family(base_selector),
					"base_resume_state": payload,
				},
			)
			monitor.heartbeat(
				detail=_selection_checkpoint_detail(
					{
						"base_resume_state": payload,
						"trace": payload.get("trace", []),
					}
				)
			)

		if fill_resume_payload is not None:
			base_result = corpus_selection_result_from_dict(
				dict(fill_resume_payload["base_result"])
			)
		elif base_result_cache is not None and base_cache_key in base_result_cache:
			base_result = base_result_cache[base_cache_key]
		else:
			with monitor.stage(
				SelectionStage.BASE_SELECTOR,
				timeout_s=_timeout_for_stage(
					execution_config, SelectionStage.BASE_SELECTOR
				),
			):
				base_result = _execute_selector_body_raw(
					selector=base_selector,
					graph=graph,
					case=case,
					budget=budget,
					family=_selection_family(base_selector),
					resume_payload=base_resume_payload,
					checkpoint_callback=_base_checkpoint,
					stop_callback=interrupt_controller.checkpoint,
				)
			if base_result_cache is not None:
				base_result_cache[base_cache_key] = base_result

		def _budget_fill_checkpoint(payload: dict[str, Any]) -> None:
			_save_selection_resume_state(
				checkpoint_store=checkpoint_store,
				item=item,
				stage=SelectionStage.BUDGET_FILL,
				payload=payload,
			)
			monitor.heartbeat(detail=_selection_checkpoint_detail(payload))

		with monitor.stage(
			SelectionStage.BUDGET_FILL,
			timeout_s=_timeout_for_stage(execution_config, SelectionStage.BUDGET_FILL),
		):
			return selector.select(
				graph,
				case,
				budget,
				base_result=base_result,
				resume_state=fill_resume_payload,
				checkpoint_callback=_budget_fill_checkpoint,
				stop_callback=interrupt_controller.checkpoint,
			)

	# Non-BudgetFill selectors (gold_support_context, full_corpus_upper_bound).
	# These produce identical results regardless of budget — cache across budgets.
	result_cache_key = (item.case_id, selector.name)
	if base_result_cache is not None and result_cache_key in base_result_cache:
		return base_result_cache[result_cache_key]

	selection_body_resume = None
	if resume_state is not None and resume_state.stage == SelectionStage.SELECTOR_BODY:
		selection_body_resume = dict(resume_state.payload)

	def _checkpoint(payload: dict[str, Any]) -> None:
		_save_selection_resume_state(
			checkpoint_store=checkpoint_store,
			item=item,
			stage=SelectionStage.SELECTOR_BODY,
			payload=payload,
		)
		monitor.heartbeat(detail=_selection_checkpoint_detail(payload))

	with monitor.stage(
		SelectionStage.SELECTOR_BODY,
		timeout_s=_timeout_for_stage(execution_config, SelectionStage.SELECTOR_BODY),
	):
		result = _execute_selector_body_raw(
			selector=selector,
			graph=graph,
			case=case,
			budget=budget,
			family=item.selector_family,
			resume_payload=selection_body_resume,
			checkpoint_callback=_checkpoint,
			stop_callback=interrupt_controller.checkpoint,
		)
	if base_result_cache is not None:
		base_result_cache[result_cache_key] = result
	return result


def _execute_selection(
	*,
	item: SelectionPlanItem,
	selector: Any,
	graph: Any,
	case: EvaluationCase,
	budget: EvaluationBudget,
	checkpoint_store: CheckpointStore,
	interrupt_controller: InterruptController,
	run_state: RunState,
	execution_config: _SelectionExecutionConfig,
	answerer: SupportsAnswer | None,
	export_graphrag_inputs: bool,
	output_dir: Path,
	base_result_cache: dict[tuple[str, str], CorpusSelectionResult] | None = None,
) -> SelectionCheckpointBundle:
	monitor = _SelectionExecutionMonitor(
		item=item,
		checkpoint_store=checkpoint_store,
		run_state=run_state,
		heartbeat_interval_s=execution_config.heartbeat_interval_s,
	)
	resume_state = checkpoint_store.load_resume_state(item.selection_key)

	if resume_state is not None and resume_state.stage in {
		SelectionStage.EXPORT_GRAPHRAG,
		SelectionStage.E2E,
		SelectionStage.FINAL_COMMIT,
	}:
		payload_case, selection = _selection_from_stage_payload(resume_state.payload)
		if payload_case.case_id != case.case_id:
			raise ValueError(f"Resume payload case mismatch for {item.selection_key}.")
		current_case = payload_case
	else:
		raw_result = _execute_selector_body(
			item=item,
			selector=selector,
			graph=graph,
			case=case,
			budget=budget,
			checkpoint_store=checkpoint_store,
			interrupt_controller=interrupt_controller,
			resume_state=resume_state,
			run_state=run_state,
			execution_config=execution_config,
			base_result_cache=base_result_cache,
		)
		selection = _selection_result_from_raw(
			graph=graph, case=case, budget=budget, raw=raw_result
		)
		current_case = case
		next_stage = (
			SelectionStage.EXPORT_GRAPHRAG
			if export_graphrag_inputs
			else SelectionStage.E2E
			if answerer is not None
			else SelectionStage.FINAL_COMMIT
		)
		_save_selection_resume_state(
			checkpoint_store=checkpoint_store,
			item=item,
			stage=next_stage,
			payload=_selection_stage_payload(current_case, selection),
		)
		interrupt_controller.checkpoint()
		resume_state = checkpoint_store.load_resume_state(item.selection_key)

	if (
		resume_state is not None
		and resume_state.stage == SelectionStage.EXPORT_GRAPHRAG
	):
		with monitor.stage(
			SelectionStage.EXPORT_GRAPHRAG,
			timeout_s=_timeout_for_stage(
				execution_config, SelectionStage.EXPORT_GRAPHRAG
			),
		):
			selection.graphrag_input_path = _write_graphrag_input(
				graph=graph,
				case_id=current_case.case_id,
				selection=selection,
				output_dir=output_dir,
			)
		next_stage = (
			SelectionStage.E2E if answerer is not None else SelectionStage.FINAL_COMMIT
		)
		_save_selection_resume_state(
			checkpoint_store=checkpoint_store,
			item=item,
			stage=next_stage,
			payload=_selection_stage_payload(current_case, selection),
		)
		interrupt_controller.checkpoint()
		resume_state = checkpoint_store.load_resume_state(item.selection_key)

	if resume_state is not None and resume_state.stage == SelectionStage.E2E:
		if answerer is None:
			raise ValueError("Encountered e2e resume state without an answerer.")
		with monitor.stage(
			SelectionStage.E2E,
			timeout_s=_timeout_for_stage(execution_config, SelectionStage.E2E),
		):
			selection.end_to_end = _run_end_to_end(
				graph=graph,
				case=current_case,
				node_ids=selection.corpus.node_ids,
				extractor=SubgraphExtractor(),
				answerer=answerer,
			)
		_save_selection_resume_state(
			checkpoint_store=checkpoint_store,
			item=item,
			stage=SelectionStage.FINAL_COMMIT,
			payload=_selection_stage_payload(current_case, selection),
		)
		interrupt_controller.checkpoint()

	with monitor.stage(
		SelectionStage.FINAL_COMMIT,
		timeout_s=_timeout_for_stage(execution_config, SelectionStage.FINAL_COMMIT),
	):
		bundle = _bundle_from_selection(current_case, selection)
		checkpoint_store.save_selection_checkpoint(bundle)
		checkpoint_store.remove_resume_state(item.selection_key)
	return bundle


def _warmup_selectors(selectors: Sequence[Any]) -> None:
	"""Pre-load heavy resources (e.g. sentence-transformer model) before the eval loop."""
	for selector in selectors:
		if hasattr(selector, "_get_embedder"):
			logger.info("Warming up embedder for selector %s", selector.name)
			selector._get_embedder()


def _run_loaded_experiment(
	*,
	graph: Any,
	graph_for_case: Callable[[EvaluationCase], Any] | None = None,
	selected_cases: Sequence[EvaluationCase],
	selectors: Sequence[Any],
	budgets: Sequence[EvaluationBudget],
	output_dir: Path,
	dataset_name: str,
	study_preset: str | None,
	selector_preset: str | None,
	token_budgets: Sequence[int] | None,
	budget_ratios: Sequence[float] | None,
	case_ids_file: str | Path | None,
	control_selector_name: str | None,
	selector_provider: str,
	selector_model: str | None,
	selector_api_key_env: str | None,
	selector_base_url: str | None,
	selector_openai_api_mode: str | None,
	mdr_home: str | Path | None,
	mdr_artifact_manifest: str | Path | None,
	sentence_transformer_model: str | None,
	sentence_transformer_device: str | None,
	with_e2e: bool,
	answerer_mode: str,
	answer_provider: str | None,
	answer_model: str | None,
	answer_api_key_env: str | None,
	answer_base_url: str | None,
	answer_cache_path: str | Path | None,
	export_graphrag_inputs: bool,
	progress_observer: ExperimentProgressObserver | None,
	split: str | None = None,
	variant: str | None = None,
	chunk_meta: dict[str, Any] | None = None,
	store_uri: str | Path | None = None,
	resume: bool = False,
	restart: bool = False,
) -> tuple[list[CaseEvaluation], ExperimentSummary]:
	output_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_store = CheckpointStore(output_dir)
	existing_artifacts_before_lock = _has_run_artifacts(output_dir)
	answerer = _build_answerer(
		with_e2e=with_e2e,
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
	)
	plan_items = _build_selection_plan(
		cases=selected_cases, budgets=budgets, selectors=selectors
	)
	planned_case_ids = [str(case.case_id) for case in selected_cases]
	runtime_config_payload = _build_runtime_config_payload(
		dataset_name=dataset_name,
		split=split,
		variant=variant,
		planned_case_ids=planned_case_ids,
		plan_items=plan_items,
		selectors=selectors,
		budgets=budgets,
		selector_provider=selector_provider,
		selector_model=selector_model,
		selector_api_key_env=selector_api_key_env,
		selector_base_url=selector_base_url,
		selector_openai_api_mode=selector_openai_api_mode,
		mdr_home=mdr_home,
		mdr_artifact_manifest=mdr_artifact_manifest,
		sentence_transformer_model=sentence_transformer_model,
		sentence_transformer_device=sentence_transformer_device,
		with_e2e=with_e2e,
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		export_graphrag_inputs=export_graphrag_inputs,
	)
	config_fingerprint = build_config_fingerprint(runtime_config_payload)
	manifest_payload = {
		"dataset_name": dataset_name,
		"split": split,
		"variant": variant,
		"study_preset": study_preset,
		"selector_preset": selector_preset,
		"resolved_selectors": [selector.name for selector in selectors],
		"resolved_token_budgets": list(token_budgets)
		if token_budgets is not None
		else None,
		"resolved_budget_ratios": list(budget_ratios)
		if budget_ratios is not None
		else None,
		"case_ids_file": str(case_ids_file) if case_ids_file is not None else None,
		"total_selected_cases": len(selected_cases),
		"control_selector_name": control_selector_name,
		"selector_provider": selector_provider,
		"selector_model": selector_model,
		"selector_api_key_env": selector_api_key_env,
		"selector_base_url": selector_base_url,
		"selector_openai_api_mode": selector_openai_api_mode,
		"mdr_home": str(mdr_home) if mdr_home is not None else None,
		"mdr_artifact_manifest": str(mdr_artifact_manifest)
		if mdr_artifact_manifest is not None
		else None,
		"sentence_transformer_model": sentence_transformer_model,
		"sentence_transformer_device": sentence_transformer_device,
		"with_e2e": with_e2e,
		"answerer_mode": answerer_mode,
		"answer_provider": answer_provider
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"answer_model": answer_model
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"answer_api_key_env": answer_api_key_env
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"answer_base_url": answer_base_url
		if with_e2e and answerer_mode == "llm_fixed"
		else None,
		"export_graphrag_inputs": export_graphrag_inputs,
		"planned_case_ids": planned_case_ids,
		"planned_selection_keys": [item.selection_key for item in plan_items],
		"config_fingerprint": config_fingerprint,
	}
	if chunk_meta is not None:
		manifest_payload.update(
			{
				"chunk_size": chunk_meta.get("chunk_size"),
				"chunk_index": chunk_meta.get("chunk_index"),
				"case_start": chunk_meta.get("case_start"),
				"case_limit": chunk_meta.get("case_limit"),
			}
		)

	with checkpoint_store.acquire_lock():
		if restart:
			_clear_run_artifacts(output_dir, checkpoint_store)

		has_existing_artifacts = existing_artifacts_before_lock and not restart
		if has_existing_artifacts and not resume and not restart:
			raise ValueError(
				f"Output directory {output_dir} already contains experiment artifacts. Use --resume or --restart."
			)
		if resume and not has_existing_artifacts:
			raise ValueError(
				f"Cannot resume because {output_dir} does not contain experiment artifacts."
			)

		checkpoint_store.ensure_layout()
		manifest_path = output_dir / "run_manifest.json"

		if resume:
			run_state = checkpoint_store.load_run_state()
			if run_state is None or not manifest_path.exists():
				raise ValueError(
					f"Cannot resume because {output_dir} is missing run_state.json or run_manifest.json."
				)
			_validate_resume_configuration(
				manifest_path=manifest_path,
				expected_manifest_payload=manifest_payload,
				run_state=run_state,
				config_fingerprint=config_fingerprint,
			)
		else:
			run_state = RunState(
				version=RUN_STATE_VERSION,
				status=RunStatus.PENDING,
				config_fingerprint=config_fingerprint,
				planned_case_ids=list(planned_case_ids),
				planned_selection_keys=[item.selection_key for item in plan_items],
			)
			checkpoint_store.save_run_state(run_state)
			_write_run_manifest(
				output_dir,
				dataset_name=dataset_name,
				split=split,
				variant=variant,
				study_preset=study_preset,
				selector_preset=selector_preset,
				resolved_selectors=[selector.name for selector in selectors],
				resolved_token_budgets=token_budgets,
				resolved_budget_ratios=budget_ratios,
				case_ids_file=case_ids_file,
				total_selected_cases=len(selected_cases),
				control_selector_name=control_selector_name,
				extra={
					key: value
					for key, value in manifest_payload.items()
					if key
					not in {
						"dataset_name",
						"split",
						"variant",
						"study_preset",
						"selector_preset",
						"resolved_selectors",
						"resolved_token_budgets",
						"resolved_budget_ratios",
						"case_ids_file",
						"total_selected_cases",
						"control_selector_name",
					}
				},
			)

		evaluations, summary = _rebuild_public_outputs(
			output_dir=output_dir,
			plan_items=plan_items,
			planned_case_ids=planned_case_ids,
			store=checkpoint_store,
			study_preset=study_preset,
			control_selector_name=control_selector_name,
			dataset_name=dataset_name,
		)

		if chunk_meta is not None:
			chunk_payload = {
				**chunk_meta,
				"store_uri": str(store_uri) if store_uri is not None else None,
				"selectors": [selector.name for selector in selectors],
				"study_preset": study_preset,
				"selector_preset": selector_preset,
				"token_budgets": list(token_budgets)
				if token_budgets is not None
				else None,
				"budget_ratios": list(budget_ratios)
				if budget_ratios is not None
				else None,
				"case_ids_file": str(case_ids_file)
				if case_ids_file is not None
				else None,
				"selector_provider": selector_provider,
				"selector_model": selector_model,
				"selector_base_url": selector_base_url,
				"with_e2e": with_e2e,
				"answerer_mode": answerer_mode,
				"answer_provider": answer_provider
				if with_e2e and answerer_mode == "llm_fixed"
				else None,
				"answer_model": answer_model
				if with_e2e and answerer_mode == "llm_fixed"
				else None,
				"answer_api_key_env": answer_api_key_env
				if with_e2e and answerer_mode == "llm_fixed"
				else None,
				"answer_base_url": answer_base_url
				if with_e2e and answerer_mode == "llm_fixed"
				else None,
				"answer_cache_path": str(answer_cache_path)
				if answer_cache_path is not None
				else None,
				"export_graphrag_inputs": export_graphrag_inputs,
			}
			atomic_write_text(
				output_dir / "chunk.json",
				json.dumps(chunk_payload, ensure_ascii=False, indent=2),
			)

		case_by_id = {str(case.case_id): case for case in selected_cases}
		selector_by_name = {selector.name: selector for selector in selectors}
		budget_by_label = {budget.budget_label: budget for budget in budgets}
		selection_keys_by_case = _selection_keys_by_case(plan_items)
		selection_keys_by_case_selector = _selection_keys_by_case_selector(plan_items)
		plan_items_by_key = {item.selection_key: item for item in plan_items}

		interrupt_controller = InterruptController()
		interrupt_controller.install()
		completed_selection_keys: set[str] = set()
		try:
			completed_selection_keys = set(checkpoint_store.list_selection_keys())
			run_state.status = RunStatus.RUNNING
			run_state.interrupted_reason = None
			run_state.last_error = None
			run_state.completed_selection_keys = _ordered_completed_selection_keys(
				plan_items, completed_selection_keys
			)
			run_state.completed_case_ids = _completed_case_ids(
				planned_case_ids=planned_case_ids,
				selection_keys_by_case=selection_keys_by_case,
				completed_selection_keys=completed_selection_keys,
			)
			checkpoint_store.save_run_state(run_state)

			initial_progress = _selection_progress_snapshot(
				plan_items=plan_items,
				completed_selection_keys=completed_selection_keys,
				selection_keys_by_case=selection_keys_by_case,
				selection_keys_by_case_selector=selection_keys_by_case_selector,
				current_case_id=None,
			)
			# Warm up selector models (e.g. sentence-transformer) before evaluation
			_notify_progress(
				progress_observer,
				dataset_name=dataset_name,
				phase="initializing",
				total_cases=len(selected_cases),
				completed_cases=0,
			)
			_warmup_selectors(selectors)
			execution_config = _load_selection_execution_config()
			logger.info(
				"Selection execution config heartbeat_interval_s=%s selection_stage_timeout_s=%s budget_fill_timeout_s=%s",
				execution_config.heartbeat_interval_s,
				execution_config.selection_stage_timeout_s,
				execution_config.budget_fill_timeout_s,
			)

			_notify_progress(
				progress_observer,
				dataset_name=dataset_name,
				phase="evaluating",
				total_cases=len(selected_cases),
				completed_cases=len(run_state.completed_case_ids),
				total_selections=initial_progress.total_selections,
				completed_selections=initial_progress.completed_selections,
				summary=summary if summary.total_cases > 0 else None,
			)

			base_result_cache: dict[tuple[str, str], CorpusSelectionResult] = {}
			current_cache_case_id: str | None = None
			for item in plan_items:
				if item.selection_key in completed_selection_keys:
					continue
				if item.case_id != current_cache_case_id:
					base_result_cache.clear()
					current_cache_case_id = item.case_id
				case = case_by_id[item.case_id]
				run_state.current_case_id = item.case_id
				run_state.current_selection_key = item.selection_key
				run_state.current_selector_name = item.selector_name
				run_state.current_budget_label = item.budget_label
				run_state.current_stage = None
				run_state.stage_started_at = None
				run_state.last_heartbeat_at = None
				checkpoint_store.save_run_state(run_state)
				current_progress = _selection_progress_snapshot(
					plan_items=plan_items,
					completed_selection_keys=completed_selection_keys,
					selection_keys_by_case=selection_keys_by_case,
					selection_keys_by_case_selector=selection_keys_by_case_selector,
					current_case_id=case.case_id,
					current_selector_name=item.selector_name,
					current_budget_label=item.budget_label,
				)
				_notify_progress(
					progress_observer,
					dataset_name=dataset_name,
					phase="evaluating",
					total_cases=len(selected_cases),
					completed_cases=len(run_state.completed_case_ids),
					current_case_id=case.case_id,
					current_query=case.query,
					total_selections=current_progress.total_selections,
					completed_selections=current_progress.completed_selections,
					current_selector_name=current_progress.current_selector_name,
					current_budget_label=current_progress.current_budget_label,
					case_total_selectors=current_progress.case_total_selectors,
					case_completed_selectors=current_progress.case_completed_selectors,
					case_total_selections=current_progress.case_total_selections,
					case_completed_selections=current_progress.case_completed_selections,
					summary=summary if summary.total_cases > 0 else None,
				)
				case_graph = (
					graph_for_case(case) if graph_for_case is not None else graph
				)
				try:
					bundle = _execute_selection(
						item=item,
						selector=selector_by_name[item.selector_name],
						graph=case_graph,
						case=case,
						budget=budget_by_label[item.budget_label],
						checkpoint_store=checkpoint_store,
						interrupt_controller=interrupt_controller,
						run_state=run_state,
						execution_config=execution_config,
						answerer=answerer,
						export_graphrag_inputs=export_graphrag_inputs,
						output_dir=output_dir,
						base_result_cache=base_result_cache,
					)
				except SelectionStageTimeout as timeout_exc:
					logger.warning(
						"Selection timed out, skipping: case=%s selector=%s budget=%s — %s",
						item.case_id,
						item.selector_name,
						item.budget_label,
						timeout_exc,
					)
					bundle = _build_timeout_result_bundle(
						item=item,
						case=case,
						budget=budget_by_label[item.budget_label],
						timeout_exc=timeout_exc,
						checkpoint_store=checkpoint_store,
					)
				completed_selection_keys.add(bundle.selection_key)
				run_state.completed_selection_keys = _ordered_completed_selection_keys(
					plan_items, completed_selection_keys
				)
				previous_completed_case_ids = set(run_state.completed_case_ids)
				run_state.completed_case_ids = _completed_case_ids(
					planned_case_ids=planned_case_ids,
					selection_keys_by_case=selection_keys_by_case,
					completed_selection_keys=completed_selection_keys,
				)
				run_state.current_case_id = None
				run_state.current_selection_key = None
				run_state.current_selector_name = None
				run_state.current_budget_label = None
				run_state.current_stage = None
				run_state.stage_started_at = None
				run_state.last_heartbeat_at = None
				checkpoint_store.save_run_state(run_state)
				completed_progress = _selection_progress_snapshot(
					plan_items=plan_items,
					completed_selection_keys=completed_selection_keys,
					selection_keys_by_case=selection_keys_by_case,
					selection_keys_by_case_selector=selection_keys_by_case_selector,
					current_case_id=case.case_id,
					current_selector_name=item.selector_name,
					current_budget_label=item.budget_label,
				)
				if set(run_state.completed_case_ids) != previous_completed_case_ids:
					evaluations, summary = _rebuild_public_outputs(
						output_dir=output_dir,
						plan_items=plan_items,
						planned_case_ids=planned_case_ids,
						store=checkpoint_store,
						study_preset=study_preset,
						control_selector_name=control_selector_name,
						dataset_name=dataset_name,
					)
				_notify_progress(
					progress_observer,
					dataset_name=dataset_name,
					phase="evaluating",
					total_cases=len(selected_cases),
					completed_cases=len(run_state.completed_case_ids),
					total_selections=completed_progress.total_selections,
					completed_selections=completed_progress.completed_selections,
					current_case_id=case.case_id,
					current_query=case.query,
					current_selector_name=completed_progress.current_selector_name,
					current_budget_label=completed_progress.current_budget_label,
					case_total_selectors=completed_progress.case_total_selectors,
					case_completed_selectors=completed_progress.case_completed_selectors,
					case_total_selections=completed_progress.case_total_selections,
					case_completed_selections=completed_progress.case_completed_selections,
					summary=summary if summary.total_cases > 0 else None,
				)
				interrupt_controller.checkpoint()

			final_progress = _selection_progress_snapshot(
				plan_items=plan_items,
				completed_selection_keys=completed_selection_keys,
				selection_keys_by_case=selection_keys_by_case,
				selection_keys_by_case_selector=selection_keys_by_case_selector,
				current_case_id=None,
			)
			_notify_progress(
				progress_observer,
				dataset_name=dataset_name,
				phase="finalizing",
				total_cases=len(selected_cases),
				completed_cases=len(run_state.completed_case_ids),
				total_selections=final_progress.total_selections,
				completed_selections=final_progress.completed_selections,
				summary=summary if summary.total_cases > 0 else None,
			)
			evaluations, summary = _rebuild_public_outputs(
				output_dir=output_dir,
				plan_items=plan_items,
				planned_case_ids=planned_case_ids,
				store=checkpoint_store,
				study_preset=study_preset,
				control_selector_name=control_selector_name,
				dataset_name=dataset_name,
			)
			run_state.status = RunStatus.COMPLETED
			run_state.current_case_id = None
			run_state.current_selection_key = None
			run_state.current_selector_name = None
			run_state.current_budget_label = None
			run_state.current_stage = None
			run_state.stage_started_at = None
			run_state.last_heartbeat_at = None
			run_state.interrupted_reason = None
			run_state.last_error = None
			run_state.completed_selection_keys = _ordered_completed_selection_keys(
				plan_items, completed_selection_keys
			)
			run_state.completed_case_ids = _completed_case_ids(
				planned_case_ids=planned_case_ids,
				selection_keys_by_case=selection_keys_by_case,
				completed_selection_keys=completed_selection_keys,
			)
			checkpoint_store.save_run_state(run_state)
		except StopRequested:
			evaluations, summary = _rebuild_public_outputs(
				output_dir=output_dir,
				plan_items=plan_items,
				planned_case_ids=planned_case_ids,
				store=checkpoint_store,
				study_preset=study_preset,
				control_selector_name=control_selector_name,
				dataset_name=dataset_name,
			)
			run_state.status = RunStatus.INTERRUPTED
			run_state.interrupted_reason = "SIGINT"
			run_state.last_error = None
			checkpoint_store.save_run_state(run_state)
			interrupted_item = (
				plan_items_by_key.get(run_state.current_selection_key)
				if run_state.current_selection_key is not None
				else None
			)
			interrupted_progress = _selection_progress_snapshot(
				plan_items=plan_items,
				completed_selection_keys=completed_selection_keys,
				selection_keys_by_case=selection_keys_by_case,
				selection_keys_by_case_selector=selection_keys_by_case_selector,
				current_case_id=run_state.current_case_id,
				current_selector_name=interrupted_item.selector_name
				if interrupted_item is not None
				else None,
				current_budget_label=interrupted_item.budget_label
				if interrupted_item is not None
				else None,
			)
			_notify_progress(
				progress_observer,
				dataset_name=dataset_name,
				phase="interrupted",
				total_cases=len(selected_cases),
				completed_cases=len(run_state.completed_case_ids),
				total_selections=interrupted_progress.total_selections,
				completed_selections=interrupted_progress.completed_selections,
				current_case_id=run_state.current_case_id,
				current_query=case_by_id[run_state.current_case_id].query
				if run_state.current_case_id in case_by_id
				else None,
				current_selector_name=interrupted_progress.current_selector_name,
				current_budget_label=interrupted_progress.current_budget_label,
				case_total_selectors=interrupted_progress.case_total_selectors,
				case_completed_selectors=interrupted_progress.case_completed_selectors,
				case_total_selections=interrupted_progress.case_total_selections,
				case_completed_selections=interrupted_progress.case_completed_selections,
				summary=summary if summary.total_cases > 0 else None,
			)
		except HardStopRequested:
			run_state.status = RunStatus.INTERRUPTED
			run_state.interrupted_reason = "SIGINT(force)"
			run_state.last_error = None
			checkpoint_store.save_run_state(run_state)
			interrupted_item = (
				plan_items_by_key.get(run_state.current_selection_key)
				if run_state.current_selection_key is not None
				else None
			)
			interrupted_progress = _selection_progress_snapshot(
				plan_items=plan_items,
				completed_selection_keys=completed_selection_keys,
				selection_keys_by_case=selection_keys_by_case,
				selection_keys_by_case_selector=selection_keys_by_case_selector,
				current_case_id=run_state.current_case_id,
				current_selector_name=interrupted_item.selector_name
				if interrupted_item is not None
				else None,
				current_budget_label=interrupted_item.budget_label
				if interrupted_item is not None
				else None,
			)
			_notify_progress(
				progress_observer,
				dataset_name=dataset_name,
				phase="interrupted",
				total_cases=len(selected_cases),
				completed_cases=len(run_state.completed_case_ids),
				total_selections=interrupted_progress.total_selections,
				completed_selections=interrupted_progress.completed_selections,
				current_case_id=run_state.current_case_id,
				current_query=case_by_id[run_state.current_case_id].query
				if run_state.current_case_id in case_by_id
				else None,
				current_selector_name=interrupted_progress.current_selector_name,
				current_budget_label=interrupted_progress.current_budget_label,
				case_total_selectors=interrupted_progress.case_total_selectors,
				case_completed_selectors=interrupted_progress.case_completed_selectors,
				case_total_selections=interrupted_progress.case_total_selections,
				case_completed_selections=interrupted_progress.case_completed_selections,
				summary=summary if summary.total_cases > 0 else None,
			)
			raise
		except Exception as exc:
			run_state.status = RunStatus.FAILED
			run_state.last_error = f"{type(exc).__name__}: {exc}"
			checkpoint_store.save_run_state(run_state)
			failed_item = (
				plan_items_by_key.get(run_state.current_selection_key)
				if run_state.current_selection_key is not None
				else None
			)
			failed_progress = _selection_progress_snapshot(
				plan_items=plan_items,
				completed_selection_keys=completed_selection_keys,
				selection_keys_by_case=selection_keys_by_case,
				selection_keys_by_case_selector=selection_keys_by_case_selector,
				current_case_id=run_state.current_case_id,
				current_selector_name=failed_item.selector_name
				if failed_item is not None
				else None,
				current_budget_label=failed_item.budget_label
				if failed_item is not None
				else None,
			)
			_notify_progress(
				progress_observer,
				dataset_name=dataset_name,
				phase="failed",
				total_cases=len(selected_cases),
				completed_cases=len(run_state.completed_case_ids),
				total_selections=failed_progress.total_selections,
				completed_selections=failed_progress.completed_selections,
				current_case_id=run_state.current_case_id,
				current_query=case_by_id[run_state.current_case_id].query
				if run_state.current_case_id in case_by_id
				else None,
				current_selector_name=failed_progress.current_selector_name,
				current_budget_label=failed_progress.current_budget_label,
				case_total_selectors=failed_progress.case_total_selectors,
				case_completed_selectors=failed_progress.case_completed_selectors,
				case_total_selections=failed_progress.case_total_selections,
				case_completed_selections=failed_progress.case_completed_selections,
				summary=summary if summary.total_cases > 0 else None,
			)
			raise
		finally:
			interrupt_controller.uninstall()

	return evaluations, summary


def _consistent_chunk_value(chunk_metas: Sequence[dict[str, Any]], key: str) -> Any:
	if not chunk_metas:
		return None
	first = chunk_metas[0].get(key)
	if all(chunk_meta.get(key) == first for chunk_meta in chunk_metas[1:]):
		return first
	return None


def _merged_resolved_selectors(
	chunk_metas: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]]
) -> list[str]:
	selectors = _consistent_chunk_value(chunk_metas, "selectors")
	if selectors is not None:
		return list(selectors)
	ordered_selectors: list[str] = []
	seen: set[str] = set()
	for record in records:
		selector_name = str(record["selector"])
		if selector_name in seen:
			continue
		seen.add(selector_name)
		ordered_selectors.append(selector_name)
	return ordered_selectors


def _build_evaluators(
	*,
	selectors,
	budgets: Sequence[EvaluationBudget],
	with_e2e: bool,
	answerer_mode: str,
	answer_provider: str | None,
	answer_model: str | None,
	answer_api_key_env: str | None,
	answer_base_url: str | None,
	answer_cache_path: str | Path | None,
) -> list[Evaluator]:
	answerer = _build_answerer(
		with_e2e=with_e2e,
		answerer_mode=answerer_mode,
		answer_provider=answer_provider,
		answer_model=answer_model,
		answer_api_key_env=answer_api_key_env,
		answer_base_url=answer_base_url,
		answer_cache_path=answer_cache_path,
	)
	return [
		Evaluator(
			selectors,
			budget=budget,
			with_e2e=with_e2e,
			answerer=answerer,
		)
		for budget in budgets
	]


def _resolve_budgets(
	*,
	token_budgets: Sequence[int] | None,
	budget_ratios: Sequence[float] | None,
) -> list[EvaluationBudget]:
	if token_budgets is not None and budget_ratios is not None:
		raise ValueError("Specify either token_budgets or budget_ratios, not both.")
	if token_budgets is not None:
		return [
			EvaluationBudget(
				token_budget_tokens=budget,
				token_budget_ratio=None,
			)
			for budget in token_budgets
		]
	if budget_ratios is not None:
		return [
			EvaluationBudget(
				token_budget_tokens=None,
				token_budget_ratio=ratio,
			)
			for ratio in budget_ratios
		]
	return [
		EvaluationBudget(
			token_budget_tokens=budget,
			token_budget_ratio=None,
		)
		for budget in DEFAULT_TOKEN_BUDGETS
	]


def _build_answerer(
	*,
	with_e2e: bool,
	answerer_mode: str,
	answer_provider: str | None,
	answer_model: str | None,
	answer_api_key_env: str | None,
	answer_base_url: str | None,
	answer_cache_path: str | Path | None,
) -> SupportsAnswer | None:
	if not with_e2e:
		return None
	if answerer_mode == "heuristic":
		return Answerer()
	if answerer_mode != "llm_fixed":
		raise ValueError(f"Unknown answerer_mode: {answerer_mode}")
	resolved_provider = cast(AnswererProvider, answer_provider or "copilot")
	config = LLMAnswererConfig(
		provider=resolved_provider,
		model=answer_model,
		api_key_env=answer_api_key_env,
		base_url=answer_base_url,
		cache_path=Path(answer_cache_path) if answer_cache_path is not None else None,
	)
	if config.provider != "copilot" and not os.environ.get(
		cast(str, config.api_key_env)
	):
		raise ValueError(
			f"Missing API key in environment variable {config.api_key_env}"
		)
	return LLMAnswerer(config=config)


def _evaluate_cases(
	*,
	graph,
	cases: Sequence[Any],
	evaluators: Sequence[Evaluator],
	description: str,
	dataset_name: str,
	progress_observer: ExperimentProgressObserver | None = None,
	on_case_complete: Callable[[CaseEvaluation, int, int], None] | None = None,
) -> list[CaseEvaluation]:
	evaluations: list[CaseEvaluation] = []
	total = len(cases)
	if total == 0:
		return evaluations
	logger.info("%s (%s cases)", description.capitalize(), total)
	_notify_progress(
		progress_observer,
		dataset_name=dataset_name,
		phase="evaluating",
		total_cases=total,
		completed_cases=0,
	)
	completed_cases = 0
	for case in _iterate_with_optional_progress(cases, description=description):
		_notify_progress(
			progress_observer,
			dataset_name=dataset_name,
			phase="evaluating",
			total_cases=total,
			completed_cases=completed_cases,
			current_case_id=getattr(case, "case_id", None),
			current_query=getattr(case, "query", None),
		)
		selections = []
		for evaluator in evaluators:
			selections.extend(evaluator.evaluate_case(graph, case).selections)
		evaluation = CaseEvaluation(case=case, selections=selections)
		evaluations.append(evaluation)
		completed_cases += 1
		if on_case_complete is not None:
			on_case_complete(evaluation, completed_cases, total)
	return evaluations


def _evaluate_single_case(
	*,
	graph,
	case,
	evaluators: Sequence[Evaluator],
) -> CaseEvaluation:
	combined_selections = []
	for evaluator in evaluators:
		combined_selections.extend(evaluator.evaluate_case(graph, case).selections)
	return CaseEvaluation(case=case, selections=combined_selections)


def _export_graphrag_inputs(
	*,
	graph,
	evaluations: Sequence[CaseEvaluation],
	output_dir: Path,
	description: str,
) -> None:
	total = sum(len(evaluation.selections) for evaluation in evaluations)
	logger.info("%s (%s exports)", description.capitalize(), total)
	for evaluation in _iterate_with_optional_progress(
		evaluations,
		total=len(evaluations),
		description=description,
	):
		_export_case_graphrag_inputs(
			graph=graph,
			evaluation=evaluation,
			output_dir=output_dir,
		)


def _export_case_graphrag_inputs(
	*,
	graph,
	evaluation: CaseEvaluation,
	output_dir: Path,
) -> None:
	for selection in evaluation.selections:
		selection.graphrag_input_path = _write_graphrag_input(
			graph=graph,
			case_id=evaluation.case.case_id,
			selection=selection,
			output_dir=output_dir,
		)


def _iterate_with_optional_progress(
	items,
	*,
	description: str,
	total: int | None = None,
):
	if total is None:
		try:
			total = len(items)
		except TypeError:
			total = None
	if not should_render_progress():
		for item in items:
			yield item
		return

	with create_progress(transient=True) as progress:
		task_id = progress.add_task(description, total=total)
		for item in items:
			yield item
			progress.advance(task_id, 1)  # ty: ignore[invalid-argument-type]  # TaskID is NewType(int)


def _write_graphrag_input(
	*,
	graph,
	case_id: str,
	selection,
	output_dir: Path,
) -> str:
	budget_slug = _budget_label_slug(selection.budget.budget_label)
	export_path = (
		output_dir
		/ "graphrag_inputs"
		/ selection.selector_name
		/ f"budget-{budget_slug}"
		/ f"{case_id}.csv"
	)
	export_path.parent.mkdir(parents=True, exist_ok=True)

	with export_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=["id", "title", "text", "url"])
		writer.writeheader()
		for node_id in selection.corpus.node_ids:
			document = graph.get_document(node_id)
			if document is None:
				continue
			url = str(graph.node_attr.get(node_id, {}).get("url", ""))
			writer.writerow(
				{
					"id": document.node_id,
					"title": document.title,
					"text": document.text,
					"url": url,
				}
			)

	return str(export_path.relative_to(output_dir))


def _budget_label_slug(value: str) -> str:
	return value.replace(".", "_")


def _notify_progress(
	observer: ExperimentProgressObserver | None,
	*,
	dataset_name: str,
	phase: ExperimentPhase,
	total_cases: int | None,
	completed_cases: int,
	total_selections: int | None = None,
	completed_selections: int | None = None,
	current_case_id: str | None = None,
	current_query: str | None = None,
	current_selector_name: str | None = None,
	current_budget_label: str | None = None,
	case_total_selectors: int | None = None,
	case_completed_selectors: int | None = None,
	case_total_selections: int | None = None,
	case_completed_selections: int | None = None,
	summary: ExperimentSummary | None = None,
) -> None:
	if observer is None:
		return
	observer(
		ExperimentProgressUpdate(
			dataset_name=dataset_name,
			phase=phase,
			total_cases=total_cases,
			completed_cases=completed_cases,
			total_selections=total_selections,
			completed_selections=completed_selections,
			current_case_id=current_case_id,
			current_query=current_query,
			current_selector_name=current_selector_name,
			current_budget_label=current_budget_label,
			case_total_selectors=case_total_selectors,
			case_completed_selectors=case_completed_selectors,
			case_total_selections=case_total_selections,
			case_completed_selections=case_completed_selections,
			summary=summary,
		)
	)


def _initialize_result_files(
	chunk_dir: Path,
) -> tuple[Path, Path, Path]:
	results_path = chunk_dir / "results.jsonl"
	selector_logs_path = chunk_dir / "selector_logs.jsonl"
	summary_path = chunk_dir / "summary.json"
	results_path.write_text("", encoding="utf-8")
	selector_logs_path.write_text("", encoding="utf-8")
	return results_path, selector_logs_path, summary_path


def _append_case_result_files(
	*,
	results_path: Path,
	selector_logs_path: Path,
	evaluation: CaseEvaluation,
) -> None:
	with results_path.open("a", encoding="utf-8") as handle:
		for selection in evaluation.selections:
			record = _selection_record(evaluation, selection)
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")
	with selector_logs_path.open("a", encoding="utf-8") as handle:
		for selection in evaluation.selections:
			for record in _selector_log_records(evaluation, selection):
				handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_summary_file(
	*,
	summary_path: Path,
	summary: ExperimentSummary,
	study_preset: str | None = None,
	control_selector_name: str | None = None,
	results_path: Path | None = None,
) -> None:
	summary_path.write_text(
		json.dumps(asdict(summary), ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	export_summary_report(summary, summary_path.with_name("summary_rows.csv"))
	export_study_comparison_report(
		summary,
		summary_path.with_name("study_comparison_rows.csv"),
		study_preset=study_preset,
		control_selector_name=control_selector_name,
	)
	if results_path is not None:
		export_subset_comparison_report(
			results_path,
			summary_path.with_name("subset_comparison_rows.csv"),
			study_preset=study_preset,
			control_selector_name=control_selector_name,
		)


def _selection_record(evaluation: CaseEvaluation, selection) -> dict[str, Any]:
	return {
		"dataset_name": evaluation.case.dataset_name,
		"case_id": evaluation.case.case_id,
		"query": evaluation.case.query,
		"expected_answer": evaluation.case.expected_answer,
		"gold_support_nodes": evaluation.case.gold_support_nodes,
		"gold_start_nodes": evaluation.case.gold_start_nodes,
		"gold_path_nodes": evaluation.case.gold_path_nodes,
		"question_type": evaluation.case.question_type,
		"selector": selection.selector_name,
		"budget_mode": selection.budget.budget_mode,
		"budget_value": selection.budget.budget_value,
		"budget_label": selection.budget.budget_label,
		"token_budget_tokens": selection.budget.token_budget_tokens,
		"token_budget_ratio": selection.budget.token_budget_ratio,
		"selector_provider": selection.selector_metadata.provider
		if selection.selector_metadata is not None
		else None,
		"selector_model": selection.selector_metadata.model
		if selection.selector_metadata is not None
		else None,
		"answerer_mode": selection.end_to_end.mode
		if selection.end_to_end is not None
		else None,
		"answer_model": selection.end_to_end.model
		if selection.end_to_end is not None
		else None,
		"selection": {
			"budget": asdict(selection.budget),
			"corpus": asdict(selection.corpus),
			"metrics": asdict(selection.metrics),
			"trace": [asdict(step) for step in selection.trace],
			"stop_reason": selection.stop_reason,
			"graphrag_input_path": selection.graphrag_input_path,
			"selector_metadata": asdict(selection.selector_metadata)
			if selection.selector_metadata is not None
			else None,
			"selector_usage": asdict(selection.selector_usage)
			if selection.selector_usage is not None
			else None,
		},
		"end_to_end": asdict(selection.end_to_end)
		if selection.end_to_end is not None
		else None,
	}


def _selector_log_records(
	evaluation: CaseEvaluation, selection
) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	for log in selection.selector_logs:
		records.append(
			{
				"dataset_name": evaluation.case.dataset_name,
				"case_id": evaluation.case.case_id,
				"query": evaluation.case.query,
				"selector": selection.selector_name,
				"budget_mode": selection.budget.budget_mode,
				"budget_value": selection.budget.budget_value,
				"budget_label": selection.budget.budget_label,
				"selector_provider": selection.selector_metadata.provider
				if selection.selector_metadata is not None
				else None,
				"selector_model": selection.selector_metadata.model
				if selection.selector_metadata is not None
				else None,
				"selector_metadata": asdict(selection.selector_metadata)
				if selection.selector_metadata is not None
				else None,
				"log": asdict(log),
			}
		)
	return records


def _slice_cases(
	cases,
	*,
	split: str,
	limit: int | None,
	case_start: int,
	case_limit: int | None,
	chunk_size: int | None,
	chunk_index: int | None,
) -> tuple[list[Any], dict[str, Any]]:
	if limit is not None:
		cases = cases[:limit]
	total_cases = len(cases)
	effective_chunk_size = (
		chunk_size
		if chunk_size is not None
		else (100 if chunk_index is not None else None)
	)
	if chunk_index is not None:
		if effective_chunk_size is None:
			raise ValueError("chunk_size must be set when chunk_index is provided")
		case_start = chunk_index * effective_chunk_size
		case_limit = effective_chunk_size
	end = (
		total_cases if case_limit is None else min(total_cases, case_start + case_limit)
	)
	selected_cases = cases[case_start:end]
	return selected_cases, {
		"split": split,
		"total_cases": total_cases,
		"case_start": case_start,
		"case_limit": end - case_start,
		"chunk_size": effective_chunk_size,
		"chunk_index": chunk_index,
	}


def _chunk_output_dir(
	*, output_root: Path, exp_name: str, chunk_meta: dict[str, Any]
) -> Path:
	chunk_index = chunk_meta.get("chunk_index")
	if chunk_index is not None:
		label = f"chunk-{int(chunk_index):05d}"
	else:
		start = int(chunk_meta.get("case_start", 0))
		end = start + int(chunk_meta.get("case_limit", 0))
		label = f"range-{start:05d}-{max(start, end - 1):05d}"
	return output_root / exp_name / "chunks" / label


def _summarize_result_records(records: Sequence[dict[str, Any]]) -> ExperimentSummary:
	if not records:
		raise ValueError("Cannot summarize empty result records.")
	groups: dict[
		tuple[str, str, str, str | None, str | None], list[dict[str, Any]]
	] = {}
	ordered_keys: list[tuple[str, str, str, str | None, str | None]] = []
	for record in records:
		key = (
			str(record["selector"]),
			str(record["budget_mode"]),
			str(record["budget_value"]),
			record.get("selector_provider"),
			record.get("selector_model"),
		)
		if key not in groups:
			groups[key] = []
			ordered_keys.append(key)
		groups[key].append(record)

	selector_budgets: list[dict[str, Any]] = []
	for key in ordered_keys:
		rows = groups[key]
		name = str(rows[0]["selector"])
		budget_mode = str(rows[0]["budget_mode"])
		budget_value_raw = rows[0]["budget_value"]
		selector_provider = rows[0].get("selector_provider")
		selector_model = rows[0].get("selector_model")
		budget_value: int | float = (
			int(budget_value_raw)
			if budget_mode == "tokens"
			else float(budget_value_raw)
		)
		budget_label = str(rows[0]["budget_label"])
		token_budget_tokens = rows[0].get("token_budget_tokens")
		token_budget_ratio = rows[0].get("token_budget_ratio")
		start_hits = [
			1.0 if row["selection"]["metrics"]["start_hit"] else 0.0
			for row in rows
			if row["selection"]["metrics"]["start_hit"] is not None
		]
		support_recall = [
			float(row["selection"]["metrics"]["support_recall"])
			for row in rows
			if row["selection"]["metrics"]["support_recall"] is not None
		]
		support_precision = [
			float(row["selection"]["metrics"]["support_precision"])
			for row in rows
			if row["selection"]["metrics"]["support_precision"] is not None
		]
		support_f1 = [
			float(row["selection"]["metrics"]["support_f1"])
			for row in rows
			if row["selection"]["metrics"]["support_f1"] is not None
		]
		support_f1_zero_on_empty = [
			float(row["selection"]["metrics"]["support_f1_zero_on_empty"])
			for row in rows
			if row["selection"]["metrics"].get("support_f1_zero_on_empty") is not None
		]
		support_set_em = [
			float(row["selection"]["metrics"]["support_set_em"])
			for row in rows
			if row["selection"]["metrics"].get("support_set_em") is not None
		]
		path_hits = [
			1.0 if row["selection"]["metrics"]["path_hit"] else 0.0
			for row in rows
			if row["selection"]["metrics"]["path_hit"] is not None
		]
		selected_nodes = [
			float(row["selection"]["metrics"]["selected_nodes_count"]) for row in rows
		]
		selected_tokens = [
			float(row["selection"]["metrics"]["selected_token_estimate"])
			for row in rows
		]
		compression = [
			float(row["selection"]["metrics"]["compression_ratio"]) for row in rows
		]
		adherence = [
			1.0 if row["selection"]["metrics"]["budget_adherence"] else 0.0
			for row in rows
		]
		utilization = [
			float(row["selection"]["metrics"].get("budget_utilization", 0.0))
			for row in rows
		]
		empty_rate = [
			1.0 if row["selection"]["metrics"].get("empty_selection", False) else 0.0
			for row in rows
		]
		runtime = [
			float(row["selection"]["metrics"]["selection_runtime_s"]) for row in rows
		]
		selector_prompt_tokens = [
			float(row["selection"]["selector_usage"]["prompt_tokens"])
			for row in rows
			if row["selection"].get("selector_usage") is not None
		]
		selector_completion_tokens = [
			float(row["selection"]["selector_usage"]["completion_tokens"])
			for row in rows
			if row["selection"].get("selector_usage") is not None
		]
		selector_total_tokens = [
			float(row["selection"]["selector_usage"]["total_tokens"])
			for row in rows
			if row["selection"].get("selector_usage") is not None
		]
		selector_runtime = [
			float(row["selection"]["selector_usage"]["runtime_s"])
			for row in rows
			if row["selection"].get("selector_usage") is not None
		]
		selector_llm_calls = [
			float(row["selection"]["selector_usage"]["llm_calls"])
			for row in rows
			if row["selection"].get("selector_usage") is not None
		]
		selector_fallback_rates = [
			float(row["selection"]["selector_usage"].get("fallback_steps", 0))
			/ float(row["selection"]["selector_usage"].get("step_count", 0))
			for row in rows
			if row["selection"].get("selector_usage") is not None
			and float(row["selection"]["selector_usage"].get("step_count", 0)) > 0
		]
		selector_parse_failure_rates = [
			float(row["selection"]["selector_usage"].get("parse_failure_steps", 0))
			/ float(row["selection"]["selector_usage"].get("step_count", 0))
			for row in rows
			if row["selection"].get("selector_usage") is not None
			and float(row["selection"]["selector_usage"].get("step_count", 0)) > 0
		]
		answer_em = [
			float(row["end_to_end"]["em"])
			for row in rows
			if row["end_to_end"] is not None and row["end_to_end"]["em"] is not None
		]
		answer_f1 = [
			float(row["end_to_end"]["f1"])
			for row in rows
			if row["end_to_end"] is not None and row["end_to_end"]["f1"] is not None
		]
		selector_budgets.append(
			{
				"name": name,
				"selector_provider": selector_provider,
				"selector_model": selector_model,
				"budget_mode": budget_mode,
				"budget_value": budget_value,
				"budget_label": budget_label,
				"token_budget_ratio": float(token_budget_ratio)
				if token_budget_ratio is not None
				else None,
				"token_budget_tokens": int(token_budget_tokens)
				if token_budget_tokens is not None
				else None,
				"num_cases": len(rows),
				"avg_start_hit": _average_or_none(start_hits),
				"avg_support_recall": _average_or_none(support_recall),
				"avg_support_precision": _average_or_none(support_precision),
				"avg_support_f1": _average_or_none(support_f1),
				"avg_support_f1_zero_on_empty": _average_or_none(
					support_f1_zero_on_empty
				),
				"avg_support_set_em": _average_or_none(support_set_em),
				"avg_path_hit": _average_or_none(path_hits),
				"avg_selected_nodes": _average_or_none(selected_nodes) or 0.0,
				"avg_selected_token_estimate": _average_or_none(selected_tokens) or 0.0,
				"avg_compression_ratio": _average_or_none(compression) or 0.0,
				"avg_budget_adherence": _average_or_none(adherence) or 0.0,
				"avg_budget_utilization": _average_or_none(utilization) or 0.0,
				"avg_empty_selection_rate": _average_or_none(empty_rate) or 0.0,
				"avg_selection_runtime_s": _average_or_none(runtime) or 0.0,
				"avg_selector_prompt_tokens": _average_or_none(selector_prompt_tokens),
				"avg_selector_completion_tokens": _average_or_none(
					selector_completion_tokens
				),
				"avg_selector_total_tokens": _average_or_none(selector_total_tokens),
				"avg_selector_runtime_s": _average_or_none(selector_runtime),
				"avg_selector_llm_calls": _average_or_none(selector_llm_calls),
				"avg_selector_fallback_rate": _average_or_none(selector_fallback_rates),
				"avg_selector_parse_failure_rate": _average_or_none(
					selector_parse_failure_rates
				),
				"avg_answer_em": _average_or_none(answer_em),
				"avg_answer_f1": _average_or_none(answer_f1),
			}
		)
	return ExperimentSummary(
		dataset_name=str(records[0]["dataset_name"]),
		total_cases=len({record["case_id"] for record in records}),
		selector_budgets=[SelectorBudgetSummary(**row) for row in selector_budgets],
	)


def _average_or_none(values: Sequence[float]) -> float | None:
	if not values:
		return None
	return sum(values) / len(values)


def _missing_chunk_indices(
	chunk_indices: Sequence[int],
	*,
	total_cases: int | None,
	chunk_size: int | None,
) -> list[int]:
	if (
		not chunk_indices
		or total_cases is None
		or chunk_size is None
		or chunk_size <= 0
	):
		return []
	expected_chunks = (total_cases + chunk_size - 1) // chunk_size
	expected = set(range(expected_chunks))
	return sorted(expected.difference(chunk_indices))
