from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence
import urllib.request
from urllib.parse import urlparse

from hypercorpus.graph import LinkContext, LinkContextGraph
from hypercorpus.text import content_tokens
from hypercorpus.walker import (
	LinkContextOverlapStepScorer,
	StepScorerMetadata,
	StepLinkScorer,
	StepScoreCard,
	_clamp_score,
)

SelectorProvider = Literal["openai", "anthropic", "gemini"]
OpenAIApiMode = Literal[
	"chat_completions",
	"responses",
	"azure_foundry_chat_completions",
	"github_models_chat_completions",
]

DEFAULT_SELECTOR_API_KEY_ENVS: dict[SelectorProvider, str] = {
	"openai": "OPENAI_API_KEY",
	"anthropic": "ANTHROPIC_API_KEY",
	"gemini": "GEMINI_API_KEY",
}

DEFAULT_SELECTOR_MODELS: dict[SelectorProvider, str] = {
	"openai": "gpt-4.1-mini",
	"anthropic": "claude-3-5-haiku-latest",
	"gemini": "gemini-2.0-flash",
}


@dataclass(slots=True)
class SelectorLLMConfig:
	provider: SelectorProvider = "openai"
	model: str | None = None
	api_key_env: str | None = None
	base_url: str | None = None
	openai_api_mode: OpenAIApiMode = "chat_completions"
	cache_path: Path | None = None
	temperature: float = 0.0
	prompt_version: str = "v1"
	candidate_prefilter_top_n: int = 8
	two_hop_prefilter_top_n: int = 4
	controller_prompt_version: str = "controller_v4"
	controller_prefilter_top_n: int = 6
	controller_future_top_n: int = 2
	controller_max_attempts: int = 3
	enable_stop: bool = True
	enable_backtrack: bool = True
	enable_scout_fork: bool = True
	max_scout_branches: int = 2
	max_backtracks_per_case: int = 1

	def __post_init__(self) -> None:
		if self.provider != "openai" and self.openai_api_mode != "chat_completions":
			raise ValueError(
				"openai_api_mode is only supported when provider is 'openai'."
			)
		if self.model is None:
			self.model = DEFAULT_SELECTOR_MODELS[self.provider]
		if self.api_key_env is None:
			self.api_key_env = DEFAULT_SELECTOR_API_KEY_ENVS[self.provider]
		if self.provider == "openai" and self.base_url is not None:
			self.base_url = _normalize_openai_transport_url(
				self.base_url, api_mode=self.openai_api_mode
			)
		if self.candidate_prefilter_top_n <= 0:
			raise ValueError("candidate_prefilter_top_n must be positive.")
		if self.two_hop_prefilter_top_n <= 0:
			raise ValueError("two_hop_prefilter_top_n must be positive.")
		if self.controller_prefilter_top_n <= 0:
			raise ValueError("controller_prefilter_top_n must be positive.")
		if self.controller_future_top_n <= 0:
			raise ValueError("controller_future_top_n must be positive.")
		if self.controller_max_attempts <= 0:
			raise ValueError("controller_max_attempts must be positive.")
		if self.max_scout_branches <= 0:
			raise ValueError("max_scout_branches must be positive.")
		if self.max_backtracks_per_case < 0:
			raise ValueError("max_backtracks_per_case must be non-negative.")


ControllerAction = Literal["stop", "choose_one", "choose_two"]


@dataclass(slots=True)
class ControllerCandidate:
	edge_id: str
	utility: float
	support_coverage: float
	direct_support: float
	bridge_potential: float
	future_potential: float | None = None
	redundancy_risk: float = 0.0
	rationale: str | None = None


@dataclass(slots=True)
class ControllerDecision:
	action: ControllerAction
	primary_edge_id: str | None
	effective_action: ControllerAction | None = None
	secondary_edge_id: str | None = None
	backup_edge_id: str | None = None
	stop_score: float = 0.0
	evidence_cluster_confidence: float = 0.0
	candidates: list[ControllerCandidate] = field(default_factory=list)
	text: str | None = None
	raw_response: str | None = None
	prompt_tokens: int | None = None
	completion_tokens: int | None = None
	total_tokens: int | None = None
	latency_s: float = 0.0
	cache_hit: bool | None = None
	fallback_reason: str | None = None
	llm_attempts: int = 1


@dataclass(slots=True)
class BackendCompletion:
	text: str
	payload: dict[str, Any] | None
	prompt_tokens: int | None
	completion_tokens: int | None
	total_tokens: int | None
	raw_response: str | None


class SelectorLLMResponseError(ValueError):
	def __init__(self, kind: str, detail: str | None = None):
		self.kind = kind
		self.detail = _compact_error_detail(detail)
		super().__init__(self.fallback_reason)

	@property
	def fallback_reason(self) -> str:
		if self.detail:
			return f"{self.kind}:{self.detail}"
		return self.kind


class BackendAdapter(Protocol):
	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema: dict[str, Any] | None = None,
	) -> BackendCompletion: ...


class JsonlSelectorCache:
	def __init__(self, path: str | Path):
		self.path = Path(path)
		self._entries: dict[str, dict[str, Any]] | None = None

	def get(self, key: str) -> dict[str, Any] | None:
		self._load()
		assert self._entries is not None
		return self._entries.get(key)

	def put(self, key: str, payload: dict[str, Any]) -> None:
		self._load()
		assert self._entries is not None
		self.path.parent.mkdir(parents=True, exist_ok=True)
		record = {"key": key, **payload}
		with self.path.open("a", encoding="utf-8") as handle:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")
		self._entries[key] = record

	def _load(self) -> None:
		if self._entries is not None:
			return
		entries: dict[str, dict[str, Any]] = {}
		if self.path.exists():
			for line in self.path.read_text(encoding="utf-8").splitlines():
				if not line.strip():
					continue
				record = json.loads(line)
				entries[str(record["key"])] = record
		self._entries = entries


class OpenAIBackendAdapter:
	def __init__(
		self,
		*,
		api_key: str,
		base_url: str | None = None,
		api_mode: OpenAIApiMode = "chat_completions",
		client_factory: Callable[..., Any] | None = None,
		http_post: Callable[[str, dict[str, Any], str, dict[str, str] | None], dict[str, Any]]
		| None = None,
	):
		self.api_key = api_key
		self.base_url = base_url
		self.api_mode = api_mode
		self._client_factory = client_factory
		self._http_post = http_post or _post_json_request
		self._client: Any | None = None

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema: dict[str, Any] | None = None,
	) -> BackendCompletion:
		if self.api_mode == "responses":
			client = self._get_client()
			del response_schema
			response = client.responses.create(
				model=model,
				temperature=temperature,
				instructions=system_prompt,
				input=user_prompt,
				text={"format": {"type": "json_object"}},
			)
			prompt_tokens, completion_tokens, total_tokens = (
				_openai_responses_usage_triplet(getattr(response, "usage", None))
			)
			return BackendCompletion(
				text=_openai_response_output_text(response),
				payload=None,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				total_tokens=total_tokens,
				raw_response=_raw_response_payload(response),
			)
		if self.api_mode == "azure_foundry_chat_completions":
			del response_schema
			if self.base_url is None:
				raise ValueError(
					"selector_base_url is required for azure_foundry_chat_completions."
				)
			payload = self._http_post(
				self.base_url,
				{
					"model": model,
					"temperature": temperature,
					"response_format": {"type": "json_object"},
					"messages": [
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_prompt},
					],
				},
				self.api_key,
			)
			prompt_tokens, completion_tokens, total_tokens = _openai_usage_triplet(
				payload.get("usage")
			)
			return BackendCompletion(
				text=_openai_message_content(
					_openai_chat_message_from_payload(payload)
				),
				payload=None,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				total_tokens=total_tokens,
				raw_response=json.dumps(payload, ensure_ascii=False),
			)
		if self.api_mode == "github_models_chat_completions":
			del response_schema
			if self.base_url is None:
				raise ValueError(
					"selector_base_url is required for github_models_chat_completions."
				)
			payload = self._http_post(
				self.base_url,
				{
					"model": model,
					"temperature": temperature,
					"response_format": {"type": "json_object"},
					"messages": [
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_prompt},
					],
				},
				self.api_key,
				{
					"Accept": "application/vnd.github+json",
					"X-GitHub-Api-Version": "2026-03-10",
				},
			)
			prompt_tokens, completion_tokens, total_tokens = _openai_usage_triplet(
				payload.get("usage")
			)
			return BackendCompletion(
				text=_openai_message_content(
					_openai_chat_message_from_payload(payload)
				),
				payload=None,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				total_tokens=total_tokens,
				raw_response=json.dumps(payload, ensure_ascii=False),
			)
		client = self._get_client()
		del response_schema
		response = client.chat.completions.create(
			model=model,
			temperature=temperature,
			response_format={"type": "json_object"},
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		)
		message = response.choices[0].message
		prompt_tokens, completion_tokens, total_tokens = _openai_usage_triplet(
			getattr(response, "usage", None)
		)
		raw_response = _raw_response_payload(response)
		return BackendCompletion(
			text=_openai_message_content(message.content),
			payload=None,
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			raw_response=raw_response,
		)

	def _get_client(self) -> Any:
		if self._client is not None:
			return self._client
		if self._client_factory is not None:
			self._client = self._client_factory(
				api_key=self.api_key, base_url=self.base_url
			)
			return self._client
		try:
			from openai import OpenAI
		except ImportError as exc:
			raise RuntimeError(
				"OpenAI selector scoring requires the openai package."
			) from exc
		kwargs: dict[str, Any] = {"api_key": self.api_key}
		if self.base_url:
			kwargs["base_url"] = self.base_url
		self._client = OpenAI(**kwargs)
		return self._client


class AnthropicBackendAdapter:
	def __init__(
		self,
		*,
		api_key: str,
		client_factory: Callable[..., Any] | None = None,
	):
		self.api_key = api_key
		self._client_factory = client_factory
		self._client: Any | None = None

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema: dict[str, Any] | None = None,
	) -> BackendCompletion:
		client = self._get_client()
		tools = None
		tool_choice = None
		if response_schema is not None:
			tools = [
				{
					"name": "score_candidates",
					"description": "Return structured candidate scores for selector edge ranking.",
					"input_schema": response_schema,
				}
			]
			tool_choice = {"type": "tool", "name": "score_candidates"}
		response = client.messages.create(
			model=model,
			max_tokens=1024,
			temperature=temperature,
			system=system_prompt,
			messages=[{"role": "user", "content": user_prompt}],
			tools=tools,
			tool_choice=tool_choice,
		)
		usage = getattr(response, "usage", None)
		prompt_tokens = _maybe_int(getattr(usage, "input_tokens", None))
		completion_tokens = _maybe_int(getattr(usage, "output_tokens", None))
		total_tokens = (
			prompt_tokens + completion_tokens
			if prompt_tokens is not None and completion_tokens is not None
			else None
		)
		text_fragments: list[str] = []
		payload: dict[str, Any] | None = None
		for item in getattr(response, "content", []):
			if (
				getattr(item, "type", None) == "tool_use"
				and getattr(item, "name", None) == "score_candidates"
			):
				tool_input = getattr(item, "input", None)
				if isinstance(tool_input, dict):
					payload = tool_input
				continue
			if getattr(item, "type", None) == "text" and getattr(item, "text", None):
				text_fragments.append(str(item.text))
		return BackendCompletion(
			text="".join(text_fragments),
			payload=payload,
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			raw_response=_raw_response_payload(response),
		)

	def _get_client(self) -> Any:
		if self._client is not None:
			return self._client
		if self._client_factory is not None:
			self._client = self._client_factory(api_key=self.api_key)
			return self._client
		try:
			from anthropic import Anthropic
		except ImportError as exc:
			raise RuntimeError(
				"Anthropic selector scoring requires the anthropic package."
			) from exc
		self._client = Anthropic(api_key=self.api_key)
		return self._client


class GeminiBackendAdapter:
	def __init__(
		self,
		*,
		api_key: str,
		client_factory: Callable[..., Any] | None = None,
	):
		self.api_key = api_key
		self._client_factory = client_factory
		self._client: Any | None = None

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema: dict[str, Any] | None = None,
	) -> BackendCompletion:
		del response_schema
		client = self._get_client()
		response = client.models.generate_content(
			model=model,
			contents=f"{system_prompt}\n\n{user_prompt}",
			config={
				"temperature": temperature,
				"response_mime_type": "application/json",
			},
		)
		usage = getattr(response, "usage_metadata", None)
		prompt_tokens = _maybe_int(
			getattr(usage, "prompt_token_count", None)
			or getattr(usage, "input_token_count", None)
		)
		completion_tokens = _maybe_int(
			getattr(usage, "candidates_token_count", None)
			or getattr(usage, "output_token_count", None)
		)
		total_tokens = _maybe_int(getattr(usage, "total_token_count", None))
		text = getattr(response, "text", None)
		if text is None:
			text = _gemini_text(response)
		return BackendCompletion(
			text=str(text or ""),
			payload=None,
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			raw_response=_raw_response_payload(response),
		)

	def _get_client(self) -> Any:
		if self._client is not None:
			return self._client
		if self._client_factory is not None:
			self._client = self._client_factory(api_key=self.api_key)
			return self._client
		try:
			from google import genai
		except ImportError as exc:
			raise RuntimeError(
				"Gemini selector scoring requires the google-genai package."
			) from exc
		self._client = genai.Client(api_key=self.api_key)
		return self._client


class LLMStepLinkScorer:
	scorer_kind = "llm"

	def __init__(
		self,
		*,
		config: SelectorLLMConfig,
		mode: Literal["single_hop", "two_hop"],
		prefilter_scorer: StepLinkScorer | None = None,
		fallback_scorer: StepLinkScorer | None = None,
		backend_factory: Callable[[SelectorLLMConfig], BackendAdapter] | None = None,
	):
		self.config = config
		self.mode = mode
		self.prefilter_scorer = prefilter_scorer or LinkContextOverlapStepScorer()
		self.fallback_scorer = fallback_scorer or LinkContextOverlapStepScorer()
		self.backend_factory = backend_factory or _default_backend_factory
		self._cache = (
			JsonlSelectorCache(config.cache_path)
			if config.cache_path is not None
			else None
		)
		self._backend: BackendAdapter | None = None
		self.metadata = StepScorerMetadata(
			scorer_kind=self.scorer_kind,
			backend=self.config.provider,
			provider=self.config.provider,
			model=self.config.model,
			prompt_version=self.config.prompt_version,
			candidate_prefilter_top_n=self.config.candidate_prefilter_top_n,
			two_hop_prefilter_top_n=self.config.two_hop_prefilter_top_n
			if self.mode == "two_hop"
			else None,
		)

	def validate_environment(self) -> None:
		api_key = os.environ.get(self.config.api_key_env or "")
		if not api_key:
			raise ValueError(
				f"Missing API key in environment variable {self.config.api_key_env}"
			)

	def score_candidates(
		self,
		*,
		query: str,
		graph: LinkContextGraph,
		current_node_id: str,
		candidate_links: Sequence[LinkContext],
		visited_nodes: set[str],
		path_node_ids: Sequence[str],
		remaining_steps: int,
	) -> list[StepScoreCard]:
		if not candidate_links:
			return []

		fallback_cards = self.fallback_scorer.score_candidates(
			query=query,
			graph=graph,
			current_node_id=current_node_id,
			candidate_links=candidate_links,
			visited_nodes=visited_nodes,
			path_node_ids=path_node_ids,
			remaining_steps=remaining_steps,
		)
		prefilter_cards = self.prefilter_scorer.score_candidates(
			query=query,
			graph=graph,
			current_node_id=current_node_id,
			candidate_links=candidate_links,
			visited_nodes=visited_nodes,
			path_node_ids=path_node_ids,
			remaining_steps=remaining_steps,
		)
		if not fallback_cards or not prefilter_cards:
			return fallback_cards

		prefiltered_indices = _prefilter_indices(
			prefilter_cards,
			top_n=min(self.config.candidate_prefilter_top_n, len(prefilter_cards)),
			query=query,
			graph=graph,
			candidate_links=candidate_links,
		)
		bundle = _candidate_bundle(
			graph=graph,
			candidate_links=candidate_links,
			score_cards=prefilter_cards,
			prefiltered_indices=prefiltered_indices,
			query=query,
			current_node_id=current_node_id,
			path_node_ids=path_node_ids,
			visited_nodes=visited_nodes,
			mode=self.mode,
			future_top_n=self.config.two_hop_prefilter_top_n,
		)
		cache_key = _selector_cache_key(
			selector_name=f"link_context_llm_{self.mode}",
			provider=self.config.provider,
			model=self.config.model or "",
			base_url=self.config.base_url,
			prompt_version=self.config.prompt_version,
			query=query,
			path_node_ids=path_node_ids,
			bundle=bundle,
		)

		cached = self._cache.get(cache_key) if self._cache is not None else None
		if cached is not None:
			try:
				cached_text = _maybe_text(cached.get("response_text"))
				cached_payload = _cached_payload(cached)
				return self._cards_from_payload(
					payload=cached_payload
					if cached_payload is not None
					else _parse_completion_payload(cached_text or ""),
					text=cached_text,
					raw_response=str(cached.get("raw_response", "")) or None,
					prompt_tokens=_maybe_int(cached.get("prompt_tokens")),
					completion_tokens=_maybe_int(cached.get("completion_tokens")),
					total_tokens=_maybe_int(cached.get("total_tokens")),
					latency_s=0.0,
					cache_hit=True,
					candidate_links=candidate_links,
					prefiltered_indices=prefiltered_indices,
				)
			except Exception as exc:  # pragma: no cover - cache corruption path
				return _cards_with_fallback(
					fallback_cards,
					provider=self.config.provider,
					model=self.config.model,
					fallback_reason=f"cache_parse_error:{exc}",
					text=str(cached.get("response_text", "")) or None,
					raw_response=str(cached.get("raw_response", "")) or None,
					prompt_tokens=_maybe_int(cached.get("prompt_tokens")),
					completion_tokens=_maybe_int(cached.get("completion_tokens")),
					total_tokens=_maybe_int(cached.get("total_tokens")),
					cache_hit=True,
				)

		started_at = time.perf_counter()
		response: BackendCompletion | None = None
		try:
			self.validate_environment()
			response = self._get_backend().complete_json(
				model=self.config.model or "",
				system_prompt=_system_prompt(self.mode),
				user_prompt=_user_prompt(query=query, bundle=bundle),
				temperature=self.config.temperature,
				response_schema=_response_schema(self.mode),
			)
		except Exception as exc:
			latency_s = time.perf_counter() - started_at
			return _cards_with_fallback(
				fallback_cards,
				provider=self.config.provider,
				model=self.config.model,
				fallback_reason=f"provider_error:{_compact_error_detail(str(exc))}",
				latency_s=latency_s,
				text=response.text if response is not None else None,
				raw_response=response.raw_response if response is not None else None,
				prompt_tokens=response.prompt_tokens if response is not None else None,
				completion_tokens=response.completion_tokens
				if response is not None
				else None,
				total_tokens=response.total_tokens if response is not None else None,
			)

		latency_s = time.perf_counter() - started_at
		assert response is not None
		try:
			payload = (
				response.payload
				if response.payload is not None
				else _parse_completion_payload(response.text)
			)
			cards = self._cards_from_payload(
				payload=payload,
				text=response.text,
				raw_response=response.raw_response,
				prompt_tokens=response.prompt_tokens,
				completion_tokens=response.completion_tokens,
				total_tokens=response.total_tokens,
				latency_s=latency_s,
				cache_hit=False,
				candidate_links=candidate_links,
				prefiltered_indices=prefiltered_indices,
			)
		except SelectorLLMResponseError as exc:
			return _cards_with_fallback(
				fallback_cards,
				provider=self.config.provider,
				model=self.config.model,
				fallback_reason=exc.fallback_reason,
				latency_s=latency_s,
				text=response.text,
				raw_response=response.raw_response,
				prompt_tokens=response.prompt_tokens,
				completion_tokens=response.completion_tokens,
				total_tokens=response.total_tokens,
			)

		if self._cache is not None:
			self._cache.put(
				cache_key,
				{
					"provider": self.config.provider,
					"model": self.config.model,
					"response_payload": payload,
					"response_text": json.dumps(payload, ensure_ascii=False),
					"raw_response": response.raw_response,
					"prompt_tokens": response.prompt_tokens,
					"completion_tokens": response.completion_tokens,
					"total_tokens": response.total_tokens,
				},
			)
		return cards

	def _get_backend(self) -> BackendAdapter:
		if self._backend is not None:
			return self._backend
		self._backend = self.backend_factory(self.config)
		return self._backend

	def _cards_from_payload(
		self,
		*,
		payload: dict[str, Any],
		text: str | None,
		raw_response: str | None,
		prompt_tokens: int | None,
		completion_tokens: int | None,
		total_tokens: int | None,
		latency_s: float,
		cache_hit: bool,
		candidate_links: Sequence[LinkContext],
		prefiltered_indices: Sequence[int],
	) -> list[StepScoreCard]:
		entries = payload.get("scores")
		if not isinstance(entries, list):
			raise SelectorLLMResponseError("schema_error", "missing_scores_list")
		parsed: dict[str, dict[str, Any]] = {}
		for entry in entries:
			if not isinstance(entry, dict):
				continue
			edge_id = str(entry.get("edge_id", "")).strip()
			if edge_id:
				parsed[edge_id] = entry

		cards: list[StepScoreCard] = []
		prefiltered_set = {str(index) for index in prefiltered_indices}
		for index, _link in enumerate(candidate_links):
			edge_id = str(index)
			if edge_id not in prefiltered_set:
				cards.append(
					StepScoreCard(
						edge_id=edge_id,
						total_score=0.0,
						subscores={"prefilter_score": 0.0},
						rationale=None,
						text=text,
						backend=self.config.provider,
						provider=self.config.provider,
						model=self.config.model,
						latency_s=latency_s,
						prompt_tokens=prompt_tokens,
						completion_tokens=completion_tokens,
						total_tokens=total_tokens,
						cache_hit=cache_hit,
						fallback_reason="prefiltered_out",
					)
				)
				continue

			record = parsed.get(edge_id, {})
			if self.mode == "single_hop":
				direct_support = _clamp_score(record.get("direct_support"))
				bridge_potential = _clamp_score(record.get("bridge_potential"))
				novelty = _clamp_score(record.get("novelty"))
				total_score = _clamp_score(
					0.45 * direct_support + 0.40 * bridge_potential + 0.15 * novelty
				)
				subscores = {
					"direct_support": direct_support,
					"bridge_potential": bridge_potential,
					"novelty": novelty,
				}
				best_next_edge_id = None
			else:
				direct_support = _clamp_score(record.get("direct_support"))
				bridge_potential = _clamp_score(record.get("bridge_potential"))
				future_potential = _clamp_score(record.get("future_potential"))
				novelty = _clamp_score(record.get("novelty"))
				total_score = _clamp_score(
					0.30 * direct_support
					+ 0.25 * bridge_potential
					+ 0.35 * future_potential
					+ 0.10 * novelty
				)
				subscores = {
					"direct_support": direct_support,
					"bridge_potential": bridge_potential,
					"future_potential": future_potential,
					"novelty": novelty,
				}
				best_next_edge_id = (
					str(record.get("best_next_edge_id"))
					if record.get("best_next_edge_id")
					else None
				)

			cards.append(
				StepScoreCard(
					edge_id=edge_id,
					total_score=total_score,
					subscores=subscores,
					rationale=_maybe_text(record.get("rationale")),
					text=text,
					backend=self.config.provider,
					provider=self.config.provider,
					model=self.config.model,
					latency_s=latency_s,
					prompt_tokens=prompt_tokens,
					completion_tokens=completion_tokens,
					total_tokens=total_tokens,
					cache_hit=cache_hit,
					fallback_reason=None,
					best_next_edge_id=best_next_edge_id,
					raw_response=raw_response,
				)
			)
		return cards


class LLMController:
	def __init__(
		self,
		*,
		config: SelectorLLMConfig,
		mode: Literal["single_hop", "two_hop"],
		prefilter_scorer: StepLinkScorer | None = None,
		fallback_scorer: StepLinkScorer | None = None,
		backend_factory: Callable[[SelectorLLMConfig], BackendAdapter] | None = None,
	):
		self.config = config
		self.mode = mode
		self.prefilter_scorer = prefilter_scorer or LinkContextOverlapStepScorer()
		self.fallback_scorer = fallback_scorer or LinkContextOverlapStepScorer()
		self.backend_factory = backend_factory or _default_backend_factory
		self._cache = (
			JsonlSelectorCache(config.cache_path)
			if config.cache_path is not None
			else None
		)
		self._backend: BackendAdapter | None = None

	def validate_environment(self) -> None:
		api_key = os.environ.get(self.config.api_key_env or "")
		if not api_key:
			raise ValueError(
				f"Missing API key in environment variable {self.config.api_key_env}"
			)

	def decide(
		self,
		*,
		query: str,
		graph: LinkContextGraph,
		current_node_id: str,
		candidate_links: Sequence[LinkContext],
		visited_nodes: set[str],
		path_node_ids: Sequence[str],
		remaining_steps: int,
		current_depth: int,
		forks_used: int = 0,
		backtracks_used: int = 0,
	) -> ControllerDecision:
		fallback_cards = self.fallback_scorer.score_candidates(
			query=query,
			graph=graph,
			current_node_id=current_node_id,
			candidate_links=candidate_links,
			visited_nodes=visited_nodes,
			path_node_ids=path_node_ids,
			remaining_steps=remaining_steps,
		)
		prefilter_cards = self.prefilter_scorer.score_candidates(
			query=query,
			graph=graph,
			current_node_id=current_node_id,
			candidate_links=candidate_links,
			visited_nodes=visited_nodes,
			path_node_ids=path_node_ids,
			remaining_steps=remaining_steps,
		)
		if not fallback_cards or not prefilter_cards:
			return self._fallback_decision(
				fallback_cards=fallback_cards,
				prefiltered_indices=[],
				current_depth=current_depth,
				forks_used=forks_used,
				backtracks_used=backtracks_used,
				fallback_reason="empty_candidates",
			)

		prefiltered_indices = _prefilter_indices(
			prefilter_cards,
			top_n=min(self.config.controller_prefilter_top_n, len(prefilter_cards)),
			query=query,
			graph=graph,
			candidate_links=candidate_links,
		)
		bundle = _candidate_bundle(
			graph=graph,
			candidate_links=candidate_links,
			score_cards=prefilter_cards,
			prefiltered_indices=prefiltered_indices,
			query=query,
			current_node_id=current_node_id,
			path_node_ids=path_node_ids,
			visited_nodes=visited_nodes,
			mode=self.mode,
			future_top_n=self.config.controller_future_top_n,
		)
		cache_key = _selector_cache_key(
			selector_name=f"link_context_llm_controller_{self.mode}",
			provider=self.config.provider,
			model=self.config.model or "",
			base_url=self.config.base_url,
			prompt_version=self.config.controller_prompt_version,
			query=query,
			path_node_ids=path_node_ids,
			bundle=bundle,
		)

		cached = self._cache.get(cache_key) if self._cache is not None else None
		if cached is not None:
			try:
				cached_text = _maybe_text(cached.get("response_text"))
				cached_payload = _cached_payload(cached)
				payload = (
					cached_payload
					if cached_payload is not None
					else _parse_completion_payload(cached_text or "")
				)
				return self._decision_from_payload(
					payload=payload,
					text=cached_text,
					raw_response=str(cached.get("raw_response", "")) or None,
					prompt_tokens=_maybe_int(cached.get("prompt_tokens")),
					completion_tokens=_maybe_int(cached.get("completion_tokens")),
					total_tokens=_maybe_int(cached.get("total_tokens")),
					latency_s=0.0,
					cache_hit=True,
					prefiltered_indices=prefiltered_indices,
					fallback_cards=fallback_cards,
					current_depth=current_depth,
					forks_used=forks_used,
					backtracks_used=backtracks_used,
					llm_attempts=1,
				)
			except Exception as exc:  # pragma: no cover - cache corruption path
				return self._fallback_decision(
					fallback_cards=fallback_cards,
					prefiltered_indices=prefiltered_indices,
					current_depth=current_depth,
					forks_used=forks_used,
					backtracks_used=backtracks_used,
					fallback_reason=f"cache_parse_error:{_compact_error_detail(str(exc))}",
					text=str(cached.get("response_text", "")) or None,
					raw_response=str(cached.get("raw_response", "")) or None,
					prompt_tokens=_maybe_int(cached.get("prompt_tokens")),
					completion_tokens=_maybe_int(cached.get("completion_tokens")),
					total_tokens=_maybe_int(cached.get("total_tokens")),
					cache_hit=True,
					llm_attempts=1,
				)

		started_at = time.perf_counter()
		self.validate_environment()
		system_prompt = _controller_system_prompt(self.mode)
		user_prompt = _controller_user_prompt(query=query, bundle=bundle, mode=self.mode)
		response_schema = _controller_response_schema(self.mode)
		response: BackendCompletion | None = None
		last_parse_error: SelectorLLMResponseError | None = None
		prompt_token_values: list[int] = []
		completion_token_values: list[int] = []
		total_token_values: list[int] = []
		for attempt_index in range(self.config.controller_max_attempts):
			try:
				response = self._get_backend().complete_json(
					model=self.config.model or "",
					system_prompt=system_prompt,
					user_prompt=user_prompt,
					temperature=self.config.temperature,
					response_schema=response_schema,
				)
			except Exception as exc:
				latency_s = time.perf_counter() - started_at
				return self._fallback_decision(
					fallback_cards=fallback_cards,
					prefiltered_indices=prefiltered_indices,
					current_depth=current_depth,
					forks_used=forks_used,
					backtracks_used=backtracks_used,
					fallback_reason=f"provider_error:{_compact_error_detail(str(exc))}",
					latency_s=latency_s,
					text=response.text if response is not None else None,
					raw_response=response.raw_response if response is not None else None,
					prompt_tokens=_aggregate_usage_total(prompt_token_values),
					completion_tokens=_aggregate_usage_total(completion_token_values),
					total_tokens=_aggregate_usage_total(total_token_values),
					llm_attempts=attempt_index + 1,
				)
			assert response is not None
			if response.prompt_tokens is not None:
				prompt_token_values.append(response.prompt_tokens)
			if response.completion_tokens is not None:
				completion_token_values.append(response.completion_tokens)
			if response.total_tokens is not None:
				total_token_values.append(response.total_tokens)
			try:
				payload = (
					response.payload
					if response.payload is not None
					else _parse_completion_payload(response.text)
				)
				latency_s = time.perf_counter() - started_at
				decision = self._decision_from_payload(
					payload=payload,
					text=response.text,
					raw_response=response.raw_response,
					prompt_tokens=_aggregate_usage_total(prompt_token_values),
					completion_tokens=_aggregate_usage_total(completion_token_values),
					total_tokens=_aggregate_usage_total(total_token_values),
					latency_s=latency_s,
					cache_hit=False,
					prefiltered_indices=prefiltered_indices,
					fallback_cards=fallback_cards,
					current_depth=current_depth,
					forks_used=forks_used,
					backtracks_used=backtracks_used,
					llm_attempts=attempt_index + 1,
				)
				break
			except SelectorLLMResponseError as exc:
				last_parse_error = exc
				if attempt_index + 1 >= self.config.controller_max_attempts:
					latency_s = time.perf_counter() - started_at
					return self._fallback_decision(
						fallback_cards=fallback_cards,
						prefiltered_indices=prefiltered_indices,
						current_depth=current_depth,
						forks_used=forks_used,
						backtracks_used=backtracks_used,
						fallback_reason=exc.fallback_reason,
						latency_s=latency_s,
						text=response.text,
						raw_response=response.raw_response,
						prompt_tokens=_aggregate_usage_total(prompt_token_values),
						completion_tokens=_aggregate_usage_total(completion_token_values),
						total_tokens=_aggregate_usage_total(total_token_values),
						llm_attempts=attempt_index + 1,
					)
				user_prompt = _controller_repair_user_prompt(
					query=query,
					bundle=bundle,
					mode=self.mode,
					previous_response=response.text,
					validation_error=exc.fallback_reason,
				)
		else:  # pragma: no cover - loop always returns or breaks
			raise AssertionError("controller retry loop exited unexpectedly")
		if last_parse_error is not None:
			del last_parse_error

		if self._cache is not None:
			self._cache.put(
				cache_key,
				{
					"provider": self.config.provider,
					"model": self.config.model,
					"response_payload": payload,
					"response_text": json.dumps(payload, ensure_ascii=False),
					"raw_response": response.raw_response,
					"prompt_tokens": response.prompt_tokens,
					"completion_tokens": response.completion_tokens,
					"total_tokens": response.total_tokens,
				},
			)
		return decision

	def _get_backend(self) -> BackendAdapter:
		if self._backend is not None:
			return self._backend
		self._backend = self.backend_factory(self.config)
		return self._backend

	def _decision_from_payload(
		self,
		*,
		payload: dict[str, Any],
		text: str | None,
		raw_response: str | None,
		prompt_tokens: int | None,
		completion_tokens: int | None,
		total_tokens: int | None,
		latency_s: float,
		cache_hit: bool,
		prefiltered_indices: Sequence[int],
		fallback_cards: Sequence[StepScoreCard],
		current_depth: int,
		forks_used: int,
		backtracks_used: int,
		llm_attempts: int,
	) -> ControllerDecision:
		entries = payload.get("candidates")
		if not isinstance(entries, list):
			raise SelectorLLMResponseError("schema_error", "missing_candidates_list")
		parsed_candidates: list[ControllerCandidate] = []
		prefiltered_set = {str(index) for index in prefiltered_indices}
		for entry in entries:
			if not isinstance(entry, dict):
				continue
			edge_id = str(entry.get("edge_id", "")).strip()
			if not edge_id or edge_id not in prefiltered_set:
				continue
			parsed_candidates.append(
				ControllerCandidate(
					edge_id=edge_id,
					utility=_clamp_score(entry.get("utility")),
					support_coverage=_clamp_score(entry.get("support_coverage")),
					direct_support=_clamp_score(entry.get("direct_support")),
					bridge_potential=_clamp_score(entry.get("bridge_potential")),
					future_potential=(
						None
						if self.mode == "single_hop"
						else _clamp_score(entry.get("future_potential"))
					),
					redundancy_risk=_clamp_score(entry.get("redundancy_risk")),
					rationale=_maybe_text(entry.get("rationale")),
				)
			)
		if not parsed_candidates:
			raise SelectorLLMResponseError("schema_error", "no_prefiltered_candidates")
		action = str(payload.get("action", "")).strip()
		if action not in {"stop", "choose_one", "choose_two"}:
			raise SelectorLLMResponseError("schema_error", "invalid_action")
		decision = ControllerDecision(
			action=action,  # type: ignore[arg-type]
			effective_action=action,  # type: ignore[arg-type]
			primary_edge_id=_maybe_text(payload.get("primary_edge_id")),
			secondary_edge_id=_maybe_text(payload.get("secondary_edge_id")),
			backup_edge_id=_maybe_text(payload.get("backup_edge_id")),
			stop_score=_clamp_score(payload.get("stop_score")),
			evidence_cluster_confidence=_clamp_score(
				payload.get("evidence_cluster_confidence")
			),
			candidates=parsed_candidates,
			text=text,
			raw_response=raw_response,
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			latency_s=latency_s,
			cache_hit=cache_hit,
			llm_attempts=llm_attempts,
		)
		return self._normalize_decision(
			decision,
			fallback_cards=fallback_cards,
			prefiltered_indices=prefiltered_indices,
			current_depth=current_depth,
			forks_used=forks_used,
			backtracks_used=backtracks_used,
		)

	def _fallback_decision(
		self,
		*,
		fallback_cards: Sequence[StepScoreCard],
		prefiltered_indices: Sequence[int],
		current_depth: int,
		forks_used: int,
		backtracks_used: int,
		fallback_reason: str,
		latency_s: float = 0.0,
		text: str | None = None,
		raw_response: str | None = None,
		prompt_tokens: int | None = None,
		completion_tokens: int | None = None,
		total_tokens: int | None = None,
		cache_hit: bool | None = None,
		llm_attempts: int = 1,
	) -> ControllerDecision:
		candidates = [
			_controller_candidate_from_card(
				fallback_cards[index],
				edge_id=str(index),
				two_hop=self.mode == "two_hop",
			)
			for index in prefiltered_indices
			if index < len(fallback_cards)
		]
		if not candidates and fallback_cards:
			candidates = [
				_controller_candidate_from_card(
					card, edge_id=card.edge_id, two_hop=self.mode == "two_hop"
				)
				for card in fallback_cards
			]
		candidates.sort(key=lambda item: (item.utility, item.edge_id), reverse=True)
		best = candidates[0] if candidates else None
		second = candidates[1] if len(candidates) > 1 else None
		action: ControllerAction = "choose_one"
		stop_score = _clamp_score(1.0 - (best.utility if best is not None else 0.0))
		if self.config.enable_stop and current_depth >= 2 and stop_score >= 0.80:
			action = "stop"
		elif _allow_choose_two(
			config=self.config,
			candidates=candidates,
			primary=best,
			secondary=second,
			forks_used=forks_used,
		):
			action = "choose_two"
		decision = ControllerDecision(
			action=action,
			effective_action=action,
			primary_edge_id=best.edge_id if best is not None else None,
			secondary_edge_id=second.edge_id
			if action == "choose_two" and second is not None
			else None,
			backup_edge_id=(
				second.edge_id
				if self.config.enable_backtrack
				and backtracks_used < self.config.max_backtracks_per_case
				and second is not None
				else None
			),
			stop_score=stop_score,
			evidence_cluster_confidence=best.utility if best is not None else 0.0,
			candidates=candidates,
			text=text,
			raw_response=raw_response,
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			latency_s=latency_s,
			cache_hit=cache_hit,
			fallback_reason=fallback_reason,
			llm_attempts=llm_attempts,
		)
		return self._normalize_decision(
			decision,
			fallback_cards=fallback_cards,
			prefiltered_indices=prefiltered_indices,
			current_depth=current_depth,
			forks_used=forks_used,
			backtracks_used=backtracks_used,
		)

	def _normalize_decision(
		self,
		decision: ControllerDecision,
		*,
		fallback_cards: Sequence[StepScoreCard],
		prefiltered_indices: Sequence[int],
		current_depth: int,
		forks_used: int,
		backtracks_used: int,
	) -> ControllerDecision:
		candidates_by_id = {
			candidate.edge_id: candidate for candidate in decision.candidates
		}
		if not candidates_by_id:
			for index in prefiltered_indices:
				if index < len(fallback_cards):
					card = fallback_cards[index]
					candidates_by_id[str(index)] = _controller_candidate_from_card(
						card,
						edge_id=str(index),
						two_hop=self.mode == "two_hop",
					)
			decision.candidates = list(candidates_by_id.values())
		ranked = sorted(
			candidates_by_id.values(),
			key=lambda item: (item.utility, item.edge_id),
			reverse=True,
		)
		if decision.primary_edge_id not in candidates_by_id:
			decision.primary_edge_id = ranked[0].edge_id if ranked else None
		effective_action = decision.effective_action or decision.action
		if effective_action == "choose_two":
			secondary = candidates_by_id.get(decision.secondary_edge_id or "")
			if not _allow_choose_two(
				config=self.config,
				candidates=ranked,
				primary=candidates_by_id.get(decision.primary_edge_id or ""),
				secondary=secondary
				if secondary is not None
				else (ranked[1] if len(ranked) > 1 else None),
				forks_used=forks_used,
			):
				decision.secondary_edge_id = None
			elif decision.secondary_edge_id not in candidates_by_id:
				decision.secondary_edge_id = (
					ranked[1].edge_id if len(ranked) > 1 else None
				)
		else:
			decision.secondary_edge_id = None
		decision.effective_action = effective_action
		if (
			not self.config.enable_backtrack
			or backtracks_used >= self.config.max_backtracks_per_case
			or decision.backup_edge_id not in candidates_by_id
			or decision.backup_edge_id == decision.primary_edge_id
			or decision.backup_edge_id == decision.secondary_edge_id
		):
			fallback_backup = next(
				(
					candidate.edge_id
					for candidate in ranked
					if candidate.edge_id
					not in {decision.primary_edge_id, decision.secondary_edge_id}
				),
				None,
			)
			decision.backup_edge_id = (
				fallback_backup if self.config.enable_backtrack else None
			)
		decision.evidence_cluster_confidence = _clamp_score(
			decision.evidence_cluster_confidence
		)
		decision.stop_score = _clamp_score(decision.stop_score)
		return decision


class LLMControllerStepScorer:
	scorer_kind = "llm_controller"

	def __init__(
		self,
		*,
		controller: LLMController,
		config: SelectorLLMConfig,
		mode: Literal["single_hop", "two_hop"],
		fallback_scorer: StepLinkScorer | None = None,
	):
		self.controller = controller
		self.config = config
		self.mode = mode
		self.fallback_scorer = fallback_scorer or LinkContextOverlapStepScorer()
		self.metadata = StepScorerMetadata(
			scorer_kind=self.scorer_kind,
			backend=self.config.provider,
			provider=self.config.provider,
			model=self.config.model,
			prompt_version=self.config.controller_prompt_version,
			candidate_prefilter_top_n=self.config.controller_prefilter_top_n,
			two_hop_prefilter_top_n=self.config.controller_future_top_n
			if self.mode == "two_hop"
			else None,
			controller_prompt_version=self.config.controller_prompt_version,
			controller_prefilter_top_n=self.config.controller_prefilter_top_n,
			controller_future_top_n=self.config.controller_future_top_n
			if self.mode == "two_hop"
			else None,
		)

	def score_candidates(
		self,
		*,
		query: str,
		graph: LinkContextGraph,
		current_node_id: str,
		candidate_links: Sequence[LinkContext],
		visited_nodes: set[str],
		path_node_ids: Sequence[str],
		remaining_steps: int,
	) -> list[StepScoreCard]:
		fallback_cards = self.fallback_scorer.score_candidates(
			query=query,
			graph=graph,
			current_node_id=current_node_id,
			candidate_links=candidate_links,
			visited_nodes=visited_nodes,
			path_node_ids=path_node_ids,
			remaining_steps=remaining_steps,
		)
		decision = self.controller.decide(
			query=query,
			graph=graph,
			current_node_id=current_node_id,
			candidate_links=candidate_links,
			visited_nodes=visited_nodes,
			path_node_ids=path_node_ids,
			remaining_steps=remaining_steps,
			current_depth=max(len(path_node_ids) - 1, 0),
		)
		return _decision_to_cards(
			decision=decision,
			fallback_cards=fallback_cards,
			candidate_links=candidate_links,
			provider=self.config.provider,
			model=self.config.model,
			mode=self.mode,
			prefilter_top_n=self.config.controller_prefilter_top_n,
		)


def _default_backend_factory(config: SelectorLLMConfig) -> BackendAdapter:
	api_key = os.environ.get(config.api_key_env or "")
	if not api_key:
		raise ValueError(
			f"Missing API key in environment variable {config.api_key_env}"
		)
	if config.provider == "openai":
		return OpenAIBackendAdapter(
			api_key=api_key,
			base_url=config.base_url,
			api_mode=config.openai_api_mode,
		)
	if config.provider == "anthropic":
		return AnthropicBackendAdapter(api_key=api_key)
	if config.provider == "gemini":
		return GeminiBackendAdapter(api_key=api_key)
	raise ValueError(f"Unknown selector provider: {config.provider}")


def _selector_cache_key(
	*,
	selector_name: str,
	provider: str,
	model: str,
	base_url: str | None,
	prompt_version: str,
	query: str,
	path_node_ids: Sequence[str],
	bundle: dict[str, Any],
) -> str:
	payload = json.dumps(
		{
			"selector_name": selector_name,
			"provider": provider,
			"model": model,
			"base_url": base_url,
			"prompt_version": prompt_version,
			"query": query,
			"path_digest": hashlib.sha256(
				json.dumps(list(path_node_ids)).encode("utf-8")
			).hexdigest(),
			"candidate_bundle_digest": hashlib.sha256(
				json.dumps(bundle, sort_keys=True, ensure_ascii=False).encode("utf-8")
			).hexdigest(),
		},
		sort_keys=True,
		ensure_ascii=False,
	)
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prefilter_indices(
	cards: Sequence[StepScoreCard],
	*,
	top_n: int,
	query: str | None = None,
	graph: LinkContextGraph | None = None,
	candidate_links: Sequence[LinkContext] | None = None,
	support_keep_n: int = 2,
) -> list[int]:
	ranked = sorted(
		enumerate(cards),
		key=lambda item: (item[1].total_score, -item[0]),
		reverse=True,
	)
	selected: set[int] = {index for index, _card in ranked[:top_n]}
	if query is None or graph is None or candidate_links is None or support_keep_n <= 0:
		return [index for index, _card in ranked[:top_n]]
	support_ranked = sorted(
		enumerate(cards),
		key=lambda item: (
			_support_coverage_hint(
				query=query,
				graph=graph,
				link=candidate_links[item[0]],
				card=item[1],
			),
			item[1].total_score,
			-item[0],
		),
		reverse=True,
	)
	added = 0
	for index, card in support_ranked:
		if index in selected:
			continue
		support_hint = _support_coverage_hint(
			query=query,
			graph=graph,
			link=candidate_links[index],
			card=card,
		)
		if support_hint < 0.55:
			continue
		selected.add(index)
		added += 1
		if added >= support_keep_n:
			break
	return [index for index, _card in ranked if index in selected]


def _candidate_bundle(
	*,
	graph: LinkContextGraph,
	candidate_links: Sequence[LinkContext],
	score_cards: Sequence[StepScoreCard],
	prefiltered_indices: Sequence[int],
	query: str,
	current_node_id: str,
	path_node_ids: Sequence[str],
	visited_nodes: set[str],
	mode: str,
	future_top_n: int,
) -> dict[str, Any]:
	entries: list[dict[str, Any]] = []
	future_scorer = LinkContextOverlapStepScorer()
	for index in prefiltered_indices:
		link = candidate_links[index]
		score_card = score_cards[index]
		target_title = _node_title(graph, link.target)
		sentence_mentions_target = _title_coverage_in_sentence(
			title=target_title,
			sentence=link.sentence,
		)
		support_coverage_hint = _support_coverage_hint(
			query=query,
			graph=graph,
			link=link,
			card=score_card,
		)
		entry: dict[str, Any] = {
			"edge_id": str(index),
			"source_title": _node_title(graph, link.source),
			"target_title": target_title,
			"anchor_text": link.anchor_text,
			"sentence": link.sentence,
			"prefilter_score": score_card.total_score,
			"query_anchor_overlap": _clamp_score(
				score_card.subscores.get("anchor_overlap", 0.0)
			),
			"query_sentence_overlap": _clamp_score(
				score_card.subscores.get("sentence_overlap", 0.0)
			),
			"query_target_overlap": _clamp_score(
				score_card.subscores.get("target_overlap", 0.0)
			),
			"support_coverage_hint": support_coverage_hint,
			"source_sentence_mentions_target_title": sentence_mentions_target >= 0.75,
		}
		if mode == "two_hop":
			next_links = [
				next_link
				for next_link in graph.links_from(link.target)
				if next_link.target not in visited_nodes
				and next_link.target != link.source
			]
			future_cards = future_scorer.score_candidates(
				query=query,
				graph=graph,
				current_node_id=link.target,
				candidate_links=next_links,
				visited_nodes=visited_nodes | {link.target},
				path_node_ids=[*path_node_ids, link.target],
				remaining_steps=1,
			)
			future_indices = _prefilter_indices(
				future_cards,
				top_n=min(future_top_n, len(future_cards)),
				query=query,
				graph=graph,
				candidate_links=next_links,
				support_keep_n=1,
			)
			entry["future_candidates"] = [
				{
					"edge_id": f"{index}-{future_index}",
					"target_title": _node_title(graph, next_links[future_index].target),
					"anchor_text": next_links[future_index].anchor_text,
					"sentence": next_links[future_index].sentence,
					"prefilter_score": future_cards[future_index].total_score,
					"query_anchor_overlap": _clamp_score(
						future_cards[future_index].subscores.get("anchor_overlap", 0.0)
					),
					"query_sentence_overlap": _clamp_score(
						future_cards[future_index].subscores.get("sentence_overlap", 0.0)
					),
					"query_target_overlap": _clamp_score(
						future_cards[future_index].subscores.get("target_overlap", 0.0)
					),
					"support_coverage_hint": _support_coverage_hint(
						query=query,
						graph=graph,
						link=next_links[future_index],
						card=future_cards[future_index],
					),
				}
				for future_index in future_indices
			]
		entries.append(entry)

	return {
		"query": query,
		"current_node_id": current_node_id,
		"path_titles": [_node_title(graph, node_id) for node_id in path_node_ids],
		"candidates": entries,
	}


def _title_coverage_in_sentence(*, title: str, sentence: str) -> float:
	title_tokens = content_tokens(title)
	if not title_tokens:
		return 0.0
	sentence_tokens = set(content_tokens(sentence))
	shared = sum(1 for token in title_tokens if token in sentence_tokens)
	return shared / max(len(title_tokens), 1)


def _support_coverage_hint(
	*,
	query: str,
	graph: LinkContextGraph,
	link: LinkContext,
	card: StepScoreCard,
) -> float:
	target_title = _node_title(graph, link.target)
	title_coverage = _title_coverage_in_sentence(title=target_title, sentence=link.sentence)
	sentence_overlap = _clamp_score(card.subscores.get("sentence_overlap", 0.0))
	target_overlap = _clamp_score(card.subscores.get("target_overlap", 0.0))
	anchor_overlap = _clamp_score(card.subscores.get("anchor_overlap", 0.0))
	query_tokens = set(content_tokens(query))
	target_tokens = content_tokens(target_title)
	target_query_overlap = (
		sum(1 for token in target_tokens if token in query_tokens)
		/ max(len(target_tokens), 1)
		if target_tokens
		else 0.0
	)
	return _clamp_score(
		0.45 * sentence_overlap
		+ 0.25 * title_coverage
		+ 0.20 * target_overlap
		+ 0.05 * anchor_overlap
		+ 0.05 * target_query_overlap
	)


def _response_schema(mode: str) -> dict[str, Any]:
	properties: dict[str, Any] = {
		"edge_id": {"type": "string"},
		"direct_support": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"bridge_potential": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"novelty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"rationale": {"type": "string"},
	}
	required = ["edge_id", "direct_support", "bridge_potential", "novelty", "rationale"]
	if mode == "two_hop":
		properties["future_potential"] = {
			"type": "number",
			"minimum": 0.0,
			"maximum": 1.0,
		}
		properties["best_next_edge_id"] = {"type": "string"}
		required.append("future_potential")
	return {
		"type": "object",
		"additionalProperties": False,
		"properties": {
			"scores": {
				"type": "array",
				"items": {
					"type": "object",
					"additionalProperties": False,
					"properties": properties,
					"required": required,
				},
			}
		},
		"required": ["scores"],
	}


def _controller_response_schema(mode: str) -> dict[str, Any]:
	candidate_properties: dict[str, Any] = {
		"edge_id": {"type": "string"},
		"utility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"support_coverage": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"direct_support": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"bridge_potential": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"redundancy_risk": {"type": "number", "minimum": 0.0, "maximum": 1.0},
		"rationale": {"type": "string"},
	}
	required = [
		"edge_id",
		"utility",
		"support_coverage",
		"direct_support",
		"bridge_potential",
		"redundancy_risk",
		"rationale",
	]
	if mode == "two_hop":
		candidate_properties["future_potential"] = {
			"type": "number",
			"minimum": 0.0,
			"maximum": 1.0,
		}
		required.append("future_potential")
	return {
		"type": "object",
		"additionalProperties": False,
		"properties": {
			"action": {"type": "string", "enum": ["stop", "choose_one", "choose_two"]},
			"primary_edge_id": {"type": "string"},
			"secondary_edge_id": {"type": "string"},
			"backup_edge_id": {"type": "string"},
			"stop_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
			"evidence_cluster_confidence": {
				"type": "number",
				"minimum": 0.0,
				"maximum": 1.0,
			},
			"candidates": {
				"type": "array",
				"items": {
					"type": "object",
					"additionalProperties": False,
					"properties": candidate_properties,
					"required": required,
				},
			},
		},
		"required": [
			"action",
			"primary_edge_id",
			"backup_edge_id",
			"stop_score",
			"evidence_cluster_confidence",
			"candidates",
		],
	}


def _system_prompt(mode: str) -> str:
	if mode == "single_hop":
		return (
			"You are scoring hyperlink choices for a single-path multi-hop retriever. "
			"Use the provided structured output channel exactly once. "
			"Each candidate needs direct_support, bridge_potential, novelty, and rationale. "
			"All subscores must be floats in [0,1]."
		)
	return (
		"You are scoring hyperlink choices for a two-hop-aware single-path multi-hop retriever. "
		"Use the provided structured output channel exactly once. "
		"Each candidate needs direct_support, bridge_potential, future_potential, novelty, rationale, "
		"and optional best_next_edge_id. All subscores must be floats in [0,1]."
	)


def _controller_system_prompt(mode: str) -> str:
	if mode == "single_hop":
		return (
			"You are controlling a budgeted hyperlink retriever. "
			"Pick the next retrieval action, not just per-edge scores. "
			"Favor explicit support-node coverage, low redundancy, and avoiding precision collapse. "
			"Even if the current page text already hints at the answer, prefer a high-value link that would "
			"add an explicit answer-bearing or support-bearing node to the retrieved evidence set instead of stopping early. "
			"Return exactly one JSON object with an action and candidate utilities. "
			"Do not emit prose, markdown, code fences, or explanations outside the JSON object. "
			"Only use choose_two when two options are genuinely tied."
		)
	return (
		"You are controlling a budgeted hyperlink retriever with limited lookahead. "
		"Pick the next retrieval action, not just per-edge scores. "
		"Favor explicit support-node coverage, bridge discovery, and low redundancy under a token budget. "
		"Even if the current page text already hints at the answer, prefer a high-value link that would "
		"add an explicit answer-bearing or support-bearing node to the retrieved evidence set instead of stopping early. "
		"Return exactly one JSON object with an action and candidate utilities. "
		"Do not emit prose, markdown, code fences, or explanations outside the JSON object. "
		"Only use choose_two for near-ties that justify a short scout branch."
	)


def _user_prompt(*, query: str, bundle: dict[str, Any]) -> str:
	return (
		f"Question:\n{query}\n\n"
		"Retriever context:\n"
		f"{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
		'Return JSON as {"scores": [{"edge_id": "...", ...}]}. '
		"Do not add markdown or extra prose."
	)


def _controller_user_prompt(*, query: str, bundle: dict[str, Any], mode: str) -> str:
	return (
		f"Question:\n{query}\n\n"
		"Retriever context:\n"
		f"{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
		"Required JSON schema:\n"
		f"{_controller_schema_instructions(mode)}\n\n"
		"Choose the next retrieval action under a budget. "
		"Use stop only if the currently retrieved nodes already make the key supporting entity or institution explicit and "
		"no remaining candidate offers materially better explicit support coverage. "
		"Treat support_coverage as a first-class utility component: if a candidate would add an explicit support-bearing node, "
		"prefer that candidate over stop unless redundancy is very high. "
		"Use choose_two only for a genuine near-tie. "
		"Return JSON only."
	)


def _controller_repair_user_prompt(
	*,
	query: str,
	bundle: dict[str, Any],
	mode: str,
	previous_response: str,
	validation_error: str,
) -> str:
	return (
		f"Question:\n{query}\n\n"
		"Retriever context:\n"
		f"{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
		"Your previous response failed validation.\n"
		f"Validation error: {validation_error}\n\n"
		"Previous response:\n"
		f"{previous_response}\n\n"
		"Return a corrected JSON object that satisfies this exact schema:\n"
		f"{_controller_schema_instructions(mode)}\n\n"
		"Return JSON only. Do not include commentary, markdown, or reasoning."
	)


def _controller_schema_instructions(mode: str) -> str:
	candidate_fields = [
		'"edge_id": "<one of the listed candidate edge ids>"',
		'"utility": 0.0-1.0',
		'"support_coverage": 0.0-1.0',
		'"direct_support": 0.0-1.0',
		'"bridge_potential": 0.0-1.0',
	]
	if mode == "two_hop":
		candidate_fields.append('"future_potential": 0.0-1.0')
	candidate_fields.extend(
		[
			'"redundancy_risk": 0.0-1.0',
			'"rationale": "<short rationale string>"',
		]
	)
	candidate_schema = ", ".join(candidate_fields)
	return (
		'{"action":"stop|choose_one|choose_two",'
		'"primary_edge_id":"<edge id or empty string>",'
		'"secondary_edge_id":"<edge id or empty string>",'
		'"backup_edge_id":"<edge id or empty string>",'
		'"stop_score":0.0-1.0,'
		'"evidence_cluster_confidence":0.0-1.0,'
		f'"candidates":[{{{candidate_schema}}}]}} '
		"Rules: candidates must be a non-empty array, every edge_id must come from the provided candidate list, "
		"action must be one of stop/choose_one/choose_two, and utility should explicitly account for support_coverage "
		"(adding an explicit support-bearing node beats an early stop when redundancy is not extreme). "
		"The response must be exactly one JSON object."
	)


def _node_title(graph: LinkContextGraph, node_id: str) -> str:
	return str(graph.node_attr.get(node_id, {}).get("title", node_id))


def _cards_with_fallback(
	cards: Sequence[StepScoreCard],
	*,
	provider: str,
	model: str | None,
	fallback_reason: str,
	latency_s: float = 0.0,
	text: str | None = None,
	raw_response: str | None = None,
	prompt_tokens: int | None = None,
	completion_tokens: int | None = None,
	total_tokens: int | None = None,
	cache_hit: bool | None = None,
) -> list[StepScoreCard]:
	return [
		StepScoreCard(
			edge_id=card.edge_id,
			total_score=card.total_score,
			subscores=dict(card.subscores),
			rationale=card.rationale,
			text=text,
			backend="overlap",
			provider=provider,
			model=model,
			latency_s=latency_s,
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			cache_hit=cache_hit,
			fallback_reason=fallback_reason,
			best_next_edge_id=card.best_next_edge_id,
			raw_response=raw_response,
		)
		for card in cards
	]


def _controller_candidate_from_card(
	card: StepScoreCard,
	*,
	edge_id: str,
	two_hop: bool,
) -> ControllerCandidate:
	return ControllerCandidate(
		edge_id=edge_id,
		utility=_clamp_score(card.total_score),
		support_coverage=_clamp_score(
			card.subscores.get(
				"support_coverage",
				card.subscores.get("support_coverage_hint", card.total_score),
			)
		),
		direct_support=_clamp_score(
			card.subscores.get("direct_support", card.total_score)
		),
		bridge_potential=_clamp_score(
			card.subscores.get("bridge_potential", card.total_score)
		),
		future_potential=(
			_clamp_score(
				card.subscores.get(
					"future_potential", card.subscores.get("future_score", 0.0)
				)
			)
			if two_hop
			else None
		),
		redundancy_risk=_clamp_score(1.0 - card.subscores.get("novelty", 1.0)),
		rationale=card.rationale,
	)


def _allow_choose_two(
	*,
	config: SelectorLLMConfig,
	candidates: Sequence[ControllerCandidate],
	primary: ControllerCandidate | None,
	secondary: ControllerCandidate | None,
	forks_used: int,
) -> bool:
	if not config.enable_scout_fork or forks_used >= max(
		config.max_scout_branches - 1, 1
	):
		return False
	if primary is None or secondary is None:
		return False
	if primary.utility < 0.55 or secondary.utility < 0.55:
		return False
	return (primary.utility - secondary.utility) <= 0.07


def _decision_to_cards(
	*,
	decision: ControllerDecision,
	fallback_cards: Sequence[StepScoreCard],
	candidate_links: Sequence[LinkContext],
	provider: str,
	model: str | None,
	mode: str,
	prefilter_top_n: int,
) -> list[StepScoreCard]:
	candidate_map = {candidate.edge_id: candidate for candidate in decision.candidates}
	top_prefiltered = set(candidate_map)
	if not top_prefiltered:
		top_prefiltered = {
			card.edge_id
			for card in sorted(
				fallback_cards,
				key=lambda item: (item.total_score, item.edge_id),
				reverse=True,
			)[:prefilter_top_n]
		}
	cards: list[StepScoreCard] = []
	for index, _link in enumerate(candidate_links):
		edge_id = str(index)
		fallback_card = fallback_cards[index] if index < len(fallback_cards) else None
		if edge_id not in top_prefiltered:
			cards.append(
				StepScoreCard(
					edge_id=edge_id,
					total_score=0.0,
					subscores={"prefilter_score": 0.0},
					rationale=None,
					text=decision.text,
					backend=provider,
					provider=provider,
					model=model,
					latency_s=decision.latency_s,
					prompt_tokens=decision.prompt_tokens,
					completion_tokens=decision.completion_tokens,
					total_tokens=decision.total_tokens,
				cache_hit=decision.cache_hit,
					fallback_reason="prefiltered_out",
					raw_response=decision.raw_response,
					llm_calls=decision.llm_attempts,
					decision_action=decision.effective_action or decision.action,
					raw_decision_action=decision.action,
					primary_edge_id=decision.primary_edge_id,
					secondary_edge_id=decision.secondary_edge_id,
						backup_edge_id=decision.backup_edge_id,
					stop_score=decision.stop_score,
					evidence_cluster_confidence=decision.evidence_cluster_confidence,
					prefiltered_candidate_count=len(top_prefiltered),
				)
			)
			continue
		candidate = candidate_map.get(edge_id)
		subscores: dict[str, float]
		total_score: float
		rationale: str | None
		if candidate is not None:
			subscores = {
				"controller_utility": candidate.utility,
				"support_coverage": candidate.support_coverage,
				"direct_support": candidate.direct_support,
				"bridge_potential": candidate.bridge_potential,
				"redundancy_risk": candidate.redundancy_risk,
			}
			if mode == "two_hop":
				subscores["future_potential"] = _clamp_score(candidate.future_potential)
			total_score = candidate.utility
			rationale = candidate.rationale
		elif fallback_card is not None:
			subscores = dict(fallback_card.subscores)
			total_score = fallback_card.total_score
			rationale = fallback_card.rationale
		else:
			subscores = {"controller_utility": 0.0}
			total_score = 0.0
			rationale = None
		cards.append(
			StepScoreCard(
				edge_id=edge_id,
				total_score=total_score,
				subscores=subscores,
				rationale=rationale,
				text=decision.text,
				backend=provider,
				provider=provider,
				model=model,
				latency_s=decision.latency_s,
				prompt_tokens=decision.prompt_tokens,
				completion_tokens=decision.completion_tokens,
				total_tokens=decision.total_tokens,
				cache_hit=decision.cache_hit,
				fallback_reason=decision.fallback_reason,
				raw_response=decision.raw_response,
				llm_calls=decision.llm_attempts,
				decision_action=decision.effective_action or decision.action,
				raw_decision_action=decision.action,
				primary_edge_id=decision.primary_edge_id,
				secondary_edge_id=decision.secondary_edge_id,
				backup_edge_id=decision.backup_edge_id,
				stop_score=decision.stop_score,
				evidence_cluster_confidence=decision.evidence_cluster_confidence,
				prefiltered_candidate_count=len(top_prefiltered),
			)
		)
	return cards


def _parse_completion_payload(text: str) -> dict[str, Any]:
	stripped = text.strip()
	if not stripped:
		raise SelectorLLMResponseError("empty_response")

	candidates: list[str] = [stripped]
	fenced = _strip_json_fence(stripped)
	if fenced is not None and fenced not in candidates:
		candidates.append(fenced)
	extracted = _extract_first_json_object(stripped)
	if extracted is not None and extracted not in candidates:
		candidates.append(extracted)

	parse_errors: list[str] = []
	for candidate in candidates:
		try:
			payload = json.loads(candidate)
		except json.JSONDecodeError as exc:
			parse_errors.append(str(exc))
			continue
		if not isinstance(payload, dict):
			raise SelectorLLMResponseError("schema_error", "top_level_not_object")
		return payload

	detail = parse_errors[0] if parse_errors else "unable_to_extract_json"
	raise SelectorLLMResponseError("json_parse_error", detail)


def _cached_payload(record: dict[str, Any]) -> dict[str, Any] | None:
	payload = record.get("response_payload")
	if isinstance(payload, dict):
		return payload
	if isinstance(payload, str):
		parsed = json.loads(payload)
		if isinstance(parsed, dict):
			return parsed
	return None


def _strip_json_fence(text: str) -> str | None:
	stripped = text.strip()
	if not stripped.startswith("```"):
		return None
	lines = stripped.splitlines()
	if len(lines) < 3:
		return None
	if not lines[-1].strip().startswith("```"):
		return None
	return "\n".join(lines[1:-1]).strip() or None


def _extract_first_json_object(text: str) -> str | None:
	start = text.find("{")
	if start < 0:
		return None
	depth = 0
	in_string = False
	escaped = False
	for index in range(start, len(text)):
		char = text[index]
		if in_string:
			if escaped:
				escaped = False
			elif char == "\\":
				escaped = True
			elif char == '"':
				in_string = False
			continue
		if char == '"':
			in_string = True
			continue
		if char == "{":
			depth += 1
			continue
		if char == "}":
			depth -= 1
			if depth == 0:
				return text[start : index + 1]
	return None


def _compact_error_detail(detail: str | None) -> str | None:
	if detail is None:
		return None
	compact = " ".join(str(detail).split())
	return compact or None


def _aggregate_usage_total(values: Sequence[int]) -> int | None:
	return sum(values) if values else None


def _openai_usage_triplet(usage: Any) -> tuple[int | None, int | None, int | None]:
	if usage is None:
		return None, None, None
	if isinstance(usage, dict):
		return (
			_maybe_int(usage.get("prompt_tokens")),
			_maybe_int(usage.get("completion_tokens")),
			_maybe_int(usage.get("total_tokens")),
		)
	return (
		_maybe_int(getattr(usage, "prompt_tokens", None)),
		_maybe_int(getattr(usage, "completion_tokens", None)),
		_maybe_int(getattr(usage, "total_tokens", None)),
	)


def _openai_responses_usage_triplet(
	usage: Any,
) -> tuple[int | None, int | None, int | None]:
	if usage is None:
		return None, None, None
	if isinstance(usage, dict):
		prompt_tokens = _maybe_int(usage.get("input_tokens"))
		completion_tokens = _maybe_int(usage.get("output_tokens"))
		total_tokens = _maybe_int(usage.get("total_tokens"))
	else:
		prompt_tokens = _maybe_int(getattr(usage, "input_tokens", None))
		completion_tokens = _maybe_int(getattr(usage, "output_tokens", None))
		total_tokens = _maybe_int(getattr(usage, "total_tokens", None))
	if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
		total_tokens = prompt_tokens + completion_tokens
	return prompt_tokens, completion_tokens, total_tokens


def _openai_message_content(content: Any) -> str:
	if content is None:
		return "{}"
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		fragments: list[str] = []
		for item in content:
			text = (
				item.get("text")
				if isinstance(item, dict)
				else getattr(item, "text", None)
			)
			if text:
				fragments.append(str(text))
		return "".join(fragments) or "{}"
	return str(content)


def _openai_chat_message_from_payload(payload: dict[str, Any]) -> Any:
	choices = payload.get("choices")
	if not isinstance(choices, list) or not choices:
		return None
	first_choice = choices[0]
	if not isinstance(first_choice, dict):
		return None
	message = first_choice.get("message")
	if not isinstance(message, dict):
		return None
	return message.get("content")


def _openai_response_output_text(response: Any) -> str:
	output_text = getattr(response, "output_text", None)
	if output_text:
		return str(output_text)
	output = getattr(response, "output", None)
	if not isinstance(output, list):
		return "{}"
	fragments: list[str] = []
	for item in output:
		content = (
			item.get("content")
			if isinstance(item, dict)
			else getattr(item, "content", None)
		)
		if not isinstance(content, list):
			continue
		for entry in content:
			if isinstance(entry, dict):
				text = entry.get("text") or entry.get("output_text")
				entry_type = entry.get("type")
			else:
				text = getattr(entry, "text", None) or getattr(
					entry, "output_text", None
				)
				entry_type = getattr(entry, "type", None)
			if entry_type in {"output_text", "text"} and text:
				fragments.append(str(text))
	return "".join(fragments) or "{}"


def _gemini_text(response: Any) -> str:
	candidates = getattr(response, "candidates", None) or []
	fragments: list[str] = []
	for candidate in candidates:
		content = getattr(candidate, "content", None)
		parts = getattr(content, "parts", None) if content is not None else None
		for part in parts or []:
			text = getattr(part, "text", None)
			if text:
				fragments.append(str(text))
	return "".join(fragments)


def _raw_response_payload(response: Any) -> str | None:
	if hasattr(response, "model_dump_json"):
		return response.model_dump_json()
	if hasattr(response, "model_dump"):
		return json.dumps(response.model_dump(), ensure_ascii=False)
	try:
		return json.dumps(response, ensure_ascii=False)
	except TypeError:
		return repr(response)


def _maybe_int(value: Any) -> int | None:
	if value is None:
		return None
	return int(value)


def _maybe_text(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _normalize_openai_base_url(base_url: str) -> str:
	return _normalize_openai_transport_url(
		base_url, api_mode="chat_completions"
	)


def _normalize_openai_transport_url(
	base_url: str,
	*,
	api_mode: OpenAIApiMode,
) -> str:
	text = base_url.strip()
	parsed = urlparse(text)
	if not parsed.scheme or not parsed.netloc:
		raise ValueError(
			"selector_base_url must be an absolute OpenAI-compatible base URL."
		)
	if api_mode == "azure_foundry_chat_completions":
		if "/models/chat/completions" in parsed.path:
			return text
		return text.rstrip("/")
	if api_mode == "github_models_chat_completions":
		if "/inference/chat/completions" not in parsed.path:
			raise ValueError(
				"GitHub Models selector_base_url must be the full REST inference endpoint, for example https://models.github.ai/inference/chat/completions or https://models.github.ai/orgs/<org>/inference/chat/completions."
			)
		if parsed.query:
			raise ValueError(
				"GitHub Models selector_base_url must not include query parameters."
			)
		return text
	if parsed.query:
		raise ValueError(
			"selector_base_url must be a base URL, not a raw endpoint with query parameters."
		)
	path = parsed.path.rstrip("/")
	if path.endswith("/responses") or path.endswith("/chat/completions") or path.endswith(
		"/completions"
	):
		raise ValueError(
			"selector_base_url must point at the OpenAI-compatible API root, for example https://<resource>.openai.azure.com/openai/v1/."
		)
	if parsed.netloc.endswith("azure.com") and path and path != "/openai/v1":
		raise ValueError(
			"Azure selector_base_url must point at the SDK-compatible API root, for example https://<resource>.openai.azure.com/openai/v1/."
		)
	return f"{text.rstrip('/')}/"


def _post_json_request(
	url: str,
	payload: dict[str, Any],
	api_key: str,
	headers: dict[str, str] | None = None,
) -> dict[str, Any]:
	body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
	request_headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}",
	}
	if headers is not None:
		request_headers.update(headers)
	request = urllib.request.Request(
		url,
		data=body,
		headers=request_headers,
		method="POST",
	)
	with urllib.request.urlopen(request) as response:
		text = response.read().decode("utf-8")
	parsed = json.loads(text)
	if not isinstance(parsed, dict):
		raise ValueError("Expected JSON object response from selector backend.")
	return parsed
