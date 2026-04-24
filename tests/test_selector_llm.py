import json
import inspect
from types import SimpleNamespace
from typing import cast

import pytest

import hypercorpus.selector_llm as selector_llm_module
from hypercorpus.copilot import CopilotSdkCompletion
from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.selector import RuntimeBudget, build_selector, select_selectors
from hypercorpus.controller_exposure import (
	ControllerCandidateBundle,
	ControllerCandidateBundleEntry,
	ControllerExposurePlan,
	answer_bearing_link_bonus,
	build_controller_candidate_bundle,
	build_controller_exposure_plan,
	generic_concept_penalty,
	is_generic_concept_candidate,
)
from hypercorpus.graph import DocumentNode, LinkContext, LinkContextGraph
from hypercorpus.selector_llm import (
	AnthropicBackendAdapter,
	BackendCompletion,
	CopilotBackendAdapter,
	LLMController,
	LLMControllerStepScorer,
	OpenAIBackendAdapter,
	SelectorLLMConfig,
	SelectorLLMFallbackError,
	_controller_response_schema,
	_controller_schema_instructions,
	_controller_user_prompt,
)
from hypercorpus.walker import StepScoreCard

SINGLE_HOP = "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_llm__lookahead_1"
TWO_HOP = "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_llm__lookahead_2"
CONTROLLER_SINGLE_PATH = "top_1_seed__lexical_overlap__hop_adaptive__single_path_walk__link_context_llm_controller__lookahead_2"
CONTROLLER_MULTIPATH = "top_1_seed__lexical_overlap__hop_adaptive__constrained_multipath__link_context_llm_controller__lookahead_2"


class FakeBackend:
	def __init__(
		self, payload: dict, *, prompt_tokens: int = 17, completion_tokens: int = 9
	):
		self.payload = payload
		self.prompt_tokens = prompt_tokens
		self.completion_tokens = completion_tokens
		self.call_count = 0

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema=None,
	) -> BackendCompletion:
		del model, system_prompt, user_prompt, temperature, response_schema
		self.call_count += 1
		return BackendCompletion(
			text=json.dumps(self.payload, ensure_ascii=False),
			payload=self.payload,
			prompt_tokens=self.prompt_tokens,
			completion_tokens=self.completion_tokens,
			total_tokens=self.prompt_tokens + self.completion_tokens,
			raw_response=json.dumps({"payload": self.payload}, ensure_ascii=False),
		)


class FakeTextBackend:
	def __init__(
		self,
		text: str,
		*,
		prompt_tokens: int = 17,
		completion_tokens: int = 9,
		raw_response: str | None = None,
	):
		self.text = text
		self.prompt_tokens = prompt_tokens
		self.completion_tokens = completion_tokens
		self.raw_response = raw_response or json.dumps(
			{"text": text}, ensure_ascii=False
		)
		self.call_count = 0

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema=None,
	) -> BackendCompletion:
		del model, system_prompt, user_prompt, temperature, response_schema
		self.call_count += 1
		return BackendCompletion(
			text=self.text,
			payload=None,
			prompt_tokens=self.prompt_tokens,
			completion_tokens=self.completion_tokens,
			total_tokens=self.prompt_tokens + self.completion_tokens,
			raw_response=self.raw_response,
		)


class FakeTextSequenceBackend:
	def __init__(
		self,
		texts: list[str],
		*,
		prompt_tokens: int = 17,
		completion_tokens: int = 9,
	):
		self.texts = texts
		self.prompt_tokens = prompt_tokens
		self.completion_tokens = completion_tokens
		self.call_count = 0
		self.user_prompts: list[str] = []
		self.system_prompts: list[str] = []

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema=None,
	) -> BackendCompletion:
		del model, temperature, response_schema
		text = self.texts[min(self.call_count, len(self.texts) - 1)]
		self.call_count += 1
		self.user_prompts.append(user_prompt)
		self.system_prompts.append(system_prompt)
		return BackendCompletion(
			text=text,
			payload=None,
			prompt_tokens=self.prompt_tokens,
			completion_tokens=self.completion_tokens,
			total_tokens=self.prompt_tokens + self.completion_tokens,
			raw_response=json.dumps({"text": text}, ensure_ascii=False),
		)


def _primary_role_fields(
	role: str = "bridge_support",
	*,
	confidence: float = 0.88,
	rationale: str = "Primary edge role for the chosen node.",
) -> dict[str, object]:
	return {
		"primary_node_role": role,
		"primary_node_role_confidence": confidence,
		"primary_node_role_rationale": rationale,
	}


def _secondary_role_fields(
	role: str = "bridge_support",
	*,
	confidence: float = 0.74,
	rationale: str = "Secondary edge role for the chosen node.",
) -> dict[str, object]:
	return {
		"secondary_node_role": role,
		"secondary_node_role_confidence": confidence,
		"secondary_node_role_rationale": rationale,
	}


def _controller_payload(
	decision: str,
	*,
	runner_up: str | None = None,
	state: str = "need_bridge",
	reason: str = "Controller decision for the visible choice set.",
) -> dict[str, object]:
	return {
		"decision": decision,
		"runner_up": runner_up,
		"state": state,
		"reason": reason,
	}


class FakeSequenceBackend:
	def __init__(
		self,
		payloads: list[dict],
		*,
		prompt_tokens: int = 17,
		completion_tokens: int = 9,
	):
		self.payloads = payloads
		self.prompt_tokens = prompt_tokens
		self.completion_tokens = completion_tokens
		self.call_count = 0

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema=None,
	) -> BackendCompletion:
		del model, system_prompt, user_prompt, temperature, response_schema
		payload = self.payloads[min(self.call_count, len(self.payloads) - 1)]
		self.call_count += 1
		return BackendCompletion(
			text=json.dumps(payload, ensure_ascii=False),
			payload=payload,
			prompt_tokens=self.prompt_tokens,
			completion_tokens=self.completion_tokens,
			total_tokens=self.prompt_tokens + self.completion_tokens,
			raw_response=json.dumps({"payload": payload}, ensure_ascii=False),
		)


class FakeRetrySequenceBackend:
	def __init__(
		self,
		items: list[dict[str, object] | Exception],
		*,
		prompt_tokens: int = 17,
		completion_tokens: int = 9,
	):
		self.items = items
		self.prompt_tokens = prompt_tokens
		self.completion_tokens = completion_tokens
		self.call_count = 0

	def complete_json(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		response_schema=None,
	) -> BackendCompletion:
		del model, system_prompt, user_prompt, temperature, response_schema
		item = self.items[min(self.call_count, len(self.items) - 1)]
		self.call_count += 1
		if isinstance(item, Exception):
			raise item
		return BackendCompletion(
			text=json.dumps(item, ensure_ascii=False),
			payload=item,
			prompt_tokens=self.prompt_tokens,
			completion_tokens=self.completion_tokens,
			total_tokens=self.prompt_tokens + self.completion_tokens,
			raw_response=json.dumps({"payload": item}, ensure_ascii=False),
		)


class FakeTypedResponsesRunner:
	def __init__(
		self,
		items: list[dict[str, object] | Exception],
		*,
		prompt_tokens: int = 17,
		completion_tokens: int = 9,
		requests: int = 1,
	):
		self.items = items
		self.prompt_tokens = prompt_tokens
		self.completion_tokens = completion_tokens
		self.requests = requests
		self.call_count = 0
		self.calls: list[dict[str, object]] = []

	def __call__(
		self,
		*,
		api_key: str,
		base_url: str | None,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float,
		output_type,
		schema_max_attempts: int,
	):
		self.calls.append(
			{
				"api_key": api_key,
				"base_url": base_url,
				"model": model,
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"temperature": temperature,
				"schema_max_attempts": schema_max_attempts,
				"output_type": output_type,
			}
		)
		item = self.items[min(self.call_count, len(self.items) - 1)]
		self.call_count += 1
		if isinstance(item, Exception):
			raise item
		output = output_type.model_validate(item)
		return selector_llm_module.StructuredBackendCompletion(
			output=output,
			prompt_tokens=self.prompt_tokens,
			completion_tokens=self.completion_tokens,
			total_tokens=self.prompt_tokens + self.completion_tokens,
			raw_response=json.dumps({"payload": item}, ensure_ascii=False),
			requests=self.requests,
		)


class CountingScorer:
	def __init__(self, base_score: float):
		self.base_score = base_score
		self.calls = 0
		self.metadata = SimpleNamespace(
			scorer_kind="counting",
			backend="counting",
			provider=None,
			model=None,
			prompt_version=None,
			candidate_prefilter_top_n=None,
			two_hop_prefilter_top_n=None,
			controller_prompt_version=None,
			controller_prefilter_top_n=None,
			controller_future_top_n=None,
		)

	def score_candidates(
		self,
		*,
		query: str,
		graph: LinkContextGraph,
		current_node_id: str,
		candidate_links: list[LinkContext],
		visited_nodes: set[str],
		path_node_ids: list[str],
		remaining_steps: int,
	) -> list[StepScoreCard]:
		del query, graph, current_node_id, visited_nodes, path_node_ids, remaining_steps
		self.calls += 1
		return [
			StepScoreCard(
				edge_id=str(index),
				total_score=max(self.base_score - index * 0.1, 0.0),
				subscores={},
				rationale=None,
				backend="counting",
				provider=None,
				model=None,
				latency_s=0.0,
				prompt_tokens=None,
				completion_tokens=None,
				total_tokens=None,
				cache_hit=None,
				fallback_reason=None,
				llm_calls=None,
				best_next_edge_id=None,
				text=None,
				raw_response=None,
			)
			for index, _link in enumerate(candidate_links)
		]


def _prepare_controller_inputs(
	controller: LLMController,
	*,
	query: str,
	graph: LinkContextGraph,
	current_node_id: str,
	candidate_links: list[LinkContext],
	visited_nodes: set[str],
	path_node_ids: list[str],
	remaining_steps: int,
) -> tuple[list[StepScoreCard], ControllerExposurePlan, ControllerCandidateBundle]:
	fallback_cards = controller.fallback_scorer.score_candidates(
		query=query,
		graph=graph,
		current_node_id=current_node_id,
		candidate_links=candidate_links,
		visited_nodes=visited_nodes,
		path_node_ids=path_node_ids,
		remaining_steps=remaining_steps,
	)
	lexical_cards = controller.prefilter_scorer.score_candidates(
		query=query,
		graph=graph,
		current_node_id=current_node_id,
		candidate_links=candidate_links,
		visited_nodes=visited_nodes,
		path_node_ids=path_node_ids,
		remaining_steps=remaining_steps,
	)
	semantic_cards = (
		controller.semantic_prefilter_scorer.score_candidates(
			query=query,
			graph=graph,
			current_node_id=current_node_id,
			candidate_links=candidate_links,
			visited_nodes=visited_nodes,
			path_node_ids=path_node_ids,
			remaining_steps=remaining_steps,
		)
		if controller.semantic_prefilter_scorer is not None
		else None
	)
	exposure_plan = build_controller_exposure_plan(
		query=query,
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=lexical_cards,
		semantic_cards=semantic_cards,
		small_page_bypass_n=controller.config.controller_small_page_bypass_n,
		lexical_top_n=controller.config.controller_lexical_top_n,
		semantic_top_n=controller.config.controller_semantic_top_n,
		bonus_keep_n=controller.config.controller_bonus_keep_n,
		visible_cap=controller.config.controller_visible_top_n,
	)
	bundle = build_controller_candidate_bundle(
		graph=graph,
		candidate_links=candidate_links,
		score_cards=lexical_cards,
		semantic_score_cards=semantic_cards,
		exposure_plan=exposure_plan,
		query=query,
		current_node_id=current_node_id,
		path_node_ids=path_node_ids,
		visited_nodes=visited_nodes,
		mode=controller.mode,
		future_top_n=controller.config.controller_future_top_n,
		generic_page_policy=controller.config.controller_generic_page_policy,
	)
	return fallback_cards, exposure_plan, bundle


class FakeToolUseContent:
	def __init__(
		self,
		*,
		type: str,
		text: str | None = None,
		name: str | None = None,
		input: dict | None = None,
	):
		self.type = type
		self.text = text
		self.name = name
		self.input = input


class FakeAnthropicUsage:
	def __init__(self, *, input_tokens: int, output_tokens: int):
		self.input_tokens = input_tokens
		self.output_tokens = output_tokens


class FakeAnthropicResponse:
	def __init__(self, *, content, usage):
		self.content = content
		self.usage = usage


class FakeAnthropicMessages:
	def __init__(self, response):
		self.response = response
		self.last_kwargs = None

	def create(self, **kwargs):
		self.last_kwargs = kwargs
		return self.response


class FakeAnthropicClient:
	def __init__(self, response):
		self.messages = FakeAnthropicMessages(response)


class FakeOpenAIResponsesAPI:
	def __init__(self, response: object):
		self.response = response
		self.last_kwargs: dict[str, object] | None = None

	def create(self, **kwargs):
		self.last_kwargs = kwargs
		return self.response


class FakeOpenAIResponsesClient:
	def __init__(self, response: object):
		self.responses = FakeOpenAIResponsesAPI(response)


def test_select_selectors_requires_selector_llm_key(monkeypatch):
	monkeypatch.delenv("MISSING_SELECTOR_KEY", raising=False)

	with pytest.raises(ValueError, match="MISSING_SELECTOR_KEY"):
		select_selectors(
			[SINGLE_HOP],
			selector_provider="anthropic",
			selector_model="claude-test",
			selector_api_key_env="MISSING_SELECTOR_KEY",
		)


def test_anthropic_backend_adapter_uses_tool_output_schema():
	response = FakeAnthropicResponse(
		content=[
			FakeToolUseContent(
				type="tool_use",
				name="score_candidates",
				input={
					"scores": [
						{
							"edge_id": "0",
							"direct_support": 0.8,
							"bridge_potential": 0.7,
							"novelty": 0.6,
							"rationale": "Bridge looks relevant.",
						}
					]
				},
			),
			FakeToolUseContent(type="text", text="unused prose"),
		],
		usage=FakeAnthropicUsage(input_tokens=21, output_tokens=13),
	)
	client = FakeAnthropicClient(response)
	adapter = AnthropicBackendAdapter(
		api_key="test-key", client_factory=lambda api_key: client
	)

	completion = adapter.complete_json(
		model="claude-haiku-test",
		system_prompt="score candidates",
		user_prompt="bundle",
		temperature=0.0,
		response_schema={
			"type": "object",
			"properties": {"scores": {"type": "array"}},
			"required": ["scores"],
		},
	)

	assert completion.payload == {
		"scores": [
			{
				"edge_id": "0",
				"direct_support": 0.8,
				"bridge_potential": 0.7,
				"novelty": 0.6,
				"rationale": "Bridge looks relevant.",
			}
		]
	}
	assert completion.text == "unused prose"
	assert completion.total_tokens == 34
	assert client.messages.last_kwargs is not None
	assert client.messages.last_kwargs["tool_choice"] == {
		"type": "tool",
		"name": "score_candidates",
	}


def test_openai_backend_adapter_responses_mode_delegates_to_typed_runner(monkeypatch):
	runner = FakeTypedResponsesRunner(
		[
			{
				"scores": [
					{
						"edge_id": "0",
						"direct_support": 0.8,
						"bridge_potential": 0.7,
						"novelty": 0.6,
						"rationale": "Bridge looks relevant.",
					}
				]
			}
		],
		prompt_tokens=11,
		completion_tokens=7,
	)
	monkeypatch.setattr(selector_llm_module, "_run_openai_responses_typed", runner)
	adapter = OpenAIBackendAdapter(
		api_key="test-key",
		base_url="https://example.cognitiveservices.azure.com",
		api_mode="responses",
	)

	completion = adapter.complete_typed(
		model="gpt-5.3-codex",
		system_prompt="score candidates",
		user_prompt="bundle",
		temperature=0.0,
		output_type=selector_llm_module._score_output_model("single_hop"),
		schema_max_attempts=3,
	)

	assert completion.output.scores[0].edge_id == "0"
	assert completion.prompt_tokens == 11
	assert completion.completion_tokens == 7
	assert completion.total_tokens == 18
	assert runner.call_count == 1
	assert runner.calls[0]["base_url"] == "https://example.cognitiveservices.azure.com"
	assert runner.calls[0]["model"] == "gpt-5.3-codex"
	assert runner.calls[0]["schema_max_attempts"] == 3


def test_openai_backend_adapter_azure_foundry_chat_mode_posts_json():
	captured: dict[str, object] = {}

	def _fake_post(
		url: str,
		payload: dict[str, object],
		api_key: str,
		headers: dict[str, str] | None = None,
	) -> dict[str, object]:
		captured["url"] = url
		captured["payload"] = payload
		captured["api_key"] = api_key
		captured["headers"] = headers
		return {
			"choices": [{"message": {"content": '{"scores": [{"edge_id": "0"}]}'}}],
			"usage": {
				"prompt_tokens": 13,
				"completion_tokens": 5,
				"total_tokens": 18,
			},
		}

	adapter = OpenAIBackendAdapter(
		api_key="azure-test-key",
		base_url="https://example.cognitiveservices.azure.com/models/chat/completions?api-version=2024-05-01-preview",
		api_mode="azure_foundry_chat_completions",
		http_post=_fake_post,
	)

	completion = adapter.complete_json(
		model="DeepSeek-V3.2-Speciale",
		system_prompt="score candidates",
		user_prompt="bundle",
		temperature=0.0,
	)

	assert completion.text == '{"scores": [{"edge_id": "0"}]}'
	assert completion.prompt_tokens == 13
	assert completion.completion_tokens == 5
	assert completion.total_tokens == 18
	assert (
		captured["url"]
		== "https://example.cognitiveservices.azure.com/models/chat/completions?api-version=2024-05-01-preview"
	)
	assert captured["api_key"] == "azure-test-key"
	payload = cast(dict[str, object], captured["payload"])
	assert payload["model"] == "DeepSeek-V3.2-Speciale"
	assert payload["response_format"] == {"type": "json_object"}
	assert payload["messages"] == [
		{"role": "system", "content": "score candidates"},
		{"role": "user", "content": "bundle"},
	]
	assert captured["headers"] is None


def test_openai_backend_adapter_github_models_chat_mode_uses_sdk_client():
	captured: dict[str, object] = {}

	class FakeUsage:
		prompt_tokens = 19
		completion_tokens = 6
		total_tokens = 25

	class FakeMessage:
		content = '{"scores": [{"edge_id": "0"}]}'

	class FakeChoice:
		message = FakeMessage()

	class FakeResponse:
		choices = [FakeChoice()]
		usage = FakeUsage()

	class FakeCompletions:
		def create(self, **kwargs: object) -> FakeResponse:
			captured["request"] = kwargs
			return FakeResponse()

	class FakeChat:
		def __init__(self) -> None:
			self.completions = FakeCompletions()

	class FakeClient:
		def __init__(self) -> None:
			self.chat = FakeChat()

	def _fake_client_factory(**kwargs: object) -> FakeClient:
		captured["client_kwargs"] = kwargs
		return FakeClient()

	adapter = OpenAIBackendAdapter(
		api_key="gh-models-key",
		base_url="https://models.github.ai/inference",
		api_mode="github_models_chat_completions",
		client_factory=_fake_client_factory,
	)

	completion = adapter.complete_json(
		model="gpt-4.1-mini",
		system_prompt="score candidates",
		user_prompt="bundle",
		temperature=0.0,
	)

	assert completion.text == '{"scores": [{"edge_id": "0"}]}'
	assert completion.prompt_tokens == 19
	assert completion.completion_tokens == 6
	assert completion.total_tokens == 25
	assert captured["client_kwargs"] == {
		"api_key": "gh-models-key",
		"base_url": "https://models.github.ai/inference",
		"default_headers": selector_llm_module.github_models_default_headers(),
	}
	assert captured["request"] == {
		"model": "gpt-4.1-mini",
		"temperature": 0.0,
		"response_format": {"type": "json_object"},
		"messages": [
			{"role": "system", "content": "score candidates"},
			{"role": "user", "content": "bundle"},
		],
	}


def test_selector_llm_config_rejects_raw_endpoint_for_sdk_modes():
	with pytest.raises(ValueError, match="must be a base URL"):
		SelectorLLMConfig(
			provider="openai",
			base_url="https://example.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
			openai_api_mode="responses",
		)


def test_selector_llm_config_accepts_azure_foundry_chat_endpoint():
	config = SelectorLLMConfig(
		provider="openai",
		base_url="https://example.cognitiveservices.azure.com/models/chat/completions?api-version=2024-05-01-preview",
		openai_api_mode="azure_foundry_chat_completions",
	)

	assert (
		config.base_url
		== "https://example.cognitiveservices.azure.com/models/chat/completions?api-version=2024-05-01-preview"
	)


def test_selector_llm_config_accepts_azure_responses_root():
	config = SelectorLLMConfig(
		provider="openai",
		base_url="https://example.cognitiveservices.azure.com",
		openai_api_mode="responses",
	)

	assert config.base_url == "https://example.cognitiveservices.azure.com"


def test_selector_llm_config_defaults_to_copilot_sdk_settings():
	config = SelectorLLMConfig(provider="copilot")

	assert config.model == "gpt-4.1"
	assert config.api_key_env is None
	assert config.base_url is None
	assert config.openai_api_mode is None


def test_selector_llm_config_rejects_copilot_transport_overrides():
	with pytest.raises(ValueError, match="selector_api_key_env"):
		SelectorLLMConfig(provider="copilot", api_key_env="GITHUB_TOKEN")
	with pytest.raises(ValueError, match="selector_base_url"):
		SelectorLLMConfig(
			provider="copilot",
			base_url="https://models.github.ai/inference",
		)
	with pytest.raises(ValueError, match="selector_openai_api_mode"):
		SelectorLLMConfig(
			provider="copilot",
			openai_api_mode="github_models_chat_completions",
		)


def test_selector_llm_config_rejects_provider_prefixed_copilot_model_name():
	with pytest.raises(ValueError, match="SDK-native ids"):
		SelectorLLMConfig(provider="copilot", model="openai/gpt-5")


def test_copilot_backend_adapter_uses_sdk_runner():
	class FakeRunner:
		def __init__(self) -> None:
			self.calls: list[dict[str, object]] = []

		def complete(
			self,
			*,
			model: str,
			system_prompt: str,
			user_prompt: str,
			timeout_s: float = 120.0,
		) -> CopilotSdkCompletion:
			self.calls.append(
				{
					"model": model,
					"system_prompt": system_prompt,
					"user_prompt": user_prompt,
					"timeout_s": timeout_s,
				}
			)
			return CopilotSdkCompletion(
				text='{"scores":[{"edge_id":"0"}]}',
				model=model,
				prompt_tokens=None,
				completion_tokens=None,
				total_tokens=None,
				raw_response='{"scores":[{"edge_id":"0"}]}',
			)

	runner = FakeRunner()
	adapter = CopilotBackendAdapter(runner=runner)

	completion = adapter.complete_json(
		model="gpt-5",
		system_prompt="score candidates",
		user_prompt="bundle",
		temperature=0.0,
	)

	assert completion.text == '{"scores":[{"edge_id":"0"}]}'
	assert completion.prompt_tokens is None
	assert completion.completion_tokens is None
	assert completion.total_tokens is None
	assert runner.calls == [
		{
			"model": "gpt-5",
			"system_prompt": "score candidates",
			"user_prompt": "bundle",
			"timeout_s": 120.0,
		}
	]


def test_single_hop_selector_records_usage_and_logs(
	sample_graph, monkeypatch, tmp_path
):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeBackend(
		{
			"scores": [
				{
					"edge_id": "0",
					"direct_support": 1.0,
					"bridge_potential": 0.8,
					"novelty": 0.6,
					"rationale": "Cape directly supports the answer.",
				},
				{
					"edge_id": "1",
					"direct_support": 0.1,
					"bridge_potential": 0.2,
					"novelty": 0.5,
					"rationale": "Director path is weak.",
				},
			]
		}
	)
	selector = build_selector(
		SINGLE_HOP,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_cache_path=str(tmp_path / "selector-cache.jsonl"),
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(case_id="q1", query="Which city hosts the launch site?")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	first = selector.select(sample_graph, case, budget)
	second = selector.select(sample_graph, case, budget)

	assert "cape" in first.selected_node_ids
	assert first.selector_metadata is not None
	assert first.selector_metadata.provider == "openai"
	assert first.selector_usage is not None
	assert first.selector_usage.llm_calls == 2
	assert first.selector_usage.total_tokens == 52
	assert len(first.selector_logs) == 2
	assert (
		first.selector_logs[0].candidates[0].rationale
		== "Cape directly supports the answer."
	)
	assert backend.call_count == 2

	assert second.selector_logs[0].cache_hit is True
	assert backend.call_count == 2


def test_two_hop_selector_records_best_next_edge_id(monkeypatch):
	monkeypatch.setenv("GEMINI_API_KEY", "test-key")
	backend = FakeBackend(
		{
			"scores": [
				{
					"edge_id": "0",
					"direct_support": 0.2,
					"bridge_potential": 0.3,
					"future_potential": 0.2,
					"novelty": 0.5,
					"rationale": "Bait dead-ends.",
					"best_next_edge_id": "0-0",
				},
				{
					"edge_id": "1",
					"direct_support": 0.4,
					"bridge_potential": 0.8,
					"future_potential": 1.0,
					"novelty": 0.6,
					"rationale": "Bridge reaches the answer node next.",
					"best_next_edge_id": "1-0",
				},
			]
		}
	)
	selector = build_selector(
		TWO_HOP,
		selector_provider="gemini",
		selector_model="gemini-test",
		selector_api_key_env="GEMINI_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(case_id="bridge", query="launch navigation root")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	result = selector.select(_build_bridge_graph(), case, budget)

	assert result.selector_logs
	assert any(
		candidate.best_next_edge_id == "1-0"
		for log in result.selector_logs
		for candidate in log.candidates
	)


@pytest.mark.parametrize(
	("response_text"),
	[
		json.dumps(
			{
				"scores": [
					{
						"edge_id": "0",
						"direct_support": 1.0,
						"bridge_potential": 0.8,
						"novelty": 0.6,
						"rationale": "Cape directly supports the answer.",
					}
				]
			},
			ensure_ascii=False,
		),
		"```json\n"
		+ json.dumps(
			{
				"scores": [
					{
						"edge_id": "0",
						"direct_support": 1.0,
						"bridge_potential": 0.8,
						"novelty": 0.6,
						"rationale": "Cape directly supports the answer.",
					}
				]
			},
			ensure_ascii=False,
		)
		+ "\n```",
		"Here is the requested JSON:\n"
		+ json.dumps(
			{
				"scores": [
					{
						"edge_id": "0",
						"direct_support": 1.0,
						"bridge_potential": 0.8,
						"novelty": 0.6,
						"rationale": "Cape directly supports the answer.",
					}
				]
			},
			ensure_ascii=False,
		),
	],
)
def test_single_hop_selector_parses_text_json_variants(
	sample_graph, monkeypatch, response_text
):
	monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
	backend = FakeTextBackend(response_text, raw_response='{"id":"mock-anthropic"}')
	selector = build_selector(
		SINGLE_HOP,
		selector_provider="anthropic",
		selector_model="claude-haiku-test",
		selector_api_key_env="ANTHROPIC_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(case_id="q-json", query="Which city hosts the launch site?")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	result = selector.select(sample_graph, case, budget)

	assert "cape" in result.selected_node_ids
	assert result.selector_usage is not None
	assert result.selector_usage.fallback_steps == 0
	assert result.selector_logs[0].backend == "anthropic"
	assert result.selector_logs[0].fallback_reason is None
	assert result.selector_logs[0].text == response_text


@pytest.mark.parametrize(
	("response_text", "expected_reason"),
	[
		("not json at all", "json_parse_error:"),
		("   ", "empty_response"),
		(json.dumps({"wrong": []}, ensure_ascii=False), "schema_error:"),
	],
)
def test_single_hop_selector_preserves_usage_on_response_failures(
	sample_graph, monkeypatch, response_text, expected_reason
):
	monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
	backend = FakeTextBackend(response_text, raw_response='{"id":"mock-failure"}')
	selector = build_selector(
		SINGLE_HOP,
		selector_provider="anthropic",
		selector_model="claude-haiku-test",
		selector_api_key_env="ANTHROPIC_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(
		case_id="q-failure", query="Which city hosts the launch site?"
	)
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	with pytest.raises(SelectorLLMFallbackError, match=expected_reason):
		selector.select(sample_graph, case, budget)


def test_controller_single_path_records_decision_and_backtrack(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeSequenceBackend(
		[
			_controller_payload(
				"edge_0",
				runner_up="edge_1",
				state="drift_recovery",
				reason="Try the flashy edge first and keep the bridge as backup.",
			),
			_controller_payload(
				"edge_0",
				runner_up="stop",
				state="need_answer_grounding",
				reason="The answer edge is explicit support.",
			),
		]
	)
	selector = build_selector(
		CONTROLLER_SINGLE_PATH,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(case_id="q-controller", query="harbor root evidence")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	result = selector.select(_build_backtrack_graph(), case, budget)

	assert result.selected_node_ids[0] == "root"
	assert "bridge" in result.selected_node_ids
	assert "answer" in result.selected_node_ids
	assert "bait" not in result.selected_node_ids
	assert backend.call_count == 2
	assert result.selector_usage is not None
	assert result.selector_usage.llm_calls == 2
	assert result.selector_usage.controller_backtrack_actions == 1
	assert result.selector_usage.controller_calls == 2
	assert any(
		log.controller is not None and log.controller.kind == "backtrack"
		for log in result.selector_logs
	)
	assert result.selector_logs[0].controller is not None
	assert result.selector_logs[0].controller.backup_edge_id == "1"


def test_controller_constrained_multipath_forks_once_and_keeps_bridge(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeSequenceBackend(
		[
			_controller_payload(
				"edge_1",
				runner_up="edge_0",
				state="need_bridge",
				reason="The bridge branch carries the best support path.",
			),
			_controller_payload(
				"edge_0",
				runner_up="stop",
				state="need_answer_grounding",
				reason="The answer edge is explicit support.",
			),
		]
	)
	selector = build_selector(
		CONTROLLER_MULTIPATH,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(case_id="q-multipath", query="launch navigation root")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	result = selector.select(_build_bridge_graph(), case, budget)

	assert "bridge" in result.selected_node_ids
	assert "answer" in result.selected_node_ids
	assert result.selector_usage is not None
	assert result.selector_usage.controller_fork_actions == 1
	assert result.selector_usage.controller_backtrack_actions == 0
	assert any(
		log.controller is not None and log.controller.effective_action == "choose_two"
		for log in result.selector_logs
	)


def test_controller_retries_schema_failure_then_succeeds(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	backend = FakeTextSequenceBackend(
		[
			'{"decision":": "}',
			json.dumps(
				_controller_payload(
					"edge_0",
					runner_up="stop",
					state="need_answer_grounding",
					reason="The bridge answer edge is explicit support.",
				),
				ensure_ascii=False,
			),
		],
	)
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: backend,
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	decision = controller.decide(
		query="launch navigation root",
		path_node_ids=["root"],
		current_depth=1,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
	)

	assert backend.call_count == 2
	assert len(backend.user_prompts) == 2
	assert "Required JSON schema:" in backend.user_prompts[0]
	assert "Your previous response failed validation." in backend.user_prompts[1]
	assert "schema_error:invalid_decision" in backend.user_prompts[1]
	assert '{"decision":": "}' in backend.user_prompts[1]
	assert decision.fallback_reason is None
	assert decision.llm_attempts == 2
	assert decision.total_tokens == 52
	assert decision.prompt_tokens == 34
	assert decision.completion_tokens == 18
	assert decision.primary_edge_id == "0"
	assert [candidate.edge_id for candidate in decision.candidates] == ["0", "1"]
	assert decision.visible_edge_ids == ["0", "1"]


def test_controller_retries_three_times_then_raises(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	backend = FakeTextSequenceBackend(
		[
			'{"decision":": "}',
			'{"decision":"edge_99"}',
			'{"wrong":[]}',
		]
	)
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: backend,
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	with pytest.raises(SelectorLLMFallbackError, match="schema_error:"):
		controller.decide(
			query="launch navigation root",
			path_node_ids=["root"],
			current_depth=1,
			fallback_cards=fallback_cards,
			exposure_plan=exposure_plan,
			bundle=bundle,
		)


def test_controller_lenient_secondary_fields(monkeypatch):
	"""Open-weight models may omit or malform runner_up, state, reason.

	The controller should degrade gracefully: primary decision must be valid,
	but secondary fields fall back to safe defaults instead of raising.
	"""
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	# Only decision is present; runner_up, state, reason all missing.
	backend = FakeBackend({"decision": "edge_0"})
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: backend,
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	decision = controller.decide(
		query="launch navigation root",
		path_node_ids=["root"],
		current_depth=1,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
	)
	assert decision.decision == "edge_0"
	assert decision.runner_up is None
	assert decision.state == "need_bridge"
	assert decision.reason is None


def test_controller_lenient_runner_up_none_string(monkeypatch):
	"""runner_up returned as 'none' or 'null' should resolve to None."""
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	for none_value in ["none", "null", "None", "NULL", "N/A"]:
		backend = FakeBackend(
			{
				"decision": "edge_0",
				"runner_up": none_value,
				"state": "need_bridge",
				"reason": "proceed",
			}
		)
		controller = LLMController(
			config=SelectorLLMConfig(
				provider="openai",
				model="gpt-test-mini",
				api_key_env="OPENAI_API_KEY",
			),
			mode="two_hop",
			backend_factory=lambda _config: backend,
		)
		candidate_links = list(graph.links_from("root"))
		fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
			controller,
			query="launch navigation root",
			graph=graph,
			current_node_id="root",
			candidate_links=candidate_links,
			visited_nodes={"root"},
			path_node_ids=["root"],
			remaining_steps=2,
		)
		decision = controller.decide(
			query="launch navigation root",
			path_node_ids=["root"],
			current_depth=1,
			fallback_cards=fallback_cards,
			exposure_plan=exposure_plan,
			bundle=bundle,
		)
		assert decision.runner_up is None, f"runner_up={none_value!r} should become None"


def test_controller_lenient_runner_up_matches_decision(monkeypatch):
	"""runner_up matching decision should be cleared to None, not raise."""
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	backend = FakeBackend(
		{
			"decision": "edge_0",
			"runner_up": "edge_0",
			"state": "need_bridge",
			"reason": "proceed",
		}
	)
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: backend,
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	decision = controller.decide(
		query="launch navigation root",
		path_node_ids=["root"],
		current_depth=1,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
	)
	assert decision.decision == "edge_0"
	assert decision.runner_up is None


def test_controller_lenient_invalid_state_falls_back(monkeypatch):
	"""Unrecognized state string should fall back, not raise."""
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	backend = FakeBackend(
		{
			"decision": "stop",
			"runner_up": None,
			"state": "exploring",
			"reason": "done",
		}
	)
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: backend,
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	decision = controller.decide(
		query="launch navigation root",
		path_node_ids=["root"],
		current_depth=1,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
	)
	assert decision.decision == "stop"
	assert decision.state == "enough_evidence"


def test_controller_retries_provider_rate_limit_then_succeeds(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	sleep_delays: list[float] = []
	monkeypatch.setattr(
		selector_llm_module.time,
		"sleep",
		lambda delay_s: sleep_delays.append(delay_s),
	)
	graph = _build_bridge_graph()
	backend = FakeRetrySequenceBackend(
		[
			RuntimeError("HTTP Error 429: Too Many Requests"),
			RuntimeError("HTTP Error 429: Too Many Requests"),
			_controller_payload(
				"edge_0",
				runner_up="stop",
				state="need_answer_grounding",
				reason="The bridge answer edge is explicit support.",
			),
		]
	)
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: backend,
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	decision = controller.decide(
		query="launch navigation root",
		path_node_ids=["root"],
		current_depth=1,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
	)

	assert backend.call_count == 3
	assert sleep_delays == [1.0, 2.0]
	assert decision.fallback_reason is None
	assert decision.llm_attempts == 3
	assert decision.primary_edge_id == "0"


def test_llm_step_scorer_retries_provider_rate_limit_then_succeeds(
	sample_graph, monkeypatch
):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	sleep_delays: list[float] = []
	monkeypatch.setattr(
		selector_llm_module.time,
		"sleep",
		lambda delay_s: sleep_delays.append(delay_s),
	)
	backend = FakeRetrySequenceBackend(
		[
			RuntimeError("HTTP Error 429: Too Many Requests"),
			{
				"scores": [
					{
						"edge_id": "0",
						"direct_support": 1.0,
						"bridge_potential": 0.8,
						"novelty": 0.6,
						"rationale": "Cape directly supports the answer.",
					},
					{
						"edge_id": "1",
						"direct_support": 0.1,
						"bridge_potential": 0.2,
						"novelty": 0.5,
						"rationale": "Director path is weak.",
					},
				]
			},
		]
	)
	scorer = selector_llm_module.LLMStepLinkScorer(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="single_hop",
		backend_factory=lambda _config: backend,
	)

	cards = scorer.score_candidates(
		query="Which city hosts the launch site?",
		graph=sample_graph,
		current_node_id="mission",
		candidate_links=list(sample_graph.links_from("mission")),
		visited_nodes={"mission"},
		path_node_ids=["mission"],
		remaining_steps=1,
	)

	assert backend.call_count == 2
	assert sleep_delays == [1.0]
	assert cards[0].fallback_reason is None
	assert cards[0].prompt_tokens == 17


def test_controller_openai_responses_schema_error_raises(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	runner = FakeTypedResponsesRunner(
		[
			selector_llm_module.SelectorLLMResponseError(
				"schema_error", "missing_candidates_list"
			)
		]
	)
	monkeypatch.setattr(selector_llm_module, "_run_openai_responses_typed", runner)
	graph = _build_bridge_graph()
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-5.3-codex",
			api_key_env="OPENAI_API_KEY",
			base_url="https://example.cognitiveservices.azure.com",
			openai_api_mode="responses",
		),
		mode="two_hop",
		backend_factory=lambda _config: OpenAIBackendAdapter(
			api_key="test-key",
			base_url="https://example.cognitiveservices.azure.com",
			api_mode="responses",
		),
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	with pytest.raises(
		SelectorLLMFallbackError, match="schema_error:missing_candidates_list"
	):
		controller.decide(
			query="launch navigation root",
			path_node_ids=["root"],
			current_depth=1,
			fallback_cards=fallback_cards,
			exposure_plan=exposure_plan,
			bundle=bundle,
		)


def test_controller_openai_responses_retries_provider_rate_limit_then_succeeds(
	monkeypatch,
):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	sleep_delays: list[float] = []
	monkeypatch.setattr(
		selector_llm_module.time,
		"sleep",
		lambda delay_s: sleep_delays.append(delay_s),
	)
	runner = FakeTypedResponsesRunner(
		[
			RuntimeError("HTTP Error 429: Too Many Requests"),
			RuntimeError("HTTP Error 429: Too Many Requests"),
			_controller_payload(
				"edge_0",
				runner_up="stop",
				state="need_answer_grounding",
				reason="The bridge answer edge is explicit support.",
			),
		]
	)
	monkeypatch.setattr(selector_llm_module, "_run_openai_responses_typed", runner)
	graph = _build_bridge_graph()
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-5.3-codex",
			api_key_env="OPENAI_API_KEY",
			base_url="https://example.cognitiveservices.azure.com",
			openai_api_mode="responses",
		),
		mode="two_hop",
		backend_factory=lambda _config: OpenAIBackendAdapter(
			api_key="test-key",
			base_url="https://example.cognitiveservices.azure.com",
			api_mode="responses",
		),
	)
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)
	decision = controller.decide(
		query="launch navigation root",
		path_node_ids=["root"],
		current_depth=1,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
	)

	assert runner.call_count == 3
	assert sleep_delays == [1.0, 2.0]
	assert decision.fallback_reason is None
	assert decision.primary_edge_id == "0"
	assert decision.llm_attempts == 3


def test_llm_step_scorer_openai_responses_retries_provider_rate_limit_then_succeeds(
	sample_graph, monkeypatch
):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	sleep_delays: list[float] = []
	monkeypatch.setattr(
		selector_llm_module.time,
		"sleep",
		lambda delay_s: sleep_delays.append(delay_s),
	)
	runner = FakeTypedResponsesRunner(
		[
			RuntimeError("HTTP Error 429: Too Many Requests"),
			{
				"scores": [
					{
						"edge_id": "0",
						"direct_support": 1.0,
						"bridge_potential": 0.8,
						"novelty": 0.6,
						"rationale": "Cape directly supports the answer.",
					},
					{
						"edge_id": "1",
						"direct_support": 0.1,
						"bridge_potential": 0.2,
						"novelty": 0.5,
						"rationale": "Director path is weak.",
					},
				]
			},
		]
	)
	monkeypatch.setattr(selector_llm_module, "_run_openai_responses_typed", runner)
	scorer = selector_llm_module.LLMStepLinkScorer(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-5.3-codex",
			api_key_env="OPENAI_API_KEY",
			base_url="https://example.cognitiveservices.azure.com",
			openai_api_mode="responses",
		),
		mode="single_hop",
		backend_factory=lambda _config: OpenAIBackendAdapter(
			api_key="test-key",
			base_url="https://example.cognitiveservices.azure.com",
			api_mode="responses",
		),
	)

	cards = scorer.score_candidates(
		query="Which city hosts the launch site?",
		graph=sample_graph,
		current_node_id="mission",
		candidate_links=list(sample_graph.links_from("mission")),
		visited_nodes={"mission"},
		path_node_ids=["mission"],
		remaining_steps=1,
	)

	assert runner.call_count == 2
	assert sleep_delays == [1.0]
	assert cards[0].fallback_reason is None
	assert cards[0].prompt_tokens == 17


def test_controller_step_scoring_happens_once_per_stage(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	backend = FakeBackend(
		_controller_payload(
			"edge_0",
			runner_up="stop",
			state="need_bridge",
			reason="The bridge edge should be followed.",
		)
	)
	fallback_scorer = CountingScorer(0.9)
	prefilter_scorer = CountingScorer(0.8)
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		prefilter_scorer=prefilter_scorer,
		fallback_scorer=fallback_scorer,
		backend_factory=lambda _config: backend,
	)
	scorer = LLMControllerStepScorer(
		controller=controller,
		config=controller.config,
		mode="two_hop",
		fallback_scorer=fallback_scorer,
	)

	result = scorer.evaluate_controller_step(
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=list(graph.links_from("root")),
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
		current_depth=1,
	)

	assert fallback_scorer.calls == 1
	assert prefilter_scorer.calls == 1
	assert result.primary is not None
	assert result.primary.edge_id == "0"


def test_controller_decide_requires_precomputed_exposure_inputs() -> None:
	parameter_names = list(inspect.signature(LLMController.decide).parameters)

	assert parameter_names == [
		"self",
		"query",
		"path_node_ids",
		"current_depth",
		"fallback_cards",
		"exposure_plan",
		"bundle",
		"forks_used",
		"backtracks_used",
	]


def test_controller_stop_decision_is_preserved_in_nested_trace(
	monkeypatch,
):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeSequenceBackend(
		[
			_controller_payload(
				"stop",
				runner_up="edge_1",
				state="enough_evidence",
				reason="The visible alternatives are not meaningful enough to justify another hop.",
			),
			_controller_payload(
				"edge_0",
				runner_up="stop",
				state="need_answer_grounding",
				reason="The answer edge is explicit support.",
			),
		]
	)
	selector = build_selector(
		CONTROLLER_MULTIPATH,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(case_id="q-raw-stop", query="launch navigation root")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	result = selector.select(_build_bridge_graph(), case, budget)

	assert backend.call_count == 1
	assert result.selected_node_ids == ["root"]
	assert result.selector_logs[0].controller is not None
	assert result.selector_logs[0].controller.decision == "stop"
	assert result.selector_logs[0].controller.runner_up == "edge_1"
	assert result.selector_logs[0].controller.effective_action == "stop"
	assert result.selector_logs[0].stop_reason == "controller_stop"


def test_controller_root_stop_is_coerced_to_answer_bearing_edge(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: FakeBackend({}),
	)
	graph = _build_geneva_doctorate_graph()
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="In what country did Bain attend doctoral seminars of Wlad Godzich?",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)

	decision = controller._decision_from_payload(
		payload=_controller_payload(
			"stop",
			runner_up="edge_1",
			state="enough_evidence",
			reason="The current page already suggests the answer.",
		),
		text='{"decision":"stop"}',
		raw_response='{"decision":"stop"}',
		prompt_tokens=10,
		completion_tokens=5,
		total_tokens=15,
		latency_s=0.0,
		cache_hit=False,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
		llm_attempts=1,
		current_depth=0,
	)

	assert decision.fallback_reason is None
	assert decision.decision == "edge_1"
	assert decision.runner_up == "stop"
	assert decision.primary_edge_id == "1"
	assert decision.state == "need_answer_grounding"
	assert "strong answer-bearing visible edge" in (decision.reason or "")


def test_controller_nested_stop_is_not_coerced(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: FakeBackend({}),
	)
	graph = _build_geneva_doctorate_graph()
	candidate_links = list(graph.links_from("root"))
	fallback_cards, exposure_plan, bundle = _prepare_controller_inputs(
		controller,
		query="In what country did Bain attend doctoral seminars of Wlad Godzich?",
		graph=graph,
		current_node_id="root",
		candidate_links=candidate_links,
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
	)

	decision = controller._decision_from_payload(
		payload=_controller_payload(
			"stop",
			runner_up="edge_1",
			state="enough_evidence",
			reason="The current page already suggests the answer.",
		),
		text='{"decision":"stop"}',
		raw_response='{"decision":"stop"}',
		prompt_tokens=10,
		completion_tokens=5,
		total_tokens=15,
		latency_s=0.0,
		cache_hit=False,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
		llm_attempts=1,
		current_depth=1,
	)

	assert decision.decision == "stop"
	assert decision.runner_up == "edge_1"
	assert decision.primary_edge_id is None
	assert decision.state == "enough_evidence"
	assert decision.reason == "The current page already suggests the answer."


def test_controller_ratio_budget_does_not_trigger_budget_pacing_stop(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeSequenceBackend(
		[
			_controller_payload(
				"edge_1",
				runner_up="stop",
				state="need_bridge",
				reason="The bridge edge should be preferred first.",
			),
			_controller_payload(
				"edge_0",
				runner_up="stop",
				state="need_answer_grounding",
				reason="The answer edge is explicit support.",
			),
		]
	)
	selector = build_selector(
		CONTROLLER_MULTIPATH,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(case_id="q-ratio-budget", query="launch navigation root")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_ratio=0.01))

	result = selector.select(_build_bridge_graph(), case, budget)

	assert backend.call_count == 2
	assert result.stop_reason != "budget_pacing_stop"
	assert all(log.stop_reason != "budget_pacing_stop" for log in result.selector_logs)


def test_lexical_controller_selector_does_not_require_embedder(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeBackend(
		_controller_payload(
			"edge_0",
			runner_up="stop",
			state="need_bridge",
			reason="Follow the bridge page.",
		)
	)
	selector = build_selector(
		CONTROLLER_SINGLE_PATH,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_backend_factory=lambda _config: backend,
		sentence_transformer_embedder_factory=lambda _config: (_ for _ in ()).throw(
			AssertionError("controller lexical path should not build an embedder")
		),
	)
	case = EvaluationCase(case_id="q-no-embedder", query="launch navigation root")
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	result = selector.select(_build_bridge_graph(), case, budget)

	assert backend.call_count >= 1
	assert "bridge" in result.selected_node_ids


def test_controller_all_dangling_page_stops_without_backend_call(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	factory_calls = 0

	def _backend_factory(_config: SelectorLLMConfig) -> FakeBackend:
		nonlocal factory_calls
		factory_calls += 1
		return FakeBackend({})

	selector = build_selector(
		CONTROLLER_SINGLE_PATH,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_backend_factory=_backend_factory,
	)
	case = EvaluationCase(
		case_id="q-all-dangling", query="Which target should we inspect next?"
	)
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_tokens=128))

	result = selector.select(_build_all_dangling_graph(), case, budget)

	assert factory_calls == 0
	assert result.selected_node_ids == ["root"]
	assert result.stop_reason == "dead_end"
	assert len(result.selector_logs) == 1
	assert result.selector_logs[0].chosen_edge_id is None
	assert all(
		candidate.exposure_status == "filtered_dangling_target"
		for candidate in result.selector_logs[0].candidates
	)


def test_controller_exposure_plan_keeps_answer_bearing_edge_and_filters_dangling_target():
	query = "In what country did Bain attend doctoral seminars of Wlad Godzich?"
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"root",
				"Thomas Bain (Orange)",
				(
					"Thomas Bain attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
				),
			),
			DocumentNode(
				"wlad", "Wlad Godzich", ("Wlad Godzich is a literary theorist.",)
			),
			DocumentNode(
				"geneva",
				"University of Geneva",
				("The University of Geneva is in Switzerland.",),
			),
			DocumentNode("bait", "Orange Order", ("Orange Order is unrelated here.",)),
		]
	)
	candidate_links = [
		LinkContext(
			source="root",
			target="wlad",
			anchor_text="doctoral seminars of Wlad Godzich",
			sentence="Thomas Bain attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
			sent_idx=0,
		),
		LinkContext(
			source="root",
			target="geneva",
			anchor_text="University of Geneva",
			sentence="Thomas Bain attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
			sent_idx=0,
		),
		LinkContext(
			source="root",
			target="26833.Wlad Godzich",
			anchor_text="26833.Wlad Godzich",
			sentence="A malformed target should never be exposed to the controller.",
			sent_idx=0,
		),
		LinkContext(
			source="root",
			target="bait",
			anchor_text="Orange Order",
			sentence="Orange Order is unrelated here.",
			sent_idx=0,
		),
	]
	cards = [
		StepScoreCard(
			edge_id="0",
			total_score=0.95,
			subscores={
				"anchor_overlap": 0.86,
				"sentence_overlap": 0.72,
				"target_overlap": 0.28,
				"novelty": 1.0,
			},
			rationale=None,
			text=None,
			backend="overlap",
			provider=None,
			model=None,
			latency_s=0.0,
			prompt_tokens=None,
			completion_tokens=None,
			total_tokens=None,
			cache_hit=None,
			fallback_reason=None,
		),
		StepScoreCard(
			edge_id="1",
			total_score=0.54,
			subscores={
				"anchor_overlap": 0.14,
				"sentence_overlap": 0.72,
				"target_overlap": 0.0,
				"novelty": 1.0,
			},
			rationale=None,
			text=None,
			backend="overlap",
			provider=None,
			model=None,
			latency_s=0.0,
			prompt_tokens=None,
			completion_tokens=None,
			total_tokens=None,
			cache_hit=None,
			fallback_reason=None,
		),
		StepScoreCard(
			edge_id="2",
			total_score=0.91,
			subscores={
				"anchor_overlap": 0.62,
				"sentence_overlap": 0.18,
				"target_overlap": 0.0,
				"novelty": 1.0,
			},
			rationale=None,
			text=None,
			backend="overlap",
			provider=None,
			model=None,
			latency_s=0.0,
			prompt_tokens=None,
			completion_tokens=None,
			total_tokens=None,
			cache_hit=None,
			fallback_reason=None,
		),
		StepScoreCard(
			edge_id="3",
			total_score=0.53,
			subscores={
				"anchor_overlap": 0.22,
				"sentence_overlap": 0.18,
				"target_overlap": 0.0,
				"novelty": 1.0,
			},
			rationale=None,
			text=None,
			backend="overlap",
			provider=None,
			model=None,
			latency_s=0.0,
			prompt_tokens=None,
			completion_tokens=None,
			total_tokens=None,
			cache_hit=None,
			fallback_reason=None,
		),
	]

	assert answer_bearing_link_bonus(
		query=query,
		graph=graph,
		link=candidate_links[1],
		card=cards[1],
	) > answer_bearing_link_bonus(
		query=query,
		graph=graph,
		link=candidate_links[3],
		card=cards[3],
	)
	plan = build_controller_exposure_plan(
		query=query,
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=cards,
		semantic_cards=None,
		small_page_bypass_n=1,
		lexical_top_n=1,
		semantic_top_n=0,
		bonus_keep_n=1,
		visible_cap=16,
	)

	assert plan.small_page_bypass is False
	assert plan.raw_candidate_count == 4
	assert plan.valid_candidate_count == 3
	assert plan.dangling_indices == [2]
	assert 1 in plan.visible_indices
	assert 2 not in plan.visible_indices


def test_controller_exposure_plan_pins_best_answer_bearing_edge_into_visible_cap():
	query = "In what country did Bain attend doctoral seminars of Wlad Godzich?"
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"root", "Thomas Bain (Orange)", ("Root mentions two candidates.",)
			),
			DocumentNode(
				"geneva",
				"University of Geneva",
				("The University of Geneva is in Switzerland.",),
			),
			DocumentNode(
				"bait", "Academic history", ("Academic history is broad context only.",)
			),
		]
	)
	candidate_links = [
		LinkContext(
			source="root",
			target="bait",
			anchor_text="academic history",
			sentence="This broad academic-history page is highly lexical but not answer-bearing.",
			sent_idx=0,
		),
		LinkContext(
			source="root",
			target="geneva",
			anchor_text="University of Geneva",
			sentence="Thomas Bain attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
			sent_idx=0,
		),
	]
	cards = [
		StepScoreCard(
			edge_id="0",
			total_score=0.97,
			subscores={
				"anchor_overlap": 0.92,
				"sentence_overlap": 0.08,
				"target_overlap": 0.0,
				"novelty": 1.0,
			},
		),
		StepScoreCard(
			edge_id="1",
			total_score=0.42,
			subscores={
				"anchor_overlap": 0.15,
				"sentence_overlap": 0.74,
				"target_overlap": 0.0,
				"novelty": 1.0,
			},
		),
	]

	plan = build_controller_exposure_plan(
		query=query,
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=cards,
		semantic_cards=None,
		small_page_bypass_n=1,
		lexical_top_n=1,
		semantic_top_n=0,
		bonus_keep_n=1,
		visible_cap=1,
	)

	assert plan.visible_indices == [1]
	assert plan.bonus_rescued_edge_ids == ["1"]


def test_controller_exposure_plan_bonus_rescue_prefers_higher_semantic_score():
	query = "Which port contains the answer?"
	graph = LinkContextGraph(
		documents=[
			DocumentNode("root", "Root", ("Root mentions two candidate ports.",)),
			DocumentNode("alpha", "Alpha Port", ("Alpha Port is a plausible answer.",)),
			DocumentNode(
				"beta", "Beta Port", ("Beta Port is another plausible answer.",)
			),
		]
	)
	candidate_links = [
		LinkContext(
			source="root",
			target="alpha",
			anchor_text="Alpha Port",
			sentence="Alpha Port may contain the answer.",
			sent_idx=0,
		),
		LinkContext(
			source="root",
			target="beta",
			anchor_text="Beta Port",
			sentence="Beta Port may contain the answer.",
			sent_idx=0,
		),
	]
	lexical_cards = [
		StepScoreCard(
			edge_id="0",
			total_score=0.50,
			subscores={
				"anchor_overlap": 0.40,
				"sentence_overlap": 0.80,
				"target_overlap": 0.50,
			},
		),
		StepScoreCard(
			edge_id="1",
			total_score=0.50,
			subscores={
				"anchor_overlap": 0.40,
				"sentence_overlap": 0.80,
				"target_overlap": 0.50,
			},
		),
	]
	semantic_cards = [
		StepScoreCard(edge_id="0", total_score=0.90),
		StepScoreCard(edge_id="1", total_score=0.70),
	]

	plan = build_controller_exposure_plan(
		query=query,
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=lexical_cards,
		semantic_cards=semantic_cards,
		small_page_bypass_n=1,
		lexical_top_n=0,
		semantic_top_n=0,
		bonus_keep_n=1,
		visible_cap=16,
	)

	assert plan.bonus_rescued_edge_ids == ["0"]
	assert plan.visible_indices == [0]


def test_controller_payload_parsing_keeps_sparse_visible_edge_ids(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	controller = LLMController(
		config=SelectorLLMConfig(
			provider="openai",
			model="gpt-test-mini",
			api_key_env="OPENAI_API_KEY",
		),
		mode="two_hop",
		backend_factory=lambda _config: FakeBackend({}),
	)
	fallback_cards = [
		StepScoreCard(edge_id=str(index), total_score=0.01 * (index + 1), subscores={})
		for index in range(31)
	]
	exposure_plan = ControllerExposurePlan(
		raw_candidate_count=31,
		valid_candidate_count=31,
		small_page_bypass=False,
		valid_indices=list(range(31)),
		dangling_indices=[],
		lexical_prefilter_edge_ids=["30"],
		semantic_prefilter_edge_ids=[],
		bonus_rescued_edge_ids=["30"],
		visible_indices=[30],
	)
	bundle = ControllerCandidateBundle(
		query="In what country did Bain attend doctoral seminars of Wlad Godzich?",
		current_node_id="root",
		path_titles=["Thomas Bain (Orange)"],
		raw_candidate_count=31,
		valid_candidate_count=31,
		small_page_bypass=False,
		dangling_edge_ids=[],
		lexical_prefilter_edge_ids=["30"],
		semantic_prefilter_edge_ids=[],
		bonus_rescued_edge_ids=["30"],
		visible_edge_ids=["30"],
		generic_page_policy="prompt_only",
		candidates=[
			ControllerCandidateBundleEntry(
				edge_id="30",
				source_title="Thomas Bain (Orange)",
				target_title="University of Geneva",
				anchor_text="University of Geneva",
				sentence="Thomas Bain attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
				prefilter_score=0.99,
				query_anchor_overlap=0.25,
				query_sentence_overlap=0.80,
				query_target_overlap=0.0,
				answer_bearing_link_bonus=0.90,
				source_sentence_mentions_target_title=True,
				semantic_prefilter_score=0.0,
				generic_concept_like=False,
				generic_concept_penalty=0.0,
			)
		],
	)

	decision = controller._decision_from_payload(
		payload=_controller_payload(
			"stop",
			runner_up="edge_30",
			state="enough_evidence",
			reason="The visible edge is already recorded as the runner-up.",
		),
		text='{"decision":"stop","runner_up":"edge_30"}',
		raw_response='{"decision":"stop","runner_up":"edge_30"}',
		prompt_tokens=10,
		completion_tokens=5,
		total_tokens=15,
		latency_s=0.0,
		cache_hit=False,
		fallback_cards=fallback_cards,
		exposure_plan=exposure_plan,
		bundle=bundle,
		llm_attempts=1,
		current_depth=1,
	)

	assert decision.fallback_reason is None
	assert decision.decision == "stop"
	assert decision.runner_up == "edge_30"
	assert decision.visible_edge_ids == ["30"]
	assert [candidate.edge_id for candidate in decision.candidates] == ["30"]


def test_controller_exposure_plan_bypasses_small_pages():
	graph = LinkContextGraph(
		documents=[
			DocumentNode("root", "Root", ("Root connects to a few valid pages.",)),
			DocumentNode("alpha", "Alpha", ("Alpha document.",)),
			DocumentNode("beta", "Beta", ("Beta document.",)),
			DocumentNode("gamma", "Gamma", ("Gamma document.",)),
		]
	)
	candidate_links = [
		LinkContext(
			source="root",
			target="alpha",
			anchor_text="Alpha",
			sentence="Root links to Alpha.",
			sent_idx=0,
		),
		LinkContext(
			source="root",
			target="beta",
			anchor_text="Beta",
			sentence="Root links to Beta.",
			sent_idx=0,
		),
		LinkContext(
			source="root",
			target="gamma",
			anchor_text="Gamma",
			sentence="Root links to Gamma.",
			sent_idx=0,
		),
	]
	cards = [
		StepScoreCard(edge_id=str(index), total_score=1.0 - index * 0.1)
		for index in range(len(candidate_links))
	]

	plan = build_controller_exposure_plan(
		query="Which page should we inspect next?",
		graph=graph,
		candidate_links=candidate_links,
		lexical_cards=cards,
		semantic_cards=None,
		small_page_bypass_n=3,
		lexical_top_n=1,
		semantic_top_n=0,
		bonus_keep_n=1,
		visible_cap=16,
	)

	assert plan.small_page_bypass is True
	assert plan.valid_indices == [0, 1, 2]
	assert plan.visible_indices == [0, 1, 2]
	assert plan.lexical_prefilter_edge_ids == []
	assert plan.semantic_prefilter_edge_ids == []
	assert plan.bonus_rescued_edge_ids == []


def test_controller_prompt_mentions_answer_bearing_bonus():
	prompt = _controller_user_prompt(
		query="In what country did Bain attend doctoral seminars of Wlad Godzich?",
		bundle={
			"query": "In what country did Bain attend doctoral seminars of Wlad Godzich?",
			"current_node_id": "root",
			"path_titles": ["Thomas Bain (Orange)"],
			"generic_page_policy": "strong_generic_penalty",
			"candidates": [
				{
					"edge_id": "1",
					"target_title": "University of Geneva",
					"answer_bearing_link_bonus": 0.82,
					"generic_concept_like": False,
					"generic_concept_penalty": 0.0,
				}
			],
		},
		mode="two_hop",
	)

	assert "answer_bearing_link_bonus" in prompt
	assert "directly helps answer the question" in prompt
	assert "Unused budget is acceptable" in prompt
	assert "Named entities or places are not automatically good follow-ups" in prompt
	assert "Lateral related entities are low utility" in prompt
	assert "materially improves answer-bearing support" in prompt
	assert "STOP" in prompt
	assert "runner_up" in prompt
	assert "state" in prompt
	assert "primary_node_role" not in prompt
	assert "generic concept or degree pages" in prompt
	assert "generic_concept_penalty" in prompt


def test_generic_concept_title_matcher_and_policy_penalties():
	card = StepScoreCard(
		edge_id="0",
		total_score=0.35,
		subscores={
			"target_overlap": 0.0,
			"anchor_overlap": 0.0,
		},
	)
	link = LinkContext(
		source="geneva",
		target="doctorate",
		anchor_text="Doctorate",
		sentence="Doctorate is a degree concept.",
		sent_idx=0,
	)

	assert is_generic_concept_candidate(
		target_title="Doctorate",
		anchor_text="Doctorate",
	)
	assert is_generic_concept_candidate(
		target_title="Master of Advanced Studies",
		anchor_text="Master of Advanced Studies",
	)
	assert not is_generic_concept_candidate(
		target_title="University of Geneva",
		anchor_text="University of Geneva",
	)
	assert not is_generic_concept_candidate(
		target_title="Wlad Godzich",
		anchor_text="Wlad Godzich",
	)
	assert (
		generic_concept_penalty(
			policy="prompt_only",
			query="What country is the University of Geneva in?",
			link=link,
			target_title="Doctorate",
			card=card,
			answer_bearing_bonus=0.0,
		)
		== 0.0
	)
	light_penalty = generic_concept_penalty(
		policy="light_generic_penalty",
		query="What country is the University of Geneva in?",
		link=link,
		target_title="Doctorate",
		card=card,
		answer_bearing_bonus=0.0,
	)
	strong_penalty = generic_concept_penalty(
		policy="strong_generic_penalty",
		query="What country is the University of Geneva in?",
		link=link,
		target_title="Doctorate",
		card=card,
		answer_bearing_bonus=0.0,
	)

	assert light_penalty > 0.0
	assert strong_penalty > light_penalty


def test_controller_schema_requires_single_stage_fields():
	schema = _controller_response_schema("two_hop")
	instructions = _controller_schema_instructions("two_hop")
	output_type = selector_llm_module._controller_output_model("two_hop")

	assert set(schema["properties"]) == {"decision", "runner_up", "state", "reason"}
	assert "runner_up" in instructions
	assert "state" in instructions
	assert "primary_node_role" not in instructions
	with pytest.raises(ValueError, match="runner_up must differ from decision"):
		output_type.model_validate(
			{
				"decision": "edge_0",
				"runner_up": "edge_0",
				"state": "need_bridge",
				"reason": "Take the support edge.",
			}
		)


def test_controller_trace_records_decision_summary(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeBackend(
		_controller_payload(
			"edge_1",
			runner_up="stop",
			state="need_answer_grounding",
			reason="University of Geneva is the explicit answer-bearing support node.",
		)
	)
	selector = build_selector(
		CONTROLLER_SINGLE_PATH,
		selector_provider="openai",
		selector_model="gpt-test-mini",
		selector_api_key_env="OPENAI_API_KEY",
		selector_backend_factory=lambda _config: backend,
	)
	case = EvaluationCase(
		case_id="q-self-label",
		query="In what country did Bain attend doctoral seminars of Wlad Godzich?",
	)
	budget = cast(RuntimeBudget, EvaluationBudget(token_budget_ratio=0.01))

	result = selector.select(_build_geneva_doctorate_graph(), case, budget)

	assert result.selector_logs[0].controller is not None
	assert result.selector_logs[0].controller.decision == "edge_1"
	assert result.selector_logs[0].controller.runner_up == "stop"
	assert result.selector_logs[0].controller.state == "need_answer_grounding"
	assert "explicit answer-bearing support node" in (
		result.selector_logs[0].controller.reason or ""
	)
	assert any(
		candidate.generic_concept_like is True
		for candidate in result.selector_logs[0].controller.candidates
	)


def _build_bridge_graph() -> LinkContextGraph:
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"root",
				"Launch Root",
				("Launch Root offers multiple navigation paths.",),
			),
			DocumentNode(
				"bait", "Bait Page", ("Bait page looks relevant but dead ends.",)
			),
			DocumentNode(
				"bridge", "Bridge Page", ("Bridge page leads onward to the answer.",)
			),
			DocumentNode(
				"answer",
				"Answer Page",
				("Answer page contains the true launch harbor location.",),
			),
		]
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="bait",
			anchor_text="harbor location",
			sentence="generic bait sentence",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="bridge",
			anchor_text="plain note",
			sentence="harbor",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="bridge",
			target="answer",
			anchor_text="harbor location",
			sentence="harbor location",
			sent_idx=0,
		)
	)
	return graph


def _build_all_dangling_graph() -> LinkContextGraph:
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"root",
				"Dangling Root",
				("Dangling Root points only to malformed or missing targets.",),
			),
		]
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="26833.Wlad Godzich",
			anchor_text="26833.Wlad Godzich",
			sentence="A malformed person target should never be exposed.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="books?as auth=Wlad+Godzich",
			anchor_text="books?as auth=Wlad+Godzich",
			sentence="A malformed query target should never be exposed.",
			sent_idx=0,
		)
	)
	return graph


def _build_geneva_doctorate_graph() -> LinkContextGraph:
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"root",
				"Thomas Bain (Orange)",
				(
					"Thomas Bain attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
				),
			),
			DocumentNode(
				"geneva",
				"University of Geneva",
				("The University of Geneva is in Switzerland.",),
			),
			DocumentNode(
				"doctorate",
				"Doctorate",
				("A doctorate is an academic degree.",),
			),
		]
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="doctorate",
			anchor_text="Doctorate",
			sentence="Doctorate is a generic degree concept.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="geneva",
			anchor_text="University of Geneva",
			sentence="Thomas Bain attended the doctoral seminars of Wlad Godzich in the University of Geneva.",
			sent_idx=0,
		)
	)
	return graph


def _build_backtrack_graph() -> LinkContextGraph:
	graph = LinkContextGraph(
		documents=[
			DocumentNode(
				"root",
				"Harbor Root",
				("Harbor Root is the place to start the evidence search.",),
			),
			DocumentNode(
				"bait",
				"Flashy Harbor",
				("Flashy Harbor sounds relevant but dead ends.",),
			),
			DocumentNode(
				"bridge",
				"Bridge Record",
				("Bridge Record leads to the real harbor answer.",),
			),
			DocumentNode(
				"answer",
				"Answer State",
				("Answer State contains the final harbor evidence.",),
			),
		]
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="bait",
			anchor_text="harbor shortcut",
			sentence="A flashy harbor shortcut looks tempting.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="root",
			target="bridge",
			anchor_text="bridge evidence",
			sentence="Bridge evidence points onward to the answer state.",
			sent_idx=0,
		)
	)
	graph.add_link(
		LinkContext(
			source="bridge",
			target="answer",
			anchor_text="answer state",
			sentence="Bridge evidence reaches the answer state.",
			sent_idx=0,
		)
	)
	return graph
