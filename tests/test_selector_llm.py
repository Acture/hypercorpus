import json
from types import SimpleNamespace
from typing import cast

import pytest

from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.selector import RuntimeBudget, build_selector, select_selectors
from hypercorpus.graph import DocumentNode, LinkContext, LinkContextGraph
from hypercorpus.selector_llm import (
	AnthropicBackendAdapter,
	BackendCompletion,
	LLMController,
	OpenAIBackendAdapter,
	SelectorLLMConfig,
	_controller_user_prompt,
	_prefilter_indices,
	_support_coverage_hint,
)
from hypercorpus.walker import StepScoreCard

SINGLE_HOP = "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_llm__lookahead_1"
TWO_HOP = "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_llm__lookahead_2"
CONTROLLER_SINGLE_PATH = "top_1_seed__lexical_overlap__hop_2__single_path_walk__link_context_llm_controller__lookahead_2"
CONTROLLER_MULTIPATH = "top_1_seed__lexical_overlap__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2"


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


def test_openai_backend_adapter_responses_mode_parses_output_text():
	response = SimpleNamespace(
		output_text='{"scores": [{"edge_id": "0"}]}',
		usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
		output=[],
	)
	client = FakeOpenAIResponsesClient(response)
	adapter = OpenAIBackendAdapter(
		api_key="test-key",
		api_mode="responses",
		client_factory=lambda **_kwargs: client,
	)

	completion = adapter.complete_json(
		model="gpt-5.3-codex",
		system_prompt="score candidates",
		user_prompt="bundle",
		temperature=0.0,
	)

	assert completion.text == '{"scores": [{"edge_id": "0"}]}'
	assert completion.prompt_tokens == 11
	assert completion.completion_tokens == 7
	assert completion.total_tokens == 18
	assert client.responses.last_kwargs is not None
	assert client.responses.last_kwargs["model"] == "gpt-5.3-codex"
	assert client.responses.last_kwargs["text"] == {"format": {"type": "json_object"}}


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
	assert captured["url"] == "https://example.cognitiveservices.azure.com/models/chat/completions?api-version=2024-05-01-preview"
	assert captured["api_key"] == "azure-test-key"
	payload = cast(dict[str, object], captured["payload"])
	assert payload["model"] == "DeepSeek-V3.2-Speciale"
	assert payload["response_format"] == {"type": "json_object"}
	assert payload["messages"] == [
		{"role": "system", "content": "score candidates"},
		{"role": "user", "content": "bundle"},
	]
	assert captured["headers"] is None


def test_openai_backend_adapter_github_models_chat_mode_posts_json():
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
				"prompt_tokens": 19,
				"completion_tokens": 6,
				"total_tokens": 25,
			},
		}

	adapter = OpenAIBackendAdapter(
		api_key="gh-models-key",
		base_url="https://models.github.ai/inference/chat/completions",
		api_mode="github_models_chat_completions",
		http_post=_fake_post,
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
	assert captured["url"] == "https://models.github.ai/inference/chat/completions"
	assert captured["api_key"] == "gh-models-key"
	assert captured["headers"] == {
		"Accept": "application/vnd.github+json",
		"X-GitHub-Api-Version": "2026-03-10",
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


def test_selector_llm_config_accepts_github_models_chat_endpoint():
	config = SelectorLLMConfig(
		provider="openai",
		base_url="https://models.github.ai/inference/chat/completions",
		openai_api_mode="github_models_chat_completions",
	)

	assert config.base_url == "https://models.github.ai/inference/chat/completions"


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

	result = selector.select(sample_graph, case, budget)

	assert "cape" in result.selected_node_ids
	assert result.selector_usage is not None
	assert result.selector_usage.llm_calls == 2
	assert result.selector_usage.total_tokens == 52
	assert result.selector_usage.fallback_steps == 2
	assert result.selector_logs[0].fallback_reason is not None
	assert result.selector_logs[0].fallback_reason.startswith(expected_reason)
	assert result.selector_logs[0].raw_response == '{"id":"mock-failure"}'


def test_controller_single_path_records_decision_and_backtrack(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeSequenceBackend(
		[
			{
				"action": "choose_one",
				"primary_edge_id": "0",
				"secondary_edge_id": "1",
				"backup_edge_id": "1",
				"stop_score": 0.20,
				"evidence_cluster_confidence": 0.42,
				"candidates": [
					{
						"edge_id": "0",
						"utility": 0.91,
						"direct_support": 0.80,
						"bridge_potential": 0.15,
						"future_potential": 0.10,
						"redundancy_risk": 0.10,
						"rationale": "Try the flashy edge first.",
					},
					{
						"edge_id": "1",
						"utility": 0.83,
						"direct_support": 0.65,
						"bridge_potential": 0.95,
						"future_potential": 1.00,
						"redundancy_risk": 0.15,
						"rationale": "Keep the bridge as backup.",
					},
				],
			},
			{
				"action": "choose_one",
				"primary_edge_id": "0",
				"secondary_edge_id": "",
				"backup_edge_id": "",
				"stop_score": 0.10,
				"evidence_cluster_confidence": 0.86,
				"candidates": [
					{
						"edge_id": "0",
						"utility": 0.86,
						"direct_support": 0.72,
						"bridge_potential": 0.98,
						"future_potential": 0.94,
						"redundancy_risk": 0.08,
						"rationale": "This finishes the bridge path.",
					}
				],
			},
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
	assert any(log.decision_action == "backtrack" for log in result.selector_logs)
	assert result.selector_logs[0].backup_edge_id == "1"


def test_controller_constrained_multipath_forks_once_and_keeps_bridge(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeSequenceBackend(
		[
			{
				"action": "choose_two",
				"primary_edge_id": "1",
				"secondary_edge_id": "0",
				"backup_edge_id": "0",
				"stop_score": 0.18,
				"evidence_cluster_confidence": 0.78,
				"candidates": [
					{
						"edge_id": "0",
						"utility": 0.79,
						"direct_support": 0.35,
						"bridge_potential": 0.40,
						"future_potential": 0.20,
						"redundancy_risk": 0.85,
						"rationale": "This is the weaker flashy branch.",
					},
					{
						"edge_id": "1",
						"utility": 0.84,
						"direct_support": 0.60,
						"bridge_potential": 0.96,
						"future_potential": 1.00,
						"redundancy_risk": 0.12,
						"rationale": "This is the bridge branch to keep.",
					},
				],
			},
			{
				"action": "choose_one",
				"primary_edge_id": "0",
				"secondary_edge_id": "",
				"backup_edge_id": "",
				"stop_score": 0.12,
				"evidence_cluster_confidence": 0.90,
				"candidates": [
					{
						"edge_id": "0",
						"utility": 0.92,
						"direct_support": 0.78,
						"bridge_potential": 0.98,
						"future_potential": 0.96,
						"redundancy_risk": 0.05,
						"rationale": "Take the answer edge.",
					}
				],
			},
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
	assert any(log.decision_action == "choose_two" for log in result.selector_logs)


def test_controller_retries_schema_failure_then_succeeds(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	backend = FakeTextSequenceBackend(
		[
			'{"action":": "}',
			json.dumps(
				{
					"action": "choose_one",
					"primary_edge_id": "0",
					"secondary_edge_id": "",
					"backup_edge_id": "",
					"stop_score": 0.10,
					"evidence_cluster_confidence": 0.86,
					"candidates": [
						{
							"edge_id": "0",
							"utility": 0.86,
							"direct_support": 0.72,
							"bridge_potential": 0.98,
							"future_potential": 0.94,
							"redundancy_risk": 0.08,
							"rationale": "This finishes the bridge path.",
						}
					],
				},
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
	decision = controller.decide(
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=graph.links_from("root"),
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
		current_depth=1,
	)

	assert backend.call_count == 2
	assert len(backend.user_prompts) == 2
	assert "Required JSON schema:" in backend.user_prompts[0]
	assert "Your previous response failed validation." in backend.user_prompts[1]
	assert "schema_error:missing_candidates_list" in backend.user_prompts[1]
	assert '{"action":": "}' in backend.user_prompts[1]
	assert decision.fallback_reason is None
	assert decision.llm_attempts == 2
	assert decision.total_tokens == 52
	assert decision.prompt_tokens == 34
	assert decision.completion_tokens == 18
	assert decision.primary_edge_id == "0"
	assert len(decision.candidates) == 1


def test_controller_retries_three_times_then_falls_back(monkeypatch):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	graph = _build_bridge_graph()
	backend = FakeTextSequenceBackend(
		[
			'{"action":": "}',
			'{"action":"choose_one"}',
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
	decision = controller.decide(
		query="launch navigation root",
		graph=graph,
		current_node_id="root",
		candidate_links=graph.links_from("root"),
		visited_nodes={"root"},
		path_node_ids=["root"],
		remaining_steps=2,
		current_depth=1,
	)

	assert backend.call_count == 3
	assert decision.llm_attempts == 3
	assert decision.total_tokens == 78
	assert decision.prompt_tokens == 51
	assert decision.completion_tokens == 27
	assert decision.fallback_reason is not None
	assert decision.fallback_reason.startswith("schema_error:")


def test_controller_preserves_raw_stop_action_while_downgrading_effective_action(
	monkeypatch,
):
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	backend = FakeSequenceBackend(
		[
			{
				"action": "stop",
				"primary_edge_id": "0",
				"secondary_edge_id": "",
				"backup_edge_id": "1",
				"stop_score": 0.95,
				"evidence_cluster_confidence": 0.95,
				"candidates": [
					{
						"edge_id": "0",
						"utility": 0.84,
						"direct_support": 0.60,
						"bridge_potential": 0.96,
						"future_potential": 1.00,
						"redundancy_risk": 0.10,
						"rationale": "Continue through the bridge.",
					},
					{
						"edge_id": "1",
						"utility": 0.40,
						"direct_support": 0.20,
						"bridge_potential": 0.20,
						"future_potential": 0.10,
						"redundancy_risk": 0.90,
						"rationale": "Weak alternative.",
					},
				],
			},
			{
				"action": "choose_one",
				"primary_edge_id": "0",
				"secondary_edge_id": "",
				"backup_edge_id": "",
				"stop_score": 0.10,
				"evidence_cluster_confidence": 0.90,
				"candidates": [
					{
						"edge_id": "0",
						"utility": 0.92,
						"direct_support": 0.80,
						"bridge_potential": 0.98,
						"future_potential": 0.95,
						"redundancy_risk": 0.05,
						"rationale": "Take the answer edge.",
					}
				],
			},
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
	assert result.selector_logs[0].raw_decision_action == "stop"
	assert result.selector_logs[0].decision_action == "stop"
	assert result.selector_logs[0].stop_reason == "controller_stop"


def test_controller_prefilter_keeps_explicit_support_edge():
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
			DocumentNode("wlad", "Wlad Godzich", ("Wlad Godzich is a literary theorist.",)),
			DocumentNode("geneva", "University of Geneva", ("The University of Geneva is in Switzerland.",)),
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

	assert _support_coverage_hint(
		query=query,
		graph=graph,
		link=candidate_links[1],
		card=cards[1],
	) >= 0.55
	assert _prefilter_indices(
		cards,
		top_n=1,
		query=query,
		graph=graph,
		candidate_links=candidate_links,
	) == [0, 1]


def test_controller_prompt_mentions_explicit_support_coverage():
	prompt = _controller_user_prompt(
		query="In what country did Bain attend doctoral seminars of Wlad Godzich?",
		bundle={
			"query": "In what country did Bain attend doctoral seminars of Wlad Godzich?",
			"current_node_id": "root",
			"path_titles": ["Thomas Bain (Orange)"],
			"candidates": [
				{
					"edge_id": "1",
					"target_title": "University of Geneva",
					"support_coverage_hint": 0.82,
				}
			],
		},
		mode="two_hop",
	)

	assert "support_coverage" in prompt
	assert "prefer that candidate over stop" in prompt
	assert "explicit support-bearing node" in prompt


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
