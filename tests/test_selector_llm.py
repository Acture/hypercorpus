import json

import pytest

from webwalker.eval import (
    EvaluationCase,
    SeedLinkContextLLMSinglePathWalkSelector,
    SeedLinkContextLLMTwoHopSinglePathWalkSelector,
    SelectionBudget,
    select_selectors,
)
from webwalker.graph import DocumentNode, LinkContext, LinkContextGraph
from webwalker.selector_llm import AnthropicBackendAdapter, BackendCompletion, SelectorLLMConfig


class StaticStartPolicy:
    def __init__(self, node_ids: list[str]):
        self.node_ids = node_ids

    def select_start(self, _graph, _query) -> list[str]:
        return list(self.node_ids)


class FakeBackend:
    def __init__(self, payload: dict, *, prompt_tokens: int = 17, completion_tokens: int = 9):
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
    def __init__(self, text: str, *, prompt_tokens: int = 17, completion_tokens: int = 9, raw_response: str | None = None):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.raw_response = raw_response or json.dumps({"text": text}, ensure_ascii=False)
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


class FakeToolUseContent:
    def __init__(self, *, type: str, text: str | None = None, name: str | None = None, input: dict | None = None):
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


def test_select_selectors_requires_selector_llm_key(monkeypatch):
    monkeypatch.delenv("MISSING_SELECTOR_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_SELECTOR_KEY"):
        select_selectors(
            ["seed__link_context_llm__single_path_walk"],
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
    adapter = AnthropicBackendAdapter(api_key="test-key", client_factory=lambda api_key: client)

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
    assert completion.prompt_tokens == 21
    assert completion.completion_tokens == 13
    assert completion.total_tokens == 34
    assert client.messages.last_kwargs is not None
    assert client.messages.last_kwargs["tool_choice"] == {"type": "tool", "name": "score_candidates"}
    assert client.messages.last_kwargs["tools"][0]["input_schema"]["required"] == ["scores"]


def test_seed_link_context_llm_single_path_walk_records_usage_and_logs(sample_graph, monkeypatch, tmp_path):
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
    selector = SeedLinkContextLLMSinglePathWalkSelector(
        llm_config=SelectorLLMConfig(
            provider="openai",
            model="gpt-test-mini",
            api_key_env="OPENAI_API_KEY",
            cache_path=tmp_path / "selector-cache.jsonl",
        ),
        start_policy_factory=lambda _top_k: StaticStartPolicy(["mission"]),
        backend_factory=lambda _config: backend,
    )
    case = EvaluationCase(case_id="q1", query="Which city hosts the launch site?")
    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_tokens=128)

    first = selector.select(sample_graph, case, budget)
    second = selector.select(sample_graph, case, budget)

    assert first.corpus.node_ids == ["mission", "cape"]
    assert first.selector_metadata is not None
    assert first.selector_metadata.provider == "openai"
    assert first.selector_metadata.model == "gpt-test-mini"
    assert first.selector_usage is not None
    assert first.selector_usage.llm_calls == 1
    assert first.selector_usage.total_tokens == 26
    assert len(first.selector_logs) == 1
    assert first.selector_logs[0].candidates[0].rationale == "Cape directly supports the answer."
    assert first.selector_logs[0].raw_response is not None
    assert backend.call_count == 1

    assert second.corpus.node_ids == ["mission", "cape"]
    assert second.selector_logs[0].cache_hit is True
    assert backend.call_count == 1


def test_seed_link_context_llm_two_hop_single_path_walk_records_best_next_edge_id(monkeypatch):
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
    selector = SeedLinkContextLLMTwoHopSinglePathWalkSelector(
        llm_config=SelectorLLMConfig(
            provider="gemini",
            model="gemini-test",
            api_key_env="GEMINI_API_KEY",
        ),
        start_policy_factory=lambda _top_k: StaticStartPolicy(["root"]),
        backend_factory=lambda _config: backend,
    )
    case = EvaluationCase(case_id="bridge", query="harbor location")
    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_tokens=128)

    result = selector.select(_build_bridge_graph(), case, budget)

    assert result.corpus.node_ids == ["root", "bridge"]
    assert result.selector_logs[0].candidates[1].best_next_edge_id == "1-0"


@pytest.mark.parametrize(
    ("response_text"),
    [
        (
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
            )
        ),
        (
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
            + "\n```"
        ),
        (
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
            )
        ),
    ],
)
def test_seed_link_context_llm_single_path_walk_parses_anthropic_json_variants(
    sample_graph, monkeypatch, response_text
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = FakeTextBackend(response_text, raw_response='{"id":"mock-anthropic"}')
    selector = SeedLinkContextLLMSinglePathWalkSelector(
        llm_config=SelectorLLMConfig(
            provider="anthropic",
            model="claude-haiku-test",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        start_policy_factory=lambda _top_k: StaticStartPolicy(["mission"]),
        backend_factory=lambda _config: backend,
    )
    case = EvaluationCase(case_id="q-json", query="Which city hosts the launch site?")
    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_tokens=128)

    result = selector.select(sample_graph, case, budget)

    assert result.corpus.node_ids == ["mission", "cape"]
    assert result.selector_usage is not None
    assert result.selector_usage.llm_calls == 1
    assert result.selector_usage.total_tokens == 26
    assert result.selector_usage.fallback_steps == 0
    assert result.selector_logs[0].backend == "anthropic"
    assert result.selector_logs[0].fallback_reason is None
    assert result.selector_logs[0].text == response_text
    assert result.selector_logs[0].raw_response == '{"id":"mock-anthropic"}'


@pytest.mark.parametrize(
    ("response_text", "payload_text", "expected_reason"),
    [
        ("not json at all", "not json at all", "json_parse_error:"),
        ("   ", "   ", "empty_response"),
        (json.dumps({"wrong": []}, ensure_ascii=False), json.dumps({"wrong": []}, ensure_ascii=False), "schema_error:"),
    ],
)
def test_seed_link_context_llm_single_path_walk_preserves_usage_on_response_failures(
    sample_graph, monkeypatch, response_text, payload_text, expected_reason
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = FakeTextBackend(response_text, raw_response='{"id":"mock-failure"}')
    selector = SeedLinkContextLLMSinglePathWalkSelector(
        llm_config=SelectorLLMConfig(
            provider="anthropic",
            model="claude-haiku-test",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        start_policy_factory=lambda _top_k: StaticStartPolicy(["mission"]),
        backend_factory=lambda _config: backend,
    )
    case = EvaluationCase(case_id="q-failure", query="Which city hosts the launch site?")
    budget = SelectionBudget(max_steps=2, top_k=1, token_budget_tokens=128)

    result = selector.select(sample_graph, case, budget)

    assert result.corpus.node_ids == ["mission", "cape"]
    assert result.selector_usage is not None
    assert result.selector_usage.llm_calls == 1
    assert result.selector_usage.total_tokens == 26
    assert result.selector_usage.fallback_steps == 1
    assert result.selector_usage.parse_failure_steps == (0 if expected_reason.startswith("provider_error:") else 1)
    assert result.selector_usage.step_count == 1
    assert result.selector_logs[0].backend == "overlap"
    assert result.selector_logs[0].fallback_reason is not None
    assert result.selector_logs[0].fallback_reason.startswith(expected_reason)
    assert result.selector_logs[0].text == payload_text
    assert result.selector_logs[0].raw_response == '{"id":"mock-failure"}'


def _build_bridge_graph() -> LinkContextGraph:
    graph = LinkContextGraph(
        documents=[
            DocumentNode("root", "Launch Root", ("Launch Root offers multiple navigation paths.",)),
            DocumentNode("bait", "Bait Page", ("Bait page looks relevant but dead ends.",)),
            DocumentNode("bridge", "Bridge Page", ("Bridge page leads onward to the answer.",)),
            DocumentNode("answer", "Answer Page", ("Answer page contains the true launch harbor location.",)),
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
