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
from webwalker.selector_llm import BackendCompletion, SelectorLLMConfig


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

    def complete_json(self, *, model: str, system_prompt: str, user_prompt: str, temperature: float) -> BackendCompletion:
        del model, system_prompt, user_prompt, temperature
        self.call_count += 1
        return BackendCompletion(
            text=json.dumps(self.payload, ensure_ascii=False),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            raw_response=json.dumps({"payload": self.payload}, ensure_ascii=False),
        )


def test_select_selectors_requires_selector_llm_key(monkeypatch):
    monkeypatch.delenv("MISSING_SELECTOR_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_SELECTOR_KEY"):
        select_selectors(
            ["seed__link_context_llm__single_path_walk"],
            selector_provider="anthropic",
            selector_model="claude-test",
            selector_api_key_env="MISSING_SELECTOR_KEY",
        )


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
