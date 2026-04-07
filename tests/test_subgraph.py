from typing import cast

from hypercorpus.copilot import CopilotSdkCompletion, DEFAULT_COPILOT_MODEL
from hypercorpus.answering import Answerer, LLMAnswerer, LLMAnswererConfig
from hypercorpus.subgraph import SubgraphExtractor


def test_subgraph_extractor_keeps_query_relevant_context(sample_graph):
	subgraph = SubgraphExtractor().extract(
		"Which city hosts the launch site?",
		sample_graph,
		["mission", "cape"],
	)
	assert set(subgraph.node_ids) == {"mission", "cape"}
	assert all(snippet.node_id in {"mission", "cape"} for snippet in subgraph.snippets)
	assert [relation.target for relation in subgraph.relations] == ["cape"]


def test_answerer_prefers_relation_target_for_where_question(sample_graph):
	subgraph = SubgraphExtractor().extract(
		"Which city hosts the launch site?",
		sample_graph,
		["mission", "cape"],
	)
	answer = Answerer().answer("Which city hosts the launch site?", subgraph)
	assert answer.answer == "Cape Canaveral"
	assert answer.evidence


def test_llm_answerer_parses_json_and_uses_cache(sample_graph, tmp_path, monkeypatch):
	subgraph = SubgraphExtractor().extract(
		"Which city hosts the launch site?",
		sample_graph,
		["mission", "cape"],
	)

	class FakeUsage:
		prompt_tokens = 11
		completion_tokens = 7
		total_tokens = 18

	class FakeMessage:
		content = '{"answer": "Cape Canaveral"}'

	class FakeChoice:
		message = FakeMessage()

	class FakeResponse:
		choices = [FakeChoice()]
		usage = FakeUsage()

	class FakeCompletions:
		def __init__(self):
			self.calls = 0

		def create(self, **_kwargs):
			self.calls += 1
			return FakeResponse()

	class FakeChat:
		def __init__(self):
			self.completions = FakeCompletions()

	class FakeClient:
		def __init__(self):
			self.chat = FakeChat()

	fake_client = FakeClient()
	monkeypatch.setenv("OPENAI_API_KEY", "test-key")
	answerer = LLMAnswerer(
		config=LLMAnswererConfig(
			provider="openai",
			model="gpt-4.1-mini",
			api_key_env="OPENAI_API_KEY",
			cache_path=tmp_path / "answer-cache.jsonl",
		),
		client_factory=lambda **_kwargs: fake_client,
	)

	first = answerer.answer("Which city hosts the launch site?", subgraph)
	second = answerer.answer("Which city hosts the launch site?", subgraph)

	assert first.answer == "Cape Canaveral"
	assert first.mode == "llm_fixed"
	assert first.model == "gpt-4.1-mini"
	assert first.prompt_tokens == 11
	assert first.completion_tokens == 7
	assert first.total_tokens == 18
	assert second.answer == "Cape Canaveral"
	assert fake_client.chat.completions.calls == 1


def test_llm_answerer_defaults_to_copilot_sdk(sample_graph):
	subgraph = SubgraphExtractor().extract(
		"Which city hosts the launch site?",
		sample_graph,
		["mission", "cape"],
	)
	captured: dict[str, object] = {}

	class FakeRunner:
		def complete(
			self,
			*,
			model: str,
			system_prompt: str,
			user_prompt: str,
			temperature: float,
			timeout_s: float = 120.0,
		) -> CopilotSdkCompletion:
			captured["request"] = {
				"model": model,
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
				"temperature": temperature,
				"timeout_s": timeout_s,
			}
			return CopilotSdkCompletion(
				text='{"answer": "Cape Canaveral"}',
				model=model,
				prompt_tokens=None,
				completion_tokens=None,
				total_tokens=None,
				raw_response='{"answer": "Cape Canaveral"}',
			)

	answerer = LLMAnswerer(
		config=LLMAnswererConfig(provider="copilot"),
		copilot_runner=FakeRunner(),
	)

	result = answerer.answer("Which city hosts the launch site?", subgraph)
	request = cast(dict[str, object], captured["request"])

	assert result.answer == "Cape Canaveral"
	assert result.model == DEFAULT_COPILOT_MODEL
	assert request["model"] == DEFAULT_COPILOT_MODEL
	assert request["system_prompt"] == (
		"Answer only from the supplied evidence context. "
		'Return JSON with a single string field: {"answer": "..."}'
	)
	assert request["temperature"] == 0.0
	assert request["timeout_s"] == 120.0
	user_prompt = cast(str, request["user_prompt"])
	assert "Question:\nWhich city hosts the launch site?" in user_prompt
	assert "Evidence context:\n" in user_prompt
