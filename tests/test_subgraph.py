from typing import cast

from hypercorpus.copilot import CopilotSdkCompletion, DEFAULT_COPILOT_MODEL
from hypercorpus.answering import Answerer, LLMAnswerer, LLMAnswererConfig
from hypercorpus.subgraph import FullDocumentExtractor, SubgraphExtractor


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
			timeout_s: float = 120.0,
		) -> CopilotSdkCompletion:
			captured["request"] = {
				"model": model,
				"system_prompt": system_prompt,
				"user_prompt": user_prompt,
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
		"Reply with the shortest exact answer span — a name, number, "
		"date, or short phrase. No full sentences, no leading articles "
		"(a/an/the), no explanations, no units unless asked. "
		'Return JSON: {"answer": "..."}'
	)
	assert request["timeout_s"] == 120.0
	user_prompt = cast(str, request["user_prompt"])
	assert "Question:\nWhich city hosts the launch site?" in user_prompt
	assert "Evidence context:\n" in user_prompt




def test_full_document_extractor_emits_every_sentence_per_node(sample_graph):
	subgraph = FullDocumentExtractor().extract(
		"Which city hosts the launch site?",
		sample_graph,
		["mission", "cape"],
	)
	# One snippet per node, with full document text joined by spaces.
	mission_snippets = [s for s in subgraph.snippets if s.node_id == "mission"]
	cape_snippets = [s for s in subgraph.snippets if s.node_id == "cape"]
	assert len(mission_snippets) == 1
	assert len(cape_snippets) == 1
	assert (
		mission_snippets[0].text
		== "Moon Launch Program uses Cape Canaveral as its launch site. "
		"The program was directed by Alice Johnson."
	)
	assert cape_snippets[0].text == "Cape Canaveral is a city in Florida."
	assert subgraph.relations == []
	assert subgraph.token_cost_estimate > 0


def test_full_document_extractor_includes_non_overlapping_sentences(sample_graph):
	query = "What year did the program get directed?"
	full = FullDocumentExtractor().extract(query, sample_graph, ["director"])
	legacy = SubgraphExtractor(max_snippets_per_node=2).extract(
		query, sample_graph, ["director"]
	)
	# Full mode emits the whole document; the relevant sentence must be inside
	# the (single) full-doc snippet for "director".
	full_text = " ".join(s.text for s in full.snippets)
	assert "Alice Johnson directed the Moon Launch Program in 1969." in full_text
	for legacy_snip in legacy.snippets:
		assert legacy_snip.text in full_text


def test_full_document_extractor_respects_token_cap(sample_graph):
	# Cap at 50 real GPT-4 tokens; should drop trailing nodes until prompt fits.
	subgraph = FullDocumentExtractor(max_input_tokens=50).extract(
		"any query",
		sample_graph,
		["mission", "cape", "director", "florida"],
	)
	# At most one or two nodes survive at this tight cap.
	assert len(subgraph.snippets) < 4
	assert all(s.text for s in subgraph.snippets)

