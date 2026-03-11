from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

from webwalker.graph import LinkContext, LinkContextGraph
from webwalker.walker import (
    LinkContextOverlapStepScorer,
    StepScorerMetadata,
    StepLinkScorer,
    StepScoreCard,
    _clamp_score,
)

SelectorProvider = Literal["openai", "anthropic", "gemini"]

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
    cache_path: Path | None = None
    temperature: float = 0.0
    prompt_version: str = "v1"
    candidate_prefilter_top_n: int = 8
    two_hop_prefilter_top_n: int = 4

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = DEFAULT_SELECTOR_MODELS[self.provider]
        if self.api_key_env is None:
            self.api_key_env = DEFAULT_SELECTOR_API_KEY_ENVS[self.provider]
        if self.candidate_prefilter_top_n <= 0:
            raise ValueError("candidate_prefilter_top_n must be positive.")
        if self.two_hop_prefilter_top_n <= 0:
            raise ValueError("two_hop_prefilter_top_n must be positive.")


@dataclass(slots=True)
class BackendCompletion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    raw_response: str | None


class BackendAdapter(Protocol):
    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> BackendCompletion:
        ...


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
        client_factory: Callable[..., Any] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._client_factory = client_factory
        self._client: Any | None = None

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> BackendCompletion:
        client = self._get_client()
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
        prompt_tokens, completion_tokens, total_tokens = _openai_usage_triplet(getattr(response, "usage", None))
        raw_response = _raw_response_payload(response)
        return BackendCompletion(
            text=_openai_message_content(message.content),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=raw_response,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(api_key=self.api_key, base_url=self.base_url)
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI selector scoring requires the openai package.") from exc
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
    ) -> BackendCompletion:
        client = self._get_client()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
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
        for item in getattr(response, "content", []):
            if getattr(item, "type", None) == "text" and getattr(item, "text", None):
                text_fragments.append(str(item.text))
        return BackendCompletion(
            text="".join(text_fragments),
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
            raise RuntimeError("Anthropic selector scoring requires the anthropic package.") from exc
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
    ) -> BackendCompletion:
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
            getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", None)
        )
        completion_tokens = _maybe_int(
            getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None)
        )
        total_tokens = _maybe_int(getattr(usage, "total_token_count", None))
        text = getattr(response, "text", None)
        if text is None:
            text = _gemini_text(response)
        return BackendCompletion(
            text=str(text or ""),
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
            raise RuntimeError("Gemini selector scoring requires the google-genai package.") from exc
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
        self._cache = JsonlSelectorCache(config.cache_path) if config.cache_path is not None else None
        self._backend: BackendAdapter | None = None
        self.metadata = StepScorerMetadata(
            scorer_kind=self.scorer_kind,
            backend=self.config.provider,
            provider=self.config.provider,
            model=self.config.model,
            prompt_version=self.config.prompt_version,
            candidate_prefilter_top_n=self.config.candidate_prefilter_top_n,
            two_hop_prefilter_top_n=self.config.two_hop_prefilter_top_n if self.mode == "two_hop" else None,
        )

    def validate_environment(self) -> None:
        api_key = os.environ.get(self.config.api_key_env or "")
        if not api_key:
            raise ValueError(f"Missing API key in environment variable {self.config.api_key_env}")

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
        )
        bundle = _candidate_bundle(
            graph=graph,
            candidate_links=candidate_links,
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
                return self._cards_from_payload(
                    payload=json.loads(str(cached["response_text"])),
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
                )

        started_at = time.perf_counter()
        try:
            self.validate_environment()
            response = self._get_backend().complete_json(
                model=self.config.model or "",
                system_prompt=_system_prompt(self.mode),
                user_prompt=_user_prompt(query=query, bundle=bundle),
                temperature=self.config.temperature,
            )
            latency_s = time.perf_counter() - started_at
            payload = json.loads(response.text or "{}")
            cards = self._cards_from_payload(
                payload=payload,
                raw_response=response.raw_response,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_s=latency_s,
                cache_hit=False,
                candidate_links=candidate_links,
                prefiltered_indices=prefiltered_indices,
            )
            if self._cache is not None:
                self._cache.put(
                    cache_key,
                    {
                        "provider": self.config.provider,
                        "model": self.config.model,
                        "response_text": json.dumps(payload, ensure_ascii=False),
                        "raw_response": response.raw_response,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                    },
                )
            return cards
        except Exception as exc:
            latency_s = time.perf_counter() - started_at
            return _cards_with_fallback(
                fallback_cards,
                provider=self.config.provider,
                model=self.config.model,
                fallback_reason=f"llm_error:{exc}",
                latency_s=latency_s,
            )

    def _get_backend(self) -> BackendAdapter:
        if self._backend is not None:
            return self._backend
        self._backend = self.backend_factory(self.config)
        return self._backend

    def _cards_from_payload(
        self,
        *,
        payload: dict[str, Any],
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
            raise ValueError("Selector LLM response must contain a scores list.")
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
                total_score = _clamp_score(0.45 * direct_support + 0.40 * bridge_potential + 0.15 * novelty)
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
                best_next_edge_id = str(record.get("best_next_edge_id")) if record.get("best_next_edge_id") else None

            cards.append(
                StepScoreCard(
                    edge_id=edge_id,
                    total_score=total_score,
                    subscores=subscores,
                    rationale=_maybe_text(record.get("rationale")),
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


def _default_backend_factory(config: SelectorLLMConfig) -> BackendAdapter:
    api_key = os.environ.get(config.api_key_env or "")
    if not api_key:
        raise ValueError(f"Missing API key in environment variable {config.api_key_env}")
    if config.provider == "openai":
        return OpenAIBackendAdapter(api_key=api_key, base_url=config.base_url)
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
            "path_digest": hashlib.sha256(json.dumps(list(path_node_ids)).encode("utf-8")).hexdigest(),
            "candidate_bundle_digest": hashlib.sha256(
                json.dumps(bundle, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prefilter_indices(cards: Sequence[StepScoreCard], *, top_n: int) -> list[int]:
    ranked = sorted(
        enumerate(cards),
        key=lambda item: (item[1].total_score, -item[0]),
        reverse=True,
    )
    return [index for index, _card in ranked[:top_n]]


def _candidate_bundle(
    *,
    graph: LinkContextGraph,
    candidate_links: Sequence[LinkContext],
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
        entry = {
            "edge_id": str(index),
            "source_title": _node_title(graph, link.source),
            "target_title": _node_title(graph, link.target),
            "anchor_text": link.anchor_text,
            "sentence": link.sentence,
        }
        if mode == "two_hop":
            next_links = [
                next_link
                for next_link in graph.links_from(link.target)
                if next_link.target not in visited_nodes and next_link.target != link.source
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
            future_indices = _prefilter_indices(future_cards, top_n=min(future_top_n, len(future_cards)))
            entry["future_candidates"] = [
                {
                    "edge_id": f"{index}-{future_index}",
                    "target_title": _node_title(graph, next_links[future_index].target),
                    "anchor_text": next_links[future_index].anchor_text,
                    "sentence": next_links[future_index].sentence,
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


def _system_prompt(mode: str) -> str:
    if mode == "single_hop":
        return (
            "You are scoring hyperlink choices for a single-path multi-hop retriever. "
            "Return JSON only. Each candidate needs direct_support, bridge_potential, novelty, and rationale. "
            "All subscores must be floats in [0,1]."
        )
    return (
        "You are scoring hyperlink choices for a two-hop-aware single-path multi-hop retriever. "
        "Return JSON only. Each candidate needs direct_support, bridge_potential, future_potential, novelty, rationale, "
        "and optional best_next_edge_id. All subscores must be floats in [0,1]."
    )


def _user_prompt(*, query: str, bundle: dict[str, Any]) -> str:
    return (
        f"Question:\n{query}\n\n"
        "Retriever context:\n"
        f"{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON as {\"scores\": [{\"edge_id\": \"...\", ...}]}. "
        "Do not add markdown or extra prose."
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
) -> list[StepScoreCard]:
    return [
        StepScoreCard(
            edge_id=card.edge_id,
            total_score=card.total_score,
            subscores=dict(card.subscores),
            rationale=card.rationale,
            backend="overlap",
            provider=provider,
            model=model,
            latency_s=latency_s,
            prompt_tokens=card.prompt_tokens,
            completion_tokens=card.completion_tokens,
            total_tokens=card.total_tokens,
            cache_hit=card.cache_hit,
            fallback_reason=fallback_reason,
            best_next_edge_id=card.best_next_edge_id,
        )
        for card in cards
    ]


def _openai_usage_triplet(usage: Any) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None
    return (
        _maybe_int(getattr(usage, "prompt_tokens", None)),
        _maybe_int(getattr(usage, "completion_tokens", None)),
        _maybe_int(getattr(usage, "total_tokens", None)),
    )


def _openai_message_content(content: Any) -> str:
    if content is None:
        return "{}"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                fragments.append(str(text))
        return "".join(fragments) or "{}"
    return str(content)


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
