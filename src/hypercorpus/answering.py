from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from hypercorpus.subgraph import QuerySubgraph
from hypercorpus.text import (
    extract_capitalized_phrases,
    extract_years,
    normalize_answer,
    normalized_token_overlap,
)


@dataclass(slots=True)
class AnswerEvidence:
    source: str
    text: str
    score: float


@dataclass(slots=True)
class AnswerWithEvidence:
    query: str
    answer: str
    confidence: float
    evidence: list[AnswerEvidence]
    mode: str = "heuristic"
    model: str | None = None
    runtime_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class SupportsAnswer(Protocol):
    def answer(self, query: str, subgraph: QuerySubgraph) -> AnswerWithEvidence:
        ...


@dataclass(slots=True)
class LLMAnswererConfig:
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    cache_path: Path | None = None
    temperature: float = 0.0
    prompt_version: str = "v1"


class JsonlAnswerCache:
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


class Answerer:
    def __init__(self, *, max_evidence: int = 3):
        self.max_evidence = max_evidence

    def answer(self, query: str, subgraph: QuerySubgraph) -> AnswerWithEvidence:
        started_at = time.perf_counter()
        candidate_scores: dict[str, float] = defaultdict(float)
        candidate_evidence: dict[str, list[AnswerEvidence]] = defaultdict(list)
        question_kind = self._question_kind(query)
        query_norm = normalize_answer(query)

        for relation in subgraph.relations:
            candidate = relation.target_title or relation.anchor_text
            score = relation.score + self._candidate_bonus(question_kind, candidate)
            self._record_candidate(
                candidate,
                score,
                AnswerEvidence(relation.source, relation.sentence, relation.score),
                candidate_scores,
                candidate_evidence,
                query_norm,
            )

        for snippet in subgraph.snippets:
            for candidate in self._snippet_candidates(question_kind, snippet.text):
                score = snippet.score + self._candidate_bonus(question_kind, candidate)
                self._record_candidate(
                    candidate,
                    score,
                    AnswerEvidence(snippet.node_id, snippet.text, snippet.score),
                    candidate_scores,
                    candidate_evidence,
                    query_norm,
                )

        if not candidate_scores:
            fallback = subgraph.snippets[0].text if subgraph.snippets else ""
            return AnswerWithEvidence(
                query=query,
                answer=fallback,
                confidence=0.0,
                evidence=[AnswerEvidence("subgraph", fallback, 0.0)] if fallback else [],
                mode="heuristic",
                model=None,
                runtime_s=time.perf_counter() - started_at,
            )

        answer, score = max(candidate_scores.items(), key=lambda item: item[1])
        evidence = sorted(
            candidate_evidence[answer],
            key=lambda item: item.score,
            reverse=True,
        )[: self.max_evidence]
        return AnswerWithEvidence(
            query=query,
            answer=answer,
            confidence=min(score, 1.0),
            evidence=evidence,
            mode="heuristic",
            model=None,
            runtime_s=time.perf_counter() - started_at,
        )

    def _record_candidate(
        self,
        candidate: str,
        score: float,
        evidence: AnswerEvidence,
        candidate_scores: dict[str, float],
        candidate_evidence: dict[str, list[AnswerEvidence]],
        query_norm: str,
    ) -> None:
        candidate = candidate.strip()
        if not candidate:
            return
        if normalize_answer(candidate) == query_norm:
            return
        candidate_scores[candidate] = max(candidate_scores[candidate], score)
        candidate_evidence[candidate].append(evidence)

    def _question_kind(self, query: str) -> str:
        query_lower = query.lower()
        if query_lower.startswith(("who", "which person", "which director")):
            return "who"
        if query_lower.startswith(("where", "which city", "which country", "which place")):
            return "where"
        if query_lower.startswith(("when", "which year", "what year")):
            return "when"
        return "generic"

    def _candidate_bonus(self, question_kind: str, candidate: str) -> float:
        if question_kind == "when" and extract_years(candidate):
            return 0.3
        if question_kind in {"who", "where"} and " " in candidate:
            return 0.15
        return 0.05

    def _snippet_candidates(self, question_kind: str, text: str) -> list[str]:
        if question_kind == "when":
            return extract_years(text)
        if question_kind in {"who", "where"}:
            return extract_capitalized_phrases(text)
        return [
            phrase
            for phrase in extract_capitalized_phrases(text)
            if normalized_token_overlap(phrase, text) >= 0
        ]


class LLMAnswerer:
    def __init__(
        self,
        *,
        config: LLMAnswererConfig | None = None,
        max_evidence: int = 3,
        client_factory: Callable[..., Any] | None = None,
    ):
        self.config = config or LLMAnswererConfig()
        self.max_evidence = max_evidence
        self._client_factory = client_factory
        self._client: Any | None = None
        self._cache = JsonlAnswerCache(self.config.cache_path) if self.config.cache_path is not None else None

    def answer(self, query: str, subgraph: QuerySubgraph) -> AnswerWithEvidence:
        context = _render_subgraph(subgraph)
        cache_key = _answer_cache_key(
            model=self.config.model,
            base_url=self.config.base_url,
            query=query,
            context=context,
            prompt_version=self.config.prompt_version,
        )
        cached = self._cache.get(cache_key) if self._cache is not None else None
        if cached is not None:
            return _cached_answer(query=query, cached=cached)

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in environment variable {self.config.api_key_env}")

        client = self._get_client(api_key)
        started_at = time.perf_counter()
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied evidence context. "
                        'Return JSON with a single string field: {"answer": "..."}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{query}\n\nEvidence context:\n{context}",
                },
            ],
        )
        runtime_s = time.perf_counter() - started_at
        message = response.choices[0].message
        payload = json.loads(_message_content(message.content))
        answer = str(payload.get("answer", "")).strip()
        prompt_tokens, completion_tokens, total_tokens = _usage_triplet(getattr(response, "usage", None))
        result = AnswerWithEvidence(
            query=query,
            answer=answer,
            confidence=1.0 if answer else 0.0,
            evidence=_default_evidence(subgraph, max_evidence=self.max_evidence),
            mode="llm_fixed",
            model=self.config.model,
            runtime_s=runtime_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        if self._cache is not None:
            self._cache.put(
                cache_key,
                {
                    "mode": result.mode,
                    "model": result.model,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "runtime_s": result.runtime_s,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                },
            )
        return result

    def _get_client(self, api_key: str) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(api_key=api_key, base_url=self.config.base_url)
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("LLMAnswerer requires the openai package to be installed.") from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client


def _render_subgraph(subgraph: QuerySubgraph) -> str:
    sections: list[str] = []
    for snippet in subgraph.snippets:
        sections.append(
            f"[snippet] {snippet.node_id} | {snippet.title}\n"
            f"score={snippet.score:.3f}\n{snippet.text}"
        )
    for relation in subgraph.relations:
        sections.append(
            f"[relation] {relation.source} -> {relation.target}\n"
            f"target_title={relation.target_title}\n"
            f"anchor={relation.anchor_text}\n"
            f"sentence={relation.sentence}"
        )
    return "\n\n".join(sections)


def _default_evidence(subgraph: QuerySubgraph, *, max_evidence: int) -> list[AnswerEvidence]:
    snippets = sorted(subgraph.snippets, key=lambda item: item.score, reverse=True)[:max_evidence]
    return [
        AnswerEvidence(source=snippet.node_id, text=snippet.text, score=snippet.score)
        for snippet in snippets
    ]


def _answer_cache_key(
    *,
    model: str,
    base_url: str | None,
    query: str,
    context: str,
    prompt_version: str,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "base_url": base_url,
            "query": query,
            "context_digest": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "prompt_version": prompt_version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_answer(*, query: str, cached: dict[str, Any]) -> AnswerWithEvidence:
    return AnswerWithEvidence(
        query=query,
        answer=str(cached.get("answer", "")),
        confidence=float(cached.get("confidence", 0.0)),
        evidence=[],
        mode=str(cached.get("mode", "llm_fixed")),
        model=str(cached["model"]) if cached.get("model") is not None else None,
        runtime_s=float(cached.get("runtime_s", 0.0)),
        prompt_tokens=_maybe_int(cached.get("prompt_tokens")),
        completion_tokens=_maybe_int(cached.get("completion_tokens")),
        total_tokens=_maybe_int(cached.get("total_tokens")),
    )


def _usage_triplet(usage: Any) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None
    return (
        _maybe_int(getattr(usage, "prompt_tokens", None)),
        _maybe_int(getattr(usage, "completion_tokens", None)),
        _maybe_int(getattr(usage, "total_tokens", None)),
    )


def _message_content(content: Any) -> str:
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


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
