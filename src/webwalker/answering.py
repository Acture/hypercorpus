from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from webwalker.subgraph import QuerySubgraph
from webwalker.text import (
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


class Answerer:
	def __init__(self, *, max_evidence: int = 3):
		self.max_evidence = max_evidence

	def answer(self, query: str, subgraph: QuerySubgraph) -> AnswerWithEvidence:
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
