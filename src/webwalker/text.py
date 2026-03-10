from __future__ import annotations

import re
import string
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

STOPWORDS = {
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"how",
	"in",
	"is",
	"it",
	"of",
	"on",
	"or",
	"that",
	"the",
	"their",
	"this",
	"to",
	"was",
	"what",
	"when",
	"where",
	"which",
	"who",
	"why",
	"with",
}


def tokenize(text: str) -> list[str]:
	return [token.lower() for token in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
	return [token for token in tokenize(text) if token not in STOPWORDS]


def normalized_token_overlap(query: str, text: str) -> float:
	query_tokens = content_tokens(query)
	text_tokens = content_tokens(text)
	if not query_tokens or not text_tokens:
		return 0.0
	query_counts = Counter(query_tokens)
	text_counts = Counter(text_tokens)
	shared = sum(min(query_counts[token], text_counts[token]) for token in query_counts)
	return shared / max(len(query_tokens), 1)


def approx_token_count(text: str) -> int:
	return len(tokenize(text))


def extract_years(text: str) -> list[str]:
	return YEAR_RE.findall(text)


def extract_capitalized_phrases(text: str) -> list[str]:
	phrases = []
	for phrase in CAPITALIZED_PHRASE_RE.findall(text):
		if len(phrase) <= 1:
			continue
		phrases.append(phrase)
	return phrases


def normalize_answer(text: str) -> str:
	text = text.lower().strip()
	text = text.translate(str.maketrans("", "", string.punctuation))
	return " ".join(text.split())
