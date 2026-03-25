from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable, Sequence

from hypercorpus.type import AnyHashable as T


@runtime_checkable
class GraphLike(Protocol[T]):
	adj: Dict[T, list[T]]
	nodes: list[T]
	node_attr: Dict[T, Dict[str, Any]]

	def neighbors(self, u: T) -> list[T]: ...


@runtime_checkable
class TopKSimilarity(Protocol[T]):
	"""
	Given a query (string or embedding) and a candidate set, return top-k most similar nodes.
	"""

	def topk_similar(
		self, q: str, candidates: Sequence[T], k: int
	) -> list[tuple[T, float]]: ...


@runtime_checkable
class EmbeddedGraphLike(GraphLike[T], TopKSimilarity[T], Protocol[T]):
	"""Graph that can do top-k similarity over a candidate subset."""

	...
