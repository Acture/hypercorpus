from __future__ import annotations


from typing import Generic, Protocol

from .protocol import EmbeddedGraphLike, GraphLike
from webwalker.type import AnyHashable as T


class StartPolicy(Protocol[T]):
	name: str
	
	def select_start(self, g: GraphLike[T], q: str) -> list[T]:
		...


class MaxPhiOverAnchors(Generic[T]):
	name = "max_phi_over_anchors"

	def __init__(self, k: int = 1):
		self.k = k
	
	def select_start(self, g: GraphLike[T], q: str) -> list[T]:
		if not g.nodes:
			raise ValueError("No anchors available to choose a start node.")
		
		def phi(v: T) -> float:
			return float(g.node_attr.get(v, {}).get("phi", float("-inf")))
		
		return sorted(g.nodes, key=phi, reverse=True)[: self.k]


class SelectByCosTopK(Generic[T]):
	name = "select_by_cos_topk"
	
	def __init__(self, k: int = 1):
		self.k = k
	
	def select_start(self, g: EmbeddedGraphLike[T], q: str) -> list[T]:
		if not g.nodes:
			raise ValueError("No anchors available.")
		res = g.topk_similar(q, g.nodes, k=self.k)
		if not res:
			raise ValueError("Similarity backend returned empty result.")
		return [node for node, _score in res]
