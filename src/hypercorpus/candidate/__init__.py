from .protocol import GraphLike
from hypercorpus.type import AnyHashable as T
from .policy import StartPolicy


def select_starting_candidates(g: GraphLike[T], q: str, start_policy: StartPolicy[T]) -> list[T]:
	return start_policy.select_start(g, q)


def select_starting_candidate(g: GraphLike[T], q: str, start_policy: StartPolicy[T]) -> T:
	candidates = start_policy.select_start(g, q)
	if not candidates:
		raise ValueError("Start policy returned no candidate nodes.")
	return candidates[0]
