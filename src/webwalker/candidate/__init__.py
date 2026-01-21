from .protocol import GraphLike
from webwalker.type import AnyHashable as T
from .policy import StartPolicy


def select_starting_candidate(g: GraphLike[T], q: str, start_policy: StartPolicy[T]) -> T:
	return start_policy.select_start(g, q)
