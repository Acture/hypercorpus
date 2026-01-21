from __future__ import annotations

from typing import Any, Dict, Hashable, Protocol, TypeVar, runtime_checkable, Sequence, Optional

from webwalker.type import AnyHashable as K
from webwalker.type import AnyObj as V


@runtime_checkable
class KVStore(Protocol[K, V]):
	def get(self, key: K) -> Optional[V]:
		...