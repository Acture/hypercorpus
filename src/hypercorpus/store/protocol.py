from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional

from hypercorpus.type import AnyHashable as K
from hypercorpus.type import AnyObj as V


@runtime_checkable
class KVStore(Protocol[K, V]):
	def get(self, key: K) -> Optional[V]: ...
