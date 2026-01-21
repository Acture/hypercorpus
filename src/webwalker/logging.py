from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Union

from rich.console import Console
from rich.logging import RichHandler


class PrefixFilter(logging.Filter):
	"""Allow only records whose logger name starts with allowed prefixes."""
	
	def __init__(self, prefixes: Iterable[str]):
		super().__init__()
		self.prefixes = tuple(p for p in prefixes if p)
	
	def filter(self, record: logging.LogRecord) -> bool:
		name = record.name
		return any(name == p or name.startswith(p + ".") for p in self.prefixes)


def setup_rich_logging(
		package: str,
		*,
		level: Union[int, str, None] = None,
		allow: Optional[Iterable[str]] = None,  # extra allowed prefixes
		console: Optional[Console] = None,
		force: bool = True,
) -> None:
	if level is None:
		level = os.getenv("LOG_LEVEL", "INFO")
	if isinstance(level, str):
		level = level.upper()
	
	console = console or Console()
	
	handler = RichHandler(
		console=console,
		rich_tracebacks=True,
		show_time=True,
		show_level=True,
		show_path=False,
		markup=True,
	)
	
	prefixes = [package]
	if allow:
		prefixes.extend(list(allow))
	handler.addFilter(PrefixFilter(prefixes))
	
	# Root handler: accepts all, but handler filter will drop non-matching
	logging.basicConfig(
		level=level,
		handlers=[handler],
		force=force,
	)
	
	# Optional hardening: keep 3rd-party quiet even if they add handlers later
	logging.getLogger().setLevel(level)  # root level
