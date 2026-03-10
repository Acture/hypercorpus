from __future__ import annotations

import logging
import os
from typing import BinaryIO, Iterable, Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
	BarColumn,
	DownloadColumn,
	Progress,
	SpinnerColumn,
	TaskProgressColumn,
	TextColumn,
	TimeElapsedColumn,
	TimeRemainingColumn,
	TransferSpeedColumn,
)


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


def progress_console(console: Optional[Console] = None) -> Console:
	return console or Console(stderr=True)


def should_render_progress(console: Optional[Console] = None) -> bool:
	return progress_console(console).is_terminal


def create_progress(
		*,
		console: Optional[Console] = None,
		transient: bool = False,
) -> Progress:
	return Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		BarColumn(bar_width=None),
		TaskProgressColumn(),
		TimeElapsedColumn(),
		TimeRemainingColumn(),
		console=progress_console(console),
		transient=transient,
	)


def create_transfer_progress(
		*,
		console: Optional[Console] = None,
		transient: bool = False,
) -> Progress:
	return Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		BarColumn(bar_width=None),
		DownloadColumn(),
		TransferSpeedColumn(),
		TimeElapsedColumn(),
		TimeRemainingColumn(),
		console=progress_console(console),
		transient=transient,
	)


def copy_stream_with_progress(
		source: BinaryIO,
		destination: BinaryIO,
		*,
		description: str,
		total: int | None = None,
		chunk_size: int = 1024 * 1024,
		console: Optional[Console] = None,
) -> int:
	total_written = 0
	if not should_render_progress(console):
		while True:
			chunk = source.read(chunk_size)
			if not chunk:
				break
			destination.write(chunk)
			total_written += len(chunk)
		return total_written

	with create_transfer_progress(console=console, transient=True) as progress:
		task_id = progress.add_task(description, total=total)
		while True:
			chunk = source.read(chunk_size)
			if not chunk:
				break
			destination.write(chunk)
			total_written += len(chunk)
			progress.advance(task_id, len(chunk))
	return total_written
