from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
from typing import BinaryIO, Iterable, Iterator, Optional, Union

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


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes if prefix)


class PackageOrWarningFilter(logging.Filter):
    """Allow first-party logs at the configured level and third-party warnings/errors."""

    def __init__(self, package: str, allow: Optional[Iterable[str]] = None):
        super().__init__()
        self.prefixes = tuple([package, *(allow or [])])

    def filter(self, record: logging.LogRecord) -> bool:
        if _matches_prefix(record.name, self.prefixes):
            return True
        return record.levelno >= logging.WARNING


@dataclass(slots=True)
class DashboardLogEntry:
    rendered: str
    logger_name: str
    level_name: str
    levelno: int


class DashboardLogBuffer:
    def __init__(self, *, max_records: int = 200):
        self.max_records = max_records
        self._entries: deque[DashboardLogEntry] = deque(maxlen=max_records)

    def append(self, entry: DashboardLogEntry) -> None:
        self._entries.append(entry)

    def tail(self, *, limit: int | None = None) -> list[DashboardLogEntry]:
        entries = list(self._entries)
        if limit is None or limit >= len(entries):
            return entries
        return entries[-limit:]


@dataclass(slots=True)
class DashboardProgressTask:
    task_id: int
    description: str
    completed: float = 0.0
    total: float | None = None
    detail: str | None = None
    finished: bool = False


class DashboardProgressState:
    def __init__(self) -> None:
        self._tasks: dict[int, DashboardProgressTask] = {}
        self._next_task_id = 1

    def add_task(self, description: str, total: float | None = None, **fields: object) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        detail = fields.get("detail")
        self._tasks[task_id] = DashboardProgressTask(
            task_id=task_id,
            description=description,
            total=total,
            detail=str(detail) if detail is not None else None,
        )
        return task_id

    def advance(self, task_id: int, advance: float = 1.0) -> None:
        self.update(task_id, advance=advance)

    def update(
        self,
        task_id: int,
        *,
        advance: float | None = None,
        completed: float | None = None,
        total: float | None = None,
        description: str | None = None,
        **fields: object,
    ) -> None:
        task = self._tasks[task_id]
        if advance is not None:
            completed = task.completed + advance
        if completed is not None:
            task.completed = completed
        if total is not None:
            task.total = total
        if description is not None:
            task.description = description
        if "detail" in fields:
            detail = fields["detail"]
            task.detail = str(detail) if detail is not None and detail != "" else None
        if task.total is not None and task.completed >= task.total:
            task.finished = True

    def snapshot(self) -> list[DashboardProgressTask]:
        return list(self._tasks.values())

    def active_tasks(self) -> list[DashboardProgressTask]:
        return [task for task in self._tasks.values() if not task.finished]

    def latest_task(self) -> DashboardProgressTask | None:
        active_tasks = self.active_tasks()
        if active_tasks:
            return active_tasks[-1]
        tasks = self.snapshot()
        return tasks[-1] if tasks else None


@dataclass(slots=True)
class _DashboardSession:
    log_buffer: DashboardLogBuffer | None = None
    progress_state: DashboardProgressState | None = None


_ACTIVE_DASHBOARD_SESSION: ContextVar[_DashboardSession | None] = ContextVar(
    "hypercorpus_dashboard_session",
    default=None,
)


def active_dashboard_session() -> _DashboardSession | None:
    return _ACTIVE_DASHBOARD_SESSION.get()


def active_dashboard_progress_state() -> DashboardProgressState | None:
    session = active_dashboard_session()
    return session.progress_state if session is not None else None


@contextmanager
def dashboard_session(
    *,
    log_buffer: DashboardLogBuffer | None = None,
    progress_state: DashboardProgressState | None = None,
) -> Iterator[_DashboardSession]:
    session = _DashboardSession(log_buffer=log_buffer, progress_state=progress_state)
    token = _ACTIVE_DASHBOARD_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_DASHBOARD_SESSION.reset(token)


class DashboardAwareRichHandler(RichHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if active_dashboard_session() is not None:
            return
        super().emit(record)


class DashboardBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        session = active_dashboard_session()
        if session is None or session.log_buffer is None:
            return
        try:
            rendered = self.format(record)
        except Exception:
            self.handleError(record)
            return
        session.log_buffer.append(
            DashboardLogEntry(
                rendered=rendered,
                logger_name=record.name,
                level_name=record.levelname,
                levelno=record.levelno,
            )
        )


class _DashboardProgressAdapter:
    def __init__(self, state: DashboardProgressState):
        self.state = state

    def __enter__(self) -> _DashboardProgressAdapter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def add_task(self, description: str, total: float | None = None, **fields: object) -> int:
        return self.state.add_task(description, total=total, **fields)

    def advance(self, task_id: int, advance: float = 1.0) -> None:
        self.state.advance(task_id, advance=advance)

    def update(self, task_id: int, **fields: object) -> None:
        self.state.update(task_id, **fields)


def setup_rich_logging(
    package: str,
    *,
    level: Union[int, str, None] = None,
    allow: Optional[Iterable[str]] = None,
    console: Optional[Console] = None,
    force: bool = True,
) -> None:
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    console = console or Console()
    policy_filter = PackageOrWarningFilter(package, allow=allow)

    rich_handler = DashboardAwareRichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
    )
    rich_handler.addFilter(policy_filter)

    buffer_handler = DashboardBufferHandler()
    buffer_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    buffer_handler.addFilter(policy_filter)

    logging.basicConfig(
        level=level,
        handlers=[rich_handler, buffer_handler],
        force=force,
    )
    logging.getLogger().setLevel(level)


def progress_console(console: Optional[Console] = None) -> Console:
    return console or Console(stderr=True)


def should_render_progress(console: Optional[Console] = None) -> bool:
    if active_dashboard_progress_state() is not None:
        return True
    return progress_console(console).is_terminal


def create_progress(
    *,
    console: Optional[Console] = None,
    transient: bool = False,
) -> Progress | _DashboardProgressAdapter:
    progress_state = active_dashboard_progress_state()
    if progress_state is not None:
        return _DashboardProgressAdapter(progress_state)
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
) -> Progress | _DashboardProgressAdapter:
    progress_state = active_dashboard_progress_state()
    if progress_state is not None:
        return _DashboardProgressAdapter(progress_state)
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
    log_interval_mb: int = 50,
) -> int:
    _logger = logging.getLogger(__name__)
    total_written = 0
    if not should_render_progress(console):
        next_log_at = log_interval_mb * 1024 * 1024
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            destination.write(chunk)
            total_written += len(chunk)
            if total_written >= next_log_at:
                written_mb = total_written / (1024 * 1024)
                if total:
                    total_mb = total / (1024 * 1024)
                    pct = total_written / total * 100
                    _logger.info("%s: %.0f / %.0f MB (%.0f%%)", description, written_mb, total_mb, pct)
                else:
                    _logger.info("%s: %.0f MB transferred", description, written_mb)
                next_log_at += log_interval_mb * 1024 * 1024
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
