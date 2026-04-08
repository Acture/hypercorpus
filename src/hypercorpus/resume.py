from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from filelock import FileLock


RUN_STATE_VERSION = 1
SELECTION_CHECKPOINT_VERSION = 1
RESUME_STATE_VERSION = 1


def _utcnow() -> str:
	return datetime.now(UTC).isoformat(timespec="seconds")


def _normalize_for_json(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {
			str(key): _normalize_for_json(inner)
			for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
		}
	if isinstance(value, (list, tuple)):
		return [_normalize_for_json(item) for item in value]
	return value


def atomic_write_text(path: Path, content: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_name(f".{path.name}.tmp")
	tmp_path.write_text(content, encoding="utf-8")
	os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
	atomic_write_text(
		path,
		json.dumps(_normalize_for_json(payload), ensure_ascii=False, indent=2),
	)


def build_selection_key(*, case_id: str, budget_label: str, selector_name: str) -> str:
	raw = f"{case_id}\n{budget_label}\n{selector_name}"
	return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_config_fingerprint(payload: dict[str, Any]) -> str:
	normalized = _normalize_for_json(payload)
	serialized = json.dumps(
		normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True
	)
	return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class RunStatus(StrEnum):
	PENDING = "PENDING"
	RUNNING = "RUNNING"
	INTERRUPTING = "INTERRUPTING"
	INTERRUPTED = "INTERRUPTED"
	FAILED = "FAILED"
	COMPLETED = "COMPLETED"


class SelectionStage(StrEnum):
	SELECTOR_BODY = "selector_body"
	EXPORT_GRAPHRAG = "export_graphrag"
	E2E = "e2e"
	FINAL_COMMIT = "final_commit"
	BASE_SELECTOR = "base_selector"
	BUDGET_FILL = "budget_fill"


@dataclass(slots=True, frozen=True)
class SelectionPlanItem:
	selection_key: str
	case_id: str
	budget_label: str
	selector_name: str
	selector_family: str


@dataclass(slots=True)
class RunState:
	version: int
	status: RunStatus
	config_fingerprint: str
	planned_case_ids: list[str]
	planned_selection_keys: list[str]
	completed_selection_keys: list[str] = field(default_factory=list)
	completed_case_ids: list[str] = field(default_factory=list)
	current_case_id: str | None = None
	current_selection_key: str | None = None
	current_selector_name: str | None = None
	current_budget_label: str | None = None
	current_stage: str | None = None
	stage_started_at: str | None = None
	last_heartbeat_at: str | None = None
	interrupted_reason: str | None = None
	last_error: str | None = None
	created_at: str = field(default_factory=_utcnow)
	updated_at: str = field(default_factory=_utcnow)

	def to_dict(self) -> dict[str, Any]:
		return {
			"version": self.version,
			"status": self.status.value,
			"config_fingerprint": self.config_fingerprint,
			"planned_case_ids": list(self.planned_case_ids),
			"planned_selection_keys": list(self.planned_selection_keys),
			"completed_selection_keys": list(self.completed_selection_keys),
			"completed_case_ids": list(self.completed_case_ids),
			"current_case_id": self.current_case_id,
			"current_selection_key": self.current_selection_key,
			"current_selector_name": self.current_selector_name,
			"current_budget_label": self.current_budget_label,
			"current_stage": self.current_stage,
			"stage_started_at": self.stage_started_at,
			"last_heartbeat_at": self.last_heartbeat_at,
			"interrupted_reason": self.interrupted_reason,
			"last_error": self.last_error,
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}

	@classmethod
	def from_dict(cls, payload: dict[str, Any]) -> "RunState":
		return cls(
			version=int(payload["version"]),
			status=RunStatus(str(payload["status"])),
			config_fingerprint=str(payload["config_fingerprint"]),
			planned_case_ids=[
				str(case_id) for case_id in payload.get("planned_case_ids", [])
			],
			planned_selection_keys=[
				str(key) for key in payload.get("planned_selection_keys", [])
			],
			completed_selection_keys=[
				str(key) for key in payload.get("completed_selection_keys", [])
			],
			completed_case_ids=[
				str(case_id) for case_id in payload.get("completed_case_ids", [])
			],
			current_case_id=None
			if payload.get("current_case_id") is None
			else str(payload["current_case_id"]),
			current_selection_key=None
			if payload.get("current_selection_key") is None
			else str(payload["current_selection_key"]),
			current_selector_name=None
			if payload.get("current_selector_name") is None
			else str(payload["current_selector_name"]),
			current_budget_label=None
			if payload.get("current_budget_label") is None
			else str(payload["current_budget_label"]),
			current_stage=None
			if payload.get("current_stage") is None
			else str(payload["current_stage"]),
			stage_started_at=None
			if payload.get("stage_started_at") is None
			else str(payload["stage_started_at"]),
			last_heartbeat_at=None
			if payload.get("last_heartbeat_at") is None
			else str(payload["last_heartbeat_at"]),
			interrupted_reason=None
			if payload.get("interrupted_reason") is None
			else str(payload["interrupted_reason"]),
			last_error=None
			if payload.get("last_error") is None
			else str(payload["last_error"]),
			created_at=str(payload.get("created_at") or _utcnow()),
			updated_at=str(payload.get("updated_at") or _utcnow()),
		)


@dataclass(slots=True)
class SelectionCheckpointBundle:
	selection_key: str
	case_id: str
	budget_label: str
	selector_name: str
	selection_record: dict[str, Any]
	selector_log_records: list[dict[str, Any]]
	completed_at: str = field(default_factory=_utcnow)

	def to_dict(self) -> dict[str, Any]:
		return {
			"version": SELECTION_CHECKPOINT_VERSION,
			"selection_key": self.selection_key,
			"case_id": self.case_id,
			"budget_label": self.budget_label,
			"selector_name": self.selector_name,
			"selection_record": self.selection_record,
			"selector_log_records": self.selector_log_records,
			"completed_at": self.completed_at,
		}

	@classmethod
	def from_dict(cls, payload: dict[str, Any]) -> "SelectionCheckpointBundle":
		return cls(
			selection_key=str(payload["selection_key"]),
			case_id=str(payload["case_id"]),
			budget_label=str(payload["budget_label"]),
			selector_name=str(payload["selector_name"]),
			selection_record=dict(payload["selection_record"]),
			selector_log_records=[
				dict(record) for record in payload.get("selector_log_records", [])
			],
			completed_at=str(payload.get("completed_at") or _utcnow()),
		)


@dataclass(slots=True)
class SelectionResumeState:
	selection_key: str
	case_id: str
	budget_label: str
	selector_name: str
	family: str
	stage: SelectionStage
	payload: dict[str, Any]
	updated_at: str = field(default_factory=_utcnow)

	def to_dict(self) -> dict[str, Any]:
		return {
			"version": RESUME_STATE_VERSION,
			"selection_key": self.selection_key,
			"case_id": self.case_id,
			"budget_label": self.budget_label,
			"selector_name": self.selector_name,
			"family": self.family,
			"stage": self.stage.value,
			"payload": self.payload,
			"updated_at": self.updated_at,
		}

	@classmethod
	def from_dict(cls, payload: dict[str, Any]) -> "SelectionResumeState":
		return cls(
			selection_key=str(payload["selection_key"]),
			case_id=str(payload["case_id"]),
			budget_label=str(payload["budget_label"]),
			selector_name=str(payload["selector_name"]),
			family=str(payload["family"]),
			stage=SelectionStage(str(payload["stage"])),
			payload=dict(payload.get("payload", {})),
			updated_at=str(payload.get("updated_at") or _utcnow()),
		)


class StopRequested(RuntimeError):
	"""Raised when execution should stop at the current checkpoint boundary."""


class HardStopRequested(KeyboardInterrupt):
	"""Raised on a second interrupt to abort immediately."""


class InterruptController:
	def __init__(self) -> None:
		self._soft_stop_requested = False
		self._hard_stop_requested = False
		self._installed = False
		self._previous_sigint = None

	@property
	def soft_stop_requested(self) -> bool:
		return self._soft_stop_requested

	@property
	def hard_stop_requested(self) -> bool:
		return self._hard_stop_requested

	def install(self) -> None:
		if self._installed:
			return
		self._previous_sigint = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGINT, self._handle_sigint)
		self._installed = True

	def uninstall(self) -> None:
		if not self._installed:
			return
		signal.signal(signal.SIGINT, self._previous_sigint)
		self._installed = False
		self._previous_sigint = None

	def request_soft_stop(self) -> None:
		self._soft_stop_requested = True

	def checkpoint(self) -> None:
		if self._hard_stop_requested:
			raise HardStopRequested()
		if self._soft_stop_requested:
			raise StopRequested()

	def _handle_sigint(self, _signum, _frame) -> None:
		if self._soft_stop_requested:
			self._hard_stop_requested = True
			raise HardStopRequested()
		self._soft_stop_requested = True


class SelectionExecutionDriver(Protocol):
	family_name: str

	def supports(self, selector: Any) -> bool: ...


class CheckpointStore:
	def __init__(self, root: Path):
		self.root = root
		self.run_state_path = self.root / "run_state.json"
		self.lock_path = self.root / ".lock"
		self.checkpoints_root = self.root / "_checkpoints"
		self.selections_root = self.checkpoints_root / "selections"
		self.resume_root = self.checkpoints_root / "resume"

	def ensure_layout(self) -> None:
		self.root.mkdir(parents=True, exist_ok=True)
		self.checkpoints_root.mkdir(parents=True, exist_ok=True)
		self.selections_root.mkdir(parents=True, exist_ok=True)
		self.resume_root.mkdir(parents=True, exist_ok=True)

	def acquire_lock(self) -> FileLock:
		self.ensure_layout()
		return FileLock(str(self.lock_path))

	def clear_artifacts(self) -> None:
		if self.run_state_path.exists():
			self.run_state_path.unlink()
		if self.lock_path.exists():
			self.lock_path.unlink()
		if self.checkpoints_root.exists():
			shutil.rmtree(self.checkpoints_root)

	def selection_path(self, selection_key: str) -> Path:
		return self.selections_root / f"{selection_key}.json"

	def resume_path(self, selection_key: str) -> Path:
		return self.resume_root / f"{selection_key}.json"

	def list_selection_keys(self) -> list[str]:
		if not self.selections_root.exists():
			return []
		return sorted(path.stem for path in self.selections_root.glob("*.json"))

	def list_resume_keys(self) -> list[str]:
		if not self.resume_root.exists():
			return []
		return sorted(path.stem for path in self.resume_root.glob("*.json"))

	def load_run_state(self) -> RunState | None:
		if not self.run_state_path.exists():
			return None
		return RunState.from_dict(
			json.loads(self.run_state_path.read_text(encoding="utf-8"))
		)

	def save_run_state(self, state: RunState) -> None:
		state.updated_at = _utcnow()
		atomic_write_json(self.run_state_path, state.to_dict())

	def load_selection_checkpoint(
		self, selection_key: str
	) -> SelectionCheckpointBundle | None:
		path = self.selection_path(selection_key)
		if not path.exists():
			return None
		return SelectionCheckpointBundle.from_dict(
			json.loads(path.read_text(encoding="utf-8"))
		)

	def save_selection_checkpoint(self, bundle: SelectionCheckpointBundle) -> None:
		atomic_write_json(self.selection_path(bundle.selection_key), bundle.to_dict())

	def load_resume_state(self, selection_key: str) -> SelectionResumeState | None:
		path = self.resume_path(selection_key)
		if not path.exists():
			return None
		return SelectionResumeState.from_dict(
			json.loads(path.read_text(encoding="utf-8"))
		)

	def save_resume_state(self, state: SelectionResumeState) -> None:
		state.updated_at = _utcnow()
		atomic_write_json(self.resume_path(state.selection_key), state.to_dict())

	def remove_resume_state(self, selection_key: str) -> None:
		self.resume_path(selection_key).unlink(missing_ok=True)
