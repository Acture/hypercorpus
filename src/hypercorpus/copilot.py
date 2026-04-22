from __future__ import annotations

import asyncio
import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from typing import Any, Protocol

DEFAULT_COPILOT_MODEL = "gpt-4.1"
GITHUB_MODELS_API_VERSION = "2022-11-28"
_COPILOT_TIMEOUT_S_DEFAULT = 120.0
_COPILOT_TIMEOUT_ENV = "HYPERCORPUS_COPILOT_TIMEOUT_S"


def _copilot_timeout_s() -> float:
	raw = os.environ.get(_COPILOT_TIMEOUT_ENV)
	if raw is not None:
		return float(raw)
	return _COPILOT_TIMEOUT_S_DEFAULT


@dataclass(frozen=True, slots=True)
class CopilotSdkCompletion:
	text: str
	prompt_tokens: int | None
	completion_tokens: int | None
	total_tokens: int | None
	raw_response: str | None
	model: str | None


class SupportsCopilotSdkRunner(Protocol):
	def complete(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		timeout_s: float | None = None,
	) -> CopilotSdkCompletion: ...

	def validate_environment(self) -> None: ...


def validate_copilot_model_name(model: str) -> str:
	text = model.strip()
	if not text:
		raise ValueError("copilot model name must not be empty.")
	if "/" in text:
		raise ValueError(
			"copilot model names must use SDK-native ids like 'gpt-4.1' or 'gpt-5', not provider-prefixed ids like 'openai/gpt-5'."
		)
	return text


def is_copilot_model_name(model: str) -> bool:
	try:
		validate_copilot_model_name(model)
	except ValueError:
		return False
	return True


class CopilotSdkRunner:
	def __init__(self, *, working_directory: str | Path | None = None):
		self._working_directory = str(
			(working_directory or Path.cwd()).expanduser().resolve()
			if isinstance(working_directory, Path)
			else Path(working_directory or Path.cwd()).expanduser().resolve()
		)
		self._lock = threading.Lock()
		self._runner: asyncio.Runner | None = None
		self._client: Any | None = None
		atexit.register(self.close)

	def validate_environment(self) -> None:
		with self._lock:
			self._ensure_started()

	def complete(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		timeout_s: float | None = None,
	) -> CopilotSdkCompletion:
		resolved_model = validate_copilot_model_name(model)
		with self._lock:
			self._ensure_started()
			assert self._runner is not None
			return self._runner.run(
				self._complete_async(
					model=resolved_model,
					system_prompt=system_prompt,
					user_prompt=user_prompt,
					timeout_s=timeout_s,
				)
			)

	async def _complete_async(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		timeout_s: float | None,
	) -> CopilotSdkCompletion:
		from copilot.types import PermissionHandler

		resolved_timeout = timeout_s if timeout_s is not None else _copilot_timeout_s()
		assert self._client is not None
		session = await self._client.create_session(
			model=model,
			system_message={"mode": "replace", "content": system_prompt},
			available_tools=[],
			working_directory=self._working_directory,
			streaming=False,
			on_permission_request=PermissionHandler.approve_all,
		)
		try:
			response = await session.send_and_wait(user_prompt, timeout=resolved_timeout)
			if response is None:
				raise RuntimeError("Copilot SDK returned no assistant message.")
			data = response.data
			text = str(getattr(data, "content", "") or "")
			prompt_tokens = _maybe_int(getattr(data, "input_tokens", None))
			completion_tokens = _maybe_int(getattr(data, "output_tokens", None))
			total_tokens = _sum_usage(prompt_tokens, completion_tokens)
			resolved_model = _maybe_text(getattr(data, "model", None))
			raw_response = json.dumps(
				{
					"content": text,
					"model": resolved_model,
					"input_tokens": prompt_tokens,
					"output_tokens": completion_tokens,
					"total_tokens": total_tokens,
				},
				ensure_ascii=False,
			)
			return CopilotSdkCompletion(
				text=text,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				total_tokens=total_tokens,
				raw_response=raw_response,
				model=resolved_model,
			)
		finally:
			await session.disconnect()

	def close(self) -> None:
		with self._lock:
			if self._runner is None:
				return
			try:
				if self._client is not None:
					self._runner.run(self._client.stop())
			except Exception:
				pass
			finally:
				self._client = None
				self._runner.close()
				self._runner = None

	def _ensure_started(self) -> None:
		if self._runner is None:
			self._runner = asyncio.Runner()
		if self._client is not None:
			return
		try:
			from copilot import CopilotClient
		except ImportError as exc:
			raise RuntimeError(
				"copilot provider requires the github-copilot-sdk package to be installed."
			) from exc
		self._client = CopilotClient()
		self._runner.run(self._client.start())


def _maybe_int(value: object) -> int | None:
	if value is None:
		return None
	return int(value)


def _maybe_text(value: object) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _sum_usage(prompt_tokens: int | None, completion_tokens: int | None) -> int | None:
	if prompt_tokens is None or completion_tokens is None:
		return None
	return prompt_tokens + completion_tokens


def github_models_default_headers() -> dict[str, str]:
	return {
		"Accept": "application/vnd.github+json",
		"X-GitHub-Api-Version": GITHUB_MODELS_API_VERSION,
	}


def normalize_github_models_base_url(base_url: str) -> str:
	text = base_url.strip()
	if not text:
		raise ValueError(
			"selector_base_url must be an absolute GitHub Models inference root."
		)
	from urllib.parse import urlparse

	parsed = urlparse(text)
	if not parsed.scheme or not parsed.netloc:
		raise ValueError(
			"selector_base_url must be an absolute GitHub Models inference root."
		)
	if parsed.query:
		raise ValueError("GitHub Models base URL must not include query parameters.")
	path = parsed.path.rstrip("/")
	if path.endswith("/chat/completions"):
		path = path[: -len("/chat/completions")]
	if not path.endswith("/inference"):
		raise ValueError(
			"GitHub Models base URL must point at the inference root, for example https://models.github.ai/inference or https://models.github.ai/orgs/<org>/inference."
		)
	return f"{parsed.scheme}://{parsed.netloc}{path}"
