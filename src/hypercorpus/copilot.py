from __future__ import annotations

from urllib.parse import urlparse

DEFAULT_COPILOT_API_KEY_ENV = "GITHUB_TOKEN"
DEFAULT_COPILOT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_COPILOT_BASE_URL = "https://models.github.ai/inference"
COPILOT_API_VERSION = "2022-11-28"


def copilot_default_headers() -> dict[str, str]:
	return {
		"Accept": "application/vnd.github+json",
		"X-GitHub-Api-Version": COPILOT_API_VERSION,
	}


def normalize_copilot_base_url(base_url: str | None) -> str:
	if base_url is None:
		return DEFAULT_COPILOT_BASE_URL
	text = base_url.strip()
	parsed = urlparse(text)
	if not parsed.scheme or not parsed.netloc:
		raise ValueError(
			"Copilot base URL must be an absolute GitHub Models inference root."
		)
	if parsed.query:
		raise ValueError("Copilot base URL must not include query parameters.")
	path = parsed.path.rstrip("/")
	if path.endswith("/chat/completions"):
		path = path[: -len("/chat/completions")]
	if not path.endswith("/inference"):
		raise ValueError(
			"Copilot base URL must point at the GitHub Models inference root, for example https://models.github.ai/inference or https://models.github.ai/orgs/<org>/inference."
		)
	return f"{parsed.scheme}://{parsed.netloc}{path}"
