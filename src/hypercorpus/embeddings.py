from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "multi-qa-MiniLM-L6-cos-v1"


def default_embedding_cache_path() -> Path:
	cache_root = os.environ.get("XDG_CACHE_HOME")
	if cache_root:
		return Path(cache_root) / "hypercorpus" / "embeddings.sqlite3"
	return Path.home() / ".cache" / "hypercorpus" / "embeddings.sqlite3"


_DEFAULT_ENCODE_BATCH_SIZE = 256


@dataclass(slots=True)
class SentenceTransformerEmbedderConfig:
	model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL
	cache_path: Path | None = None
	device: str | None = None
	encode_batch_size: int = _DEFAULT_ENCODE_BATCH_SIZE

	def __post_init__(self) -> None:
		if self.cache_path is None:
			self.cache_path = default_embedding_cache_path()


class TextEmbedder(Protocol):
	backend_name: str
	model_name: str

	def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SQLiteEmbeddingCache:
	def __init__(self, path: Path):
		self.path = path
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self._conn: sqlite3.Connection | None = None
		self._initialize()

	@property
	def _connection(self) -> sqlite3.Connection:
		if self._conn is None:
			self._conn = sqlite3.connect(self.path)
		return self._conn

	def close(self) -> None:
		if self._conn is not None:
			self._conn.close()
			self._conn = None

	def get(self, *, model_name: str, text: str) -> list[float] | None:
		key = _cache_key(model_name=model_name, text=text)
		row = self._connection.execute(
			"SELECT embedding_json FROM embeddings WHERE cache_key = ?",
			(key,),
		).fetchone()
		if row is None:
			return None
		payload = json.loads(str(row[0]))
		return [float(value) for value in payload]

	def put(self, *, model_name: str, text: str, embedding: Sequence[float]) -> None:
		key = _cache_key(model_name=model_name, text=text)
		payload = json.dumps([float(value) for value in embedding], ensure_ascii=False)
		self._connection.execute(
			"""
            INSERT INTO embeddings(cache_key, model_name, text_hash, embedding_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET embedding_json = excluded.embedding_json
            """,
			(key, model_name, _text_hash(text), payload),
		)
		self._connection.commit()

	def _initialize(self) -> None:
		self._connection.execute(
			"""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
		)
		self._connection.commit()


class SentenceTransformerEmbedder:
	backend_name = "sentence_transformer"
	_model_cache: dict[tuple[str, str | None], object] = {}

	def __init__(self, config: SentenceTransformerEmbedderConfig):
		self.config = config
		self.model_name = config.model_name
		self.cache = (
			SQLiteEmbeddingCache(config.cache_path)
			if config.cache_path is not None
			else None
		)

	def encode(self, texts: Sequence[str]) -> list[list[float]]:
		if not texts:
			return []
		cached_vectors: dict[int, list[float]] = {}
		missing_indices: list[int] = []
		missing_texts: list[str] = []
		if self.cache is not None:
			for index, text in enumerate(texts):
				cached = self.cache.get(model_name=self.model_name, text=text)
				if cached is None:
					missing_indices.append(index)
					missing_texts.append(text)
				else:
					cached_vectors[index] = cached
		else:
			missing_indices = list(range(len(texts)))
			missing_texts = list(texts)

		if missing_texts:
			model = self._get_model()
			batch_size = self.config.encode_batch_size
			total_batches = (len(missing_texts) + batch_size - 1) // batch_size
			if total_batches > 1:
				logger.info(
					"Encoding %d texts in %d batches (batch_size=%d, device=%s)",
					len(missing_texts),
					total_batches,
					batch_size,
					self.config.device or "cpu",
				)
			for batch_start in range(0, len(missing_texts), batch_size):
				batch_end = min(batch_start + batch_size, len(missing_texts))
				if total_batches > 1:
					logger.info(
						"  batch %d/%d (%d texts)",
						batch_start // batch_size + 1,
						total_batches,
						batch_end - batch_start,
					)
				encoded = model.encode(
					missing_texts[batch_start:batch_end],
					normalize_embeddings=True,
					show_progress_bar=False,
				)
				for offset, vector in enumerate(encoded):
					values = [float(value) for value in list(vector)]
					abs_idx = batch_start + offset
					cached_vectors[missing_indices[abs_idx]] = values
					if self.cache is not None:
						self.cache.put(
							model_name=self.model_name,
							text=missing_texts[abs_idx],
							embedding=values,
						)

		return [cached_vectors[index] for index in range(len(texts))]

	def _get_model(self):
		cache_key = (self.model_name, self.config.device)
		if cache_key in self._model_cache:
			return self._model_cache[cache_key]
		try:
			from sentence_transformers import SentenceTransformer
		except (
			ImportError
		) as exc:  # pragma: no cover - exercised only when dependency missing
			raise RuntimeError(
				"sentence-transformers is required for sentence_transformer selectors."
			) from exc
		kwargs = {}
		if self.config.device is not None:
			kwargs["device"] = self.config.device
		model = SentenceTransformer(self.model_name, **kwargs)
		self._model_cache[cache_key] = model
		return model


def _text_hash(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(*, model_name: str, text: str) -> str:
	return hashlib.sha256(
		f"{model_name}:{_text_hash(text)}".encode("utf-8")
	).hexdigest()


__all__ = [
	"DEFAULT_SENTENCE_TRANSFORMER_MODEL",
	"SentenceTransformerEmbedder",
	"SentenceTransformerEmbedderConfig",
	"TextEmbedder",
	"default_embedding_cache_path",
]
