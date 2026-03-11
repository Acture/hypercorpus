from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Protocol, Sequence

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "multi-qa-MiniLM-L6-cos-v1"


def default_embedding_cache_path() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root) / "webwalker" / "embeddings.sqlite3"
    return Path.home() / ".cache" / "webwalker" / "embeddings.sqlite3"


@dataclass(slots=True)
class SentenceTransformerEmbedderConfig:
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL
    cache_path: Path | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.cache_path is None:
            self.cache_path = default_embedding_cache_path()


class TextEmbedder(Protocol):
    backend_name: str
    model_name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SQLiteEmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get(self, *, model_name: str, text: str) -> list[float] | None:
        key = _cache_key(model_name=model_name, text=text)
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
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
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO embeddings(cache_key, model_name, text_hash, embedding_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET embedding_json = excluded.embedding_json
                """,
                (key, model_name, _text_hash(text), payload),
            )
            connection.commit()

    def _initialize(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            connection.commit()


class SentenceTransformerEmbedder:
    backend_name = "sentence_transformer"
    _model_cache: dict[tuple[str, str | None], object] = {}

    def __init__(self, config: SentenceTransformerEmbedderConfig):
        self.config = config
        self.model_name = config.model_name
        self.cache = SQLiteEmbeddingCache(config.cache_path) if config.cache_path is not None else None

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
            encoded = model.encode(
                list(missing_texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for offset, vector in enumerate(encoded):
                values = [float(value) for value in list(vector)]
                text = missing_texts[offset]
                index = missing_indices[offset]
                cached_vectors[index] = values
                if self.cache is not None:
                    self.cache.put(model_name=self.model_name, text=text, embedding=values)

        return [cached_vectors[index] for index in range(len(texts))]

    def _get_model(self):
        cache_key = (self.model_name, self.config.device)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
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
    return hashlib.sha256(f"{model_name}:{_text_hash(text)}".encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
    "SentenceTransformerEmbedder",
    "SentenceTransformerEmbedderConfig",
    "TextEmbedder",
    "default_embedding_cache_path",
]
