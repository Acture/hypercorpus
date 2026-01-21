from __future__ import annotations

import bz2
import json
import sqlite3
from typing import Iterable, Callable, Optional

from pydantic import BaseModel, Field, model_validator


class SQLiteKVStore(BaseModel):
	db_path: str
	table: str = "kv"
	
	# 可选：复用连接（更快）；但注意线程/进程场景需谨慎
	conn: sqlite3.Connection | None = Field(default=None, exclude=True)
	
	@model_validator(mode="after")
	def _ensure_table(self) -> "SQLiteKVStore":
		with sqlite3.connect(self.db_path) as conn:
			conn.execute(
				f"CREATE TABLE IF NOT EXISTS {self.table} (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
			)
			conn.execute(
				f"CREATE INDEX IF NOT EXISTS idx_{self.table}_k ON {self.table}(k)"
			)
			conn.commit()
		return self
	
	@classmethod
	def build_from_bz2(
			cls,
			db_path: str,
			bz2_paths: Iterable[str],
			key_fn: Callable[[dict], str],
			val_fn: Callable[[dict], dict] = lambda obj: obj,  # type: ignore
			table: str = "kv",
			encoding: str = "utf-8",
			commit_every: int = 5000,
	) -> "SQLiteKVStore":
		store = cls(db_path=db_path, table=table)
		
		with sqlite3.connect(db_path) as conn:
			cur = conn.cursor()
			n = 0
			for path in bz2_paths:
				with bz2.open(path, "rt", encoding=encoding) as f:
					for line in f:
						line = line.strip()
						if not line:
							continue
						obj = json.loads(line)
						k = key_fn(obj)
						v = val_fn(obj)
						cur.execute(
							f"INSERT OR REPLACE INTO {table}(k, v) VALUES (?, ?)",
							(k, json.dumps(v, ensure_ascii=False)),
						)
						n += 1
						if n % commit_every == 0:
							conn.commit()
			conn.commit()
		
		return store
	
	def get(self, key: str) -> Optional[dict]:
		with sqlite3.connect(self.db_path) as conn:
			row = conn.execute(
				f"SELECT v FROM {self.table} WHERE k = ?",
				(key,),
			).fetchone()
		return None if row is None else json.loads(row[0])
