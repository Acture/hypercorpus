from __future__ import annotations

import os
import bz2
import json
import sqlite3
import tarfile
from dataclasses import dataclass
from typing import Optional, Callable

from rich.progress import (
	Progress, SpinnerColumn, TextColumn, BarColumn,
	TimeElapsedColumn, TimeRemainingColumn
)

from hypercorpus.store.protocol import KVStore
from hypercorpus.type import AnyHashable as K, AnyObj as V


def _iter_jsonl_from_inner_bz2(inner_file, encoding: str = "utf-8"):
	decomp = bz2.BZ2Decompressor()
	buf = b""
	while True:
		chunk = inner_file.read(1 << 20)  # 1MB
		if not chunk:
			if buf.strip():
				for line in buf.split(b"\n"):
					if line.strip():
						yield json.loads(line.decode(encoding))
			break
		
		buf += decomp.decompress(chunk)
		while b"\n" in buf:
			line, buf = buf.split(b"\n", 1)
			if line.strip():
				yield json.loads(line.decode(encoding))


@dataclass
class SQLiteKVStore(KVStore[K, V]):
	db_path: str
	table: str
	
	def __post_init__(self) -> None:
		with sqlite3.connect(self.db_path) as conn:
			conn.execute(
				f"CREATE TABLE IF NOT EXISTS {self.table} (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
			)
			conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_k ON {self.table}(k)")
			conn.commit()

	def get(self, key: K) -> Optional[V]:
		with sqlite3.connect(self.db_path) as conn:
			row = conn.execute(
				f"SELECT v FROM {self.table} WHERE k = ?",
				(str(key),),
			).fetchone()
		if row is None:
			return None
		return json.loads(row[0])
	
	@classmethod
	def build_from_tar_bz2(
			cls,
			db_path: str,
			tar_bz2_path: str,
			key_fn: Callable[[dict], str],
			val_fn: Callable[[dict], dict] = lambda obj: obj,  # type: ignore
			table: str = "kv",
			encoding: str = "utf-8",
			commit_every: int = 5000,
			member_pred: Optional[Callable[[str], bool]] = None,
	) -> "SQLiteKVStore":
		store = cls(db_path=db_path, table=table)
		
		total_bytes = os.path.getsize(tar_bz2_path)
		
		progress = Progress(
			SpinnerColumn(),
			TextColumn("[bold]{task.description}[/bold]"),
			BarColumn(),
			TextColumn("{task.percentage:>6.2f}%"),
			TextColumn("•"),
			TimeElapsedColumn(),
			TextColumn("•"),
			TimeRemainingColumn(),
			TextColumn("•"),
			TextColumn("[cyan]{task.fields[detail]}[/cyan]"),
		)
		
		bytes_task = progress.add_task("tar bytes", total=total_bytes, detail="")
		rows_task = progress.add_task("rows", total=None, detail="")  # unknown total
		files_task = progress.add_task("members", total=None, detail="")
		
		with progress:
			with sqlite3.connect(db_path) as conn:
				cur = conn.cursor()
				batch: list[tuple[str, str]] = []
				rows = 0
				
				# 用 fileobj + r|bz2 真流式：不构建 member 列表
				with open(tar_bz2_path, "rb") as raw, tarfile.open(fileobj=raw, mode="r|bz2") as tf:
					last_pos = raw.tell()
					
					for m in tf:
						# 更新压缩包字节进度（位置变化是“已读取的压缩字节”）
						pos = raw.tell()
						if pos > last_pos:
							progress.update(bytes_task, advance=(pos - last_pos))
							last_pos = pos
						
						if not m.isfile():
							continue
						if not m.name.endswith(".bz2"):
							continue
						if member_pred is not None and not member_pred(m.name):
							continue
						
						progress.update(bytes_task, detail=m.name)
						progress.update(files_task, advance=1, detail=m.name)
						
						f = tf.extractfile(m)
						if f is None:
							continue
						
						for obj in _iter_jsonl_from_inner_bz2(f, encoding=encoding):
							k = key_fn(obj)
							v = val_fn(obj)
							batch.append((k, json.dumps(v, ensure_ascii=False)))
							rows += 1
							
							if rows % commit_every == 0:
								cur.executemany(
									f"INSERT OR REPLACE INTO {table}(k, v) VALUES (?, ?)",
									batch,
								)
								conn.commit()
								batch.clear()
								progress.update(rows_task, advance=commit_every, detail=f"{rows:,}")
					
					# flush remaining
					if batch:
						cur.executemany(
							f"INSERT OR REPLACE INTO {table}(k, v) VALUES (?, ?)",
							batch,
						)
						conn.commit()
						progress.update(rows_task, advance=len(batch), detail=f"{rows:,}")
						batch.clear()
		
		return store
