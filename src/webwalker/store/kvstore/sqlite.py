from __future__ import annotations

import bz2
import json
import sqlite3
import tarfile
from dataclasses import dataclass
from typing import Optional, Iterable, Callable
from webwalker.type import AnyHashable as K, AnyObj as V
from webwalker.store.protocol import KVStore

from rich.progress import (
	Progress, SpinnerColumn, TextColumn, BarColumn,
	MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
)


def _iter_jsonl_from_inner_bz2(inner_file, encoding: str = "utf-8"):
	"""
	inner_fileobj: file-like object containing *bz2-compressed* JSONL bytes
	yields decoded JSON objects line by line (streaming).
	"""
	decomp = bz2.BZ2Decompressor()
	buf = b""
	while True:
		chunk = inner_file.read(1 << 20)  # 1MB
		if not chunk:
			# flush remaining
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
		
		# 1) 预扫描 members（不解压内容）
		with tarfile.open(tar_bz2_path, "r:bz2") as tf:
			members = [
				m for m in tf.getmembers()
				if	m.isfile()
					and m.name.endswith(".bz2")
					and (member_pred(m.name) if member_pred else True)
			]
		
		total_files = len(members)
		if total_files == 0:
			raise ValueError("No matching .bz2 members found in tar.")
		
		# 2) 真正构建：一边处理一边显示进度
		progress = Progress(
			SpinnerColumn(),
			TextColumn("[bold]{task.description}[/bold]"),
			BarColumn(),
			MofNCompleteColumn(),
			TextColumn("•"),
			TimeElapsedColumn(),
			TextColumn("•"),
			TimeRemainingColumn(),
			TextColumn("•"),
			TextColumn("[cyan]{task.fields[detail]}[/cyan]"),
		)
		
		files_task = progress.add_task("members", total=total_files, detail="")
		rows_task = progress.add_task("rows", total=None, detail="")  # total unknown
		
		with progress:
			with sqlite3.connect(db_path) as conn, tarfile.open(tar_bz2_path, "r:bz2") as tf:
				cur = conn.cursor()
				
				batch: list[tuple[str, str]] = []
				rows_written = 0
				
				for i, m in enumerate(members, start=1):
					f = tf.extractfile(m)
					if f is None:
						progress.update(files_task, advance=1, detail=f"skip: {m.name}")
						continue
					
					progress.update(files_task, detail=m.name)
					
					for obj in _iter_jsonl_from_inner_bz2(f, encoding=encoding):
						k = key_fn(obj)
						v = val_fn(obj)
						batch.append((k, json.dumps(v, ensure_ascii=False)))
						rows_written += 1
						
						if rows_written % commit_every == 0:
							cur.executemany(
								f"INSERT OR REPLACE INTO {table}(k, v) VALUES (?, ?)",
								batch,
							)
							conn.commit()
							batch.clear()
							
							# rows 进度：用 advance 更新（更准确显示速率）
							progress.update(rows_task, advance=commit_every, detail=f"last commit @ {rows_written:,}")
					
					# 每处理完一个 member，推进文件进度
					progress.update(files_task, advance=1)
				
				# flush remaining
				if batch:
					cur.executemany(
						f"INSERT OR REPLACE INTO {table}(k, v) VALUES (?, ?)",
						batch,
					)
					conn.commit()
					progress.update(rows_task, advance=len(batch), detail=f"final commit (+{len(batch)})")
					batch.clear()
		
		return store


def get(self, key: str) -> Optional[dict]:
	with sqlite3.connect(self.db_path) as conn:
		row = conn.execute(
			f"SELECT v FROM {self.table} WHERE k = ?",
			(key,),
		).fetchone()
	if row is None:
		return None
	return json.loads(row[0])
