from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
import logging
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
JsonObj = Mapping[str, Any]
Getter = Union[
	str,  # dot-path like "a.b.0.c"
	Sequence[Union[str, int]],  # explicit path parts
	Callable[[JsonObj], Any],  # lambda obj: ...
	Tuple[Union[str, Sequence[Union[str, int]]], Callable[[Any, JsonObj], Any]],  # (path, transform)
]


def _get_in(obj: Any, path: Union[str, Sequence[Union[str, int]]], default: Any = None) -> Any:
	if isinstance(path, str):
		parts: list[Union[str, int]] = []
		for p in path.split("."):
			if p.isdigit():
				parts.append(int(p))
			else:
				parts.append(p)
	else:
		parts = list(path)
	
	cur = obj
	for key in parts:
		try:
			if isinstance(key, int):
				if isinstance(cur, (list, tuple)) and 0 <= key < len(cur):
					cur = cur[key]
				else:
					return default
			else:
				if isinstance(cur, Mapping) and key in cur:
					cur = cur[key]
				else:
					return default
		except Exception:
			return default
	return cur


def _join_list_str(x: Any, sep: str = " ") -> Any:
	# only auto-join list/tuple of strings (or stringify-able)
	if isinstance(x, (list, tuple)):
		parts = []
		for s in x:
			if s is None:
				continue
			s = str(s).strip()
			if s:
				parts.append(s)
		return sep.join(parts)
	return x


def jsonl_to_csv(
		in_path: PathLike,
		out_path: PathLike,
		columns: Dict[str, Getter],
		*,
		encoding: str = "utf-8",
		delimiter: str = ",",
		auto_join_str_lists: bool = True,  # e.g., "text": ["a", "b"] -> "a b"
		join_sep: str = " ",
		on_error: str = "skip",  # "skip" | "raise"
		write_header: bool = True,
) -> Dict[str, int]:
	"""
	Convert JSONL -> CSV.

	columns: mapping output_col -> getter
	  getter can be:
		- "a.b.0.c" (dot-path)
		- ["a","b",0,"c"]
		- lambda obj: ...
		- ("path", transform) where transform(value, obj) -> new_value

	Returns stats: {"read": N, "written": M, "skipped": K}
	"""
	in_path = Path(in_path)
	out_path = Path(out_path)
	
	fieldnames = list(columns.keys())
	stats = {"read": 0, "written": 0, "skipped": 0}
	
	with in_path.open("r", encoding=encoding) as fin, out_path.open("w", encoding=encoding, newline="") as fout:
		writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=delimiter)
		if write_header:
			writer.writeheader()
		
		for lineno, line in enumerate(fin, 1):
			line = line.strip()
			if not line:
				continue
			stats["read"] += 1
			try:
				obj = json.loads(line)
				row: Dict[str, Any] = {}
				
				for out_col, getter in columns.items():
					if callable(getter):
						v = getter(obj)
					elif isinstance(getter, tuple) and len(getter) == 2 and callable(getter[1]):
						path, tf = getter
						v0 = _get_in(obj, path, None)
						v = tf(v0, obj)
					else:
						v = _get_in(obj, getter, None)
					
					if auto_join_str_lists:
						v = _join_list_str(v, sep=join_sep)
					
					row[out_col] = v
				
				writer.writerow(row)
				stats["written"] += 1
			
			except Exception:
				stats["skipped"] += 1
				if on_error == "raise":
					raise RuntimeError(f"Failed at {in_path}:{lineno}") from None
	
	logger.info(f"Wrote {stats['written']} rows to {out_path} (read={stats['read']}, skipped={stats['skipped']})")
	return stats
