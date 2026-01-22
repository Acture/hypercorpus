from __future__ import annotations

from pathlib import Path
from typing import Optional

from webwalker.store.kvstore.sqlite import SQLiteKVStore
from webwalker.utils.jsonl2csv import jsonl_to_csv as jsonl2csv_impl
from webwalker.utils.fetch_wiki import fetch_wiki as fetch_wiki_impl

import typer

import logging

logger = logging.getLogger(__name__)

utils_app = typer.Typer(
	name="webwalker utils",
	help="webwalker utilities",
	add_completion=False,
)


@utils_app.command("jsonl2csv")
def jsonl2csv(
		in_path: Path = typer.Argument(..., exists=True, dir_okay=False),
		out_path: Path = typer.Argument(..., dir_okay=False),
		*,
		id_key: str = typer.Option("id", help="JSON key for id"),
		title_key: str = typer.Option("title", help="JSON key for title"),
		url_key: str = typer.Option("url", help="JSON key for url"),
		text_key: str = typer.Option("text", help="JSON key for text (list[str] or str)"),
		encoding: str = typer.Option("utf-8", help="File encoding"),
):
	"""
	Convert JSONL -> CSV (id,title,url,curid,text).
	"""
	out_path.parent.mkdir(parents=True, exist_ok=True)
	
	jsonl2csv_impl(
		in_path, out_path,
		{id_key: "id", title_key: "title", url_key: "url", text_key: "text"},
		encoding=encoding
	)


@utils_app.command("fetch_wiki")
def fetch_wiki(
		output_file: Path = typer.Argument(..., dir_okay=False),
		target_count: int = typer.Option(100, "-c", "--count", help="Number of articles to fetch"),
		min_char_length: Optional[int] = typer.Option(None, help="Minimum character length for article text"),
):
	"""
	Fetch Wikipedia articles and save to JSONL file.
	"""
	fetch_wiki_impl(output_file, target_count, min_char_length)



@utils_app.command("hotqa2db")
def hotqa2db(
		db_path: Path = typer.Argument(..., dir_okay=False, writable=True),
		tar_bz2_path: Path = typer.Argument(..., dir_okay=False, exists=True),
		table: str = typer.Option("hotqa", "-t", "--table", help="SQLite table name"),
		key_name: str = typer.Option("title", help="SQLite column name for key"),
):
	"""
	Fetch Wikipedia articles and save to JSONL file.
	"""
	store = SQLiteKVStore.build_from_tar_bz2(
		str(db_path.resolve()),
		tar_bz2_path=str(tar_bz2_path.resolve()),
		table=table,
		key_fn=lambda o: o[key_name],
	)
	
	assert store is not None
	
	logger.info(f"Loaded into {db_path}")