import json
import sqlite3

from webwalker.store.kvstore.sqlite import SQLiteKVStore


def test_build_from_bz(tmp_path, tiny_tar_bz2):
	test_db_path = tmp_path / "test.db"
	store = SQLiteKVStore.build_from_tar_bz2(
		str(test_db_path.resolve()),
		tar_bz2_path=str(tiny_tar_bz2.resolve()),
		table="hotqa",
		key_fn=lambda o: o["title"],
	)
	
	assert store is not None
	assert store.get("Moon Launch Program") == {
		"title": "Moon Launch Program",
		"text": "Uses Cape Canaveral.",
	}
	assert store.get("missing") is None


def test_get_reads_json_payload(tmp_path):
	db_path = tmp_path / "direct.db"
	store = SQLiteKVStore(str(db_path), "kv")

	with sqlite3.connect(db_path) as conn:
		conn.execute(
			"INSERT OR REPLACE INTO kv(k, v) VALUES (?, ?)",
			("alpha", json.dumps({"title": "Alpha"})),
		)
		conn.commit()

	assert store.get("alpha") == {"title": "Alpha"}
