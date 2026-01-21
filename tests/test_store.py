from pathlib import Path

from webwalker.store.kvstore.sqlite import SQLiteKVStore



def test_build_from_bz(project_dir: Path, hotqa_dir: Path):
	test_db_path = project_dir / "test.db"
	bz2_path = hotqa_dir / "enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
	
	store = SQLiteKVStore.build_from_tar_bz2(
		str(test_db_path.resolve()),
		tar_bz2_path=str(bz2_path.resolve()),
		table="hotqa",
		key_fn=lambda o: o["title"],
	)
	
	assert store is not None
	print(store)
