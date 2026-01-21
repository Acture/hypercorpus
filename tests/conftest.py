from pathlib import Path

from pytest import fixture


@fixture
def test_dir() -> Path:
	return Path(__file__).parent


@fixture
def project_dir(test_dir: Path) -> Path:
	return test_dir.parent


@fixture
def dataset_dir(project_dir: Path) -> Path:
	return project_dir / "dataset"


@fixture
def hotqa_dir(dataset_dir: Path) -> Path:
	return dataset_dir / "hotqa"