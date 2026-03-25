from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hypercorpus.baselines import (
	EXTERNAL_MDR_SELECTOR_NAME,
	ExternalMDRSelector,
	MDRArtifactManifest,
	export_iirc_store_to_mdr,
)
from hypercorpus.baselines.mdr import _resolve_mdr_home
from hypercorpus.datasets.store import prepare_store_from_records
from hypercorpus.eval import EvaluationBudget, EvaluationCase
from hypercorpus.graph import DocumentNode, LinkContext, LinkContextGraph
from hypercorpus_cli import app


class _FakeMDRBackend:
	def __init__(self, paths):
		self._paths = paths

	def retrieve(self, query: str, *, beam_size: int, topk_paths: int):
		from hypercorpus.baselines.mdr import MDRQueryResult

		del query, beam_size, topk_paths
		return MDRQueryResult(paths=list(self._paths), runtime_s=0.01)


def _prepare_iirc_train_dev_store(tmp_path: Path) -> Path:
	store_root = tmp_path / "iirc-store"
	questions_root = store_root / "questions"
	train_path = questions_root / "train.json"
	dev_path = questions_root / "dev.json"
	questions_root.mkdir(parents=True, exist_ok=True)
	train_questions = [
		{
			"case_id": "train-a",
			"question": "Which city is the launch site?",
			"answer": "Cape Canaveral",
			"gold_support_nodes": ["Moon Launch Program", "Cape Canaveral"],
			"gold_start_nodes": ["Moon Launch Program"],
			"gold_path_nodes": ["Moon Launch Program", "Cape Canaveral"],
		},
		{
			"case_id": "train-b",
			"question": "Which state contains the launch city?",
			"answer": "Florida",
			"gold_support_nodes": ["Moon Launch Program", "Cape Canaveral", "Florida"],
			"gold_start_nodes": ["Moon Launch Program"],
			"gold_path_nodes": ["Moon Launch Program", "Cape Canaveral", "Florida"],
		},
		{
			"case_id": "train-comp",
			"question": "Which program launches rockets and which agency runs missions?",
			"answer": "Moon Launch Program and NASA",
			"gold_support_nodes": ["Moon Launch Program", "NASA"],
			"gold_start_nodes": ["Moon Launch Program"],
		},
		{
			"case_id": "train-drop",
			"question": "Which program launches rockets?",
			"answer": "Moon Launch Program",
			"gold_support_nodes": ["Moon Launch Program"],
			"gold_start_nodes": ["Moon Launch Program"],
		},
	]
	dev_questions = [
		{
			"case_id": "dev-a",
			"question": "Which state contains the launch city?",
			"answer": "Florida",
			"gold_support_nodes": ["Moon Launch Program", "Cape Canaveral", "Florida"],
			"gold_start_nodes": ["Moon Launch Program"],
			"gold_path_nodes": ["Moon Launch Program", "Cape Canaveral", "Florida"],
		}
	]
	train_path.write_text(
		json.dumps(train_questions, ensure_ascii=False), encoding="utf-8"
	)
	dev_path.write_text(json.dumps(dev_questions, ensure_ascii=False), encoding="utf-8")

	records = [
		{
			"node_id": "Moon Launch Program",
			"title": "Moon Launch Program",
			"sentences": ["Moon Launch Program launches from Cape Canaveral."],
			"links": [
				{
					"target": "Cape Canaveral",
					"anchor_text": "Cape Canaveral",
					"sentence": "Moon Launch Program launches from Cape Canaveral.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Cape Canaveral",
			"title": "Cape Canaveral",
			"sentences": ["Cape Canaveral is in Florida."],
			"links": [
				{
					"target": "Florida",
					"anchor_text": "Florida",
					"sentence": "Cape Canaveral is in Florida.",
					"sent_idx": 0,
				}
			],
		},
		{
			"node_id": "Florida",
			"title": "Florida",
			"sentences": ["Florida is a state."],
			"links": [],
		},
		{
			"node_id": "NASA",
			"title": "NASA",
			"sentences": ["NASA operates launch missions."],
			"links": [],
		},
		{
			"node_id": "Houston",
			"title": "Houston",
			"sentences": ["Houston supports mission control."],
			"links": [],
		},
	]
	(store_root / "index").mkdir(parents=True, exist_ok=True)
	(store_root / "shards").mkdir(parents=True, exist_ok=True)
	prepare_store_from_records(
		output_root=store_root,
		dataset_name="iirc",
		normalized_records=records,
		target_shard_size_bytes=1024,
		questions_source="questions-source",
		graph_source="graph-source",
		questions_paths={"train": train_path, "dev": dev_path},
	)
	return store_root


def test_export_iirc_store_to_mdr_writes_expected_schema_and_is_stable(tmp_path):
	store_root = _prepare_iirc_train_dev_store(tmp_path)
	export_dir_a = tmp_path / "export-a"
	export_dir_b = tmp_path / "export-b"

	manifest_a = export_iirc_store_to_mdr(store_uri=store_root, output_dir=export_dir_a)
	export_iirc_store_to_mdr(store_uri=store_root, output_dir=export_dir_b)

	assert manifest_a.corpus_documents == 5
	assert manifest_a.total_train_cases == 4
	assert manifest_a.train_examples + manifest_a.val_examples == 3
	assert manifest_a.eligible_train_cases == 3
	assert manifest_a.dev_examples == 1
	assert manifest_a.distinct_support_ge_2_cases == 3
	assert manifest_a.dropped_no_bridge == 1
	assert manifest_a.dropped_no_bridge_with_distinct_support_ge_2 == 0
	assert manifest_a.export_manifest_path is not None
	assert Path(manifest_a.export_manifest_path).exists()
	assert manifest_a.dropped_cases_path is not None
	assert Path(manifest_a.dropped_cases_path).exists()
	assert (
		sum(manifest_a.train_type_counts.values())
		+ sum(manifest_a.val_type_counts.values())
		== 3
	)
	assert (
		manifest_a.train_type_counts.get("comparison", 0)
		+ manifest_a.val_type_counts.get("comparison", 0)
		== 1
	)

	train_lines_a = (
		(export_dir_a / "train.jsonl").read_text(encoding="utf-8").splitlines()
	)
	val_lines_a = (export_dir_a / "val.jsonl").read_text(encoding="utf-8").splitlines()
	train_lines_b = (
		(export_dir_b / "train.jsonl").read_text(encoding="utf-8").splitlines()
	)
	val_lines_b = (export_dir_b / "val.jsonl").read_text(encoding="utf-8").splitlines()
	assert train_lines_a == train_lines_b
	assert val_lines_a == val_lines_b

	exported_samples = [json.loads(line) for line in [*train_lines_a, *val_lines_a]]
	sample = next(item for item in exported_samples if item["_id"] == "train-a")
	assert sample["type"] == "bridge"
	assert sample["bridge"] == "Cape Canaveral"
	assert len(sample["pos_paras"]) == 2
	assert len(sample["neg_paras"]) == 2
	assert sample["pos_paras"][0]["title"] == "Moon Launch Program"
	assert sample["sp"] == ["Moon Launch Program", "Cape Canaveral"]

	comparison_sample = next(
		item for item in exported_samples if item["_id"] == "train-comp"
	)
	assert comparison_sample["type"] == "comparison"
	assert comparison_sample["bridge"] == "NASA"
	assert comparison_sample["sp"] == ["Moon Launch Program", "NASA"]

	corpus_sample = json.loads(
		(export_dir_a / "corpus.jsonl").read_text(encoding="utf-8").splitlines()[0]
	)
	assert set(corpus_sample) == {"title", "text"}

	dropped_cases = [
		json.loads(line)
		for line in (export_dir_a / "dropped_cases.jsonl")
		.read_text(encoding="utf-8")
		.splitlines()
	]
	assert dropped_cases == [
		{
			"case_id": "train-drop",
			"drop_reason": "no_bridge",
			"support_nodes": ["Moon Launch Program"],
			"support_nodes_in_store": ["Moon Launch Program"],
			"start_doc": "Moon Launch Program",
			"candidate_bridge_docs": [],
			"question_type_guess": "bridge",
		}
	]


def test_resolve_mdr_home_defaults_to_repo_submodule(monkeypatch):
	monkeypatch.delenv("WEBWALKER_MDR_HOME", raising=False)
	resolved = _resolve_mdr_home(None)
	assert resolved == Path("/Users/acture/repos/hypercorpus/baselines/mdr")


def test_external_mdr_selector_flattens_paths_and_applies_budget():
	graph = LinkContextGraph(
		documents=[
			DocumentNode(node_id="doc-a", title="doc-a", sentences=("alpha beta",)),
			DocumentNode(node_id="doc-b", title="doc-b", sentences=("gamma delta",)),
			DocumentNode(node_id="doc-c", title="doc-c", sentences=("epsilon zeta",)),
		],
		links=[
			LinkContext(
				source="doc-a",
				target="doc-b",
				anchor_text="b",
				sentence="go to b",
				sent_idx=0,
			),
			LinkContext(
				source="doc-b",
				target="doc-c",
				anchor_text="c",
				sentence="go to c",
				sent_idx=0,
			),
		],
	)
	artifact_manifest = MDRArtifactManifest(
		mdr_home="/tmp/mdr",
		mdr_commit="commit",
		mdr_patch_sha="patch",
		export_manifest_path="/tmp/export.json",
		train_manifest_path=None,
		checkpoint_path="/tmp/checkpoint.pt",
		model_name="roberta-base",
		shared_encoder=False,
		corpus_embeddings_path="/tmp/corpus.npy",
		id2doc_path="/tmp/id2doc.json",
		faiss_index_path=None,
		max_q_len=70,
		max_c_len=300,
		max_q_sp_len=350,
		beam_size=5,
		topk_paths=10,
		doc_count=3,
		embedding_dim=768,
		artifact_manifest_path="/tmp/mdr_artifact_manifest.json",
	)
	from hypercorpus.baselines.mdr import MDRPathResult

	selector = ExternalMDRSelector(
		artifact_manifest=artifact_manifest,
		backend_factory=lambda manifest: _FakeMDRBackend(
			[
				MDRPathResult(score=9.0, node_ids=["doc-a", "doc-b"]),
				MDRPathResult(score=8.0, node_ids=["doc-b", "doc-c"]),
			]
		),
	)
	case = EvaluationCase(
		case_id="case-1",
		query="where should we go next",
		gold_support_nodes=["doc-a", "doc-b"],
		gold_start_nodes=["doc-a"],
	)
	budget = EvaluationBudget(token_budget_tokens=4)

	result = selector.select(graph, case, budget)

	assert result.selected_node_ids == ["doc-a", "doc-b"]
	assert [node.node_id for node in result.ranked_nodes] == ["doc-a", "doc-b", "doc-c"]
	assert result.token_cost_estimate == 4
	assert len(result.selected_links) == 1
	assert result.selected_links[0].source == "doc-a"
	assert result.selector_metadata is not None
	assert result.selector_metadata.backend == "mdr_official"
	assert result.selector_metadata.model == "roberta-base"
	assert result.selector_usage is not None
	assert result.selector_usage.total_tokens == 0
	assert result.selector_usage.runtime_s >= 0.0


def test_run_iirc_store_cli_requires_mdr_artifact_manifest_for_external_selector(
	prepared_iirc_store, tmp_path
):
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-iirc-store",
			"--store",
			str(prepared_iirc_store.root),
			"--exp-name",
			"mdr",
			"--output-root",
			str(tmp_path / "runs"),
			"--selectors",
			EXTERNAL_MDR_SELECTOR_NAME,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code != 0
	assert "--mdr-artifact-manifest" in result.output


def test_export_mdr_iirc_cli_smoke(tmp_path):
	store_root = _prepare_iirc_train_dev_store(tmp_path)
	export_dir = tmp_path / "export-cli"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"baselines",
			"export-mdr-iirc",
			"--store",
			str(store_root),
			"--output-dir",
			str(export_dir),
		],
	)

	assert result.exit_code == 0, result.output
	assert (export_dir / "corpus.jsonl").exists()
	assert (export_dir / "train.jsonl").exists()
	assert (export_dir / "val.jsonl").exists()
	assert (export_dir / "mdr_export_manifest.json").exists()
	assert (export_dir / "dropped_cases.jsonl").exists()
