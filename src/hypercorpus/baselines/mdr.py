from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Protocol, cast

import numpy as np

from hypercorpus.datasets.store import ShardedDocumentStore

MDR_PINNED_COMMIT = "4872b392203d5731254e0e3be774d8837bb40caf"
EXTERNAL_MDR_SELECTOR_NAME = "external__mdr__iirc_finetuned"

_MDR_HOME_ENV = "WEBWALKER_MDR_HOME"
_PATCH_TARGET = "import pdb; pdb.set_trace()"
_PATCH_SHA = hashlib.sha256(
	b"mdr/retrieval/data/mhop_dataset.py:remove:import pdb; pdb.set_trace()"
).hexdigest()
_REQUIRED_MDR_PATHS: tuple[str, ...] = (
	"scripts/train_mhop.py",
	"scripts/encode_corpus.py",
	"mdr/retrieval/data/mhop_dataset.py",
	"mdr/retrieval/models/mhop_retriever.py",
	"mdr/retrieval/utils/utils.py",
)


def _default_repo_mdr_home() -> Path | None:
	candidate = Path(__file__).resolve().parents[3] / "baselines" / "mdr"
	if candidate.exists():
		return candidate
	return None


@dataclass(slots=True)
class MDRExportManifest:
	dataset_name: str
	store_uri: str
	store_version: str
	corpus_path: str
	train_path: str
	val_path: str
	dev_eval_path: str | None
	corpus_documents: int
	train_examples: int
	val_examples: int
	dev_examples: int
	dropped_missing_start: int
	dropped_no_bridge: int
	dropped_no_negatives: int
	export_manifest_path: str | None = None
	dropped_cases_path: str | None = None
	total_train_cases: int = 0
	eligible_train_cases: int = 0
	distinct_support_ge_2_cases: int = 0
	dropped_no_bridge_with_distinct_support_ge_2: int = 0
	train_type_counts: dict[str, int] = field(default_factory=dict)
	val_type_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class MDRTrainManifest:
	mdr_home: str
	mdr_commit: str
	mdr_patch_sha: str
	export_manifest_path: str
	output_dir: str
	model_name: str
	init_checkpoint: str | None
	shared_encoder: bool
	command: list[str]
	extra_args: list[str]
	checkpoint_path: str
	train_manifest_path: str | None = None


@dataclass(slots=True)
class MDRArtifactManifest:
	mdr_home: str
	mdr_commit: str
	mdr_patch_sha: str
	export_manifest_path: str
	train_manifest_path: str | None
	checkpoint_path: str
	model_name: str
	shared_encoder: bool
	corpus_embeddings_path: str
	id2doc_path: str
	faiss_index_path: str | None
	max_q_len: int
	max_c_len: int
	max_q_sp_len: int
	beam_size: int
	topk_paths: int
	doc_count: int
	embedding_dim: int
	artifact_manifest_path: str | None = None


@dataclass(slots=True)
class MDRPathResult:
	score: float
	node_ids: list[str]


@dataclass(slots=True)
class MDRQueryResult:
	paths: list[MDRPathResult]
	runtime_s: float


@dataclass(slots=True)
class MDRCaseSupportAnalysis:
	start_doc: str | None
	second_doc: str | None
	question_type: str
	support_nodes: list[str]
	candidate_bridge_docs: list[str]
	distinct_support_ge_2: bool


class _MDRRetrievalBackend(Protocol):
	def retrieve(
		self,
		query: str,
		*,
		beam_size: int,
		topk_paths: int,
	) -> MDRQueryResult: ...


def load_mdr_export_manifest(path: str | Path) -> MDRExportManifest:
	payload = json.loads(Path(path).read_text(encoding="utf-8"))
	return MDRExportManifest(**payload)


def load_mdr_train_manifest(path: str | Path) -> MDRTrainManifest:
	payload = json.loads(Path(path).read_text(encoding="utf-8"))
	return MDRTrainManifest(**payload)


def load_mdr_artifact_manifest(path: str | Path) -> MDRArtifactManifest:
	payload = json.loads(Path(path).read_text(encoding="utf-8"))
	return MDRArtifactManifest(**payload)


def export_iirc_store_to_mdr(
	*,
	store_uri: str | Path,
	output_dir: str | Path,
	cache_dir: str | Path | None = None,
) -> MDRExportManifest:
	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	store = ShardedDocumentStore(store_uri, cache_dir=cache_dir)
	if store.manifest.dataset_name != "iirc":
		raise ValueError(
			f"export-mdr-iirc requires an IIRC store, got {store.manifest.dataset_name!r}"
		)

	try:
		train_cases = list(store.load_questions("train"))
	except FileNotFoundError as exc:
		raise ValueError(
			f"IIRC store {store_uri!s} is missing the train split; re-prepare the store with train/dev questions."
		) from exc

	dev_cases: list[Any] = []
	try:
		dev_cases = list(store.load_questions("dev"))
	except FileNotFoundError:
		dev_cases = []

	node_ids = list(store.nodes)
	corpus_path = output_root / "corpus.jsonl"
	train_path = output_root / "train.jsonl"
	val_path = output_root / "val.jsonl"
	dev_eval_path = output_root / "dev_eval.jsonl"
	manifest_path = output_root / "mdr_export_manifest.json"
	dropped_cases_path = output_root / "dropped_cases.jsonl"

	corpus_records = [_corpus_record(store, node_id) for node_id in node_ids]
	_write_jsonl(corpus_path, corpus_records)

	train_records: list[dict[str, Any]] = []
	val_records: list[dict[str, Any]] = []
	dropped_case_records: list[dict[str, Any]] = []
	dropped_missing_start = 0
	dropped_no_bridge = 0
	dropped_no_negatives = 0
	total_train_cases = len(train_cases)
	eligible_train_cases = 0
	distinct_support_ge_2_cases = 0
	dropped_no_bridge_with_distinct_support_ge_2 = 0
	train_type_counts: dict[str, int] = {}
	val_type_counts: dict[str, int] = {}
	node_set = set(node_ids)

	for case in train_cases:
		analysis = _analyze_case_support(store, case, node_set=node_set)
		if analysis.distinct_support_ge_2:
			distinct_support_ge_2_cases += 1

		sample, drop_reason = _build_training_sample(
			store,
			case,
			node_ids=node_ids,
			analysis=analysis,
		)
		if sample is None:
			if drop_reason == "missing_start":
				dropped_missing_start += 1
			elif drop_reason == "no_bridge":
				dropped_no_bridge += 1
				if analysis.distinct_support_ge_2:
					dropped_no_bridge_with_distinct_support_ge_2 += 1
			elif drop_reason == "no_negatives":
				dropped_no_negatives += 1
			dropped_case_records.append(
				_build_dropped_case_record(case, analysis, drop_reason)
			)
			continue
		eligible_train_cases += 1
		if _is_val_case(case.case_id):
			val_records.append(sample)
			_increment_count(val_type_counts, sample["type"])
		else:
			train_records.append(sample)
			_increment_count(train_type_counts, sample["type"])

	_write_jsonl(train_path, train_records)
	_write_jsonl(val_path, val_records)
	_write_jsonl(dropped_cases_path, dropped_case_records)

	dev_records = [_build_eval_sample(store, case) for case in dev_cases]
	if dev_records:
		_write_jsonl(dev_eval_path, dev_records)

	manifest = MDRExportManifest(
		dataset_name="iirc",
		store_uri=str(store_uri),
		store_version=store.manifest.version,
		corpus_path=str(corpus_path.resolve()),
		train_path=str(train_path.resolve()),
		val_path=str(val_path.resolve()),
		dev_eval_path=str(dev_eval_path.resolve()) if dev_records else None,
		corpus_documents=len(corpus_records),
		train_examples=len(train_records),
		val_examples=len(val_records),
		dev_examples=len(dev_records),
		dropped_missing_start=dropped_missing_start,
		dropped_no_bridge=dropped_no_bridge,
		dropped_no_negatives=dropped_no_negatives,
		export_manifest_path=str(manifest_path.resolve()),
		dropped_cases_path=str(dropped_cases_path.resolve()),
		total_train_cases=total_train_cases,
		eligible_train_cases=eligible_train_cases,
		distinct_support_ge_2_cases=distinct_support_ge_2_cases,
		dropped_no_bridge_with_distinct_support_ge_2=dropped_no_bridge_with_distinct_support_ge_2,
		train_type_counts=train_type_counts,
		val_type_counts=val_type_counts,
	)
	_write_manifest(manifest_path, manifest)
	return manifest


def train_mdr_model(
	*,
	export_manifest_path: str | Path,
	output_dir: str | Path,
	mdr_home: str | Path | None = None,
	init_checkpoint: str | Path | None = None,
	model_name: str = "roberta-base",
	shared_encoder: bool = False,
	extra_args: list[str] | None = None,
) -> MDRTrainManifest:
	export_manifest = load_mdr_export_manifest(export_manifest_path)
	resolved_home, commit_sha, patch_sha = _validate_and_patch_mdr_checkout(mdr_home)

	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	command = [
		sys.executable,
		"scripts/train_mhop.py",
		"--do_train",
		"--train_file",
		export_manifest.train_path,
		"--predict_file",
		export_manifest.val_path,
		"--output_dir",
		str(output_root.resolve()),
		"--model_name",
		model_name,
	]
	if init_checkpoint is not None:
		command.extend(["--init_checkpoint", str(Path(init_checkpoint).resolve())])
	if shared_encoder:
		command.append("--shared-encoder")
	if extra_args:
		command.extend(extra_args)

	_run_official_mdr_command(command, cwd=resolved_home)
	checkpoint_path = _discover_checkpoint(output_root)
	manifest_path = output_root / "mdr_train_manifest.json"
	manifest = MDRTrainManifest(
		mdr_home=str(resolved_home),
		mdr_commit=commit_sha,
		mdr_patch_sha=patch_sha,
		export_manifest_path=str(Path(export_manifest_path).resolve()),
		output_dir=str(output_root.resolve()),
		model_name=model_name,
		init_checkpoint=None
		if init_checkpoint is None
		else str(Path(init_checkpoint).resolve()),
		shared_encoder=shared_encoder,
		command=command,
		extra_args=list(extra_args or ()),
		checkpoint_path=str(checkpoint_path.resolve()),
		train_manifest_path=str(manifest_path.resolve()),
	)
	_write_manifest(manifest_path, manifest)
	return manifest


def build_mdr_index(
	*,
	export_manifest_path: str | Path,
	output_dir: str | Path,
	checkpoint_path: str | Path | None = None,
	train_manifest_path: str | Path | None = None,
	mdr_home: str | Path | None = None,
	model_name: str | None = None,
	shared_encoder: bool | None = None,
	max_q_len: int = 70,
	max_c_len: int = 300,
	max_q_sp_len: int = 350,
	beam_size: int = 5,
	topk_paths: int = 10,
	extra_args: list[str] | None = None,
) -> MDRArtifactManifest:
	export_manifest = load_mdr_export_manifest(export_manifest_path)
	train_manifest = (
		load_mdr_train_manifest(train_manifest_path)
		if train_manifest_path is not None
		else None
	)
	resolved_home, commit_sha, patch_sha = _validate_and_patch_mdr_checkout(
		mdr_home or (train_manifest.mdr_home if train_manifest else None)
	)

	resolved_checkpoint: Path | None = None
	if checkpoint_path is not None:
		resolved_checkpoint = Path(checkpoint_path).resolve()
	elif train_manifest is not None:
		resolved_checkpoint = Path(train_manifest.checkpoint_path).resolve()
	if resolved_checkpoint is None:
		raise ValueError("build-mdr-index requires --checkpoint or --train-manifest.")

	resolved_model_name = model_name or (
		train_manifest.model_name if train_manifest is not None else "roberta-base"
	)
	resolved_shared_encoder = (
		shared_encoder
		if shared_encoder is not None
		else (train_manifest.shared_encoder if train_manifest is not None else False)
	)

	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	embed_prefix = output_root / "corpus_vectors"
	embed_prefix.mkdir(parents=True, exist_ok=True)

	command = [
		sys.executable,
		"scripts/encode_corpus.py",
		"--do_predict",
		"--predict_file",
		export_manifest.corpus_path,
		"--init_checkpoint",
		str(resolved_checkpoint),
		"--embed_save_path",
		str(embed_prefix.resolve()),
		"--model_name",
		resolved_model_name,
		"--max_q_len",
		str(max_q_len),
		"--max_c_len",
		str(max_c_len),
	]
	if resolved_shared_encoder:
		command.append("--shared-encoder")
	if extra_args:
		command.extend(extra_args)

	_run_official_mdr_command(command, cwd=resolved_home)

	embeddings_path = Path(f"{embed_prefix.resolve()}.npy")
	if not embeddings_path.exists():
		raise FileNotFoundError(
			f"Official MDR encoding did not produce embeddings at {embeddings_path}"
		)
	id2doc_path = embed_prefix / "id2doc.json"
	if not id2doc_path.exists():
		raise FileNotFoundError(
			f"Official MDR encoding did not produce id2doc.json at {id2doc_path}"
		)

	embeddings = np.load(embeddings_path).astype("float32")
	index_path = _maybe_build_faiss_index(
		output_root=output_root, embeddings=embeddings
	)
	manifest_path = output_root / "mdr_artifact_manifest.json"
	manifest = MDRArtifactManifest(
		mdr_home=str(resolved_home),
		mdr_commit=commit_sha,
		mdr_patch_sha=patch_sha,
		export_manifest_path=str(Path(export_manifest_path).resolve()),
		train_manifest_path=None
		if train_manifest_path is None
		else str(Path(train_manifest_path).resolve()),
		checkpoint_path=str(resolved_checkpoint),
		model_name=resolved_model_name,
		shared_encoder=resolved_shared_encoder,
		corpus_embeddings_path=str(embeddings_path.resolve()),
		id2doc_path=str(id2doc_path.resolve()),
		faiss_index_path=None if index_path is None else str(index_path.resolve()),
		max_q_len=max_q_len,
		max_c_len=max_c_len,
		max_q_sp_len=max_q_sp_len,
		beam_size=beam_size,
		topk_paths=topk_paths,
		doc_count=int(embeddings.shape[0]),
		embedding_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
		artifact_manifest_path=str(manifest_path.resolve()),
	)
	_write_manifest(manifest_path, manifest)
	return manifest


class ExternalMDRSelector:
	def __init__(
		self,
		*,
		name: str = EXTERNAL_MDR_SELECTOR_NAME,
		artifact_manifest_path: str | Path | None = None,
		artifact_manifest: MDRArtifactManifest | None = None,
		mdr_home: str | Path | None = None,
		beam_size: int | None = None,
		topk_paths: int | None = None,
		backend_factory: Callable[[MDRArtifactManifest], _MDRRetrievalBackend]
		| None = None,
	) -> None:
		if artifact_manifest is None and artifact_manifest_path is None:
			raise ValueError("ExternalMDRSelector requires an MDR artifact manifest.")

		from hypercorpus.selector import SelectorSpec

		self.name = name
		self._artifact_manifest_path = (
			None
			if artifact_manifest_path is None
			else Path(artifact_manifest_path).resolve()
		)
		self._artifact_manifest = artifact_manifest
		self._mdr_home = None if mdr_home is None else Path(mdr_home).resolve()
		self._beam_size = beam_size
		self._topk_paths = topk_paths
		self._backend_factory = backend_factory
		self._backend: _MDRRetrievalBackend | None = None
		self.spec = SelectorSpec(
			canonical_name=name,
			base_canonical_name=name,
			family="baseline",
		)

	def select(
		self,
		graph: Any,
		case: Any,
		budget: Any,
	) -> Any:
		from hypercorpus.selector import (
			CorpusSelectionResult,
			ScoredLink,
			ScoredNode,
			SelectionMode,
			SelectionTraceStep,
			SelectorBudget,
			SelectorMetadata,
			SelectorUsage,
			_apply_budget_to_ranked_nodes,
			_link_key,
			_query_tokens,
			_runtime_budget_token_limit,
			_selection_coverage_ratio,
		)

		backend = self._get_backend()
		artifact_manifest = self._get_artifact_manifest()
		beam_size = self._beam_size or artifact_manifest.beam_size
		topk_paths = self._topk_paths or artifact_manifest.topk_paths

		started_at = time.perf_counter()
		retrieval = backend.retrieve(
			case.query, beam_size=beam_size, topk_paths=topk_paths
		)
		ranked_nodes: list[ScoredNode] = []
		ranked_links: list[ScoredLink] = []
		trace: list[SelectionTraceStep] = []
		debug_trace: list[str] = []
		root_node_ids: list[str] = []
		seen_nodes: set[str] = set()
		seen_links: set[tuple[str, str, str, str, int, str | None]] = set()

		for path_index, path in enumerate(retrieval.paths):
			debug_trace.append(
				f"path[{path_index}] score={path.score:.4f} nodes={path.node_ids}"
			)
			if path.node_ids:
				root_node_ids.append(path.node_ids[0])
			for hop_index, node_id in enumerate(path.node_ids):
				if node_id in seen_nodes:
					continue
				seen_nodes.add(node_id)
				score = float(path.score - (path_index * 1e-3) - (hop_index * 1e-6))
				ranked_nodes.append(
					ScoredNode(
						node_id=node_id,
						score=score,
						source_strategy="external_mdr",
						selected_reason="ranked_path",
					)
				)
				trace.append(
					SelectionTraceStep(index=len(trace), node_id=node_id, score=score)
				)

			for source, target in zip(path.node_ids, path.node_ids[1:]):
				links = graph.links_between(source, target)
				if not links:
					continue
				scored_link = ScoredLink.from_link(
					links[0],
					score=float(path.score),
					source_strategy="external_mdr",
					selected_reason="retrieved_path",
				)
				key = _link_key(scored_link)
				if key in seen_links:
					continue
				seen_links.add(key)
				ranked_links.append(scored_link)

		runtime_budget = SelectorBudget(
			max_nodes=None,
			max_hops=2,
			max_tokens=_runtime_budget_token_limit(graph, budget),
		)
		selected_node_ids, token_cost_estimate = _apply_budget_to_ranked_nodes(
			graph, ranked_nodes, runtime_budget
		)
		selected_node_set = set(selected_node_ids)
		selected_links = [
			link
			for link in ranked_links
			if link.source in selected_node_set and link.target in selected_node_set
		]
		runtime_s = max(time.perf_counter() - started_at, retrieval.runtime_s)
		metadata = SelectorMetadata(
			scorer_kind="baseline",
			backend="mdr_official",
			provider="official",
			model=artifact_manifest.model_name,
			seed_strategy="mdr_two_hop_dense",
			seed_backend="mdr_official",
			seed_model=artifact_manifest.model_name,
			seed_top_k=beam_size,
			hop_budget=2,
			search_structure="two_hop_dense_retrieval",
		)
		usage = SelectorUsage(
			runtime_s=runtime_s,
			llm_calls=0,
			prompt_tokens=0,
			completion_tokens=0,
			total_tokens=0,
		)
		return CorpusSelectionResult(
			selector_name=self.name,
			query=case.query,
			ranked_nodes=ranked_nodes,
			ranked_links=ranked_links,
			selected_node_ids=selected_node_ids,
			selected_links=selected_links,
			token_cost_estimate=token_cost_estimate,
			strategy="external_mdr",
			mode=SelectionMode.STANDALONE,
			debug_trace=debug_trace,
			coverage_ratio=_selection_coverage_ratio(
				_query_tokens(case.query), selected_links
			),
			root_node_ids=_dedupe(root_node_ids),
			trace=trace,
			stop_reason="topk_paths" if retrieval.paths else "no_paths",
			selector_metadata=metadata,
			selector_usage=usage,
			selector_logs=[],
		)

	def _get_artifact_manifest(self) -> MDRArtifactManifest:
		if self._artifact_manifest is None:
			assert self._artifact_manifest_path is not None
			self._artifact_manifest = load_mdr_artifact_manifest(
				self._artifact_manifest_path
			)
		return self._artifact_manifest

	def _resolved_mdr_home(self) -> Path:
		manifest = self._get_artifact_manifest()
		candidate = self._mdr_home
		if candidate is None:
			env_value = os.environ.get(_MDR_HOME_ENV)
			if env_value:
				candidate = Path(env_value).resolve()
			elif manifest.mdr_home:
				candidate = Path(manifest.mdr_home).resolve()
			else:
				candidate = _default_repo_mdr_home()
		if candidate is None:
			raise ValueError(
				f"External MDR selector requires --mdr-home, {_MDR_HOME_ENV}, or a checkout at ./baselines/mdr; no official MDR checkout was resolved."
			)
		return candidate

	def _get_backend(self) -> _MDRRetrievalBackend:
		if self._backend is not None:
			return self._backend
		manifest = self._get_artifact_manifest()
		backend_factory = self._backend_factory or (
			lambda resolved_manifest: _OfficialMDRBackend(
				manifest=resolved_manifest,
				mdr_home=self._resolved_mdr_home(),
			)
		)
		self._backend = backend_factory(manifest)
		return self._backend


class _OfficialMDRBackend:
	def __init__(
		self,
		*,
		manifest: MDRArtifactManifest,
		mdr_home: str | Path,
	) -> None:
		self.manifest = manifest
		self.mdr_home = Path(mdr_home).resolve()
		self.embeddings = np.load(self.manifest.corpus_embeddings_path).astype(
			"float32"
		)
		self.docs = _load_id2doc_entries(self.manifest.id2doc_path)
		if len(self.docs) != len(self.embeddings):
			raise ValueError(
				f"MDR id2doc size {len(self.docs)} does not match embedding rows {len(self.embeddings)}."
			)
		self.index = _load_faiss_index(self.manifest.faiss_index_path)
		self._device = None
		self._model = None
		self._tokenizer = None

	def retrieve(
		self,
		query: str,
		*,
		beam_size: int,
		topk_paths: int,
	) -> MDRQueryResult:
		started_at = time.perf_counter()
		if self.embeddings.size == 0:
			return MDRQueryResult(paths=[], runtime_s=0.0)

		model, tokenizer, torch = self._load_model_stack()
		topk = max(1, min(int(beam_size), len(self.docs)))

		batch_q = tokenizer(
			[query.rstrip("?")],
			max_length=self.manifest.max_q_len,
			padding="max_length",
			truncation=True,
			return_tensors="pt",
		)
		batch_q = _move_batch_to_device(batch_q, self._device, torch)
		with torch.no_grad():
			q_embed = model.encode_q(
				batch_q["input_ids"],
				batch_q["attention_mask"],
				batch_q.get("token_type_ids"),
			)
		first_scores, first_indices = self._search(
			q_embed.detach().cpu().numpy().astype("float32"), topk
		)
		first_scores_row = first_scores[0].copy()
		first_indices_row = first_indices[0]

		query_pairs: list[tuple[str, str]] = []
		for rank, doc_index in enumerate(first_indices_row):
			doc = self.docs[int(doc_index)]
			doc_text = str(doc["text"]).strip()
			if "roberta" in self.manifest.model_name.lower() and not doc_text:
				doc_text = str(doc["title"])
				first_scores_row[rank] = float("-inf")
			query_pairs.append((query.rstrip("?"), doc_text))

		batch_q_sp = tokenizer(
			query_pairs,
			max_length=self.manifest.max_q_sp_len,
			padding="max_length",
			truncation=True,
			return_tensors="pt",
		)
		batch_q_sp = _move_batch_to_device(batch_q_sp, self._device, torch)
		with torch.no_grad():
			q_sp_embed = model.encode_q(
				batch_q_sp["input_ids"],
				batch_q_sp["attention_mask"],
				batch_q_sp.get("token_type_ids"),
			)
		second_scores, second_indices = self._search(
			q_sp_embed.detach().cpu().numpy().astype("float32"), topk
		)
		second_scores = second_scores.reshape(topk, topk)
		second_indices = second_indices.reshape(topk, topk)
		path_scores = np.expand_dims(first_scores_row, axis=1) + second_scores

		ranked_pairs = np.vstack(
			np.unravel_index(np.argsort(path_scores.ravel())[::-1], path_scores.shape)
		).transpose()

		paths: list[MDRPathResult] = []
		seen_paths: set[tuple[str, str]] = set()
		for hop1_rank, hop2_rank in ranked_pairs[
			: max(1, min(topk_paths, path_scores.size))
		]:
			hop1_doc_index = int(first_indices_row[int(hop1_rank)])
			hop2_doc_index = int(second_indices[int(hop1_rank), int(hop2_rank)])
			hop1_node_id = str(self.docs[hop1_doc_index]["title"])
			hop2_node_id = str(self.docs[hop2_doc_index]["title"])
			path_key = (hop1_node_id, hop2_node_id)
			if path_key in seen_paths:
				continue
			seen_paths.add(path_key)
			paths.append(
				MDRPathResult(
					score=float(path_scores[int(hop1_rank), int(hop2_rank)]),
					node_ids=[hop1_node_id, hop2_node_id],
				)
			)
			if len(paths) >= topk_paths:
				break

		return MDRQueryResult(paths=paths, runtime_s=time.perf_counter() - started_at)

	def _load_model_stack(self) -> tuple[Any, Any, Any]:
		if (
			self._model is not None
			and self._tokenizer is not None
			and self._device is not None
		):
			return self._model, self._tokenizer, cast(Any, sys.modules["torch"])

		with _temporary_sys_path(self.mdr_home):
			import torch
			from transformers import AutoConfig, AutoTokenizer

			from mdr.retrieval.models.mhop_retriever import RobertaRetriever  # ty: ignore[unresolved-import]
			from mdr.retrieval.utils.utils import load_saved  # ty: ignore[unresolved-import]

			config = AutoConfig.from_pretrained(self.manifest.model_name)
			tokenizer = AutoTokenizer.from_pretrained(self.manifest.model_name)
			args = SimpleNamespace(model_name=self.manifest.model_name)
			model = RobertaRetriever(config, args)
			model = load_saved(model, self.manifest.checkpoint_path, exact=False)
			if torch.cuda.is_available():
				device = torch.device("cuda")
			elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				device = torch.device("mps")
			else:
				device = torch.device("cpu")
			model.to(device)
			model.eval()

		self._device = device
		self._model = model
		self._tokenizer = tokenizer
		return model, tokenizer, torch

	def _search(
		self, query_vectors: np.ndarray, topk: int
	) -> tuple[np.ndarray, np.ndarray]:
		k = max(1, min(int(topk), len(self.docs)))
		vectors = np.asarray(query_vectors, dtype="float32")
		if self.index is not None:
			scores, indices = self.index.search(vectors, k)
			return scores.astype("float32"), indices.astype("int64")

		score_matrix = vectors @ self.embeddings.T
		if k >= score_matrix.shape[1]:
			indices = np.argsort(score_matrix, axis=1)[:, ::-1][:, :k]
		else:
			partitions = np.argpartition(score_matrix, -k, axis=1)[:, -k:]
			partition_scores = np.take_along_axis(score_matrix, partitions, axis=1)
			order = np.argsort(partition_scores, axis=1)[:, ::-1]
			indices = np.take_along_axis(partitions, order, axis=1)
		scores = np.take_along_axis(score_matrix, indices, axis=1)
		return scores.astype("float32"), indices.astype("int64")


def _validate_and_patch_mdr_checkout(
	mdr_home: str | Path | None,
) -> tuple[Path, str, str]:
	resolved_home = _resolve_mdr_home(mdr_home)
	for relative_path in _REQUIRED_MDR_PATHS:
		candidate = resolved_home / relative_path
		if not candidate.exists():
			raise ValueError(
				f"Official MDR checkout is missing required path: {candidate}"
			)

	commit = _git_head_commit(resolved_home)
	if commit != MDR_PINNED_COMMIT:
		raise ValueError(
			f"Official MDR checkout must be pinned to {MDR_PINNED_COMMIT}, but {resolved_home} is at {commit}."
		)

	dataset_path = resolved_home / "mdr" / "retrieval" / "data" / "mhop_dataset.py"
	source = dataset_path.read_text(encoding="utf-8")
	patched, replacements = re.subn(
		r"(?m)^[ \t]*import pdb; pdb\.set_trace\(\)\n?", "", source
	)
	if replacements:
		dataset_path.write_text(patched, encoding="utf-8")
	elif "pdb.set_trace()" in source:
		raise ValueError(f"Unexpected debugger statement shape in {dataset_path}")
	return resolved_home, commit, _PATCH_SHA


def _resolve_mdr_home(mdr_home: str | Path | None) -> Path:
	raw = mdr_home or os.environ.get(_MDR_HOME_ENV)
	path = _default_repo_mdr_home() if raw is None else Path(raw).resolve()
	if path is None:
		raise ValueError(
			f"MDR checkout path is required. Pass --mdr-home, set {_MDR_HOME_ENV}, or add the pinned checkout at ./baselines/mdr."
		)
	if not path.exists():
		raise FileNotFoundError(f"MDR checkout does not exist: {path}")
	return path


def _git_head_commit(repo_root: Path) -> str:
	completed = subprocess.run(
		["git", "-C", str(repo_root), "rev-parse", "HEAD"],
		check=True,
		capture_output=True,
		text=True,
	)
	return completed.stdout.strip()


def _run_official_mdr_command(command: list[str], *, cwd: Path) -> None:
	env = dict(os.environ)
	existing_pythonpath = env.get("PYTHONPATH")
	env["PYTHONPATH"] = (
		str(cwd)
		if not existing_pythonpath
		else f"{cwd}{os.pathsep}{existing_pythonpath}"
	)
	subprocess.run(command, cwd=cwd, env=env, check=True)


def _discover_checkpoint(output_root: Path) -> Path:
	candidates = sorted(output_root.rglob("checkpoint_best.pt"))
	if not candidates:
		candidates = sorted(output_root.rglob("checkpoint_last.pt"))
	if not candidates:
		raise FileNotFoundError(
			f"No checkpoint_best.pt or checkpoint_last.pt found under {output_root}"
		)
	candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
	return candidates[0]


def _maybe_build_faiss_index(
	*, output_root: Path, embeddings: np.ndarray
) -> Path | None:
	try:
		import faiss  # type: ignore[import-not-found]
	except ModuleNotFoundError:
		return None

	if embeddings.ndim != 2:
		raise ValueError(
			f"Expected a 2D embedding matrix, got shape {embeddings.shape}"
		)
	index = faiss.IndexFlatIP(int(embeddings.shape[1]))
	index.add(embeddings.astype("float32"))
	index_path = output_root / "corpus_vectors.faiss"
	faiss.write_index(index, str(index_path))
	return index_path


def _load_faiss_index(path: str | None) -> Any | None:
	if path is None:
		return None
	try:
		import faiss  # type: ignore[import-not-found]
	except ModuleNotFoundError:
		return None
	resolved = Path(path)
	if not resolved.exists():
		return None
	return faiss.read_index(str(resolved))


def _build_training_sample(
	store: ShardedDocumentStore,
	case: Any,
	*,
	node_ids: list[str],
	analysis: MDRCaseSupportAnalysis | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
	support = analysis or _analyze_case_support(store, case, node_set=set(node_ids))
	if support.start_doc is None:
		return None, "missing_start"

	if support.second_doc is None:
		return None, "no_bridge"

	negatives = _select_negative_docs(
		store,
		case,
		node_ids=node_ids,
		excluded={
			*{
				str(node_id)
				for node_id in (getattr(case, "gold_support_nodes", ()) or ())
			},
			support.start_doc,
			support.second_doc,
		},
		start_doc=support.start_doc,
	)
	if len(negatives) < 2:
		return None, "no_negatives"

	return {
		"_id": str(case.case_id),
		"question": str(case.query),
		"type": support.question_type,
		"bridge": support.second_doc,
		"pos_paras": [
			_doc_payload(store, support.start_doc),
			_doc_payload(store, support.second_doc),
		],
		"neg_paras": [_doc_payload(store, node_id) for node_id in negatives[:2]],
		"sp": [support.start_doc, support.second_doc],
	}, None


def _build_eval_sample(store: ShardedDocumentStore, case: Any) -> dict[str, Any]:
	analysis = _analyze_case_support(store, case, node_set=set(store.nodes))
	sp = [
		node for node in (analysis.start_doc, analysis.second_doc) if node is not None
	]
	payload = {
		"_id": str(case.case_id),
		"question": str(case.query),
		"type": analysis.question_type,
		"sp": sp,
	}
	expected_answer = getattr(case, "expected_answer", None)
	if expected_answer:
		payload["answer"] = [str(expected_answer)]
	return payload


def _select_start_doc(case: Any, node_set: set[str]) -> str | None:
	for node_id in list(getattr(case, "gold_start_nodes", ()) or ()) + list(
		getattr(case, "gold_support_nodes", ()) or ()
	):
		candidate = str(node_id)
		if candidate in node_set:
			return candidate
	return None


def _select_bridge_doc(
	store: ShardedDocumentStore,
	case: Any,
	*,
	start_doc: str | None,
	node_set: set[str],
) -> str | None:
	analysis = _analyze_case_support(
		store, case, node_set=node_set, start_doc=start_doc
	)
	return analysis.second_doc


def _analyze_case_support(
	store: ShardedDocumentStore,
	case: Any,
	*,
	node_set: set[str],
	start_doc: str | None = None,
) -> MDRCaseSupportAnalysis:
	resolved_start = (
		start_doc if start_doc is not None else _select_start_doc(case, node_set)
	)
	support_nodes = _ordered_unique(
		str(node_id)
		for node_id in (getattr(case, "gold_support_nodes", ()) or ())
		if str(node_id) in node_set
	)
	support_candidates = [
		node_id for node_id in support_nodes if node_id != resolved_start
	]
	path_candidates = _ordered_unique(
		str(node_id)
		for node_id in list(getattr(case, "gold_path_nodes", ()) or ())[1:]
		if str(node_id) != resolved_start and str(node_id) in node_set
	)

	candidate_bridge_docs = _ordered_unique(
		[
			*path_candidates,
			*_rank_support_candidates(store, resolved_start, support_candidates),
		]
	)
	distinct_support_ge_2 = len(support_nodes) >= 2
	question_type = (
		"bridge"
		if path_candidates
		else ("comparison" if distinct_support_ge_2 else "bridge")
	)
	second_doc = candidate_bridge_docs[0] if candidate_bridge_docs else None
	return MDRCaseSupportAnalysis(
		start_doc=resolved_start,
		second_doc=second_doc,
		question_type=question_type,
		support_nodes=support_nodes,
		candidate_bridge_docs=candidate_bridge_docs,
		distinct_support_ge_2=distinct_support_ge_2,
	)


def _rank_support_candidates(
	store: ShardedDocumentStore,
	start_doc: str | None,
	support_candidates: list[str],
) -> list[str]:
	if start_doc is None or not support_candidates:
		return list(support_candidates)

	direct_neighbors = list(store.neighbors(start_doc))
	direct_neighbor_set = set(direct_neighbors)
	direct_hits = [
		node_id for node_id in support_candidates if node_id in direct_neighbor_set
	]

	second_hop_set: set[str] = set()
	for neighbor in direct_neighbors:
		second_hop_set.update(store.neighbors(neighbor))
	second_hop_hits = [
		node_id
		for node_id in support_candidates
		if node_id not in direct_neighbor_set and node_id in second_hop_set
	]
	remaining_hits = [
		node_id
		for node_id in support_candidates
		if node_id not in direct_neighbor_set and node_id not in second_hop_set
	]
	return [*direct_hits, *second_hop_hits, *remaining_hits]


def _ordered_unique(values: list[str] | tuple[str, ...] | Any) -> list[str]:
	ordered: list[str] = []
	seen: set[str] = set()
	for value in values:
		candidate = str(value)
		if candidate in seen:
			continue
		ordered.append(candidate)
		seen.add(candidate)
	return ordered


def _build_dropped_case_record(
	case: Any,
	analysis: MDRCaseSupportAnalysis,
	drop_reason: str | None,
) -> dict[str, Any]:
	raw_support_nodes = _ordered_unique(
		str(node_id) for node_id in (getattr(case, "gold_support_nodes", ()) or ())
	)
	return {
		"case_id": str(case.case_id),
		"drop_reason": drop_reason,
		"support_nodes": raw_support_nodes,
		"support_nodes_in_store": list(analysis.support_nodes),
		"start_doc": analysis.start_doc,
		"candidate_bridge_docs": list(analysis.candidate_bridge_docs),
		"question_type_guess": _guess_question_type(case),
	}


def _increment_count(counter: dict[str, int], key: str) -> None:
	counter[key] = counter.get(key, 0) + 1


def _guess_question_type(case: Any) -> str:
	raw_path_candidates = _ordered_unique(
		str(node_id) for node_id in list(getattr(case, "gold_path_nodes", ()) or ())[1:]
	)
	if raw_path_candidates:
		return "bridge"
	raw_support_nodes = _ordered_unique(
		str(node_id) for node_id in (getattr(case, "gold_support_nodes", ()) or ())
	)
	if len(raw_support_nodes) >= 2:
		return "comparison"
	return "bridge"


def _select_negative_docs(
	store: ShardedDocumentStore,
	case: Any,
	*,
	node_ids: list[str],
	excluded: set[str],
	start_doc: str,
) -> list[str]:
	negatives: list[str] = []
	seen: set[str] = set()
	retrieval_k = min(len(node_ids), max(16, len(excluded) + 8))
	for node_id, _score in store.topk_similar(case.query, node_ids, k=retrieval_k):
		if node_id in excluded or node_id in seen:
			continue
		negatives.append(node_id)
		seen.add(node_id)
		if len(negatives) >= 2:
			return negatives

	for neighbor in store.neighbors(start_doc):
		if neighbor in excluded or neighbor in seen:
			continue
		negatives.append(neighbor)
		seen.add(neighbor)
		if len(negatives) >= 2:
			return negatives

	for node_id in node_ids:
		if node_id in excluded or node_id in seen:
			continue
		negatives.append(node_id)
		seen.add(node_id)
		if len(negatives) >= 2:
			return negatives
	return negatives


def _corpus_record(store: ShardedDocumentStore, node_id: str) -> dict[str, Any]:
	document = store.get_document(node_id)
	text = "" if document is None else document.text
	return {
		"title": node_id,
		"text": text,
	}


def _doc_payload(store: ShardedDocumentStore, node_id: str) -> dict[str, Any]:
	document = store.get_document(node_id)
	return {
		"title": node_id,
		"text": "" if document is None else document.text,
	}


def _is_val_case(case_id: str) -> bool:
	digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
	return int(digest, 16) % 10 == 0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
	with path.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, manifest: Any) -> None:
	path.write_text(
		json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
	)


def _load_id2doc_entries(path: str | Path) -> list[dict[str, Any]]:
	payload = json.loads(Path(path).read_text(encoding="utf-8"))
	items = sorted(payload.items(), key=lambda item: int(item[0]))
	docs: list[dict[str, Any]] = []
	for _index, value in items:
		if isinstance(value, dict):
			title = str(value.get("title", ""))
			text = str(value.get("text", ""))
			intro = bool(value.get("intro", False))
		elif isinstance(value, list | tuple):
			raw = list(value)
			title = str(raw[0]) if raw else ""
			text = str(raw[1]) if len(raw) > 1 else ""
			intro = bool(raw[2]) if len(raw) > 2 else False
		else:
			raise ValueError(f"Unsupported id2doc entry: {value!r}")
		docs.append({"title": title, "text": text, "intro": intro})
	return docs


def _move_batch_to_device(batch: Any, device: Any, torch: Any) -> dict[str, Any]:
	moved: dict[str, Any] = {}
	for key, value in dict(batch).items():
		if torch.is_tensor(value):
			moved[key] = value.to(device)
		else:
			moved[key] = value
	return moved


def _dedupe(values: list[str]) -> list[str]:
	seen: set[str] = set()
	ordered: list[str] = []
	for value in values:
		if value in seen:
			continue
		seen.add(value)
		ordered.append(value)
	return ordered


@contextmanager
def _temporary_sys_path(path: Path):
	path_str = str(path)
	sys.path.insert(0, path_str)
	try:
		yield
	finally:
		if path_str in sys.path:
			sys.path.remove(path_str)
