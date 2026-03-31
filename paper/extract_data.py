"""Extract paper table data from experiment runs into curated CSVs.

Usage:
    uv run python paper/extract_data.py

Reads from runs/ directories and writes to paper/data/*.csv.
These CSVs are the single source of truth for paper tables.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "paper" / "data"


METRIC_COLS = [
	"support_recall",
	"support_precision",
	"support_f1",
	"support_f1_zero_on_empty",
	"support_set_em",
	"budget_utilization",
	"budget_adherence",
	"empty_selection_rate",
	"selected_nodes",
	"selected_corpus_mass",
	"selection_runtime_s",
	"selector_total_tokens",
]

# Columns that identify a unique selector+budget combination.
KEY_COLS = [
	"dataset",
	"selector",
	"selector_family",
	"selector_provider",
	"selector_model",
	"seed_strategy",
	"seed_top_k",
	"hop_budget",
	"baseline",
	"search_structure",
	"edge_scorer",
	"lookahead_depth",
	"profile_name",
	"budget_fill_mode",
	"budget_mode",
	"budget_value",
	"budget_label",
	"selector_budget_tokens",
	"selector_budget_ratio",
]

LEGACY_ROW_ALIASES = {
	"selector_budget_tokens": "token_budget_tokens",
	"selector_budget_ratio": "token_budget_ratio",
	"selected_corpus_mass": "selected_token_estimate",
}


def _normalize_summary_row(row: dict[str, str]) -> dict[str, str]:
	normalized = dict(row)
	for canonical_key, legacy_key in LEGACY_ROW_ALIASES.items():
		if normalized.get(canonical_key, "") == "" and normalized.get(legacy_key, "") != "":
			normalized[canonical_key] = normalized[legacy_key]
	return normalized


def _weighted_avg(rows: list[dict]) -> dict:
	"""Compute weighted average of metric columns across chunk rows."""
	total_n = sum(int(r["num_cases"]) for r in rows)
	if total_n == 0:
		return rows[0]
	merged = dict(rows[0])  # copy non-metric fields from first row
	merged["num_cases"] = str(total_n)
	for col in METRIC_COLS:
		vals = []
		for r in rows:
			v = r.get(col, "")
			if v != "":
				vals.append((int(r["num_cases"]), float(v)))
		if vals:
			merged[col] = str(sum(n * v for n, v in vals) / sum(n for n, _ in vals))
		else:
			merged[col] = ""
	return merged


def read_summary_rows(run_dir: Path) -> list[dict]:
	"""Read summary_rows.csv from a run.

	If the run has a top-level CSV (already aggregated), use it directly.
	If only chunk-level CSVs exist, aggregate by weighted average on num_cases.
	"""
	top = run_dir / "summary_rows.csv"
	if top.exists():
		with open(top) as f:
			return [_normalize_summary_row(row) for row in csv.DictReader(f)]

	# Chunked format: read all chunks, then aggregate
	raw_rows: list[dict] = []
	chunks_dir = run_dir / "chunks"
	if chunks_dir.exists():
		for chunk in sorted(chunks_dir.iterdir()):
			csv_path = chunk / "summary_rows.csv"
			if csv_path.exists():
				with open(csv_path) as f:
					raw_rows.extend(_normalize_summary_row(row) for row in csv.DictReader(f))

	if not raw_rows:
		return []

	# Group by key columns, then weighted-average metrics
	from collections import defaultdict

	groups: dict[tuple, list[dict]] = defaultdict(list)
	for r in raw_rows:
		key = tuple(r.get(c, "") for c in KEY_COLS)
		groups[key].append(r)

	aggregated = []
	for key, chunk_rows in groups.items():
		total_n = sum(int(r["num_cases"]) for r in chunk_rows)
		merged = _weighted_avg(chunk_rows)
		print(
			f"    Aggregated {len(chunk_rows)} chunks, n={total_n}: "
			f"{merged.get('baseline', merged.get('edge_scorer', '?'))}"
		)
		aggregated.append(merged)

	return aggregated


def extract_table3():
	"""Table 3: 2Wiki calibration baseline ordering (budget=256)."""
	ablation_rows = read_summary_rows(RUNS / "2wiki-single-path-edge-ablation-s100-v1")
	baseline_rows = read_summary_rows(RUNS / "2wiki-baseline-retest-s100-v1")

	# Define which selectors go into Table 3 and their display names
	table_spec = [
		# (source, match_fn, display_name)
		(
			"ablation",
			lambda r: r.get("selector_family") == "diagnostic"
			and "gold_support" in r.get("selector", ""),
			"Oracle",
		),
		(
			"ablation",
			lambda r: r.get("edge_scorer") == "link_context_sentence_transformer"
			and r.get("profile_name") == "st_future_heavy"
			and r.get("lookahead_depth") == "2",
			"Single-Path Walk (ST, la2)",
		),
		(
			"baseline",
			lambda r: r.get("baseline") == "dense" and r.get("seed_top_k") == "1",
			"Dense (top-1)",
		),
		(
			"baseline",
			lambda r: r.get("baseline") == "iterative_dense"
			and r.get("seed_top_k") == "3",
			"Iter. Dense (top-3)",
		),
		(
			"baseline",
			lambda r: r.get("baseline") == "mdr_light" and r.get("seed_top_k") == "3",
			"MDR-Light (top-3)",
		),
		(
			"baseline",
			lambda r: r.get("baseline") == "dense" and r.get("seed_top_k") == "3",
			"Dense (top-3)",
		),
		(
			"ablation",
			lambda r: r.get("selector_family") == "diagnostic"
			and "full_corpus" in r.get("selector", ""),
			"Full Corpus",
		),
	]

	sources = {"ablation": ablation_rows, "baseline": baseline_rows}
	budget = "256"

	out_rows = []
	for source_key, match_fn, display_name in table_spec:
		candidates = [
			r
			for r in sources[source_key]
			if match_fn(r) and r.get("budget_value") == budget
		]
		if not candidates:
			# Try matching without budget filter and report
			print(f"  WARNING: No match for '{display_name}' at budget {budget}")
			out_rows.append(
				{
					"selector": display_name,
					"support_f1": "",
					"support_precision": "",
					"support_recall": "",
					"budget_utilization": "",
				}
			)
			continue

		row = candidates[0]
		out_rows.append(
			{
				"selector": display_name,
				"support_f1": row.get("support_f1_zero_on_empty", ""),
				"support_precision": row.get("support_precision", ""),
				"support_recall": row.get("support_recall", ""),
				"budget_utilization": row.get("budget_utilization", ""),
			}
		)

	out_path = DATA / "table3-2wiki-baseline.csv"
	with open(out_path, "w", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"selector",
				"support_f1",
				"support_precision",
				"support_recall",
				"budget_utilization",
			],
		)
		writer.writeheader()
		writer.writerows(out_rows)
	print(f"Wrote {out_path} ({len(out_rows)} rows)")


def extract_table6():
	"""Table 6: Edge-scorer ablation (budget=128)."""
	rows = read_summary_rows(RUNS / "2wiki-single-path-edge-ablation-s100-v1")

	# Filter: path_search family, budget=128
	budget = "128"
	filtered = [
		r
		for r in rows
		if r.get("selector_family") == "path_search" and r.get("budget_value") == budget
	]

	# Spec: 6 rows matching tables-and-figures.md
	spec = [
		("link_context_sentence_transformer", "st_future_heavy", "2"),
		("link_context_sentence_transformer", "st_balanced", "2"),
		("link_context_sentence_transformer", "st_direct_heavy", "2"),
		("link_context_sentence_transformer", "st_balanced", "1"),
		("link_context_overlap", "overlap_balanced", "1"),
		("anchor_overlap", "", "1"),
	]

	scorer_display = {
		"link_context_sentence_transformer": "ST embedding",
		"link_context_overlap": "Overlap (ctx)",
		"anchor_overlap": "Anchor overlap",
	}

	out_rows = []
	for edge_scorer, profile, la in spec:
		candidates = [
			r
			for r in filtered
			if r.get("edge_scorer") == edge_scorer
			and r.get("lookahead_depth") == la
			and (not profile or r.get("profile_name", "") == profile)
		]
		if not candidates:
			print(f"  WARNING: No match for {edge_scorer}/{profile}/la{la}")
			continue

		row = candidates[0]
		out_rows.append(
			{
				"edge_scorer": scorer_display.get(edge_scorer, edge_scorer),
				"profile": profile or "---",
				"lookahead": la,
				"support_f1": row.get("support_f1_zero_on_empty", ""),
				"support_recall": row.get("support_recall", ""),
				"support_precision": row.get("support_precision", ""),
			}
		)

	out_path = DATA / "table6-edge-scorer-ablation.csv"
	with open(out_path, "w", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"edge_scorer",
				"profile",
				"lookahead",
				"support_f1",
				"support_recall",
				"support_precision",
			],
		)
		writer.writeheader()
		writer.writerows(out_rows)
	print(f"Wrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
	DATA.mkdir(parents=True, exist_ok=True)
	print("Extracting Table 3 (2Wiki baseline)...")
	extract_table3()
	print()
	print("Extracting Table 6 (edge-scorer ablation)...")
	extract_table6()
