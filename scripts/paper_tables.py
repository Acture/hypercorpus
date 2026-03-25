#!/usr/bin/env python3
"""Generate paper-facing CSVs from merged experiment results.

Reads merged CSVs produced by `hypercorpus experiments merge-iirc-results`
and outputs filtered/pivoted tables matching paper/tables-and-figures.md.

Usage:
    uv run python scripts/paper_tables.py --run-dir runs/iirc-shortlist-full-v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Selector display names — map long internal names to short paper labels
# ---------------------------------------------------------------------------
SELECTOR_SHORT_NAMES: dict[str, str] = {
	"top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop": "Dense",
	"top_1_seed__sentence_transformer__hop_2__mdr_light__budget_fill_relative_drop": "MDR-light",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop": "SPW-ST",
	"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_llm_controller__lookahead_2": "SPW-LLM",
	"top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2": "CMP-LLM",
	"external_mdr": "MDR",
	"gold_support_context": "Oracle",
}

TABLE1_SELECTORS = list(SELECTOR_SHORT_NAMES.keys())
TABLE1_BUDGETS = {"tokens-384", "tokens-512"}

TABLE1_COLUMNS = [
	"selector_name",
	"budget_label",
	"avg_support_f1",
	"avg_support_set_em",
	"avg_support_precision",
	"avg_support_recall",
	"avg_path_hit",
]

FIGURE_SUMMARY_COLUMNS = [
	"selector",
	"budget_value",
	"support_f1_zero_on_empty",
	"support_precision",
	"support_recall",
	"budget_utilization",
	"selection_runtime_s",
	"selector_total_tokens",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline="", encoding="utf-8") as f:
		return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
		w.writeheader()
		w.writerows(rows)
	print(f"  wrote {path} ({len(rows)} rows)")


def _add_short_name(row: dict[str, str], key: str = "selector_name") -> dict[str, str]:
	row = dict(row)
	row["short_name"] = SELECTOR_SHORT_NAMES.get(row.get(key, ""), row.get(key, ""))
	return row


def generate_table1(run_dir: Path, out_dir: Path) -> None:
	"""Table 1: main comparison from study_comparison_rows.csv."""
	src = run_dir / "study_comparison_rows.csv"
	if not src.exists():
		print(f"  SKIP table1: {src} not found")
		return
	rows = _read_csv(src)
	filtered = [
		_add_short_name(r)
		for r in rows
		if r["selector_name"] in TABLE1_SELECTORS
		and r["budget_label"] in TABLE1_BUDGETS
	]
	fields = ["short_name"] + TABLE1_COLUMNS
	_write_csv(out_dir / "table1_main_comparison.csv", filtered, fields)


def generate_table2(run_dir: Path, out_dir: Path) -> None:
	"""Table 2: hard-subset comparison from subset_comparison_rows.csv."""
	src = run_dir / "subset_comparison_rows.csv"
	if not src.exists():
		print(f"  SKIP table2: {src} not found")
		return
	rows = _read_csv(src)
	target_subsets = {"bridge", "comparison"}
	filtered = [
		_add_short_name(r)
		for r in rows
		if r.get("subset_label", "") in target_subsets
		and r["selector_name"] in TABLE1_SELECTORS
		and r["budget_label"] in TABLE1_BUDGETS
	]
	fields = (
		["short_name", "subset_label"]
		+ TABLE1_COLUMNS
		+ [
			"delta_support_f1_vs_dense_control",
		]
	)
	_write_csv(out_dir / "table2_hard_subset.csv", filtered, fields)


def generate_figure_csvs(run_dir: Path, out_dir: Path) -> None:
	"""Figure 2 (cost-quality) and Figure 3 (budget sensitivity) from summary_rows.csv."""
	src = run_dir / "summary_rows.csv"
	if not src.exists():
		print(f"  SKIP figures: {src} not found")
		return
	rows = _read_csv(src)
	enriched = [_add_short_name(r, key="selector") for r in rows]

	# Figure 2: all selectors × all budgets — cost vs quality
	fig2_fields = ["short_name"] + FIGURE_SUMMARY_COLUMNS
	_write_csv(out_dir / "figure2_cost_quality.csv", enriched, fig2_fields)

	# Figure 3: budget sensitivity for key selectors
	key_selectors = {
		"top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop",
		"top_1_seed__sentence_transformer__hop_2__single_path_walk__link_context_sentence_transformer__lookahead_2__profile_st_future_heavy__budget_fill_relative_drop",
		"top_1_seed__sentence_transformer__hop_2__constrained_multipath__link_context_llm_controller__lookahead_2",
	}
	fig3 = [r for r in enriched if r.get("selector", "") in key_selectors]
	fig3_fields = [
		"short_name",
		"selector",
		"budget_value",
		"support_f1_zero_on_empty",
		"support_precision",
		"support_recall",
	]
	_write_csv(out_dir / "figure3_budget_sensitivity.csv", fig3, fig3_fields)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument(
		"--run-dir",
		type=Path,
		required=True,
		help="Merged run directory (e.g. runs/iirc-shortlist-full-v1)",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path("paper/data"),
		help="Output directory for paper CSVs",
	)
	args = parser.parse_args()

	if not args.run_dir.is_dir():
		print(f"ERROR: {args.run_dir} is not a directory", file=sys.stderr)
		sys.exit(1)

	print(f"Generating paper tables from {args.run_dir} → {args.out_dir}")
	generate_table1(args.run_dir, args.out_dir)
	generate_table2(args.run_dir, args.out_dir)
	generate_figure_csvs(args.run_dir, args.out_dir)
	print("Done.")


if __name__ == "__main__":
	main()
