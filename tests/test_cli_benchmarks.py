from typer.testing import CliRunner

from hypercorpus_cli import app


CANONICAL_DENSE = "top_1_seed__lexical_overlap__hop_0__dense"


def test_run_musique_cli_smoke(musique_files, tmp_path):
	questions_path, graph_path = musique_files
	output_dir = tmp_path / "musique-cli-out"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-musique",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "musique summary" in result.stdout


def test_run_hotpotqa_distractor_cli_smoke(hotpotqa_distractor_file, tmp_path):
	output_dir = tmp_path / "hotpot-distractor-cli-out"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-hotpotqa",
			"--questions",
			str(hotpotqa_distractor_file),
			"--output",
			str(output_dir),
			"--variant",
			"distractor",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "hotpotqa-distractor summary" in result.stdout


def test_run_hotpotqa_fullwiki_cli_requires_graph_records(
	hotpotqa_fullwiki_files, tmp_path
):
	questions_path, _graph_path = hotpotqa_fullwiki_files
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-hotpotqa",
			"--questions",
			str(questions_path),
			"--output",
			str(tmp_path / "hotpot-fullwiki-cli-out"),
			"--variant",
			"fullwiki",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code != 0


def test_run_hotpotqa_fullwiki_cli_smoke(hotpotqa_fullwiki_files, tmp_path):
	questions_path, graph_path = hotpotqa_fullwiki_files
	output_dir = tmp_path / "hotpot-fullwiki-cli-out"
	runner = CliRunner()

	result = runner.invoke(
		app,
		[
			"experiments",
			"run-hotpotqa",
			"--questions",
			str(questions_path),
			"--graph-records",
			str(graph_path),
			"--output",
			str(output_dir),
			"--variant",
			"fullwiki",
			"--selectors",
			CANONICAL_DENSE,
			"--token-budgets",
			"128",
			"--no-e2e",
			"--no-export-graphrag-inputs",
		],
	)

	assert result.exit_code == 0, result.stdout
	assert "hotpotqa-fullwiki summary" in result.stdout


def test_run_store_cli_commands_smoke(
	prepared_iirc_store, prepared_musique_store, prepared_hotpotqa_store, tmp_path
):
	runner = CliRunner()
	commands = [
		("run-iirc-store", prepared_iirc_store.root, "iirc"),
		("run-musique-store", prepared_musique_store.root, "musique"),
		("run-hotpotqa-store", prepared_hotpotqa_store.root, "hotpot"),
	]
	for command_name, store_root, exp_name in commands:
		result = runner.invoke(
			app,
			[
				"experiments",
				command_name,
				"--store",
				str(store_root),
				"--exp-name",
				exp_name,
				"--output-root",
				str(tmp_path / "runs"),
				"--selectors",
				CANONICAL_DENSE,
				"--token-budgets",
				"128",
				"--no-e2e",
				"--no-export-graphrag-inputs",
			],
		)
		assert result.exit_code == 0, result.stdout
		assert "chunk_dir" in result.stdout


def test_merge_new_store_results_cli_smoke(
	prepared_iirc_store, prepared_musique_store, prepared_hotpotqa_store, tmp_path
):
	runner = CliRunner()
	command_specs = [
		("run-iirc-store", prepared_iirc_store.root, "iirc", "merge-iirc-results"),
		(
			"run-musique-store",
			prepared_musique_store.root,
			"musique",
			"merge-musique-results",
		),
		(
			"run-hotpotqa-store",
			prepared_hotpotqa_store.root,
			"hotpot",
			"merge-hotpotqa-results",
		),
	]
	for run_command, store_root, exp_name, merge_command in command_specs:
		run_result = runner.invoke(
			app,
			[
				"experiments",
				run_command,
				"--store",
				str(store_root),
				"--exp-name",
				exp_name,
				"--output-root",
				str(tmp_path / "runs"),
				"--selectors",
				CANONICAL_DENSE,
				"--token-budgets",
				"128",
				"--no-e2e",
				"--no-export-graphrag-inputs",
			],
		)
		assert run_result.exit_code == 0, run_result.stdout
		merge_result = runner.invoke(
			app,
			[
				"experiments",
				merge_command,
				"--run-dir",
				str(tmp_path / "runs" / exp_name),
			],
		)
		assert merge_result.exit_code == 0, merge_result.stdout
		assert "merged summary.json" in merge_result.stdout
