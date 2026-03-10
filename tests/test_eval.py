from webwalker.eval import EvaluationCase, Evaluator


def test_evaluator_runs_all_pipelines(sample_graph):
	case = EvaluationCase(
		case_id="launch-site",
		query="Which city hosts the launch site?",
		expected_answer="Cape Canaveral",
		supporting_node_ids=["mission", "cape"],
	)
	evaluation = Evaluator().evaluate_case(sample_graph, case)
	results = {result.name: result for result in evaluation.results}

	assert set(results) == {
		"baseline_graphrag",
		"dense_rag",
		"webwalker",
		"semantic_beam",
		"semantic_astar",
		"semantic_gbfs",
		"semantic_ucs",
		"semantic_ppr",
		"semantic_beam_ppr",
		"semantic_astar_ppr",
		"semantic_gbfs_ppr",
		"semantic_ucs_ppr",
	}
	assert results["webwalker"].correct is True
	assert results["webwalker"].walk is not None
	assert all(result.metrics.visited_steps >= 1 for result in evaluation.results)
	assert results["semantic_beam"].selection is not None
	assert results["semantic_beam"].metrics.support_recall is not None


def test_case_evaluation_selection_report_sorts_by_recall_then_cost(sample_graph):
	case = EvaluationCase(
		case_id="launch-site",
		query="Which city hosts the launch site?",
		expected_answer="Cape Canaveral",
		supporting_node_ids=["mission", "cape"],
	)
	evaluation = Evaluator().evaluate_case(sample_graph, case)
	report = evaluation.selection_report()

	assert report.case_id == "launch-site"
	assert report.rows
	assert report.rows[0].support_recall is not None
	assert report.rows[0].support_recall >= report.rows[-1].support_recall
	for earlier, later in zip(report.rows, report.rows[1:], strict=False):
		if earlier.support_recall == later.support_recall:
			assert earlier.token_cost_estimate <= later.token_cost_estimate
