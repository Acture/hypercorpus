from webwalker.eval import EvaluationCase, Evaluator


def test_evaluator_runs_all_pipelines(sample_graph):
	case = EvaluationCase(
		case_id="launch-site",
		query="Which city hosts the launch site?",
		expected_answer="Cape Canaveral",
	)
	evaluation = Evaluator().evaluate_case(sample_graph, case)
	results = {result.name: result for result in evaluation.results}

	assert set(results) == {"baseline_graphrag", "dense_rag", "webwalker"}
	assert results["webwalker"].correct is True
	assert results["webwalker"].walk is not None
	assert all(result.metrics.visited_steps >= 1 for result in evaluation.results)
