from hypercorpus.walker import DynamicWalker, StopReason, WalkBudget


def test_dynamic_walker_follows_best_scoring_path(sample_graph):
	walk = DynamicWalker(sample_graph).walk(
		"Which city hosts the launch site?",
		start_nodes=["mission", "director"],
		budget=WalkBudget(max_steps=3, min_score=0.05),
	)
	assert walk.visited_nodes == ["mission", "cape", "florida"]
	assert walk.steps[1].anchor_text == "Cape Canaveral"
	assert walk.stop_reason == StopReason.BUDGET_EXHAUSTED


def test_dynamic_walker_avoids_cycles(cyclic_graph):
	walk = DynamicWalker(cyclic_graph).walk(
		"Where is the final target?",
		start_nodes=["a"],
		budget=WalkBudget(max_steps=4, min_score=0.01),
	)
	assert walk.visited_nodes == ["a", "b", "c"]
	assert walk.stop_reason == StopReason.DEAD_END


def test_dynamic_walker_stops_when_score_is_too_low(sample_graph):
	walk = DynamicWalker(sample_graph).walk(
		"Unrelated astronomy question",
		start_nodes=["mission"],
		budget=WalkBudget(max_steps=3, min_score=0.9),
	)
	assert walk.visited_nodes == ["mission"]
	assert walk.stop_reason == StopReason.SCORE_BELOW_THRESHOLD
