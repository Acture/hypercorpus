from hypercorpus.eval import EvaluationBudget, EvaluationCase, Evaluator
from hypercorpus.selector import (
	SelectorBudget,
	available_selector_names,
	build_selector,
	parse_selector_spec,
)

__all__ = [
	"EvaluationBudget",
	"EvaluationCase",
	"Evaluator",
	"SelectorBudget",
	"available_selector_names",
	"build_selector",
	"parse_selector_spec",
]


def main() -> None:
	print(
		"hypercorpus exposes canonical selectors via hypercorpus.selector and evaluation via hypercorpus.eval."
	)
