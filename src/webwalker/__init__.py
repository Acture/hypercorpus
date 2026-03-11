from webwalker.eval import EvaluationBudget, EvaluationCase, Evaluator
from webwalker.selector import SelectorBudget, available_selector_names, build_selector, parse_selector_spec

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
    print("webwalker exposes canonical selectors via webwalker.selector and evaluation via webwalker.eval.")
