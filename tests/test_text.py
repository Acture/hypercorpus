from webwalker.text import answer_f1


def test_answer_f1_scores_partial_overlap():
    assert answer_f1("Cape Canaveral, Florida", "Cape Canaveral") == 0.8


def test_answer_f1_handles_missing_gold_answer():
    assert answer_f1("anything", None) is None
