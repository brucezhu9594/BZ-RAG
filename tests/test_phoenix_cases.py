from evaluation.phoenix.cases import CASES, CASE_IDS, SMOKE_IDS, example_id


def test_example_id_is_stable_and_16_hex():
    a = example_id("禾蛙平台是什么类型的平台？")
    b = example_id("禾蛙平台是什么类型的平台？")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_example_id_ignores_surrounding_whitespace():
    assert example_id(" 同一个问题 ") == example_id("同一个问题")


def test_example_id_differs_for_different_questions():
    assert example_id("问题一") != example_id("问题二")


def test_cases_loaded_and_ids_unique():
    assert len(CASES) >= 20
    assert len(CASE_IDS) == len(CASES)
    assert len(set(CASE_IDS)) == len(CASE_IDS)
    assert CASE_IDS == [c.example_id for c in CASES]


def test_smoke_subset_is_small_and_is_a_subset():
    assert 0 < len(SMOKE_IDS) < len(CASES)
    assert SMOKE_IDS <= set(CASE_IDS)
