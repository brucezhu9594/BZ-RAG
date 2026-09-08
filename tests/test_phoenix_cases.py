from evaluation.phoenix.cases import CASE_IDS, CASES, SMOKE_IDS, example_id, smoke_ids


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


def test_smoke_subset_is_invariant_to_input_order():
    """smoke 子集必须只依赖内容，不依赖 test_dataset.json 的行序。

    两个顺序都喂给生产路径上那个 smoke_ids()，不在测试里重实现选取规则——
    否则测试和实现会各自演化、迟早跑偏而测试毫无察觉。
    """
    import random

    shuffled = CASES[:]
    random.Random(0).shuffle(shuffled)
    assert [c.example_id for c in shuffled] != CASE_IDS  # 确认真的打乱了
    assert smoke_ids(shuffled) == smoke_ids(CASES)
    assert smoke_ids(CASES) == SMOKE_IDS  # 模块级常量与函数结果一致
