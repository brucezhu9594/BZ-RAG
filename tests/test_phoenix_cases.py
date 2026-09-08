import random

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


def test_smoke_subset_is_invariant_to_input_order():
    """smoke 子集必须只依赖内容（example_id 排序），不依赖 test_dataset.json 的行序。

    构造一份被打乱的 CASES 副本，用与生产代码相同的排序规则独立算出"打乱后应该选出
    的 smoke 子集"，断言它和模块真正导出的 SMOKE_IDS 一致。这里不直接调用 cases.py
    的内部函数，是为了不让测试和实现共享同一段选取逻辑代码——如果 SMOKE_IDS 退化回
    按文件位置切片（本次要修的那个 bug），打乱后的"内容排序前三"几乎必然和文件位置
    意义上的"原始前三"不是同一组 id，这里就会炸。
    """
    shuffled = CASES[:]
    random.Random(0).shuffle(shuffled)
    ids_from_shuffled = frozenset(sorted(c.example_id for c in shuffled)[: len(SMOKE_IDS)])
    assert ids_from_shuffled == SMOKE_IDS
