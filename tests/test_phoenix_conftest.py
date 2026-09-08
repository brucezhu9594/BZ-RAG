"""C2（最终整支 review）：Phoenix 插件三处早退会让 acceptance.record() 一次都不被
调用——不是 errored 第三态，是彻底不存在，且 acceptance 本身没有"本该有几条"的
概念兜底。conftest.py 里新加的 `_completeness_fails` 拿"实际执行的 item 数"（由
`pytest_runtest_logreport` 数出来，见 conftest.py 的注释）跟每个 annotation 实际
记录到的条数比对，缺了就报出来。

这里离线验证它本身：直接操作 acceptance._RECORDS 喂/不喂记录，再调
`_completeness_fails`，不需要真的跑一次 pytest session（本任务明确要求不为此跑
全量、smoke 也只在必要时跑一次）。
"""

import pytest

from evaluation.phoenix import acceptance
from evaluation.phoenix.conftest import _completeness_fails


@pytest.fixture(autouse=True)
def _clean():
    acceptance.reset()
    yield
    acceptance.reset()


def test_no_shortfall_when_every_annotation_has_the_expected_count():
    # (i) 的离线等价物：3 个 item 全部正常落库，每个 annotation 都记满 3 条，
    # 不应该被误报缺样本。
    for name in ("faithfulness", "answer_relevancy", "refusal_check"):
        for _ in range(3):
            acceptance.record(name, 1.0)
    assert _completeness_fails(["faithfulness", "answer_relevancy", "refusal_check"], 3) == []


def test_shortfall_detected_when_one_annotation_is_missing_records():
    # (ii)：人为制造缺样本——faithfulness 只落了 1 条，其余该有的 3 条不见了
    # （对应 Phoenix 插件三处早退中的任一处：log_run 抛异常/experiment_id 缺失/
    # offline client 为 None，都会让 acceptance.record() 完全不被调用）。
    acceptance.record("faithfulness", 1.0)
    for _ in range(3):
        acceptance.record("answer_relevancy", 1.0)
    fails = _completeness_fails(["faithfulness", "answer_relevancy"], 3)
    assert len(fails) == 1
    assert "faithfulness" in fails[0]
    assert "期望 3" in fails[0]
    assert "实际 1" in fails[0]
    assert "缺 2" in fails[0]


def test_shortfall_detected_when_annotation_has_zero_records():
    # 极端情形：该 annotation 一条都没落库（三处早退命中最彻底的那种）。
    fails = _completeness_fails(["refusal_check"], 3)
    assert len(fails) == 1
    assert "期望 3" in fails[0]
    assert "实际 0" in fails[0]
    assert "缺 3" in fails[0]


def test_errored_records_still_count_toward_the_expected_total():
    # errored（第三态）记录本身是 acceptance.record() 真的被调用过的证据——
    # 落库失败是"record 完全没被调用"，跟"record 被调用、但判官报错"是两回事
    # （分别对应 C2 与既有的 max_error_rate 闸门）。errored 记录必须照样计入
    # _completeness_fails 的"实际条数"，不能被当成缺样本。
    acceptance.record("faithfulness", 1.0)
    acceptance.record("faithfulness", None, error="judge timeout")
    acceptance.record("faithfulness", 1.0)
    assert _completeness_fails(["faithfulness"], 3) == []


def test_expected_zero_never_false_positives():
    # 没有任何 item 真的跑过（比如 --collect-only）时 expected=0，不应该对任何
    # annotation 报缺样本——即便该 annotation 一条记录都没有。
    assert _completeness_fails(["faithfulness", "answer_relevancy"], 0) == []
