import pytest

from evaluation.phoenix import acceptance as acc


@pytest.fixture(autouse=True)
def _clean():
    acc.reset()
    yield
    acc.reset()


def _crit(**kw):
    base = dict(annotation="faithfulness", metric="average", threshold=0.8)
    base.update(kw)
    return acc.Criterion(**base)


def test_average_passes_when_mean_clears_threshold():
    for s in (1.0, 1.0, 0.5):
        acc.record("faithfulness", s)
    (out,) = acc.evaluate_all([_crit(threshold=0.8)])
    assert out.passed
    assert out.observed == pytest.approx(0.8333, abs=1e-3)
    assert out.samples == 3


def test_average_fails_when_mean_below_threshold():
    for s in (0.5, 0.5, 1.0):
        acc.record("faithfulness", s)
    (out,) = acc.evaluate_all([_crit(threshold=0.8)])
    assert not out.passed


def test_direction_minimize_inverts_comparison():
    for s in (100.0, 200.0):
        acc.record("latency_ms", s)
    (out,) = acc.evaluate_all(
        [acc.Criterion(annotation="latency_ms", metric="average",
                       threshold=800, direction="minimize")]
    )
    assert out.passed


def test_pass_rate_uses_pass_when_expression():
    for s in (1.0, 1.0, 1.0, 0.0):
        acc.record("faithfulness", s)
    (out,) = acc.evaluate_all(
        [acc.Criterion(annotation="faithfulness", metric="pass_rate",
                       pass_when="score >= 0.5", min_pass_rate=0.9)]
    )
    assert not out.passed
    assert out.observed == pytest.approx(0.75)


def test_pass_when_can_read_label():
    acc.record("recall", 0.5, label="partial")
    acc.record("recall", 0.0, label="incorrect")
    (out,) = acc.evaluate_all(
        [acc.Criterion(annotation="recall", metric="pass_rate",
                       pass_when="label != 'incorrect'", min_pass_rate=1.0)]
    )
    assert not out.passed


# ——— 四条取舍，逐条锁死 ———

def test_missing_annotation_fails_rather_than_passing_vacuously():
    acc.record("something_else", 1.0)
    (out,) = acc.evaluate_all([_crit()])
    assert not out.passed
    assert "no faithfulness" in out.reason


def test_errored_judgement_is_a_third_state_not_a_zero():
    acc.record("faithfulness", 1.0)
    acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all([_crit(threshold=0.9)])
    # errored 不参与均值（否则等于判 0 分），但要在 reason 里显式报出来
    assert out.observed == pytest.approx(1.0)
    assert out.samples == 1
    assert "1 errored" in out.reason


def test_all_criteria_are_evaluated_even_when_the_first_fails():
    acc.record("a", 0.0)
    acc.record("b", 0.0)
    outs = acc.evaluate_all(
        [acc.Criterion(annotation="a", metric="average", threshold=0.5),
         acc.Criterion(annotation="b", metric="average", threshold=0.5)]
    )
    assert len(outs) == 2
    assert not any(o.passed for o in outs)


def test_min_samples_yields_insufficient_not_pass():
    acc.record("faithfulness", 1.0)
    (out,) = acc.evaluate_all([_crit(min_samples=5)])
    assert not out.passed
    assert "insufficient samples" in out.reason


def test_load_criteria_reads_the_named_section(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "offline:\n"
        "  - annotation: faithfulness\n"
        "    metric: average\n"
        "    threshold: 0.8\n"
        "online:\n"
        "  - annotation: faithfulness\n"
        "    metric: pass_rate\n"
        "    pass_when: \"score >= 0.5\"\n"
        "    min_pass_rate: 0.85\n",
        encoding="utf-8",
    )
    offline = acc.load_criteria(str(p), "offline")
    online = acc.load_criteria(str(p), "online")
    assert len(offline) == 1 and offline[0].metric == "average"
    assert len(online) == 1 and online[0].min_pass_rate == 0.85


def test_scoreboard_reports_observed_required_and_samples():
    acc.record("faithfulness", 1.0)
    board = acc.format_scoreboard(acc.evaluate_all([_crit(threshold=0.8)]))
    assert "faithfulness" in board
    assert "1.000" in board


# ——— 审视 _eval_pass_when：补充测试 ———
#
# brief 给的 11 个测试没有覆盖链式比较、score=None 落到比较分支、BoolOp、以及
# 白名单是否真的挡住危险节点。下面逐条补上，其中最后一条对应一个真实发现的缺陷
# （见 acceptance.py 里 _eval_pass_when 的注释）：pass_when 写成 `is` / `in` 等
# 未登记的比较运算符时，原实现会抛出裸 KeyError 而不是可读的 ValueError，并且
# 这个异常不会被任何地方吞掉——会一路炸穿 evaluate_all，让同一批里所有其他
# criteria 也判不出来，直接违反「全部跑完再判」。已在 acceptance.py 里修复为
# 抛出 ValueError。


def test_pass_when_chained_comparison_is_evaluated_correctly():
    # 0.5 <= score <= 1.0 ——链式比较在 ast 里是 left=0.5, ops=[LtE, LtE],
    # comparators=[score, 1.0]；逐对滚动比较，语义应等价于 0.5<=score and score<=1.0。
    rec_in = acc.Record(score=0.7)
    rec_high = acc.Record(score=1.5)
    rec_low = acc.Record(score=0.3)
    assert acc._eval_pass_when("0.5 <= score <= 1.0", rec_in) is True
    assert acc._eval_pass_when("0.5 <= score <= 1.0", rec_high) is False
    assert acc._eval_pass_when("0.5 <= score <= 1.0", rec_low) is False


def test_pass_when_none_score_in_comparison_is_false_not_a_crash():
    # errored 记录（score=None）如果真的被喂进 _eval_pass_when（正常路径下
    # _evaluate_one 会先用 usable 过滤掉 error 不为空的记录，但 score 本身
    # 单独为 None 是可能出现的边界输入），比较分支必须安全返回 False，
    # 不能因为 None 参与比较而抛 TypeError。
    rec = acc.Record(score=None)
    assert acc._eval_pass_when("score >= 0.5", rec) is False
    assert acc._eval_pass_when("0.5 <= score <= 1.0", rec) is False


def test_pass_when_boolop_and_or_combine_correctly():
    ok = acc.Record(score=0.9, label="ok")
    low_score = acc.Record(score=0.1, label="ok")
    bad_label = acc.Record(score=0.9, label="meh")
    assert acc._eval_pass_when("score >= 0.5 and label == 'ok'", ok) is True
    assert acc._eval_pass_when("score >= 0.5 and label == 'ok'", low_score) is False
    assert acc._eval_pass_when("score >= 0.5 and label == 'ok'", bad_label) is False
    assert acc._eval_pass_when("score >= 0.9 or label == 'ok'", low_score) is True
    assert acc._eval_pass_when("score >= 0.9 or label == 'ok'", bad_label) is True


def test_pass_when_whitelist_rejects_code_injection_attempt():
    # 白名单只走 ast.parse + 手写解释器，从不调用 eval/exec，所以任何不是
    # BoolOp/Compare 的节点（比如函数调用）都应该在还没执行任何东西之前
    # 就被拒绝，而不是被当成危险代码执行。
    rec = acc.Record(score=1.0)
    with pytest.raises(ValueError):
        acc._eval_pass_when("__import__('os').system('echo pwned')", rec)


def test_pass_when_unsupported_comparison_operator_raises_value_error():
    # 回归测试：修复前，`is` / `in` 这类语法合法但未登记进 _CMP 的运算符
    # （最容易发生在有人把 == 手滑写成 is）会让 _CMP[type(op)] 抛出裸
    # KeyError，且没有任何地方捕获它——这个异常会一路炸穿 evaluate_all，
    # 使同一批里所有其他 criteria 也判不出来，违反「全部跑完再判」的取舍。
    # 修复后应统一收敛成可读的 ValueError。
    rec = acc.Record(score=1.0, label="ok")
    with pytest.raises(ValueError):
        acc._eval_pass_when("label is 'ok'", rec)
    with pytest.raises(ValueError):
        acc._eval_pass_when("1.0 is score", rec)
