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
    # Ruling R21：两条都 FAIL 无法区分"第二条真的被独立求值"与"只是拿到一个
    # 占位 Outcome"。改成第一条 FAIL、第二条 PASS，断言 outs[1].passed 为真，
    # 才是真的在验证"全部跑完再判"而不是"反正都不通过看不出差别"。
    acc.record("a", 0.0)
    acc.record("b", 1.0)
    outs = acc.evaluate_all(
        [acc.Criterion(annotation="a", metric="average", threshold=0.5),
         acc.Criterion(annotation="b", metric="average", threshold=0.5)]
    )
    assert len(outs) == 2
    assert not outs[0].passed
    assert outs[1].passed


def test_average_includes_legitimate_zero_scores_not_just_truthy_ones():
    # I1（最终整支 review）：acceptance.py 里 `if r.score is not None` 是唯一
    # 支撑"哪些记录算进均值"的判据。0.0 是合法分数（faithfulness 的 incorrect、
    # refusal_check 的 refused/empty 都映射到 0.0），不是"没有值"，必须被计入。
    # 用 (1.0, 1.0, 0.0, 0.0)：`is not None` 版本四条全计入，均值 0.5，FAIL
    # （threshold 0.8）。如果有人把判据"优化"成 `if r.score`（把 falsy 的 0.0
    # 当成"没有值"排除），就只剩两条 1.0，均值变成 1.0，错误地 PASS——这条测试
    # 就是用来锁死不能退化成那个版本。
    for s in (1.0, 1.0, 0.0, 0.0):
        acc.record("faithfulness", s)
    (out,) = acc.evaluate_all([_crit(threshold=0.8)])
    assert out.samples == 4
    assert out.observed == pytest.approx(0.5)
    assert not out.passed


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


# ——— Task review (opus) 复核后的四条修复：R18-R21 ———


def test_average_with_only_non_numeric_scores_fails_with_clear_reason():
    # Ruling R21 (a)：取舍 2 点名的情形——"average 无任何数值"这个具名失败态
    # 之前 0 测试覆盖。record 了但 score 本身是 None（不是 errored，只是没有
    # 数值），必须落进 "no ... numeric scores found" 分支而不是被当成 0 或
    # 被别的分支悄悄吞掉。
    acc.record("faithfulness", None)
    (out,) = acc.evaluate_all([_crit()])
    assert not out.passed
    assert "numeric" in out.reason


def test_scoreboard_surfaces_errored_count_even_when_criterion_passes():
    # Ruling R18 (a)：即使 metric 逻辑本身判 PASS（这里只有 1 条有效样本，
    # min_samples=1，均值 1.0 达标），只要这批记录里有 errored，记分卡也必须
    # 无条件呈现——否则"19 次判官报错、只有 1 次有效"这种事在人看得见的输出
    # 里彻底隐形（这正是 review 实测出的原始漏洞：19/20 errored 时两条
    # faithfulness criteria 都 PASS，且 "errored" in board 是 False）。
    acc.record("faithfulness", 1.0)
    for _ in range(19):
        acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all([_crit(threshold=0.8, min_samples=1)])
    assert out.passed
    board = acc.format_scoreboard([out])
    assert "errored" in board


def test_max_error_rate_out_of_range_rejected_at_construction():
    with pytest.raises(ValueError):
        _crit(max_error_rate=1.5)
    with pytest.raises(ValueError):
        _crit(max_error_rate=-0.1)


def test_max_error_rate_boundary_exactly_at_threshold_does_not_trigger_the_gate():
    # Ruling R18 (b) 边界：errored=1, usable=4 → error_rate=0.2，恰好等于
    # max_error_rate，不应该触发这道门（用严格 >），交给后面的 metric 逻辑
    # 正常判定；4 个 1.0 的均值仍然 1.0 >= 0.8。
    for s in (1.0, 1.0, 1.0, 1.0):
        acc.record("faithfulness", s)
    acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all([_crit(threshold=0.8, max_error_rate=0.2)])
    assert "exceeds max_error_rate" not in out.reason
    assert out.passed


def test_max_error_rate_boundary_just_above_threshold_triggers_the_gate():
    # Ruling R18 (b) 边界：errored=2, usable=4 → error_rate=0.333 > 0.2，
    # 必须直接 FAIL——即使 usable=4 早就够 min_samples，也不能被 min_samples
    # 或 metric 逻辑盖过去。这正是 review 指出的"min_samples 在规模变大后失效，
    # 需要一个与规模无关的判据"。
    for s in (1.0, 1.0, 1.0, 1.0):
        acc.record("faithfulness", s)
    acc.record("faithfulness", None, error="judge timeout")
    acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all(
        [_crit(threshold=0.8, max_error_rate=0.2, min_samples=2)]
    )
    assert not out.passed
    assert "exceeds max_error_rate" in out.reason
    assert "errored" in out.reason


def test_max_error_rate_blocks_the_19_of_20_errored_scenario():
    # Ruling R18 核心场景复现：20 次判官调用，19 次 errored + 1 次拿到 1.0。
    # 修复前两条 faithfulness criteria 都判 PASS 且记分卡完全看不出 errored。
    # 修复后：max_error_rate 挡住它，两条都判 FAIL，且 errored 计数在 reason
    # 与记分卡里都清楚可见。
    acc.record("faithfulness", 1.0)
    for _ in range(19):
        acc.record("faithfulness", None, error="judge timeout")
    outs = acc.evaluate_all(
        [
            acc.Criterion(annotation="faithfulness", metric="average",
                          threshold=0.8, max_error_rate=0.2),
            acc.Criterion(annotation="faithfulness", metric="pass_rate",
                          pass_when="score >= 0.5", min_pass_rate=0.9,
                          max_error_rate=0.2),
        ]
    )
    assert len(outs) == 2
    assert not any(o.passed for o in outs)
    assert all("errored" in o.reason for o in outs)
    board = acc.format_scoreboard(outs)
    assert "FAIL" in board
    assert "errored" in board


def test_pass_when_syntax_error_is_caught_at_construction_not_after_the_fact():
    # Ruling R20 (a)：YAML 里把 pass_when 写残了（比如少打一半），必须在
    # Criterion 构造期就报 SyntaxError，而不是烧完所有判官调用、跑到
    # evaluate_all 最后一步才引爆、把已经算好的其它 criteria 结果一并丢光。
    with pytest.raises(SyntaxError):
        acc.Criterion(annotation="faithfulness", metric="pass_rate",
                      pass_when="score >=", min_pass_rate=0.9)


def test_pass_when_bad_operator_is_caught_at_construction():
    # Ruling R20 (a)：白名单外的运算符（is/in 等）同样要在构造期就报
    # ValueError，不用等到 evaluate_all 才发现。
    with pytest.raises(ValueError):
        acc.Criterion(annotation="faithfulness", metric="pass_rate",
                      pass_when="label is 'ok'", min_pass_rate=0.9)


def test_evaluate_all_survives_a_pass_when_type_error_and_still_scores_the_rest():
    # Ruling R19/R20 (b) 核心证据：pass_when 里常量类型和 score 的实际类型
    # 对不上（"score >= 'abc'"）这种问题没法在构造期发现——必须真的比较到
    # 具体值才会抛 TypeError。这是三类逃逸异常里唯一一个前移校验覆盖不到的，
    # 只能在 _evaluate_one 里用运行时 try/except 兜底。这里验证 evaluate_all
    # 真的不会被它炸穿：第一条判成"配置有问题"的 FAIL，第二条正常算出 PASS，
    # 不受第一条连累——这才是"就地显形而不是拖垮整批"这句话第一次成立。
    acc.record("a", 0.9)
    acc.record("b", 1.0)
    outs = acc.evaluate_all(
        [
            acc.Criterion(annotation="a", metric="pass_rate",
                          pass_when="score >= 'abc'", min_pass_rate=0.9),
            acc.Criterion(annotation="b", metric="average", threshold=0.5),
        ]
    )
    assert len(outs) == 2
    assert not outs[0].passed
    assert "invalid pass_when" in outs[0].reason
    assert outs[1].passed


# ——— Task review 第二轮复核：R22-R24 ———


def test_max_error_rate_tolerates_one_transient_failure_at_smoke_scale():
    # Ruling R22：smoke 规模 N=3，纯比例语义下 max_error_rate=0.2 会零容忍
    # （1/3=0.333 早就超标），一次判官瞬时超时就把 PR 判红。改成"绝对下限
    # 1 次 + 超出后按比例"之后，N=3 时 1 次 errored 必须被容忍。
    acc.record("faithfulness", 1.0)
    acc.record("faithfulness", 1.0)
    acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all(
        [_crit(threshold=0.8, max_error_rate=0.2, min_samples=2)]
    )
    assert "exceeds max_error_rate" not in out.reason
    assert out.passed


def test_max_error_rate_still_fails_at_smoke_scale_with_two_errors():
    # N=3，2 次 errored——超出"容忍 1 次"的绝对下限，必须判 FAIL（不能被
    # min_samples 的宽松掩盖）。
    acc.record("faithfulness", 1.0)
    acc.record("faithfulness", None, error="judge timeout")
    acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all(
        [_crit(threshold=0.8, max_error_rate=0.2, min_samples=1)]
    )
    assert not out.passed
    assert "exceeds max_error_rate" in out.reason


def test_max_error_rate_tolerates_one_transient_failure_at_master_scale():
    # master 全量规模 N=48，比例下限是 max(1, 0.2*48)=9.6；1 次 errored 远
    # 低于这个下限，必须容忍——这是 min_samples 在大规模下失效、错误率闸门
    # 接管的场景（Ruling R18 (b) 的原始动机），这里确认新公式没有把它改坏。
    for _ in range(47):
        acc.record("faithfulness", 1.0)
    acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all(
        [_crit(threshold=0.8, max_error_rate=0.2, min_samples=2)]
    )
    assert "exceeds max_error_rate" not in out.reason
    assert out.passed


def test_max_error_rate_fails_at_master_scale_with_ten_errors():
    # N=48，10 次 errored：10 > max(1, 0.2*48=9.6)，必须判 FAIL。
    for _ in range(38):
        acc.record("faithfulness", 1.0)
    for _ in range(10):
        acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all(
        [_crit(threshold=0.8, max_error_rate=0.2, min_samples=2)]
    )
    assert not out.passed
    assert "exceeds max_error_rate" in out.reason


def test_scoreboard_marker_is_gbk_encodable():
    # Ruling R23：本机默认输出编码是 gbk，U+26A0（⚠）编不进去。经 pytest 的
    # terminalwriter 输出时会把整块记分卡转义成一行，裸 print 则直接
    # UnicodeEncodeError 崩溃——记分卡是门禁面向人的唯一输出，可见性在目标
    # 平台上碎掉等于没加。改成纯 ASCII 标记后，board 必须能在 gbk 下正常
    # 编码，不能再抛异常。
    acc.record("faithfulness", 1.0)
    for _ in range(19):
        acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all([_crit(threshold=0.8, min_samples=1)])
    board = acc.format_scoreboard([out])
    assert "errored" in board
    board.encode("gbk")  # 不抛 UnicodeEncodeError 即为通过


def test_criterion_survives_asdict_and_json_dumps():
    # Ruling R24：_pass_when_ast 曾经是 dataclass field，
    # dataclasses.asdict(criterion) 会把里面的 ast.Compare 一起带出来，
    # json.dumps 直接 TypeError。Outcome 内嵌 criterion，期 2/3 的 monitor
    # 把 outcomes 落成 CI artifact 时就会炸。
    import dataclasses
    import json

    c = acc.Criterion(annotation="faithfulness", metric="pass_rate",
                       pass_when="score >= 0.5", min_pass_rate=0.9)
    payload = json.dumps(dataclasses.asdict(c))
    assert "faithfulness" in payload
    assert "score >= 0.5" in payload
