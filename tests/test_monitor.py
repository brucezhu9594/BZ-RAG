"""三态判定。

最要紧的一条：**hold 优先于 rollback**。样本不足必须独立成态，
否则刚部署完没几条 trace 就会被判"全绿"（spec §4.9 原话）。
判"样本不足"用 Outcome.samples < criterion.min_samples，
**不匹配 reason 字符串**——那是给人看的显示文本，改一个字就会把判定改坏。
"""

from evaluation.phoenix.acceptance import Criterion, Outcome
from evaluation.phoenix.monitor import decide


def _outcome(passed, samples, min_samples=20, annotation="faithfulness"):
    c = Criterion(
        annotation=annotation,
        metric="pass_rate",
        pass_when="score >= 0.5",
        min_pass_rate=0.85,
        min_samples=min_samples,
    )
    return Outcome(
        criterion=c,
        passed=passed,
        observed=0.9 if passed else 0.5,
        required=0.85,
        samples=samples,
        reason="",
    )


class TestThreeStates:
    def test_all_pass_with_enough_samples_is_promote(self):
        d = decide([_outcome(True, 30), _outcome(True, 25, annotation="refusal_check")])
        assert d.state == "promote"

    def test_quality_failure_with_enough_samples_is_rollback(self):
        d = decide([_outcome(True, 30), _outcome(False, 25, annotation="refusal_check")])
        assert d.state == "rollback"
        assert "refusal_check" in d.reason

    def test_insufficient_samples_is_hold(self):
        d = decide([_outcome(False, 3)])
        assert d.state == "hold"

    def test_hold_wins_over_rollback(self):
        """一个够样本且失败、一个样本不足——必须 hold，不能 rollback。

        理由：样本不足意味着"还判不了"，此时任何质量结论都不可信，
        贸然 rollback 会把好版本也打回去。
        """
        d = decide([_outcome(False, 30), _outcome(False, 2, annotation="refusal_check")])
        assert d.state == "hold"

    def test_zero_samples_is_hold_not_promote(self):
        """刚部署完一条 trace 都没有时，绝不能判 promote。"""
        d = decide([_outcome(False, 0)])
        assert d.state == "hold"

    def test_empty_outcomes_is_hold(self):
        assert decide([]).state == "hold"


class TestReasonText:
    def test_rollback_reason_names_the_failing_criterion(self):
        d = decide([_outcome(False, 30, annotation="faithfulness")])
        assert "faithfulness" in d.reason

    def test_hold_reason_says_how_many_short(self):
        d = decide([_outcome(False, 7, min_samples=20)])
        assert "7" in d.reason and "20" in d.reason
