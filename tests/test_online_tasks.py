"""线上任务声明的校验规则。

这些用例钉住的是"配置错了必须立刻炸"——线上 worker 是被调度起来的，
没人盯着终端，一个被静默接受的坏配置会安静地产出几小时垃圾 annotation，
比直接崩难查得多。这与期 1 把判官配置缺失做成 import 期失败是同一个取舍。
"""

import pytest

from evaluation.phoenix.online_tasks import ONLINE_EVALUATORS, load_tasks

VALID = """
- name: canary-faithfulness
  project: bz-rag-canary
  span_kind: AGENT
  evaluators: [faithfulness, answer_relevancy]
  sampling_rate: 0.2
  cadence: continuous
  window_minutes: 30
"""


def _write(tmp_path, text):
    p = tmp_path / "online_tasks.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestValid:
    def test_loads_and_normalizes(self, tmp_path):
        (t,) = load_tasks(_write(tmp_path, VALID))
        assert t.name == "canary-faithfulness"
        assert t.project == "bz-rag-canary"
        assert t.span_kind == "AGENT"
        assert t.evaluators == ("faithfulness", "answer_relevancy")
        assert t.sampling_rate == 0.2
        assert t.cadence == "continuous"
        assert t.window_minutes == 30

    def test_online_evaluator_set_excludes_ground_truth_judges(self):
        """线上没有 ground truth，需要 expected 的两个判官不可用。"""
        assert ONLINE_EVALUATORS == frozenset(
            {"faithfulness", "answer_relevancy", "refusal_check"}
        )


class TestRejects:
    def test_unknown_evaluator(self, tmp_path):
        bad = VALID.replace("[faithfulness, answer_relevancy]", "[nonesuch]")
        with pytest.raises(ValueError, match="nonesuch"):
            load_tasks(_write(tmp_path, bad))

    def test_ground_truth_evaluator_rejected(self, tmp_path):
        """contextual_recall 需要 expected，线上拿不到——必须拒绝而不是跑出一堆 errored。"""
        bad = VALID.replace("[faithfulness, answer_relevancy]", "[contextual_recall]")
        with pytest.raises(ValueError, match="contextual_recall"):
            load_tasks(_write(tmp_path, bad))

    def test_sampling_rate_out_of_range(self, tmp_path):
        bad = VALID.replace("sampling_rate: 0.2", "sampling_rate: 1.5")
        with pytest.raises(ValueError, match="sampling_rate"):
            load_tasks(_write(tmp_path, bad))

    def test_unknown_cadence(self, tmp_path):
        bad = VALID.replace("cadence: continuous", "cadence: hourly")
        with pytest.raises(ValueError, match="cadence"):
            load_tasks(_write(tmp_path, bad))

    def test_duplicate_task_name(self, tmp_path):
        with pytest.raises(ValueError, match="重复"):
            load_tasks(_write(tmp_path, VALID + VALID))

    def test_missing_required_key(self, tmp_path):
        bad = VALID.replace("  project: bz-rag-canary\n", "")
        with pytest.raises(ValueError, match="project"):
            load_tasks(_write(tmp_path, bad))
