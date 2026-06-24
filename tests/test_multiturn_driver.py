"""测试有状态 predict_fn 包装器：按 session 累积真实历史、键记录、preflight 安全。"""

from evaluation.multiturn_driver import make_stateful_predict


def _fake_pipeline(calls):
    """返回一个记录每次调用 (query, session_id, history) 的假管线，answer 固定可预测。"""

    def pipeline(query, session_id=None, history=None):
        calls.append((query, session_id, list(history or [])))
        return f"ans:{query}"

    return pipeline


def _data(*rows):
    return [{"inputs": {"query": q, "session_id": s}, "expectations": {}} for q, s in rows]


class TestMakeStatefulPredict:
    def test_first_turn_has_empty_history(self):
        """会话首轮历史为空。"""
        calls = []
        predict = make_stateful_predict(_data(("Q1", "s")), _fake_pipeline(calls))
        predict("Q1", "s")
        assert calls == [("Q1", "s", [])]

    def test_history_accumulates_in_order(self):
        """同会话逐轮调用：第 N 轮历史含前 N-1 轮的真实问答，按序。"""
        calls = []
        predict = make_stateful_predict(
            _data(("Q1", "s"), ("Q2", "s"), ("Q3", "s")), _fake_pipeline(calls)
        )
        predict("Q1", "s")
        predict("Q2", "s")
        predict("Q3", "s")
        assert calls[1][2] == [("Q1", "ans:Q1")]
        assert calls[2][2] == [("Q1", "ans:Q1"), ("Q2", "ans:Q2")]

    def test_sessions_are_isolated(self):
        """不同 session 的历史互不串台。"""
        calls = []
        predict = make_stateful_predict(
            _data(("Q1", "a"), ("Q1b", "b"), ("Q2", "a")), _fake_pipeline(calls)
        )
        predict("Q1", "a")
        predict("Q1b", "b")
        predict("Q2", "a")
        assert calls[2][2] == [("Q1", "ans:Q1")]  # 只含会话 a 的历史

    def test_rerun_same_turn_does_not_duplicate(self):
        """preflight 重复跑首轮：键记录覆盖而非 append，后续轮历史不出现重复。"""
        calls = []
        predict = make_stateful_predict(
            _data(("Q1", "s"), ("Q2", "s")), _fake_pipeline(calls)
        )
        predict("Q1", "s")  # preflight
        predict("Q1", "s")  # 正式
        predict("Q2", "s")
        assert calls[2][2] == [("Q1", "ans:Q1")]  # 仅一条 Q1

    def test_unknown_query_is_graceful(self):
        """query 不在该 session 的 turn 列表里 → 当首轮处理，历史为空，不抛异常。"""
        calls = []
        predict = make_stateful_predict(_data(("Q1", "s")), _fake_pipeline(calls))
        predict("不存在的问题", "s")
        assert calls == [("不存在的问题", "s", [])]
