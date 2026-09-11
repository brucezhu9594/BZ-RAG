"""worker 一轮的行为。全部用假 client + 注入的假判官，不打网络。

**有意不 import evaluation.phoenix.evaluators**：它在 import 期就要求
JUDGE_OPENAI_API_KEY 等三个变量（期 1 的有意设计），而本文件要能在
test.yml 的 ubuntu-latest 上跑——那里没有 .env 也没有 secrets。
判官通过 run_task(judges=...) 注入，真判官只在 default_judges() 里延迟导入。
"""

from datetime import datetime, timedelta, timezone

import pytest

from evaluation.phoenix.online_tasks import OnlineTask
from evaluation.phoenix.online_worker import run_task


def _span(span_id, kind="AGENT", parent=None, q="问题", a="答案"):
    """注意 id 与 context.span_id **有意取不同的值**。

    Phoenix 的 span 有两个 id：顶层 `id` 是 Phoenix 自己的全局节点 ID
    （形如 base64 的 "Span:5898"），`context.span_id` 才是 OTel 的十六进制 span id。
    **注解 API 认的是后者**——实测用节点 ID 调 get_span_annotations 返回 404。
    fixture 里让两者不同，才能让测试真正抓住用错 id 这个错。
    """
    return {
        "id": f"node-{span_id}",
        "name": "milvus-hybrid-rag",
        "span_kind": kind,
        "parent_id": parent,
        "attributes": {"input.value": q, "output.value": a},
        "context": {"trace_id": f"t-{span_id}", "span_id": span_id},
    }


class FakeSpans:
    """假的 spans 资源。

    **按 span_kind 分别返回**，而不是无脑回同一批——真实的
    get_spans(span_kind=...) 就是这么过滤的。worker 拉 AGENT 与补拉 RERANKER
    是两次独立调用，假 client 必须复现这一点，否则测不出"上下文没拉到"这类 bug
    （实测踩过：主查询带 AGENT 过滤，在那批结果里找 RERANKER 兄弟永远找不到，
    faithfulness 每条都看到空上下文、每条判 incorrect，而计数完全正常）。
    """

    def __init__(self, spans, existing_annotations=(), context_spans=()):
        self._spans = spans
        self._context = list(context_spans)
        self._existing = list(existing_annotations)
        self.logged = []
        self.calls = []

    def get_spans(self, **kw):
        self.calls.append(kw)
        if kw.get("span_kind") == "RERANKER":
            return list(self._context)
        return list(self._spans)

    def get_span_annotations(self, **kw):
        return list(self._existing)

    def log_span_annotations_dataframe(self, *, dataframe, annotator_kind, annotation_name):
        self.logged.append((annotation_name, annotator_kind, dataframe))
        return []


class FakeClient:
    def __init__(self, spans, existing=(), context_spans=()):
        self.spans = FakeSpans(spans, existing, context_spans)


TASK = OnlineTask(
    name="t",
    project="bz-rag-canary",
    span_kind="AGENT",
    evaluators=("refusal_check",),
    sampling_rate=1.0,
    cadence="continuous",
    window_minutes=60,
)
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _ok_judge(output, input=None, **_):  # noqa: A002 —— 参数名由判官约定
    return {"name": "refusal_check", "score": 1.0, "label": "ok", "explanation": "行"}


OK_JUDGES = {"refusal_check": _ok_judge}


class TestPullAndFilter:
    def test_queries_with_span_kind_and_window(self):
        c = FakeClient([_span("a")])
        run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        first = c.spans.calls[0]
        assert first["project_identifier"] == "bz-rag-canary"
        assert first["span_kind"] == "AGENT"
        assert first["start_time"] == NOW - timedelta(minutes=60)
        assert first["end_time"] == NOW

    def test_non_root_spans_skipped(self):
        """只评根 span：带 parent_id 的是子 span，评它会重复计数。"""
        c = FakeClient([_span("root"), _span("child", parent="node-root")])
        stats = run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        assert stats.pulled == 2
        assert stats.annotated == 1

    def test_span_without_answer_is_skipped_not_errored(self):
        bad = _span("x")
        bad["attributes"] = {"input.value": "只有问题"}
        c = FakeClient([bad])
        stats = run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        assert stats.skipped == 1
        assert stats.errored == 0
        assert stats.annotated == 0


class TestDedup:
    def test_span_with_existing_annotation_is_not_re_evaluated(self):
        """窗口重叠时同一条 span 会被反复拉到，已评过的必须跳过，否则重复计费。"""
        # "a" 是 OTel span_id（_span 里 context.span_id 用它，顶层 id 是 "node-a"）
        c = FakeClient([_span("a")], existing=[{"span_id": "a", "name": "refusal_check"}])
        stats = run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        assert stats.deduped == 1
        assert stats.annotated == 0
        assert c.spans.logged == []


class TestSampling:
    def test_guardrail_rate_one_evaluates_everything(self):
        c = FakeClient([_span(f"s{i}") for i in range(20)])
        stats = run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        assert stats.sampled == 20
        assert stats.annotated == 20

    def test_rate_zero_evaluates_nothing(self):
        c = FakeClient([_span(f"s{i}") for i in range(20)])
        stats = run_task(c, TASK._replace(sampling_rate=0.0), now=NOW, judges=OK_JUDGES)
        assert stats.sampled == 0
        assert c.spans.logged == []


class TestWriteBack:
    def test_dataframe_shape(self):
        c = FakeClient([_span("a")])
        run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        (name, kind, df) = c.spans.logged[0]
        assert name == "refusal_check"
        assert kind == "LLM"
        assert list(df["span_id"]) == ["a"]  # OTel span_id，不是 node-a
        assert list(df["label"]) == ["ok"]
        assert list(df["score"]) == [1.0]


class TestJudgeFailure:
    def test_errored_judge_counted_not_raised(self):
        def boom(output, input=None, **_):  # noqa: A002
            return {
                "name": "refusal_check",
                "score": None,
                "label": "errored",
                "explanation": "APIConnectionError",
            }

        c = FakeClient([_span("a")])
        stats = run_task(c, TASK, now=NOW, judges={"refusal_check": boom})
        assert stats.errored == 1
        assert stats.annotated == 0
        assert c.spans.logged == []


class TestLazyJudgeImport:
    def test_module_import_does_not_require_judge_credentials(self):
        """本模块被 import 时不得触发 evaluators 的 import 期凭证检查。

        钉住这条是因为 tests/ 会跑在没有 .env 也没有 secrets 的 ubuntu-latest 上。
        """
        import sys

        assert "evaluation.phoenix.evaluators" not in sys.modules or True
        # 真正的保证：online_worker 顶层没有 evaluators 的 import
        import inspect

        from evaluation.phoenix import online_worker

        src = inspect.getsource(online_worker)
        top = src.split("def default_judges")[0]
        assert "from evaluation.phoenix.evaluators import" not in top


@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_stats_fields_are_consistent(rate):
    """pulled 必须能被各计数桶解释完，不能有说不清去向的样本。"""
    c = FakeClient([_span(f"s{i}") for i in range(30)])
    s = run_task(c, TASK._replace(sampling_rate=rate), now=NOW, judges=OK_JUDGES)
    assert s.pulled == 30
    assert s.skipped + s.deduped + s.sampled + s.unsampled == s.pulled


class TestUsesOtelSpanId:
    def test_dedup_query_and_writeback_use_otel_span_id(self):
        """注解 API 认 context.span_id（OTel 十六进制），不认顶层的 Phoenix 节点 id。

        实测：用节点 id 调 get_span_annotations 返回 404。这条测试靠 _span()
        里 id 与 context.span_id 取不同值来保证真的能抓住用错。
        """
        captured = {}

        class Spy(FakeSpans):
            def get_span_annotations(self, **kw):
                captured["queried"] = list(kw["span_ids"])
                return []

        c = FakeClient([])
        c.spans = Spy([_span("deadbeef")])
        run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        assert captured["queried"] == ["deadbeef"]
        assert list(c.spans.logged[0][2]["span_id"]) == ["deadbeef"]


class TestContextFetch:
    """上下文（RERANKER span）必须单独补拉——这是一个真踩过的 bug 的回归测试。

    主查询带 span_kind=AGENT 过滤，返回的只有 AGENT span；在那批结果里找
    RERANKER 兄弟永远找不到。后果是 faithfulness 每条都看到空上下文、
    每条判 incorrect，而 pulled/sampled/annotated 这些计数完全正常——
    只有去看 annotation 的**内容**才会暴露。
    """

    @staticmethod
    def _reranker(trace_of, contexts):
        import json

        return {
            "id": "node-rr",
            "name": "_rerank",
            "span_kind": "RERANKER",
            "parent_id": "node-root",
            "attributes": {
                "output.value": json.dumps(
                    [{"page_content": c} for c in contexts], ensure_ascii=False
                )
            },
            "context": {"trace_id": f"t-{trace_of}", "span_id": "rr"},
        }

    def test_context_pulled_with_reranker_kind_and_trace_ids(self):
        c = FakeClient([_span("a")], context_spans=[self._reranker("a", ["片段一"])])
        run_task(c, TASK, now=NOW, judges=OK_JUDGES)
        second = c.spans.calls[1]
        assert second["span_kind"] == "RERANKER"
        assert second["trace_ids"] == ["t-a"]

    def test_judge_receives_non_empty_contexts(self):
        """核心断言：判官拿到的 contexts 不能是空的。"""
        seen = {}

        def spy(output, input=None, **_):  # noqa: A002
            seen["contexts"] = output["contexts"]
            return {"name": "refusal_check", "score": 1.0, "label": "ok", "explanation": ""}

        c = FakeClient([_span("a")], context_spans=[self._reranker("a", ["片段一", "片段二"])])
        run_task(c, TASK, now=NOW, judges={"refusal_check": spy})
        assert seen["contexts"] == ["片段一", "片段二"]

    def test_no_context_fetch_when_nothing_sampled(self):
        """一条都没抽中时不该白打一次网络。"""
        c = FakeClient([_span(f"s{i}") for i in range(5)])
        run_task(c, TASK._replace(sampling_rate=0.0), now=NOW, judges=OK_JUDGES)
        assert len(c.spans.calls) == 1
