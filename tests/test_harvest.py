"""把线上失败样本捞进人工分诊队列。

**不写进 bz-rag-golden，写进独立的 bz-rag-harvested。** 两个独立原因：

1. 会被抹掉——门禁的真实数据源是本地 evaluation/test_dataset.json
   （cases.py 从它读），Phoenix 上的 bz-rag-golden 只是它的镜像，
   而 dataset.py 用的 create_dataset(example_id_key=...) 是**全量替换**语义
   （"上传里没有的 example 会被删"，该文件顶部注释原话）。
2. 没有 ground truth——线上 span 上没有标准答案，而 expected_response 是
   contextual_precision / contextual_recall 两个判官的必需输入。

所以回灌这条环由人闭合：人看过、补上标准答案，再手工加进 test_dataset.json。
"""

from evaluation.phoenix.harvest import failed_spans, harvest


def _span(sid, q="问题", a="答案"):
    return {
        "id": f"node-{sid}",
        "name": "milvus-hybrid-rag",
        "span_kind": "AGENT",
        "parent_id": None,
        "attributes": {"input.value": q, "output.value": a},
        "context": {"trace_id": f"t-{sid}", "span_id": sid},
    }


class FakeSpans:
    def __init__(self, spans, annotations):
        self._spans = spans
        self._anns = annotations

    def get_spans(self, **kw):
        if kw.get("span_kind") == "RERANKER":
            return []
        return list(self._spans)

    def get_span_annotations(self, **kw):
        return list(self._anns)


class FakeDatasets:
    def __init__(self):
        self.calls = []

    def add_examples_to_dataset(self, **kw):
        self.calls.append(kw)
        return type("D", (), {"name": kw.get("dataset"), "example_count": 1})()


class FakeClient:
    def __init__(self, spans, annotations):
        self.spans = FakeSpans(spans, annotations)
        self.datasets = FakeDatasets()


def _ann(span_id, name, label):
    return {"span_id": span_id, "name": name, "result": {"label": label, "score": 0.0}}


class TestFailedSelection:
    def test_picks_refused(self):
        c = FakeClient(
            [_span("a"), _span("b")],
            [_ann("a", "refusal_check", "refused"), _ann("b", "refusal_check", "ok")],
        )
        got = failed_spans(c, "bz-rag-canary", 180)
        assert [s["context"]["span_id"] for s in got] == ["a"]

    def test_picks_incorrect_faithfulness(self):
        c = FakeClient(
            [_span("a"), _span("b")],
            [
                _ann("a", "faithfulness", "correct"),
                _ann("b", "faithfulness", "incorrect"),
            ],
        )
        got = failed_spans(c, "bz-rag-canary", 180)
        assert [s["context"]["span_id"] for s in got] == ["b"]

    def test_partial_faithfulness_is_not_failure(self):
        """partial 是 0.5 分，pass_when 是 score >= 0.5，算通过——不该被捞。"""
        c = FakeClient([_span("a")], [_ann("a", "faithfulness", "partial")])
        assert failed_spans(c, "bz-rag-canary", 180) == []

    def test_span_with_no_annotation_is_not_harvested(self):
        """没评过不等于失败——不能把未评估的样本当成失败捞回来。"""
        c = FakeClient([_span("a")], [])
        assert failed_spans(c, "bz-rag-canary", 180) == []

    def test_all_good_yields_nothing(self):
        c = FakeClient([_span("a")], [_ann("a", "refusal_check", "ok")])
        assert failed_spans(c, "bz-rag-canary", 180) == []

    def test_answer_relevancy_does_not_trigger_harvest(self):
        """answer_relevancy 不进回灌口径——不切题往往是问题本身模糊。"""
        c = FakeClient([_span("a")], [_ann("a", "answer_relevancy", "incorrect")])
        assert failed_spans(c, "bz-rag-canary", 180) == []


class TestHarvestWrite:
    def test_writes_to_harvested_dataset_not_golden(self):
        c = FakeClient([_span("a")], [_ann("a", "refusal_check", "refused")])
        n = harvest(c, "bz-rag-canary", 180)
        assert n == 1
        call = c.datasets.calls[0]
        assert call["dataset"] == "bz-rag-harvested"
        assert call["dataset"] != "bz-rag-golden"

    def test_example_uses_nested_input_output_metadata_shape(self):
        """examples 路径要嵌套的 {input, output, metadata}。

        扁平写法 + input_keys/output_keys 只对 dataframe 路径生效，
        用错会被真 API 拒：
        "examples must be a single dictionary with required 'input' and 'output' keys"。
        实测踩过——假 client 不校验形状，所以这条必须显式断言嵌套结构。
        """
        c = FakeClient(
            [_span("a", q="蛙贝能提现吗", a="不清楚")],
            [_ann("a", "refusal_check", "refused")],
        )
        harvest(c, "bz-rag-canary", 180)
        call = c.datasets.calls[0]
        assert "input_keys" not in call, "examples 路径不该传 input_keys"
        ex = call["examples"][0]
        assert set(ex) == {"input", "output", "metadata"}
        assert ex["input"] == {"question": "蛙贝能提现吗"}
        # 标准答案留空由人补——写成空串而不是省略，让分诊的人一眼看到要填。
        assert ex["output"] == {"expected_response": ""}
        assert ex["metadata"]["observed_answer"] == "不清楚"
        assert ex["metadata"]["source_span_id"] == "a"
        assert ex["metadata"]["origin"] == "harvested"

    def test_nothing_to_harvest_makes_no_write(self):
        c = FakeClient([_span("a")], [_ann("a", "refusal_check", "ok")])
        assert harvest(c, "bz-rag-canary", 180) == 0
        assert c.datasets.calls == []
