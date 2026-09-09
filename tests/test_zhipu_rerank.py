"""测试智谱 rerank 的排序与截断。

这些用例钉住的是一个真实踩过的坑：智谱 rerank 在同话题候选上分数会饱和
（实测 hybrid 返回的 6 条真实片段全部拿 1.0），而 API 对并列项返回的是**输入的倒序**。
原实现照抄 API 顺序并把 top_n 交给服务端截断，整条重排因此等价于
documents[::-1][:top_n]，系统性地选中上游 RRF 排名最差的几条 —— 三条金标问题的
金块召回率实测 0/3。
"""

import pytest
from langchain_core.documents import Document

from common.zhipu_rerank import rerank


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")


def _docs(n):
    return [Document(page_content=f"doc-{i}") for i in range(n)]


def _patch_post(monkeypatch, results, captured=None):
    """让 requests.post 返回固定的 results，并记录发出去的 payload。"""

    def fake_post(url, headers=None, json=None, timeout=None):
        if captured is not None:
            captured.update(json)
        return _FakeResponse({"results": results})

    monkeypatch.setattr("common.zhipu_rerank.requests.post", fake_post)


class TestTieBreaking:
    def test_all_scores_tied_preserves_input_order(self, monkeypatch):
        """分数全并列时必须保留输入（上游 RRF）的名次，而不是 API 的返回顺序。

        这是原 bug 的最小复现：API 按 5,4,3,2,1,0 倒序回，金块在输入的 0 号位。
        """
        results = [{"index": i, "relevance_score": 1.0} for i in (5, 4, 3, 2, 1, 0)]
        _patch_post(monkeypatch, results)

        out = rerank("q", _docs(6), top_n=2)

        assert [d.page_content for d in out] == ["doc-0", "doc-1"]

    def test_real_score_differences_win_over_input_order(self, monkeypatch):
        """模型有区分度时听模型的，不能被原始下标压过去。"""
        results = [
            {"index": 0, "relevance_score": 0.3},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.6},
        ]
        _patch_post(monkeypatch, results)

        out = rerank("q", _docs(3), top_n=2)

        assert [d.page_content for d in out] == ["doc-1", "doc-2"]

    def test_partial_tie_breaks_by_original_index(self, monkeypatch):
        """部分并列：并列的那一组内部按原始下标，组间仍按分数。"""
        results = [
            {"index": 2, "relevance_score": 1.0},
            {"index": 0, "relevance_score": 1.0},
            {"index": 1, "relevance_score": 0.5},
        ]
        _patch_post(monkeypatch, results)

        out = rerank("q", _docs(3), top_n=3)

        assert [d.page_content for d in out] == ["doc-0", "doc-2", "doc-1"]


class TestTruncation:
    def test_requests_all_documents_not_top_n(self, monkeypatch):
        """必须向 API 要回全部候选：服务端先截断的话，排序再对也救不回被切掉的。"""
        captured = {}
        results = [{"index": i, "relevance_score": 1.0} for i in range(6)]
        _patch_post(monkeypatch, results, captured)

        rerank("q", _docs(6), top_n=2)

        assert captured["top_n"] == 6

    def test_truncates_locally_to_top_n(self, monkeypatch):
        results = [{"index": i, "relevance_score": 1.0} for i in range(6)]
        _patch_post(monkeypatch, results)

        assert len(rerank("q", _docs(6), top_n=3)) == 3

    def test_top_n_larger_than_documents_returns_all(self, monkeypatch):
        results = [{"index": i, "relevance_score": 1.0} for i in range(2)]
        _patch_post(monkeypatch, results)

        assert len(rerank("q", _docs(2), top_n=10)) == 2


class TestEdgeCases:
    def test_empty_documents_short_circuits(self, monkeypatch):
        """空输入不该打网络。"""

        def explode(*a, **k):
            raise AssertionError("空文档列表不应发起请求")

        monkeypatch.setattr("common.zhipu_rerank.requests.post", explode)

        assert rerank("q", [], top_n=2) == []

    def test_rerank_score_written_into_metadata(self, monkeypatch):
        results = [{"index": 0, "relevance_score": 0.77}]
        _patch_post(monkeypatch, results)

        out = rerank("q", _docs(1), top_n=1)

        assert out[0].metadata["rerank_score"] == 0.77

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        monkeypatch.setattr("common.zhipu_rerank.requests.post", lambda *a, **k: None)

        with pytest.raises(ValueError, match="ZHIPUAI_API_KEY"):
            rerank("q", _docs(1), top_n=1)
