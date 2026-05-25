"""evaluation/deepeval_judge.py 中 JSON 提取工具的单元测试。"""
import pytest

from evaluation.deepeval_judge import MiniMaxJudge


class TestExtractJson:
    def test_pure_json_passthrough(self):
        text = '{"score": 0.9, "reason": "ok"}'
        assert MiniMaxJudge._extract_json(text) == text

    def test_markdown_wrapped_json(self):
        text = '```json\n{"a": 1}\n```'
        result = MiniMaxJudge._extract_json(text)
        assert result.strip().startswith("{")
        assert '"a": 1' in result

    def test_prose_with_embedded_json(self):
        text = '好的，评分如下：{"score": 0.5}\n仅供参考。'
        result = MiniMaxJudge._extract_json(text)
        assert result == '{"score": 0.5}'

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="无法解析 JSON"):
            MiniMaxJudge._extract_json("纯文本没有 JSON")
