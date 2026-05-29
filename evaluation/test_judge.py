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

    def test_thinking_model_strips_think_block(self):
        text = (
            '<think>用户想要 JSON {"foo":"bar"}。我应该返回 {"answer": 1}。</think>\n'
            '{"answer": 1}'
        )
        result = MiniMaxJudge._extract_json(text)
        assert result == '{"answer": 1}'

    def test_picks_last_balanced_block_when_multiple(self):
        text = '前缀 {"junk": "x"} 中间 {"truths": ["a"]}'
        result = MiniMaxJudge._extract_json(text)
        assert result == '{"truths": ["a"]}'

    def test_repairs_broken_json_with_unescaped_quotes(self):
        # MiniMax M2.7 实际产出过这种 reason 里嵌未转义引号的 broken JSON。
        text = '```json\n{\n  "reason": "score 0 because node says "禾蛙" without details"\n}\n```'
        result = MiniMaxJudge._extract_json(text)
        import json as _json
        parsed = _json.loads(result)
        assert "reason" in parsed
        assert "禾蛙" in parsed["reason"]


class TestRetryOnRateLimit:
    @staticmethod
    def _make_rate_limit_err():
        """Build a minimal openai.RateLimitError without a real httpx.Response."""
        from unittest.mock import MagicMock
        from openai import RateLimitError

        fake_response = MagicMock()
        fake_response.status_code = 429
        fake_response.request = MagicMock()
        return RateLimitError(message="429", response=fake_response, body=None)

    def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        """When _model.invoke raises RateLimitError twice then succeeds, generate() should retry and return final content."""
        import os
        os.environ.setdefault("MODEL_ID", "test")
        os.environ.setdefault("OPENAI_BASE_URL", "http://localhost")
        os.environ.setdefault("OPENAI_API_KEY", "test")

        from evaluation.deepeval_judge import MiniMaxJudge
        from unittest.mock import MagicMock

        judge = MiniMaxJudge()
        err = self._make_rate_limit_err()

        success_msg = MagicMock()
        success_msg.content = '{"ok": 1}'

        calls = {"count": 0}
        side_effects = [err, err, success_msg]

        def fake_invoke(self_inner, *args, **kwargs):
            result = side_effects[calls["count"]]
            calls["count"] += 1
            if isinstance(result, Exception):
                raise result
            return result

        # ChatOpenAI is a Pydantic model — patch at the class level, not instance level.
        monkeypatch.setattr(type(judge._model), "invoke", fake_invoke)

        # Speed up the retry waits to keep the test fast.
        import tenacity
        monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

        out = judge.generate("hi")
        assert out == '{"ok": 1}'
        assert calls["count"] == 3

    def test_retries_exhausted_raises(self, monkeypatch):
        """When _model.invoke always rate-limits, generate() should give up after max attempts and re-raise."""
        import os
        os.environ.setdefault("MODEL_ID", "test")
        os.environ.setdefault("OPENAI_BASE_URL", "http://localhost")
        os.environ.setdefault("OPENAI_API_KEY", "test")

        from evaluation.deepeval_judge import MiniMaxJudge
        from openai import RateLimitError

        judge = MiniMaxJudge()
        err = self._make_rate_limit_err()

        calls = {"count": 0}

        def fake_invoke(self_inner, *args, **kwargs):
            calls["count"] += 1
            raise err

        monkeypatch.setattr(type(judge._model), "invoke", fake_invoke)

        import tenacity
        monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

        with pytest.raises(RateLimitError):
            judge.generate("hi")
        # Max attempts is 6 per the spec.
        assert calls["count"] == 6
