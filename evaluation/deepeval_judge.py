"""把 OpenAI 兼容 LLM（默认智谱 GLM）包装成 deepeval 可用的 Judge LLM。"""
import asyncio
import json
import os
import re

from deepeval.models import DeepEvalBaseLLM
from json_repair import repair_json
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def _is_retryable(exc: BaseException) -> bool:
    # 限流 / 超时 / 连接中断都重试 —— 智谱 GLM 偶发 APITimeoutError，
    # 不重试会让 DeepEval metric 阶段整轮崩掉。
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if status in (429, 500, 502, 503, 504):
            return True
    return False


class GLMJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._model = ChatOpenAI(
            model=os.environ["MODEL_ID"],
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            request_timeout=150,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def load_model(self):
        return self._model

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=12),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    def generate(self, prompt: str, schema=None):
        msg = self._model.invoke([{"role": "user", "content": prompt}])
        text = msg.content or ""
        if schema is not None:
            return schema.model_validate_json(self._extract_json(text))
        return text

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return f"GLM({os.environ['MODEL_ID']})"

    @staticmethod
    def _extract_json(text: str) -> str:
        # 部分 thinking 模型会在 JSON 前夹 <think>...</think>，先剥掉。
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        # 兜底：扫描所有平衡 {...} 块，从最后一个开始挑能解析的。
        candidates = []
        stack: list[int] = []
        for i, c in enumerate(text):
            if c == "{":
                stack.append(i)
            elif c == "}" and stack:
                start = stack.pop()
                if not stack:
                    candidates.append(text[start : i + 1])
        for cand in reversed(candidates):
            try:
                json.loads(cand)
                return cand
            except Exception:
                continue
        # 最后兜底：json_repair 修复未转义引号等常见 LLM 输出问题。
        repaired = repair_json(text)
        if repaired and repaired != '""':
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass
        raise ValueError(f"无法解析 JSON: {text[:200]}")
