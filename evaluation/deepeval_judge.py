"""把 MiniMax 包装成 deepeval 可用的 Judge LLM。"""
import json
import os
import re

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


class MiniMaxJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._model = ChatOpenAI(
            model=os.environ["MODEL_ID"],
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            request_timeout=60,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def load_model(self):
        return self._model

    def generate(self, prompt: str, schema=None):
        msg = self._model.invoke([{"role": "user", "content": prompt}])
        text = msg.content or ""
        if schema is not None:
            return schema.model_validate_json(self._extract_json(text))
        return text

    async def a_generate(self, prompt: str, schema=None):
        import asyncio

        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return f"MiniMax({os.environ['MODEL_ID']})"

    @staticmethod
    def _extract_json(text: str) -> str:
        try:
            json.loads(text)
            return text
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return m.group(0)
            raise ValueError(f"无法解析 JSON: {text[:200]}")
