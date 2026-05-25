"""把 MiniMax 包装成 deepeval 可用的 Judge LLM（先占位，逻辑下一步补）。"""
import json
import re


class MiniMaxJudge:
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
