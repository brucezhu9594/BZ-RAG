"""多查询扩展：使用 LLM 将用户问题拆分为多个检索角度的子查询。"""

import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

EXPANSION_PROMPT = (
    "你是一个查询扩展专家。请将用户的问题从不同角度拆分为多个子查询，用于知识库检索。\n"
    "规则：\n"
    "1. 生成 2-3 个子查询，每个子查询覆盖问题的不同方面\n"
    "2. 子查询之间用换行分隔，每行一个\n"
    "3. 只输出子查询，不要编号、不要解释\n"
    "4. 保持子查询简洁，适合检索\n"
    "5. 第一个子查询应保留原始问题的核心意图"
)


def expand_query(query: str, model: str | None = None) -> list[str]:
    """将用户问题扩展为多个子查询。"""
    llm = ChatOpenAI(
        model=model or os.environ["MODEL_ID"],
        temperature=0,
    )
    msg = llm.invoke(
        [
            {"role": "system", "content": EXPANSION_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    content = (msg.content or "").strip()
    # 过滤推理模型的 <think> 思考链
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # 按换行拆分，过滤空行和编号前缀
    queries = []
    for line in content.split("\n"):
        line = re.sub(r"^\d+[.、)\s]+", "", line).strip()
        if line:
            queries.append(line)

    return queries if queries else [query]
