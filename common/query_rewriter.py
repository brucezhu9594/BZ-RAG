"""查询改写：使用 LLM 将用户原始问题优化为更适合检索的查询。"""

import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

REWRITE_PROMPT = (
    "你是一个查询改写专家。请将用户的口语化问题改写为更适合知识库检索的精确查询。\n"
    "规则：\n"
    "1. 补全缩写和口语表达，使语义更明确\n"
    "2. 保留核心意图，不要改变问题含义\n"
    "3. 只输出改写后的查询，不要解释\n"
    "4. 如果问题已经足够清晰，原样返回即可"
)


def rewrite_query(query: str, model: str | None = None) -> str:
    """将用户原始查询改写为更适合检索的形式。"""
    llm = ChatOpenAI(
        model=model or os.environ["MODEL_ID"],
        temperature=0,
    )
    msg = llm.invoke(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    rewritten = (msg.content or "").strip()
    # 过滤推理模型的 <think> 思考链
    rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()
    return rewritten if rewritten else query
