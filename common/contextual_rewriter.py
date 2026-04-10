"""历史感知查询改写：结合对话历史，将追问补全为独立查询。"""

import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

REWRITE_PROMPT = (
    "你是一个查询改写专家。根据对话历史，将用户最新的问题改写为一个独立的、"
    "适合知识库检索的完整查询。\n"
    "规则：\n"
    "1. 解决指代词（他、这个、上面的等），补全省略的主语和宾语\n"
    "2. 保留核心意图，不要改变问题含义\n"
    "3. 只输出改写后的查询，不要解释\n"
    "4. 如果问题已经是独立完整的，原样返回即可"
)


def contextual_rewrite(
    query: str,
    history: list[dict],
    model: str | None = None,
) -> str:
    """结合对话历史改写当前查询。

    Args:
        query: 用户当前问题
        history: 对话历史，格式 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        model: LLM 模型名称

    Returns:
        改写后的独立查询
    """
    if not history:
        return query

    llm = ChatOpenAI(
        model=model or os.environ["MODEL_ID"],
        temperature=0,
    )

    messages = [{"role": "system", "content": REWRITE_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"请将以下问题改写为独立查询：{query}"})

    msg = llm.invoke(messages)
    rewritten = (msg.content or "").strip()
    # 过滤推理模型的 <think> 思考链
    rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()
    return rewritten if rewritten else query
