"""多轮 RAG 的纯函数助手：查询改写 prompt 组装、带历史的 chat messages 组装。

抽到独立模块（不 import mlflow/langchain），让这两个最容易写错的纯转换可在不连
MLflow/Milvus 的情况下单测。api/milvus_rag_mlflow.py 调用它们，外面再包 LLM 调用与 span。
history 形如 [(user_query, assistant_answer), ...]。
"""


def build_rewrite_prompt(query: str, history: list[tuple[str, str]]) -> str:
    """把追问改写成独立问题的 prompt：指令 + 对话历史 + 后续问题。"""
    hist_text = "\n".join(f"用户：{q}\n助手：{a}" for q, a in history)
    return (
        "下面是对话历史和用户的后续问题。请结合历史，把后续问题改写成一个"
        "不依赖历史、单独就能完整理解的问题。只输出改写后的问题，不要任何解释。\n\n"
        f"--- 对话历史 ---\n{hist_text}\n\n--- 后续问题 ---\n{query}"
    )


def build_chat_messages(
    system_prompt: str, query: str, history: list[tuple[str, str]] | None
) -> list[dict]:
    """生成阶段的 messages：system + 历史(交替 user/assistant) + 当前 user 问题。

    history 为空/None 时返回 [system, user]，与单轮完全一致（向后兼容）。
    """
    messages = [{"role": "system", "content": system_prompt}]
    for user_q, assistant_a in history or []:
        messages.append({"role": "user", "content": user_q})
        messages.append({"role": "assistant", "content": assistant_a})
    messages.append({"role": "user", "content": query})
    return messages
