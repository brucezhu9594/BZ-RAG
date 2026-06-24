"""有状态 predict_fn 包装器：在 mlflow.genai.evaluate 逐行调用间按 session 累积真实历史。

evaluate 的 predict_fn 逐行独立调用，但多轮追问需要上一轮的真实答案。本包装器按
session_id 在进程内累积已产生的答案，靠 MLFLOW_GENAI_EVAL_MAX_WORKERS=1 的串行顺序成立。
按 (session_id, turn_idx) 键记录（覆盖而非 append）→ predict_fn 预检(preflight)重复跑首行无副作用。
前提：同一 session 内的 query 必须唯一（turn 定位、历史穿线按 query 匹配），重复会在构造期 raise ValueError。
"""

from collections.abc import Callable


def make_stateful_predict(
    data: list[dict],
    pipeline_fn: Callable[..., str],
) -> Callable[[str, str | None], str]:
    # 预排每个 session 的 turn 顺序（data 内同 session 按出现顺序）。
    order: dict[str, list[str]] = {}
    for row in data:
        sid = row["inputs"]["session_id"]
        query = row["inputs"]["query"]
        turns = order.setdefault(sid, [])
        if query in turns:
            raise ValueError(
                f"session {sid!r} 内出现重复问题 {query!r}：同会话 query 必须唯一"
                "（turn 顺序与历史穿线按 query 定位）。"
            )
        turns.append(query)

    recorded: dict[tuple[str, int], str] = {}  # (session_id, turn_idx) -> answer

    def predict(query: str, session_id: str | None = None) -> str:
        """按 session 累积历史后调用 pipeline_fn；第 N 轮带前 N-1 轮的真实问答。"""
        turns = order.get(session_id, [])
        turn_idx = turns.index(query) if query in turns else 0
        history = [
            (turns[i], recorded[(session_id, i)])
            for i in range(turn_idx)
            if (session_id, i) in recorded
        ]
        answer = pipeline_fn(query, session_id=session_id, history=history)
        recorded[(session_id, turn_idx)] = answer  # 键记录：重跑只覆盖自己，preflight 安全
        return answer

    return predict
