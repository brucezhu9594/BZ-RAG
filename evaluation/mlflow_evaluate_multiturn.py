"""BZ-RAG MLflow 多轮（session 级）评估入口。

见 docs/superpowers/specs/2026-06-22-mlflow-multiturn-eval-design.md。

只演示 MLflow 3.13 的 session 级评估机制：脚本化几段对话（同会话 turn 共享 session_id），
mlflow.genai.evaluate 逐 turn 跑管线产生 trace、自动按 session 分组，会话级 scorer
（带 session 参数的函数自动识别）对整段对话评一次连贯性。

管线 api/milvus_rag_mlflow.py 保持无状态，本入口不改它：predict_fn 已在 session_id 非空时
update_current_trace(session_id=) 写 mlflow.trace.session 元数据，分组即据此。
"""

import os
import pathlib
import sys
from datetime import datetime, timezone

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

# 连 localhost:5000 / Milvus 走 NO_PROXY，避开 Privoxy（同单轮入口）。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
# 必须串行：并发会打乱同会话内 turn 的 request_time 顺序，scorer 重建对话就乱了。
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")

import mlflow
from mlflow.entities import SpanType, Trace
from mlflow.genai.scorers import scorer

# 复用单轮入口的 judge 工厂、4 个单轮 scorer 与管线，避免重复定义。
from evaluation.mlflow_evaluate import (
    JUDGE_MODEL_ID,
    _make_glm_judge,
    answer_relevancy,
    contextual_precision,
    contextual_recall,
    faithfulness,
    milvus_rag_mlflow_query,
)

# 用 holistic「连贯性」prompt 实测会被 glm-4-flash 钻空子：它把无状态管线逢追问就礼貌地
# 答「不知道」判成连贯、给满分。解法是给 judge 硬锚点——把每轮标准答案要点喂进对话文本
# （见 _conversation_text），让它做「实际答 vs 要点」的对比，而非凭整体感觉打分。
_FOLLOWUP = _make_glm_judge(
    "followup_resolution",
    "下面 {{ inputs }} 中的 conversation 是按时间顺序的多轮对话，每轮给出：用户问题、"
    "该问题的【标准答案要点】、以及助手实际回答。评估助手对【依赖前文的追问】"
    "（第 2 轮起带「它/那/这个」等指代或省略主语的问题）处理得如何。\n"
    "逐轮对比助手实际回答与【标准答案要点】：\n"
    "- 实际回答覆盖了要点 = 该轮成功；\n"
    "- **【标准答案要点】有实质内容，但助手回答「不知道 / 检索结果中没有 / 无法回答 / 建议联系客服」"
    "= 该轮失败（说明没接住追问，绝不能因语气礼貌或没说错话就算成功）**；\n"
    "- 答错指代对象或答非所问 = 该轮失败。\n"
    "分数 = 依赖前文的追问中成功的比例。全部成功给 1.0，全部失败给 0.0。"
    "返回 0.0 到 1.0 之间的分数。",
)


def _conversation_text(session: list[Trace]) -> str:
    """按时间序从各 trace 的 AGENT 根 span 重建对话，每轮附「标准答案要点」做 judge 硬锚点。

    要点按 query 从 CONVERSATIONS 查（标准答案与 eval 定义同处）；judge 据此对比助手实际
    回答，识别「要点有实质内容却答不知道」的未接住追问。
    """
    expected = {c["inputs"]["query"]: c["expectations"]["expected_response"] for c in CONVERSATIONS}
    turns = sorted(session, key=lambda t: t.info.request_time)
    lines = []
    for t in turns:
        agent_spans = t.search_spans(span_type=SpanType.AGENT)
        if not agent_spans:
            continue
        root = agent_spans[0]
        question = (root.inputs or {}).get("query", "")
        answer = root.outputs or ""
        lines.append(
            f"用户：{question}\n【标准答案要点】：{expected.get(question, '（无）')}\n助手：{answer}"
        )
    return "\n\n".join(lines)


@scorer
def followup_resolution(session: list[Trace]):
    # 带 session 参数 → mlflow 自动识别为会话级 scorer，每个 session 调一次。
    return _FOLLOWUP(inputs={"conversation": _conversation_text(session)})


# 2 段对话 × 3 轮，每段第 2/3 轮是带指代的追问（"它""那"）。
# expected_response 取自知识库真值（见 spec；与上次单轮评估 rationale 中出现的内容一致）。
CONVERSATIONS = [
    {
        "inputs": {"query": "迅致是哪家公司的品牌？", "session_id": "conv-brand"},
        "expectations": {"expected_response": "迅致是科锐国际（300662.SZ）产业互联网平台“禾蛙”旗下品牌。"},
    },
    {
        "inputs": {"query": "它是什么时候正式启动的？", "session_id": "conv-brand"},
        "expectations": {"expected_response": "迅致于 2021 年 12 月正式启动。"},
    },
    {
        "inputs": {"query": "它依托哪些能力？", "session_id": "conv-brand"},
        "expectations": {
            "expected_response": "依托科锐国际 26 年行业积累和禾蛙平台的行业资源整合能力，"
            "通过“业务赋能系统 + 互联网平台 + 培训体系”构建。"
        },
    },
    {
        "inputs": {"query": "收到一条用户超差评扣多少蛙贝？", "session_id": "conv-waybei"},
        "expectations": {"expected_response": "收到 1 条用户超差评扣除 5 蛙贝。"},
    },
    {
        "inputs": {"query": "那好评有奖励吗？", "session_id": "conv-waybei"},
        "expectations": {"expected_response": "有，月度好评顾问奖励 50 蛙贝（不区分接发单双方）。"},
    },
    {
        "inputs": {"query": "奖励的条件是什么？", "session_id": "conv-waybei"},
        "expectations": {"expected_response": "每个自然月用户收到评价 ≥5 条且五星好评率 100%。"},
    },
]


def _print_metrics(label: str, run_id: str, metrics: dict) -> None:
    print(f"\n{label} run_id: {run_id}")
    for metric, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")


def main() -> None:
    # mlflow.genai.evaluate 不允许 predict_fn 与 session 级 scorer 同一次调用，必须两阶段：
    base = f"mlflow-eval-multiturn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"多轮评测 {base}（judge: {JUDGE_MODEL_ID}，{len(CONVERSATIONS)} turns / 2 会话）")

    # 阶段 1：逐 turn 跑管线，产生带 mlflow.trace.session 元数据的 trace + 单轮指标。
    with mlflow.start_run(run_name=base + "-turns") as run:
        turn_res = mlflow.genai.evaluate(
            data=CONVERSATIONS,
            predict_fn=milvus_rag_mlflow_query,
            scorers=[faithfulness, answer_relevancy, contextual_precision, contextual_recall],
        )
    exp_id = run.info.experiment_id
    turns_run_id = run.info.run_id

    # 阶段 2：取回这些 trace（flush=True 确保异步导出已落库），不带 predict_fn，按 session 评连贯性。
    traces = mlflow.search_traces(locations=[exp_id], run_id=turns_run_id, flush=True)
    print(f"\n取回 {len(traces)} 条 trace 做会话级评估")
    with mlflow.start_run(run_name=base + "-session") as sess_run:
        sess_res = mlflow.genai.evaluate(data=traces, scorers=[followup_resolution])

    _print_metrics("单轮", turns_run_id, turn_res.metrics)
    _print_metrics("会话级", sess_run.info.run_id, sess_res.metrics)
    print(f"\n详情见 {mlflow.get_tracking_uri()} 对应实验的 Evaluations 页（Traces 按 session 分组）。")


if __name__ == "__main__":
    main()
