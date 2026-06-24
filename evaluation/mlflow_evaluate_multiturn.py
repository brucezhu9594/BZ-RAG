"""BZ-RAG MLflow 多轮（session 级）评估入口。

见 docs/superpowers/specs/2026-06-22-mlflow-multiturn-eval-design.md。

演示 MLflow 3.13 的 session 级评估机制：脚本化几段同主题对话（同会话 turn 共享 session_id），
两阶段评估——阶段1 逐 turn 跑管线产生 trace + 单轮指标，阶段2 search_traces 取回 trace
按 session 分组、跑会话级 scorer（带 session 参数的函数自动识别）。

对话用黄金集（test_dataset.json）里验证过能检索到答案的问题，每段同一主题、各 turn 自包含
（不含指代追问）——因为管线 api/milvus_rag_mlflow.py 是无状态的，自包含问题才能逐轮答出。
会话级指标 answer_coverage = 整段对话「答出率」，对全可答对话接近 1、对答不出的对话趋 0。

注：管线无状态本入口不改它。predict_fn 在 session_id 非空时 update_current_trace(session_id=)
写 mlflow.trace.session 元数据，分组即据此。session_id 每次运行加时间戳后缀，避免多次运行
往同一 session 堆 trace。
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
from mlflow.entities import Feedback, SpanType, Trace
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

# 会话级答出率。实测让 glm-4-flash 通读整段、逐轮判、再聚合不可靠——它会数错轮次、甚至误读某轮
# （把逐字给了答案的轮判成「没答」）。改为「逐轮单独判 + Python 聚合」：每轮只问 judge 一个
# 「该轮答出了吗」，计数交给确定性代码；judge 同时换更强的 glm-4-air（非 thinking，比 flash 准）。
# rationale 强制一句话内：glm-4-air 比 flash 啰嗦，rationale 太长会撞输出上限被截，
# make_judge 解析截断的 JSON 会抛 SCORER_ERROR（实测偶发）。短 rationale 防截断。
_TURN_ANSWERED = _make_glm_judge(
    "turn_answered",
    "判断助手对单轮问题是否实质答出。{{ inputs }} 含：question（用户问题）、"
    "expected（该问题的标准答案要点）、answer（助手实际回答）。"
    "若 answer 实质覆盖了 expected 的关键信息 → 返回 1.0；"
    "若 answer 回答「不知道 / 检索结果中没有 / 无法回答 / 建议联系客服」、或答错、答非所问 → 返回 0.0。"
    "只返回 1.0 或 0.0，rationale 控制在一句话内。",
    model="glm-4-air",
)


@scorer
def answer_coverage(session: list[Trace]) -> Feedback:
    # 带 session 参数 → mlflow 自动识别为会话级 scorer。逐轮单独判，Python 聚合成答出率，
    # 避免弱 judge 通读整段时数错/误读。要点按 query 查 CONVERSATIONS（不受运行后缀影响）。
    expected = {c["inputs"]["query"]: c["expectations"]["expected_response"] for c in CONVERSATIONS}
    flags, skipped = [], 0
    for t in sorted(session, key=lambda t: t.info.request_time):
        spans = t.search_spans(span_type=SpanType.AGENT)
        if not spans:
            continue
        question = (spans[0].inputs or {}).get("query", "")
        answer = spans[0].outputs or ""
        try:
            val = _TURN_ANSWERED(
                inputs={"question": question, "expected": expected.get(question, ""), "answer": answer}
            ).value
        except Exception:
            val = None
        if val is None:  # judge 调用/解析失败：跳过该轮不计入，避免一次瞬时失败拖垮整段
            skipped += 1
            continue
        flags.append(1.0 if val >= 0.5 else 0.0)
    if not flags:
        return Feedback(value=0.0, rationale=f"全部 {skipped} 轮 judge 调用失败")
    detail = " ".join(f"Q{i + 1}{'✓' if f else '✗'}" for i, f in enumerate(flags))
    note = f"（{skipped} 轮 judge 失败已跳过）" if skipped else ""
    return Feedback(value=sum(flags) / len(flags), rationale=f"逐轮答出 {int(sum(flags))}/{len(flags)} 轮：{detail}{note}")


# 2 段同主题对话 × 3 轮，问题取自黄金集 test_dataset.json 且经 probe 确认管线能干净答出
# （非"检索结果中没有"）。各 turn 自包含（无指代）——无状态管线下这样才能逐轮答出。
# expected_response 按管线实际能检索到的关键事实写（诚实反映答到了什么，不硬套黄金原文措辞）。
CONVERSATIONS = [
    # 会话 1：发单 / 找职位
    {
        "inputs": {"query": "在禾蛙平台上发布职位时，顾问需要提供哪些信息？", "session_id": "conv-job"},
        "expectations": {
            "expected_response": "需上传有效（盖章）客户合同（含服务费率、保证期、退款条款等），"
            "并完整准确填写职位信息（职位名称/职责/任职要求/行业职能/年薪/城市/面试流程等）及客户信息。"
        },
    },
    {
        "inputs": {"query": "在禾蛙平台上如何提高职位信息的曝光度？", "session_id": "conv-job"},
        "expectations": {
            "expected_response": "提高职位信息完整度——职位完整度越高曝光率越高（完整度过低会被判虚假职位自动下架）。"
        },
    },
    {
        "inputs": {"query": "在禾蛙平台上，如何搜索自己想接单的职位？", "session_id": "conv-job"},
        "expectations": {
            "expected_response": "在禾蛙网页端首页“职位专区”/“职位”列表页或“禾蛙+”小程序搜索查找想接单的职位。"
        },
    },
    # 会话 2：平台与账号规则
    {
        "inputs": {"query": "禾蛙平台是什么类型的平台？", "session_id": "conv-platform"},
        "expectations": {
            "expected_response": "禾蛙是链接猎企职位空缺与交付能力的撮合交易平台：发单方发布职位、接单方接单交付。"
        },
    },
    {
        "inputs": {"query": "如何修改在禾蛙平台绑定的手机号码？", "session_id": "conv-platform"},
        "expectations": {
            "expected_response": "先在契约锁申请变更手机号，再用禾蛙注册邮箱将原手机号及新手机号发至 "
            "hewausc@careerintlinc.com，工作人员 1-3 个工作日内协助完成变更。"
        },
    },
    {
        "inputs": {"query": "如果在禾蛙平台上发送违规消息，会被扣减多少蛙贝？", "session_id": "conv-platform"},
        "expectations": {"expected_response": "发送违规消息，经平台核实后扣减 30 蛙贝。"},
    },
]


def _runify(conversations: list[dict], suffix: str) -> list[dict]:
    """给每条 turn 的 session_id 加运行后缀，避免多次运行往同一 session 堆 trace。"""
    return [
        {
            "inputs": {**c["inputs"], "session_id": f'{c["inputs"]["session_id"]}-{suffix}'},
            "expectations": c["expectations"],
        }
        for c in conversations
    ]


def _print_metrics(label: str, run_id: str, metrics: dict) -> None:
    print(f"\n{label} run_id: {run_id}")
    for metric, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")


def main() -> None:
    # mlflow.genai.evaluate 不允许 predict_fn 与 session 级 scorer 同一次调用，必须两阶段。
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"mlflow-eval-multiturn-{run_ts}"
    data = _runify(CONVERSATIONS, run_ts)  # session_id 加本次运行后缀
    n_sessions = len({c["inputs"]["session_id"] for c in data})
    print(f"多轮评测 {base}（judge: {JUDGE_MODEL_ID}，{len(data)} turns / {n_sessions} 会话）")

    # 阶段 1：逐 turn 跑管线，产生带 mlflow.trace.session 元数据的 trace + 单轮指标。
    with mlflow.start_run(run_name=base + "-turns") as run:
        turn_res = mlflow.genai.evaluate(
            data=data,
            predict_fn=milvus_rag_mlflow_query,
            scorers=[faithfulness, answer_relevancy, contextual_precision, contextual_recall],
        )
    exp_id = run.info.experiment_id
    turns_run_id = run.info.run_id

    # 阶段 2：取回这些 trace（flush=True 确保异步导出已落库），不带 predict_fn，按 session 评答出率。
    traces = mlflow.search_traces(locations=[exp_id], run_id=turns_run_id, flush=True)
    print(f"\n取回 {len(traces)} 条 trace 做会话级评估")
    with mlflow.start_run(run_name=base + "-session") as sess_run:
        sess_res = mlflow.genai.evaluate(data=traces, scorers=[answer_coverage])

    _print_metrics("单轮", turns_run_id, turn_res.metrics)
    _print_metrics("会话级", sess_run.info.run_id, sess_res.metrics)
    print(f"\n详情见 {mlflow.get_tracking_uri()} 对应实验的 Evaluations 页（Traces 按 session 分组）。")


if __name__ == "__main__":
    main()
