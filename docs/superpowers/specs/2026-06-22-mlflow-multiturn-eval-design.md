# MLflow 多轮（session 级）评估设计

**日期**：2026-06-22
**目标**：在现有 MLflow 评估链路上「先跑通多轮评估机制」—— 用脚本化的几段短对话，演示 MLflow 3.13 的 session 级 scorer：按会话分组 trace，对整段对话判一个会话级指标（最终落为 `followup_resolution`，见「实测发现」），与单轮指标一并产出。管线**保持无状态**，本设计不改 RAG 管线。

## 关键技术事实（已查 mlflow 3.13.0 源码确认）

- `mlflow.genai.evaluate` 自动把 scorer 分两类：单轮 scorer（每 turn 一次）与 **session 级 scorer**（每会话一次）。判定方式：**scorer 函数带 `session` 参数即自动识别为 session 级**（`mlflow/genai/scorers/base.py` 的 `scorer` 装饰器文档），无需手动设 `is_session_level_scorer`。
- **（实测纠错）`evaluate` 不允许 `predict_fn` 与 session 级 scorer 同一次调用**：`session_utils.validate_session_level_evaluation_inputs` 会报 `Session-level scorers require traces with session IDs`。session 级评估必须**两阶段**：(1) 先用 predict_fn 跑出带 session 的 trace；(2) 再 `mlflow.search_traces(...)` 取回 trace 作为 `data`（**不带** predict_fn）跑 session 级 scorer。另一条路是 `ConversationSimulator`（本设计未用）。
- 取回刚产生的 trace 用 `mlflow.search_traces(locations=[exp_id], run_id=..., flush=True)`：`flush=True` 确保异步导出已落库（旧的 `flush_trace_async_logging()` 在 3.13 已不存在）；按 `run_id` 过滤到本次产生的 trace，`locations` 必须含该 run 所在实验（否则报 location 不匹配）。
- **`session` 参数只能与 `expectations` 组合**；与 `inputs`/`outputs`/`trace` 同用会报错。故会话级 scorer 必须**纯从 `session: list[Trace]` 重建对话**。
- session 级 scorer 收到 `session: list[Trace]` —— 同一 `mlflow.trace.session` 元数据下的全部 trace。分组逻辑在 `evaluation/session_utils.py:group_traces_by_session`：先读 trace 元数据 `mlflow.trace.session`，回退到 dataset 记录的 `source_data.session_id`。
- 会话级评分结果挂到该会话**按 `request_time` 时间序最早**的那条 trace 上（`get_first_trace_in_session`）。
- 现有管线 `api/milvus_rag_mlflow.py:milvus_rag_mlflow_query(query, session_id)` **已经**在 `session_id` 非空时 `update_current_trace(session_id=session_id)` 写 session 元数据 —— 故 predict_fn 零改动即可被正确分组。
- 智谱 judge 约束沿用 [[bz-rag-mlflow-eval-stack]]：内置 Conversational* scorer 走 `openai:/` 原生 adapter、有忽略 base_url 的回归，**不可直接用**；自定义 scorer 内用 `make_judge(..., base_url=完整 chat/completions URL)` 指向智谱（与现有单轮 4 指标同款写法）。judge 默认 `glm-4-flash`。

## 范围

**做**：
- 新增 `evaluation/mlflow_evaluate_multiturn.py`（独立入口，不动现有单轮 `mlflow_evaluate.py`）。
- 内联手搓 **2 段对话 × 3 轮** 的评测数据（不注册 EvaluationDataset，最快可试）。
- 新增 1 个会话级 scorer `followup_resolution`（GLM judge + 标准答案锚点）。
- 阶段 1 复用现有 4 个单轮 scorer（每 turn 出单轮分），阶段 2 出会话级分。

**不做**（YAGNI）：
- 不改 `api/milvus_rag_mlflow.py`（管线保持无状态）。
- 不改现有 `mlflow_evaluate.py` 与单轮数据集。
- 不用 `ConversationSimulator`（自动生成多轮对话，进阶项，跑通本机制后再评估是否需要）。
- 不做有状态管线（generate 带历史 / query 改写）—— 留作后续，本设计 followup_resolution 的 0 分正是其动因。

## 数据形态（内联 list）

每行是一个 turn，同一会话的多个 turn 共享 `session_id`；turn 在 list 中**按会话顺序排列**。`MLFLOW_GENAI_EVAL_MAX_WORKERS=1`（沿用现有设置）保证串行执行、同会话内 `request_time` 递增，scorer 可据此重建对话顺序。

```python
[
  # 会话 1：迅致品牌（第 2 轮带指代「它」）
  {"inputs": {"query": "迅致是哪家公司的品牌？", "session_id": "conv-1"},
   "expectations": {"expected_response": "<填>"}},
  {"inputs": {"query": "它是什么时候正式启动的？", "session_id": "conv-1"},
   "expectations": {"expected_response": "<填>"}},
  {"inputs": {"query": "它依托哪些能力？",         "session_id": "conv-1"},
   "expectations": {"expected_response": "<填>"}},
  # 会话 2：蛙贝差评（第 2、3 轮追问）
  {"inputs": {"query": "收到一条超差评扣多少蛙贝？", "session_id": "conv-2"},
   "expectations": {"expected_response": "<填>"}},
  {"inputs": {"query": "那好评有奖励吗？",           "session_id": "conv-2"},
   "expectations": {"expected_response": "<填>"}},
  {"inputs": {"query": "奖励的条件是什么？",         "session_id": "conv-2"},
   "expectations": {"expected_response": "<填>"}},
]
```

- `inputs` 的 key 必须匹配 predict_fn 形参：`query`、`session_id`。
- `expected_response` 用现有知识库真值填（蛙贝=5、好评月度奖励 50 蛙贝、迅致=科锐国际「禾蛙」旗下、2021-12 启动 等，已在前次评估的 rationale 中出现）。

## 会话级 scorer（实现核心）

指标名 **`followup_resolution`**（指代消解 + 上下文利用），不是最初设想的「连贯性」——见下「实测发现」。

```python
from mlflow.entities import SpanType, Trace
from mlflow.genai.scorers import scorer

# judge 看「实际答 vs 每轮标准答案要点」做对比，而非凭整体感觉。
_FOLLOWUP = _make_glm_judge(
    "followup_resolution",
    "...每轮给出：用户问题、该问题的【标准答案要点】、助手实际回答。逐轮对比："
    "实际覆盖要点=成功；【要点】有实质内容但助手答「不知道/无法回答」=失败（"
    "绝不能因语气礼貌就算成功）；答错指代=失败。分数=依赖前文的追问中成功的比例。",
)

def _conversation_text(session: list[Trace]) -> str:
    # 每轮附【标准答案要点】（按 query 查 CONVERSATIONS）给 judge 硬锚点。
    expected = {c["inputs"]["query"]: c["expectations"]["expected_response"] for c in CONVERSATIONS}
    turns = sorted(session, key=lambda t: t.info.request_time)
    lines = []
    for t in turns:
        root = t.search_spans(span_type=SpanType.AGENT)[0]   # 实测取法：AGENT 根 span
        q = (root.inputs or {}).get("query", "")
        a = root.outputs or ""
        lines.append(f"用户：{q}\n【标准答案要点】：{expected.get(q, '（无）')}\n助手：{a}")
    return "\n\n".join(lines)

@scorer
def followup_resolution(session: list[Trace]):   # 带 session → 自动 session 级
    return _FOLLOWUP(inputs={"conversation": _conversation_text(session)})
```

> 注：`session` 不能与 `inputs/outputs/trace` 同用，对话只能从 `session` 重建（根 span 取法实测为 `t.search_spans(SpanType.AGENT)[0]`，`.inputs.query` / `.outputs`）。标准答案要点来自 `CONVERSATIONS`（与 eval 定义同处），scorer 自查，不走 mlflow expectations 管道。

## 实测发现（重要，2026-06-22）

最初的 holistic「连贯性」judge prompt **被 glm-4-flash 钻空子**：无状态管线逢追问就礼貌地答「检索结果中没有/无法回答」，judge 把这判成「连贯、不矛盾」给 **1.0**（即便 prompt 明写「别因礼貌给高分」也无效）。根因：这些对话是**「答漏了」而非「答错了」**——没说假话、没矛盾，弱 judge 做整体打分时不会扣分。**解法不是改措辞，是给硬锚点**：把每轮标准答案要点喂进对话文本，让 judge 做「实际 vs 要点」对比。换锚点后两会话正确判 **0.0**，judge 明确拒绝礼貌借口。教训：LLM-judged 会话级指标对「omission 型」失败不可靠，必须给 ground-truth 参照。

## 运行流程（两阶段）

- **阶段 1（产生 trace + 单轮指标）**：`mlflow.genai.evaluate(data=<内联list>, predict_fn=milvus_rag_mlflow_query, scorers=[4 个单轮])`，包在 `start_run(name=...-turns)`。逐 turn 跑 predict_fn，trace 自带 session 元数据。
- **阶段 2（会话级指标）**：`traces = mlflow.search_traces(locations=[exp_id], run_id=<阶段1 run_id>, flush=True)` 取回 trace；`mlflow.genai.evaluate(data=traces, scorers=[followup_resolution])`（**无 predict_fn**），包在 `start_run(name=...-session)`。mlflow 按 session 分组、每会话评一次。

## 实测结果（2026-06-22）

- run 状态 FINISHED；两阶段（-turns / -session）都正常落 MLflow。
- 单轮 4 指标（这批含指代追问的对话）：answer_relevancy 0.47、faithfulness 0.42、contextual_precision/recall 0.50 —— 明显低于单轮黄金集（0.70/0.75/0.91/0.88），如实反映无状态管线在追问上答得差。
- `followup_resolution`：两段会话均 **0.0**（带锚点的 judge）。原始回答显示每个追问基本答成「检索结果中没有/无法回答」，包括 conv-waybei 第 3 轮问的奖励条件其实第 2 轮已出现 —— 坐实「无状态接不住追问」。为后续有状态改造 / `ConversationSimulator` 铺垫。

## 风险 / 注意

- **必须串行**（`MLFLOW_GENAI_EVAL_MAX_WORKERS=1`）：并发会打乱同会话 turn 的 `request_time` 顺序，scorer 重建对话就乱了。
- 会话级 scorer 依赖 `_conversation_text` 把标准答案要点喂给 judge；改对话数据集时记得 `expected_response` 要填准，否则锚点失真。
- Privoxy：连 localhost:5000 / Milvus 走 `NO_PROXY=localhost,127.0.0.1`（同 [[bz-rag-milvus-proxy-gotcha]]，脚本已 setdefault）。
- 评分消耗智谱额度（单轮 4 指标 ×6 turn + followup_resolution ×2 会话）。
