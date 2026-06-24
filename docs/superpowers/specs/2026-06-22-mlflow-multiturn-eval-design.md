# MLflow 多轮（session 级）评估设计

**日期**：2026-06-22
**目标**：在现有 MLflow 评估链路上「先跑通多轮评估机制」—— 用脚本化的几段短对话，演示 MLflow 3.13 的 session 级 scorer：按会话分组 trace，对整段对话判一个会话级指标，与单轮指标一并产出。管线**保持无状态**，本设计不改 RAG 管线。

> 指标经三版演进（见「实测发现」），最终为 **`answer_coverage`**（整段对话答出率，逐轮判 + Python 聚合 + glm-4-air judge）。对话也从「指代追问」改为「黄金集自包含问题」（见「数据形态」与「实测发现 §4」）。

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
- 内联 **2 段同主题对话 × 3 轮** 的评测数据（不注册 EvaluationDataset，最快可试）。
- 新增 1 个会话级 scorer `answer_coverage`（逐轮判 + Python 聚合，judge 用 glm-4-air）。
- 阶段 1 复用现有 4 个单轮 scorer（每 turn 出单轮分），阶段 2 出会话级分。
- 给 `mlflow_evaluate.py:_make_glm_judge` 加可选 `model` 参（向后兼容），让会话级 judge 能用 glm-4-air。

**不做**（YAGNI）：
- 不改 `api/milvus_rag_mlflow.py`（管线保持无状态）。
- 不改现有单轮数据集。
- 不用 `ConversationSimulator`（自动生成多轮对话，进阶项，跑通本机制后再评估是否需要）。
- 不做有状态管线（generate 带历史 / query 改写）—— 留作后续。

## 数据形态（内联 list）

每行是一个 turn，同一会话的多个 turn 共享 `session_id`；turn 在 list 中**按会话顺序排列**。`MLFLOW_GENAI_EVAL_MAX_WORKERS=1`（沿用现有设置）保证串行执行、同会话内 `request_time` 递增，scorer 可据此重建对话顺序。

- **问题取自黄金集 `test_dataset.json` 且经 probe 确认管线能干净答出**（非「检索结果中没有」）。各 turn **自包含、无指代**——无状态管线下这样才能逐轮答出（详见「实测发现 §4」：真多轮 + 全可答在无状态管线上不可兼得）。两段主题：①发单/找职位 ②平台与账号规则。
- `expected_response` 按**管线实际能检索到的关键事实**填（诚实反映答到了什么，不硬套黄金 ground_truth 原文措辞——管线检索到的 chunk 与黄金原文常有出入）。
- `inputs` 的 key 必须匹配 predict_fn 形参：`query`、`session_id`。
- **session_id 每次运行加时间戳后缀**（`_runify`）：否则多次运行会往同一 `mlflow.trace.session` 堆 trace，UI 里看到同一会话下重复问题（实测踩过）。

## 会话级 scorer（实现核心）

指标名 **`answer_coverage`**（整段对话答出率）。**逐轮单独判 + Python 聚合**：每轮只问 judge 一个「该轮答出了吗（1/0）」，计数交给确定性代码——把 judge 最易出错的「通读整段、数轮次、聚合」环节剥掉（见「实测发现 §2/§3」）。

```python
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai.scorers import scorer

# judge 用 glm-4-air（比 flash 准），并强制 rationale 一句话内防截断（见实测发现 §3）。
_TURN_ANSWERED = _make_glm_judge(
    "turn_answered",
    "判断助手对单轮问题是否实质答出。{{ inputs }} 含 question / expected（标准答案要点）/ answer。"
    "answer 实质覆盖 expected → 1.0；答「不知道/检索结果中没有/无法回答」或答错/答非所问 → 0.0。"
    "只返回 1.0 或 0.0，rationale 控制在一句话内。",
    model="glm-4-air",
)

@scorer
def answer_coverage(session: list[Trace]) -> Feedback:   # 带 session → 自动 session 级
    expected = {c["inputs"]["query"]: c["expectations"]["expected_response"] for c in CONVERSATIONS}
    flags, skipped = [], 0
    for t in sorted(session, key=lambda t: t.info.request_time):
        spans = t.search_spans(span_type=SpanType.AGENT)   # 根 span 取 query/answer
        if not spans:
            continue
        q = (spans[0].inputs or {}).get("query", "")
        try:
            val = _TURN_ANSWERED(inputs={"question": q, "expected": expected.get(q, ""),
                                         "answer": spans[0].outputs or ""}).value
        except Exception:
            val = None
        if val is None:        # judge 调用/解析失败 → 跳过该轮，不拖垮整段
            skipped += 1; continue
        flags.append(1.0 if val >= 0.5 else 0.0)
    return Feedback(value=sum(flags) / len(flags) if flags else 0.0,
                    rationale=f"逐轮答出 {int(sum(flags))}/{len(flags)} 轮")
```

> 注：`session` 不能与 `inputs/outputs/trace` 同用，对话只能从 `session` 重建（根 span 取法实测为 `t.search_spans(SpanType.AGENT)[0]`，`.inputs.query` / `.outputs`）。标准答案要点来自 `CONVERSATIONS`（与 eval 定义同处），scorer 自查，不走 mlflow expectations 管道。`try/except` + 跳过：glm-4-air 偶发输出截断导致解析失败，单轮失败不应使整段 SCORER_ERROR。

## 实测发现（重要，2026-06-22）

指标经四次迭代才可靠，每次都暴露一个 LLM-judge 的坑：

**§1 holistic「连贯性」被 flash 钻空子 → 加硬锚点。** 最初的整体「连贯性」prompt：无状态管线逢追问礼貌答「检索结果中没有」，glm-4-flash 判成「连贯、不矛盾」给 **1.0**（prompt 明写「别因礼貌给高分」也无效）。根因：这些是**「答漏了」而非「答错了」**——没说假话、没矛盾，弱 judge 整体打分不扣分。解法不是改措辞，是**给硬锚点**：把每轮标准答案要点喂进去做「实际 vs 要点」对比。换锚点（`followup_resolution`）后含指代追问的对话正确判 **0.0**。

**§2 整段聚合判不可靠 → 逐轮判 + Python 聚合。** 即便有锚点，让**一次 judge 调用通读整段、数轮次、算成功比例**仍出错：glm-4-flash 把 3 轮**数成 4 轮**、还把一轮**逐字给了答案**的回答误读成「没给具体信息」，把 answer_coverage 错拉到 0.75。解法：**每轮单独问 judge 一个 1/0，计数交给 Python**——剥离它最易错的聚合/数数环节。

**§3 air 更聪明但更啰嗦 → 截断。** 把 per-turn judge 换成 glm-4-air（比 flash 准）后偶发 `SCORER_ERROR: Failed to parse response`：air 的 rationale 太长，撞输出 token 上限被截，make_judge 解析截断的 JSON 失败。解法：**强制 rationale 一句话内 + scorer 内 try/except 跳过失败轮**。教训：**air 聪明但啰嗦易截断，flash 简洁更稳**；对「逐轮判」这种简单任务两者都行，关键是短 rationale + 容错。judge 须非 thinking（air/flash 均满足，见 [[bz-rag-mlflow-eval-stack]]）。

**§4 无状态下「真多轮」与「全可答」不可兼得。** 含指代追问（「它/那」）能测上下文携带，但无状态管线必答不出（§1 的 0.0）；自包含问题能逐轮答出，但不测上下文携带。二选一。本设计按需求选**自包含黄金问题**（happy-path）。另注：**并非所有黄金问题都能检索到答案**（管线 recall ~0.88，事实藏长文档里 rerank top2 会漏），故先 `probe` 逐个试跑、只留能干净答出的，再组对话。

## 运行流程（两阶段）

- **阶段 1（产生 trace + 单轮指标）**：`mlflow.genai.evaluate(data=<内联list>, predict_fn=milvus_rag_mlflow_query, scorers=[4 个单轮])`，包在 `start_run(name=...-turns)`。逐 turn 跑 predict_fn，trace 自带 session 元数据。
- **阶段 2（会话级指标）**：`traces = mlflow.search_traces(locations=[exp_id], run_id=<阶段1 run_id>, flush=True)` 取回 trace；`mlflow.genai.evaluate(data=traces, scorers=[answer_coverage])`（**无 predict_fn**），包在 `start_run(name=...-session)`。mlflow 按 session 分组、每会话评一次。

## 实测结果（2026-06-22）

两组数据点（同一机制、不同对话），都印证了设计：

**A. 含指代追问对话（§1/§4 阶段，暴露无状态短板）**：单轮 answer_relevancy 0.47 / faithfulness 0.42 / precision·recall 0.50（明显低于单轮黄金集）；会话级（带锚点的 `followup_resolution`）两段均 **0.0**——每个追问答成「检索结果中没有」，含「奖励条件」其实上一轮已出现，坐实无状态接不住追问。

**B. 黄金集自包含对话（最终 happy-path，全可答）**：单轮 answer_relevancy **1.0** / faithfulness **1.0** / precision **0.98** / recall **1.0**；会话级 `answer_coverage`（逐轮 + glm-4-air）**1.0**，6/6 全答出，无 SCORER_ERROR。run 状态 FINISHED，两阶段（-turns / -session）正常落 MLflow。

## 风险 / 注意

- **必须串行**（`MLFLOW_GENAI_EVAL_MAX_WORKERS=1`）：并发会打乱同会话 turn 的 `request_time` 顺序，scorer 重建对话就乱了。
- 改对话数据集时：① 新问题先 `probe` 确认管线能答出（recall ~0.88，并非都能答）；② `expected_response` 按管线实际能答到的事实填，否则 judge 对比失真。
- 会话级 judge 用 glm-4-air：偶发输出截断（rationale 太长）→ 已用「短 rationale + try/except 跳过」兜底；换更啰嗦的模型需留意。
- Privoxy：连 localhost:5000 / Milvus 走 `NO_PROXY=localhost,127.0.0.1`（同 [[bz-rag-milvus-proxy-gotcha]]，脚本已 setdefault）。
- 评分消耗智谱额度（单轮 4 指标 ×6 turn[flash] + answer_coverage 逐轮 6 次[air]）。
