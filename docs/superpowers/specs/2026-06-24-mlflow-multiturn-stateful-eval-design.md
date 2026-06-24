# MLflow 真·多轮（有状态管线）评估设计

**日期**：2026-06-24
**目标**：把 RAG 管线升级为**有状态**（查询改写 + 生成带历史），让被测系统真能接住**指代追问**；评测数据换成「脱离历史就答不出」的真正多轮对话，使会话级指标 `answer_coverage` 从「永远 3/3」变成**真能区分多轮能力**的指标。

> 承接 [2026-06-22 旧设计](2026-06-22-mlflow-multiturn-eval-design.md) 的「不做」项「有状态管线（generate 带历史 / query 改写）—— 留作后续」。旧设计在**无状态**前提下只能用自包含问题（§4「真多轮与全可答不可兼得」），导致 `answer_coverage` 退化为非区分性指标。本设计解除该前提。
>
> **决策（已与用户确认）**：① 目标 = 升级管线 + 多轮评估（happy-path，不做 stateless/stateful A/B 对比）；② 只升级管线函数 `milvus_rag_mlflow_query`，HTTP 端点不动；③ 有状态机制 = 查询改写 + 生成带历史。

## 关键技术事实

**有状态机制（本设计新增）**
- RAG 的瓶颈是检索：指代追问（「它要多少钱」）直接拿去检索会检不到任何有用片段。故检索前必须先把追问**改写成独立问题**（condense-question / history-aware retriever，LangChain 经典模式）。
- 管线 `milvus_rag_mlflow_query` 加可选 `history` 参（`list[tuple[str, str]]`，每项 `(user_query, assistant_answer)`）：
  - `history` **非空** → 先跑 `_rewrite_query_span` 把追问改写成独立问题，用改写后的问题检索；`_generate_span` 把 `history` 作为前序 messages + 原始追问一起喂 LLM。
  - `history` **为空/None** → **完全跳过 rewrite span**，走原路径。第 1 轮对话历史为空也走此路径。
- **向后兼容是硬约束**：`history` 空时管线行为与 trace 结构**逐字节不变** → 单轮 `mlflow_evaluate.py`、现有 HTTP 端点 `/api/milvus/query-mlflow`、现有 4 个单轮 scorer 全部零影响。

**评估两阶段机制（沿用旧设计，已查 mlflow 3.13.0 源码确认）**
- `mlflow.genai.evaluate` 自动分两类 scorer：单轮（每 turn 一次）与 **session 级**（每会话一次，函数带 `session` 参数即自动识别）。
- `evaluate` **不允许** `predict_fn` 与 session 级 scorer 同一次调用 → 必须两阶段：(1) predict_fn 跑出带 session 的 trace + 单轮指标；(2) `mlflow.search_traces(locations=[exp_id], run_id=..., flush=True)` 取回 trace 作 `data`（**不带** predict_fn）跑 session 级 scorer。
- session 级 scorer 收 `session: list[Trace]`；`session` 只能与 `expectations` 组合，不能与 `inputs/outputs/trace` 同用 → 对话只能从 trace 重建（根 span `t.search_spans(SpanType.AGENT)[0]` 取 `.inputs.query` / `.outputs`）。
- 会话级评分挂到该会话 `request_time` 最早的 trace 上。
- 智谱 judge 约束沿用 [[bz-rag-mlflow-eval-stack]]：内置 Conversational* scorer 走 `openai:/` 原生 adapter 有忽略 base_url 的回归，不可直接用；自定义 scorer 内 `make_judge(..., base_url=完整 chat/completions URL)` 指向智谱。judge 须非 thinking。

**有状态历史在 evaluate 中如何穿线（本设计核心难点）**
- `predict_fn` 逐行独立调用，但追问轮需要上一轮的**真实答案**。
- 解法：包一层有状态 predict_fn 包装器，按 `session_id` 在进程内累积已产生的答案。依赖既有 `MLFLOW_GENAI_EVAL_MAX_WORKERS=1`（串行、按对话顺序）→ 处理某轮时该 session 之前轮的答案已就绪。
- 按 `(session_id, turn_idx)` **键记录**答案（非盲目 append）→ 重跑同一轮只覆盖自己的槽位，**predict_fn 预检（preflight，禁用 tracing 试跑首行）重复跑也安全**（首轮历史本就为空）。

## 范围

**做**：
- 改 `api/milvus_rag_mlflow.py`：加 `history` 参；新增 `_rewrite_query_span`；`_generate_span` 接受并使用 `history`。
- 改 `evaluation/mlflow_evaluate_multiturn.py`：CONVERSATIONS 换成真正的指代追问；stage 1 的 `predict_fn` 换成**有状态包装器**。会话级 `answer_coverage` 与两阶段结构保持不变。

**不做**（YAGNI）：
- 不改 HTTP 端点 `/api/milvus/query-mlflow`（仍单发，不传 history）。
- 不改单轮 `mlflow_evaluate.py` 与单轮数据集。
- 不做 stateless vs stateful 的 A/B 对比 run（happy-path 即可）。
- 不在服务端维护会话状态存储（按 thread_id 存历史）—— 超出「评估」需求。
- 不用 `ConversationSimulator`。

## 管线改动（`api/milvus_rag_mlflow.py`）

```python
@mlflow.trace(span_type=SpanType.LLM)
def _rewrite_query_span(query: str, history: list[tuple[str, str]]) -> str:
    # 把依赖上文的追问改写成独立问题，供检索用。history 非空才调用。
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.0, request_timeout=60)
    hist_text = "\n".join(f"用户：{q}\n助手：{a}" for q, a in history)
    prompt = (
        "下面是对话历史和用户的后续问题。请结合历史，把后续问题改写成一个"
        "不依赖历史、单独就能完整理解的问题。只输出改写后的问题，不要任何解释。\n\n"
        f"--- 对话历史 ---\n{hist_text}\n\n--- 后续问题 ---\n{query}"
    )
    msg = llm.invoke([{"role": "user", "content": prompt}])
    return (msg.content or "").strip() or query  # 兜底：改写失败退回原问题


@mlflow.trace(span_type=SpanType.LLM)
def _generate_span(query: str, context: str, history: list[tuple[str, str]] | None = None) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system_prompt = (...)  # 不变
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in history or []:          # 历史作为前序对话注入
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": query})  # 原始追问
    return (llm.invoke(messages).content) or ""


@mlflow.trace(name="milvus-hybrid-rag", span_type=SpanType.AGENT)
def milvus_rag_mlflow_query(query, session_id=None, history=None) -> str:
    if session_id:
        mlflow.update_current_trace(session_id=session_id)
    search_query = query
    if history:                          # 空历史完全跳过改写 → 单轮行为不变
        search_query = _rewrite_query_span(query, history)
    docs = _retrieve_span(search_query)
    reranked = _rerank_span(search_query, docs)
    context = "\n\n".join(...)
    return _generate_span(query, context, history)   # 生成用原始 query + 历史
```

- 检索/rerank 用**改写后**的 `search_query`；生成用**原始** `query` + `history`（保留自然对话语气）。
- rewrite 用 `temperature=0.0`（要稳定）；rewrite 失败退回原问题兜底。
- trace 树：`AGENT(milvus-hybrid-rag) → [rewrite(LLM, 仅追问轮), retrieve(RETRIEVER), rerank(RERANKER), generate(LLM)]`。

## 评估驱动（`evaluation/mlflow_evaluate_multiturn.py`）

有状态 predict_fn 包装器（替换 stage 1 的 `predict_fn=milvus_rag_mlflow_query`）：

```python
def _make_stateful_predict(data: list[dict]):
    # 按 session 预排 turn 顺序，建 query -> (session_id, turn_idx) 映射。
    order: dict[str, list[str]] = {}
    for c in data:
        order.setdefault(c["inputs"]["session_id"], []).append(c["inputs"]["query"])
    recorded: dict[tuple[str, int], str] = {}   # (session_id, turn_idx) -> answer

    def predict(query: str, session_id: str | None = None) -> str:
        turns = order.get(session_id, [])
        turn_idx = turns.index(query) if query in turns else 0
        history = [(turns[i], recorded[(session_id, i)])
                   for i in range(turn_idx) if (session_id, i) in recorded]
        answer = milvus_rag_mlflow_query(query, session_id=session_id, history=history)
        recorded[(session_id, turn_idx)] = answer   # 键记录：重跑只覆盖自己，preflight 安全
        return answer

    return predict
```

- 阶段 1：`mlflow.genai.evaluate(data=data, predict_fn=_make_stateful_predict(data), scorers=[4 个单轮])`。
- 阶段 2：`search_traces(...) → evaluate(data=traces, scorers=[answer_coverage])` —— **与旧设计完全一致，不改**。

## 评测数据（真正的指代追问）

2 段同主题 × 3 轮；**每个追问轮脱离历史就答不出**（指标有意义的前提）。候选（最终措辞以 probe 为准）：

- **会话 1 / 发单**：T1「在禾蛙平台上发布职位时，顾问需要提供哪些信息？」→ T2「那合同必须盖章吗？」（那合同=上轮客户合同）→ T3「上传后大概多久审核通过？」（上传=合同/职位）。
- **会话 2 / 平台规则**：T1 独立问；T2/T3 用「它/那个/上面说的」指代上轮实体。

约束（沿用旧设计 §4 纪律）：
- 每条追问先 `probe`：确认**改写后**管线能干净检索并答出（非「检索结果中没有」）。改写质量是成败关键，probe 不过就调措辞/prompt。
- `expected_response` 写**上下文解析后**的事实（管线实际能检索到的关键事实）。若管线没用上历史、把「它」答错，judge 对比 expected 就会判 0。
- `inputs` key 匹配 predict_fn 形参：`query`、`session_id`。`session_id` 每次运行加时间戳后缀（`_runify`，沿用）。

## 会话级 scorer（`answer_coverage`，保持不变，语义变得有意义）

- 实现**不动**：逐轮单独判 1/0（glm-4-air judge，rationale 一句话内防截断）+ Python 聚合成答出率，rationale 形如「逐轮答出 3/3 轮：Q1✓ Q2✓ Q3✓」。其稳健性经旧设计 §2/§3 三版打磨。
- 语义变化：数据是指代追问 → 只有管线真用上历史才能高 coverage；否则追问答错 → judge 判 0 → coverage 掉。指标恢复区分性。
- 附带定性证据：**rewrite span 的输出（改写后的独立问题）在 trace 里直观可见**「上下文被解析了」，肉眼即可核验。

## 运行流程（两阶段，沿用）

- **阶段 1**：`start_run(...-turns)` 内 `evaluate(data, predict_fn=有状态包装器, scorers=[faithfulness, answer_relevancy, contextual_precision, contextual_recall])` → 逐 turn 跑管线，trace 带 session 元数据 + 单轮指标。
- **阶段 2**：`traces = search_traces(locations=[exp_id], run_id=阶段1 run_id, flush=True)`；`start_run(...-session)` 内 `evaluate(data=traces, scorers=[answer_coverage])`。

## 风险 / 注意

- **改写质量决定检索成败**：rewrite prompt 要稳（temperature=0.0、失败退回原问题），probe 兜底；改写跑偏会连累全链。
- **必须串行**（`MLFLOW_GENAI_EVAL_MAX_WORKERS=1`）：并发会打乱同会话 turn 的 `request_time` 顺序，且有状态包装器依赖串行顺序累积历史。
- **preflight 安全**：predict_fn 预检会试跑首行，键记录（覆盖而非 append）保证重复跑无副作用。
- 改对话数据时：① 新追问先 probe 确认改写后能答出；② `expected_response` 按管线实际能答到的事实填。
- 每个追问轮 +1 次 rewrite LLM 调用，额外消耗智谱额度。
- Privoxy：连 localhost:5000 / Milvus 走 `NO_PROXY=localhost,127.0.0.1`（[[bz-rag-milvus-proxy-gotcha]]，脚本已 setdefault）。
