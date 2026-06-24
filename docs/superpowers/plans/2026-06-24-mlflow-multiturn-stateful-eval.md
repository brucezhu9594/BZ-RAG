# 真·多轮（有状态管线）评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 RAG 管线升级为有状态（查询改写 + 生成带历史），让被测系统真能接住指代追问，使多轮评估的 `answer_coverage` 恢复区分性。

**Architecture:** 两处改动互不耦合。① 管线 `api/milvus_rag_mlflow.py` 加可选 `history` 参，非空时先用历史把追问改写成独立问题再检索、生成时带历史；空时走原路径（向后兼容）。② 评估 `evaluation/mlflow_evaluate_multiturn.py` 把对话换成真正的指代追问，用一个有状态 predict_fn 包装器在 evaluate 逐行调用间按 session 累积真实历史。最容易写错的纯逻辑（prompt/messages 组装、历史穿线）抽到 import-light 模块单测。

**Tech Stack:** Python（运行时 3.14，target py310）、pytest 9、ruff、MLflow 3.13、LangChain（ChatOpenAI 指向智谱 OpenAI 兼容端点）、Milvus。

设计依据：`docs/superpowers/specs/2026-06-24-mlflow-multiturn-stateful-eval-design.md`。

## Global Constraints

- 管线签名固定为 `milvus_rag_mlflow_query(query, session_id=None, history=None) -> str`。
- `history` 类型固定为 `list[tuple[str, str]]`，每项 `(user_query, assistant_answer)`。
- **向后兼容硬约束**：`history` 为空/None 时，管线行为与 trace 结构与现状逐字节一致——不跑 rewrite span。
- 评测必须串行：`MLFLOW_GENAI_EVAL_MAX_WORKERS=1`（有状态包装器依赖串行顺序；并发会乱序）。
- 连 `localhost:5000` / Milvus 必须设 `NO_PROXY=localhost,127.0.0.1`（Privoxy 会拦 localhost）。
- judge 必须非 thinking 模型；会话级 `answer_coverage` 用 `glm-4-air`，rationale 强制一句话内。
- ruff：line-length 100、double quotes、import 排序；`evaluation/*` 允许 E402。新文件须 `ruff check` 通过。
- 测试：pytest，`pythonpath=["."]`，class-based + 中文 docstring（仿 `tests/test_keyword_expansion.py`）。单测不得在收集期 import `api.milvus_rag_mlflow` 或 `evaluation.mlflow_evaluate`（二者 import 即连 MLflow/读 env，会在无服务环境失败）。

---

### Task 1: 纯历史助手 `api/history_utils.py`

最易写错的两个纯转换——改写 prompt 组装、带历史的 chat messages 组装——抽到不依赖 mlflow/langchain 的模块，可在不连服务下单测。`api/__init__.py` 为空、无副作用，测试可干净 import。

**Files:**
- Create: `api/history_utils.py`
- Test: `tests/test_history_utils.py`

**Interfaces:**
- Produces:
  - `build_rewrite_prompt(query: str, history: list[tuple[str, str]]) -> str`
  - `build_chat_messages(system_prompt: str, query: str, history: list[tuple[str, str]] | None) -> list[dict]`

- [ ] **Step 1: 写失败测试**

`tests/test_history_utils.py`:

```python
"""测试多轮 RAG 的纯历史助手：改写 prompt 组装、带历史的 chat messages 组装。"""

from api.history_utils import build_chat_messages, build_rewrite_prompt


class TestBuildRewritePrompt:
    def test_prompt_contains_query_and_history(self):
        """改写 prompt 应同时含后续问题与历史里的问答。"""
        history = [("禾蛙是什么平台？", "禾蛙是撮合交易平台")]
        prompt = build_rewrite_prompt("它怎么收费？", history)
        assert "它怎么收费？" in prompt
        assert "禾蛙是什么平台？" in prompt
        assert "禾蛙是撮合交易平台" in prompt

    def test_prompt_has_rewrite_instruction(self):
        """prompt 应包含「改写成独立问题、只输出问题」的指令。"""
        prompt = build_rewrite_prompt("它怎么收费？", [("禾蛙是什么？", "平台")])
        assert "改写" in prompt
        assert "只输出" in prompt


class TestBuildChatMessages:
    def test_empty_history_yields_system_plus_user(self):
        """history 为空列表 → 只有 system + 当前 user，两条，与单轮一致。"""
        msgs = build_chat_messages("SYS", "问题", [])
        assert msgs == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "问题"},
        ]

    def test_none_history_same_as_empty(self):
        """history 为 None 与空列表行为一致（向后兼容）。"""
        assert build_chat_messages("SYS", "问题", None) == build_chat_messages("SYS", "问题", [])

    def test_history_expands_to_alternating_roles(self):
        """两轮历史 → system, user, assistant, user, assistant, 当前 user，共 6 条且顺序正确。"""
        history = [("Q1", "A1"), ("Q2", "A2")]
        msgs = build_chat_messages("SYS", "Q3", history)
        assert [m["role"] for m in msgs] == [
            "system", "user", "assistant", "user", "assistant", "user",
        ]
        assert msgs[0]["content"] == "SYS"
        assert msgs[1]["content"] == "Q1"
        assert msgs[2]["content"] == "A1"
        assert msgs[-1]["content"] == "Q3"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_history_utils.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'api.history_utils'`）

- [ ] **Step 3: 写实现**

`api/history_utils.py`:

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_history_utils.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: ruff 检查**

Run: `python -m ruff check api/history_utils.py tests/test_history_utils.py`
Expected: `All checks passed!`

- [ ] **Step 6: 提交**

```bash
git add -f api/history_utils.py tests/test_history_utils.py
git commit -m "feat(api): 多轮历史纯助手 build_rewrite_prompt / build_chat_messages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 有状态 predict_fn 包装器 `evaluation/multiturn_driver.py`

评估两阶段中阶段 1 的难点：`mlflow.genai.evaluate` 逐行独立调用 predict_fn，但追问轮需要上一轮的**真实答案**。包一层按 session 累积真实历史的包装器，靠串行顺序成立；按 `(session_id, turn_idx)` 键记录（覆盖而非 append）→ preflight 重复跑首行无副作用。pipeline_fn 依赖注入，纯逻辑可单测（`evaluation` 是命名空间包，import 此模块不触发 `mlflow_evaluate` 的副作用）。

**Files:**
- Create: `evaluation/multiturn_driver.py`
- Test: `tests/test_multiturn_driver.py`

**Interfaces:**
- Produces: `make_stateful_predict(data: list[dict], pipeline_fn: Callable[..., str]) -> Callable[[str, str | None], str]`
  - 返回的 `predict(query, session_id=None)` 会以 `pipeline_fn(query, session_id=session_id, history=history)` 形式调用注入的管线。
- Consumes（运行期，非测试期）：Task 3 的 `milvus_rag_mlflow_query`（签名匹配）。

- [ ] **Step 1: 写失败测试**

`tests/test_multiturn_driver.py`:

```python
"""测试有状态 predict_fn 包装器：按 session 累积真实历史、键记录、preflight 安全。"""

from evaluation.multiturn_driver import make_stateful_predict


def _fake_pipeline(calls):
    """返回一个记录每次调用 (query, session_id, history) 的假管线，answer 固定可预测。"""

    def pipeline(query, session_id=None, history=None):
        calls.append((query, session_id, list(history or [])))
        return f"ans:{query}"

    return pipeline


def _data(*rows):
    return [{"inputs": {"query": q, "session_id": s}, "expectations": {}} for q, s in rows]


class TestMakeStatefulPredict:
    def test_first_turn_has_empty_history(self):
        """会话首轮历史为空。"""
        calls = []
        predict = make_stateful_predict(_data(("Q1", "s")), _fake_pipeline(calls))
        predict("Q1", "s")
        assert calls == [("Q1", "s", [])]

    def test_history_accumulates_in_order(self):
        """同会话逐轮调用：第 N 轮历史含前 N-1 轮的真实问答，按序。"""
        calls = []
        predict = make_stateful_predict(
            _data(("Q1", "s"), ("Q2", "s"), ("Q3", "s")), _fake_pipeline(calls)
        )
        predict("Q1", "s")
        predict("Q2", "s")
        predict("Q3", "s")
        assert calls[1][2] == [("Q1", "ans:Q1")]
        assert calls[2][2] == [("Q1", "ans:Q1"), ("Q2", "ans:Q2")]

    def test_sessions_are_isolated(self):
        """不同 session 的历史互不串台。"""
        calls = []
        predict = make_stateful_predict(
            _data(("Q1", "a"), ("Q1b", "b"), ("Q2", "a")), _fake_pipeline(calls)
        )
        predict("Q1", "a")
        predict("Q1b", "b")
        predict("Q2", "a")
        assert calls[2][2] == [("Q1", "ans:Q1")]  # 只含会话 a 的历史

    def test_rerun_same_turn_does_not_duplicate(self):
        """preflight 重复跑首轮：键记录覆盖而非 append，后续轮历史不出现重复。"""
        calls = []
        predict = make_stateful_predict(
            _data(("Q1", "s"), ("Q2", "s")), _fake_pipeline(calls)
        )
        predict("Q1", "s")  # preflight
        predict("Q1", "s")  # 正式
        predict("Q2", "s")
        assert calls[2][2] == [("Q1", "ans:Q1")]  # 仅一条 Q1

    def test_unknown_query_is_graceful(self):
        """query 不在该 session 的 turn 列表里 → 当首轮处理，历史为空，不抛异常。"""
        calls = []
        predict = make_stateful_predict(_data(("Q1", "s")), _fake_pipeline(calls))
        predict("不存在的问题", "s")
        assert calls == [("不存在的问题", "s", [])]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_multiturn_driver.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'evaluation.multiturn_driver'`）

- [ ] **Step 3: 写实现**

`evaluation/multiturn_driver.py`:

```python
"""有状态 predict_fn 包装器：在 mlflow.genai.evaluate 逐行调用间按 session 累积真实历史。

evaluate 的 predict_fn 逐行独立调用，但多轮追问需要上一轮的真实答案。本包装器按
session_id 在进程内累积已产生的答案，靠 MLFLOW_GENAI_EVAL_MAX_WORKERS=1 的串行顺序成立。
按 (session_id, turn_idx) 键记录（覆盖而非 append）→ predict_fn 预检(preflight)重复跑首行无副作用。
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
        order.setdefault(sid, []).append(row["inputs"]["query"])

    recorded: dict[tuple[str, int], str] = {}  # (session_id, turn_idx) -> answer

    def predict(query: str, session_id: str | None = None) -> str:
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_multiturn_driver.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: ruff 检查**

Run: `python -m ruff check evaluation/multiturn_driver.py tests/test_multiturn_driver.py`
Expected: `All checks passed!`

- [ ] **Step 6: 提交**

```bash
git add -f evaluation/multiturn_driver.py tests/test_multiturn_driver.py
git commit -m "feat(eval): 有状态 predict_fn 包装器，按 session 穿线真实历史

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 管线升级为有状态 `api/milvus_rag_mlflow.py`

加 `history` 参；新增 `_rewrite_query_span`；`_generate_span` 用 `build_chat_messages` 带历史；`milvus_rag_mlflow_query` 在 history 非空时先改写再检索。空 history 完全跳过改写 → 单轮行为不变。本任务涉及 LLM/Milvus/MLflow，靠 smoke 验证（纯组装逻辑已在 Task 1 测过）。

**Files:**
- Modify: `api/milvus_rag_mlflow.py`（import 段、`_generate_span`、新增 `_rewrite_query_span`、`milvus_rag_mlflow_query`）
- Create（临时验证，验完删）：`<scratchpad>/smoke_stateful_pipeline.py`

**Interfaces:**
- Consumes: Task 1 的 `build_chat_messages`、`build_rewrite_prompt`。
- Produces: `milvus_rag_mlflow_query(query, session_id=None, history=None) -> str`（Task 2 运行期、Task 4 消费）；`_rewrite_query_span(query, history) -> str`（smoke / probe 用）。

前置：MLflow server 已起（`http://localhost:5000`）、Milvus 在 `localhost:19530`、`.env` 含 `MODEL_ID` / `OPENAI_BASE_URL` 等。

- [ ] **Step 1: 改 import 段**

在 `api/milvus_rag_mlflow.py` 顶部 import 区（`from pymilvus import ...` 之后）加：

```python
from api.history_utils import build_chat_messages, build_rewrite_prompt
```

- [ ] **Step 2: 新增 `_rewrite_query_span`**

在 `_retrieve_span` 定义之前（紧接常量/`set_experiment` 之后）插入：

```python
@mlflow.trace(span_type=SpanType.LLM)
def _rewrite_query_span(query: str, history: list[tuple[str, str]]) -> str:
    # 用对话历史把指代追问改写成独立问题，供检索用。history 非空才会被调用。
    # temperature=0：改写要稳定可复现。改写失败（空串）退回原问题兜底。
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.0, request_timeout=60)
    msg = llm.invoke([{"role": "user", "content": build_rewrite_prompt(query, history)}])
    return (msg.content or "").strip() or query
```

- [ ] **Step 3: 改 `_generate_span` 接受并使用 history**

把现有 `_generate_span` 整体替换为（system_prompt 文本一字不改，仅改签名与 invoke）：

```python
@mlflow.trace(span_type=SpanType.LLM)
def _generate_span(query: str, context: str, history: list[tuple[str, str]] | None = None) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{context}"
    )
    msg = llm.invoke(build_chat_messages(system_prompt, query, history))
    return msg.content or ""
```

- [ ] **Step 4: 改 `milvus_rag_mlflow_query` 串联改写**

把现有 `milvus_rag_mlflow_query` 整体替换为：

```python
@mlflow.trace(name="milvus-hybrid-rag", span_type=SpanType.AGENT)
def milvus_rag_mlflow_query(
    query: str, session_id: str | None = None, history: list[tuple[str, str]] | None = None
) -> str:
    # session_id 写入 trace metadata（mlflow.trace.session），多轮评估按它分组。
    if session_id:
        mlflow.update_current_trace(session_id=session_id)
    # history 非空 → 先把追问改写成独立问题再检索；空则走原路径（向后兼容，不跑 rewrite span）。
    search_query = _rewrite_query_span(query, history) if history else query
    docs = _retrieve_span(search_query)
    reranked = _rerank_span(search_query, docs)
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', '')}\nContent: {d.page_content}" for d in reranked
    )
    return _generate_span(query, context, history)
```

- [ ] **Step 5: 写 smoke 脚本**

`<scratchpad>/smoke_stateful_pipeline.py`（把 `<scratchpad>` 换成实际 scratchpad 目录）:

```python
import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
from dotenv import load_dotenv

load_dotenv()
from api.milvus_rag_mlflow import _rewrite_query_span, milvus_rag_mlflow_query

# 1) 改写：指代追问 + 历史 → 含主题的独立问题
rewritten = _rewrite_query_span(
    "它是什么类型的平台？", [("禾蛙平台能做什么？", "禾蛙是猎企撮合交易平台")]
)
print("REWRITE:", rewritten)

# 2) 单轮（history=None）：走原路径，能正常答出
print("SINGLE:", milvus_rag_mlflow_query("禾蛙平台是什么类型的平台？")[:120])

# 3) 多轮追问（带历史）：不报错、能答出
multi = milvus_rag_mlflow_query(
    "在它上面发违规消息会被扣多少蛙贝？",
    session_id="smoke",
    history=[("禾蛙平台是什么类型的平台？", "禾蛙是猎企撮合交易平台")],
)
print("MULTI:", multi[:120])
```

- [ ] **Step 6: 跑 smoke**

Run: `python <scratchpad>/smoke_stateful_pipeline.py`
Expected:
- `REWRITE:` 行输出一个**含「禾蛙」/「平台」的独立问题**（指代被解析），而非原样的「它是什么类型的平台？」。
- `SINGLE:` 行输出关于禾蛙是撮合/交易平台的答案（证明 history=None 原路径不变）。
- `MULTI:` 行输出含「30 蛙贝」的答案（证明追问被改写后检索到正确事实）。
- 无异常。

若 `REWRITE` 没解析指代或 `MULTI` 答非所问：调 `build_rewrite_prompt` 措辞或确认 `MODEL_ID`，重跑。

- [ ] **Step 7: ruff 检查 + 删 smoke**

Run: `python -m ruff check api/milvus_rag_mlflow.py`
Expected: `All checks passed!`
然后删除 smoke 脚本：`rm <scratchpad>/smoke_stateful_pipeline.py`

- [ ] **Step 8: 提交**

```bash
git add api/milvus_rag_mlflow.py
git commit -m "feat(api): 管线支持多轮——查询改写 + 生成带历史

history 非空时先改写追问再检索，空时走原路径（单轮零影响）。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 指代追问对话 + 接入有状态驱动 `evaluation/mlflow_evaluate_multiturn.py`

把 CONVERSATIONS 换成真正的指代追问（T1 独立、T2/T3 用「它/那/那份」指代上轮实体），stage 1 的 predict_fn 换成有状态包装器；新增 `--probe` 模式逐轮打印改写+答案以核验可答性。会话级 `answer_coverage` 实现不动。

**Files:**
- Modify: `evaluation/mlflow_evaluate_multiturn.py`（import、docstring、`CONVERSATIONS`、新增 `probe()`、`main()` 的 predict_fn 与 CLI 分发）

**Interfaces:**
- Consumes: Task 2 `make_stateful_predict`、Task 3 `milvus_rag_mlflow_query`。

前置：服务同 Task 3。

- [ ] **Step 1: 改 import 段**

在 `evaluation/mlflow_evaluate_multiturn.py` 现有 `from evaluation.mlflow_evaluate import (...)` 之后加：

```python
from evaluation.multiturn_driver import make_stateful_predict
```

并确认顶部 `import sys` 已有（用于 `--probe` 分发；现有文件已 import sys）。

- [ ] **Step 2: 替换 CONVERSATIONS 为指代追问**

把现有 `CONVERSATIONS = [...]`（自包含问题那段）整体替换为：

```python
# 2 段同主题对话 × 3 轮。T1 自包含；T2/T3 为指代追问（「它/那/那份」指代上轮实体），
# 脱离历史无法独立检索——只有管线真用上历史（改写后检索）才能答出，answer_coverage 才有区分性。
# expected_response 写「上下文解析后」的事实，取自旧设计已 probe 验证过的检索事实（recall=1.0 那批）。
# 改数据后务必先跑 `--probe` 复核改写后能干净答出，再据实调 expected_response。
CONVERSATIONS = [
    # 会话 1：发单（T2「那份合同」=T1 客户合同；T3「它」=发布的职位）
    {
        "inputs": {"query": "在禾蛙平台上发布职位时，顾问需要提供哪些信息？", "session_id": "conv-job"},
        "expectations": {
            "expected_response": "需上传有效（盖章）客户合同（含服务费率、保证期、退款条款等），"
            "并完整准确填写职位信息（职位名称/职责/任职要求/行业职能/年薪/城市/面试流程等）及客户信息。"
        },
    },
    {
        "inputs": {"query": "那份合同必须盖章吗？", "session_id": "conv-job"},
        "expectations": {"expected_response": "是，发布职位必须上传有效（盖章）的客户合同。"},
    },
    {
        "inputs": {"query": "它要怎么提高曝光度？", "session_id": "conv-job"},
        "expectations": {
            "expected_response": "提高职位信息完整度——完整度越高曝光率越高（完整度过低会被判虚假职位自动下架）。"
        },
    },
    # 会话 2：平台与账号规则（T2「它」=禾蛙平台；T3「那」承接平台/账号话题）
    {
        "inputs": {"query": "禾蛙平台是什么类型的平台？", "session_id": "conv-platform"},
        "expectations": {
            "expected_response": "禾蛙是链接猎企职位空缺与交付能力的撮合交易平台：发单方发布职位、接单方接单交付。"
        },
    },
    {
        "inputs": {"query": "在它上面发送违规消息会被扣多少蛙贝？", "session_id": "conv-platform"},
        "expectations": {"expected_response": "发送违规消息，经平台核实后扣减 30 蛙贝。"},
    },
    {
        "inputs": {"query": "那要怎么修改绑定的手机号码？", "session_id": "conv-platform"},
        "expectations": {
            "expected_response": "先在契约锁申请变更手机号，再用禾蛙注册邮箱将原手机号及新手机号发至 "
            "hewausc@careerintlinc.com，工作人员 1-3 个工作日内协助完成变更。"
        },
    },
]
```

- [ ] **Step 3: 更新模块 docstring**

把文件顶部 docstring 第 9-15 行那段「对话用黄金集……自包含……」改为反映新形态（示意，保留前面机制说明）：

```python
# 在 docstring 里把描述对话形态的那句改为：
# 对话为「同主题指代追问」：T1 自包含，T2/T3 用「它/那/那份」指代上轮实体；管线已升级为有状态
# （改写追问→检索、生成带历史，见 api/milvus_rag_mlflow.py），故追问能逐轮答出。
# answer_coverage = 整段答出率：只有真用上历史才高，否则追问答错 → 掉分（恢复区分性）。
```

- [ ] **Step 4: main() 的 predict_fn 换成有状态包装器**

把 `main()` 阶段 1 里的 `predict_fn=milvus_rag_mlflow_query` 改为先构造包装器：

```python
    # 阶段 1：逐 turn 跑管线，产生带 mlflow.trace.session 元数据的 trace + 单轮指标。
    stateful_predict = make_stateful_predict(data, milvus_rag_mlflow_query)
    with mlflow.start_run(run_name=base + "-turns") as run:
        turn_res = mlflow.genai.evaluate(
            data=data,
            predict_fn=stateful_predict,
            scorers=[faithfulness, answer_relevancy, contextual_precision, contextual_recall],
        )
```

- [ ] **Step 5: 新增 probe() 并在 `__main__` 分发**

在 `main()` 之后、`if __name__ == "__main__":` 之前加 `probe()`：

```python
def probe() -> None:
    """逐轮驱动对话、打印改写后的检索问题与答案，肉眼核验追问改写后能否干净答出。"""
    from api.milvus_rag_mlflow import _rewrite_query_span

    predict = make_stateful_predict(CONVERSATIONS, milvus_rag_mlflow_query)
    history: dict[str, list[tuple[str, str]]] = {}
    for c in CONVERSATIONS:
        q, sid = c["inputs"]["query"], c["inputs"]["session_id"]
        hist = history.setdefault(sid, [])
        rewritten = _rewrite_query_span(q, hist) if hist else q
        answer = predict(q, sid)
        hist.append((q, answer))
        print(f"\n[{sid}] 原问题：{q}")
        if rewritten != q:
            print(f"  改写→：{rewritten}")
        print(f"  答案：{answer[:200]}")
```

把文件末尾的 `if __name__ == "__main__":` 块改为：

```python
if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    else:
        main()
```

- [ ] **Step 6: 跑 probe 核验**

Run: `set NO_PROXY=localhost,127.0.0.1 && python evaluation/mlflow_evaluate_multiturn.py --probe`
（bash：`NO_PROXY=localhost,127.0.0.1 python evaluation/mlflow_evaluate_multiturn.py --probe`）
Expected：6 轮逐个打印；T2/T3 有「改写→」行且改写后问题含被指代的实体（如「禾蛙」「客户合同」「职位」）；每轮「答案」是实质回答而非「检索结果中没有」。
若某轮答不出：据实调该轮 `query` 措辞 / `expected_response`，或改 `build_rewrite_prompt`，重跑 probe 直至 6 轮都干净答出。

- [ ] **Step 7: ruff 检查**

Run: `python -m ruff check evaluation/mlflow_evaluate_multiturn.py`
Expected: `All checks passed!`

- [ ] **Step 8: 提交**

```bash
git add evaluation/mlflow_evaluate_multiturn.py
git commit -m "feat(eval): 多轮评估改用指代追问 + 有状态驱动，加 --probe

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 端到端跑通 + 验证 + 记录

跑完整两阶段评估，确认 run FINISHED、`answer_coverage` 反映多轮能力、trace 里追问轮有 rewrite span，把实测结果补进设计文档。

**Files:**
- Modify: `docs/superpowers/specs/2026-06-24-mlflow-multiturn-stateful-eval-design.md`（加「实测结果」段）

前置：服务同上；`tests/` 全绿。

- [ ] **Step 1: 全量单测回归**

Run: `python -m pytest -q`
Expected: 现有 + 新增测试全 PASS（含 `test_keyword_expansion`、`test_history_utils`、`test_multiturn_driver`）。

- [ ] **Step 2: 跑端到端多轮评估**

Run: `NO_PROXY=localhost,127.0.0.1 python evaluation/mlflow_evaluate_multiturn.py`
Expected（控制台）：
- 打印 `多轮评测 mlflow-eval-multiturn-<ts>（judge: glm-4-flash，6 turns / 2 会话）`
- 阶段 1 后打印「取回 6 条 trace 做会话级评估」
- 末尾打印「单轮」4 指标 + 「会话级」`answer_coverage`（happy-path 期望两会话各 3/3、聚合 1.0；现在因数据是指代追问，这个 1.0 是「管线真接住了追问」的证据）
- 无异常、两个 run（-turns / -session）均 FINISHED。

- [ ] **Step 3: UI 核验 rewrite span 与分组**

打开 `http://localhost:5000` → 实验 `bz-rag-milvus` → Sessions → 找到本次 `conv-job-<ts>` / `conv-platform-<ts>`。
核验：① 3 轮按 session 分组；② **T2/T3 的 trace 树里有 `_rewrite_query_span`（LLM）节点，其输出是被解析过的独立问题**（肉眼可见上下文携带）；③ T1 无 rewrite span（向后兼容）；④ Session assessments 的 `answer_coverage` rationale 形如「逐轮答出 3/3 轮：Q1✓ Q2✓ Q3✓」。

- [ ] **Step 4: 补「实测结果」到设计文档**

在 `docs/superpowers/specs/2026-06-24-mlflow-multiturn-stateful-eval-design.md` 末尾「风险 / 注意」之前插入一段「## 实测结果（2026-06-24）」，**据 Step 2/3 的真实输出填写**：单轮 4 指标数值、会话级 `answer_coverage` 数值与 N/N、rewrite span 抽样改写示例（原问题→改写后）、run 状态。

- [ ] **Step 5: 提交**

```bash
git add -f docs/superpowers/specs/2026-06-24-mlflow-multiturn-stateful-eval-design.md
git commit -m "docs: 真·多轮评估实测结果

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: 更新记忆索引（可选，若实测有新坑）**

若实测暴露了新的可复用经验（如改写 prompt 的某种失败模式），在 `bz-rag-mlflow-eval-stack` 记忆文件补一行；否则跳过。

---

## Self-Review

**1. Spec 覆盖**（逐节对照设计文档）：
- 有状态机制（history 参 / rewrite / 生成带历史 / 空跳过）→ Task 1（纯助手）+ Task 3（管线）。✓
- 向后兼容硬约束 → Task 1 `test_*_history_same_as_empty` + Task 3 `if history` 守卫 + smoke SINGLE 行 + Task 5 全量回归。✓
- evaluate 历史穿线（包装器 / 键记录 / preflight 安全 / 串行）→ Task 2（含 5 个单测）+ Global Constraints。✓
- 指代追问数据 + probe + expected=解析后事实 → Task 4。✓
- `answer_coverage` 不动、语义恢复区分 → Task 4 未改 scorer；Task 5 Step 3 核验。✓
- 两阶段运行流程 → Task 4 Step 4（stage1 换 predict_fn）+ Task 5 Step 2。✓
- 范围「不做」（不改 HTTP 端点 / 单轮 eval / 不做 A/B）→ 计划未触及这些文件。✓
- 风险（改写质量 / 串行 / preflight / 额度 / Privoxy）→ Global Constraints + Task 3/4 验证步骤。✓

**2. 占位符扫描**：无 TBD/TODO。Task 4 的 `expected_response` 给的是旧设计已 probe 验证过的具体事实 + Step 6 据实调整，非占位符；Task 5 Step 4 据真实输出填实测段，是正常的「跑完记录」而非占位。✓

**3. 类型/命名一致性**：
- `milvus_rag_mlflow_query(query, session_id=None, history=None)` 在 Task 3 定义、Task 2/4 以同签名调用（`history=` 关键字）。✓
- `make_stateful_predict(data, pipeline_fn)` 在 Task 2 定义、Task 4 Step 4/5 同名调用。✓
- `build_chat_messages` / `build_rewrite_prompt` 在 Task 1 定义、Task 3 import 同名。✓
- `history` 类型处处为 `list[tuple[str, str]]`。✓
