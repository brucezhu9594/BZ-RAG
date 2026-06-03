# Milvus 混合检索 API + DeepEval Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `api/main.py` 新增 `POST /api/milvus/query`，由 `api/milvus_rag.py` 用 DeepEval `@observe` 把 milvus 混合检索拆成 retrieve/rerank/generate 三个 span（顶层 agent span = trace，thread_id = 会话分组），推送到 Confident AI Observatory。

**Architecture:** 新建 `api/milvus_rag.py` 独立编排检索（复用 `ZhipuAIEmbeddings` / `MilvusClient` / `RRFRanker` / `common.zhipu_rerank.rerank` 底层，不调 `hybrid_search.rag`），每步包 DeepEval span；`api/main.py` 加端点 + thread_id 字段 + NO_PROXY 设置。

**Tech Stack:** FastAPI、deepeval.tracing (4.0.3)、langchain_openai、pymilvus、langchain_community ZhipuAIEmbeddings、python-dotenv。

**Spec:** `docs/superpowers/specs/2026-06-03-milvus-tracing-api-design.md`

**前置已落地**：
- `common/zhipu_rerank.py` 的 `rerank(query, docs, top_n)`（项目原有）
- `app/milvus/hybrid_search.py` 的 milvus 检索参数与逻辑可参照（commit `53001b8` 起带 langfuse 装饰）
- `.env` 有 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_ID` / `ZHIPUAI_API_KEY` / `CONFIDENT_API_KEY`
- deepeval tracing API 已实测：`observe(type=...)`、`update_current_trace(thread_id=,input=,output=,name=)`、`update_current_span(input=,output=,retrieval_context=)`

---

## File Structure

- **Create**：`api/milvus_rag.py` — DeepEval-traced milvus 混合检索流水线（4 函数）
- **Create**：`api/test_milvus_rag.py` — mock 外部依赖的单测
- **Modify**：`api/main.py` — NO_PROXY 设置 + QueryRequest 加 thread_id + 新端点 `/api/milvus/query`
- **Unchanged**：现有 `/`、`/api/health`、`/api/query`（chroma）

---

## Task 1: api/milvus_rag.py 骨架 + thread_id 透传 TDD

**Files:**
- Create: `E:\wwwroot\BZ\BZ-RAG\api\test_milvus_rag.py`
- Create: `E:\wwwroot\BZ\BZ-RAG\api\milvus_rag.py`

本任务用 mock 把三个 span 函数与 DeepEval tracing 隔离，验证顶层编排逻辑（串联顺序 + thread_id 透传 + 返回值），不触碰真实 milvus/LLM/Confident。

- [ ] **Step 1: 写失败的测试**

Create `E:\wwwroot\BZ\BZ-RAG\api\test_milvus_rag.py`:

```python
"""api/milvus_rag.py 单测：mock 外部依赖，验证编排与 thread_id 透传。"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from api import milvus_rag


def _doc(text, source="helpContent/10006"):
    return Document(page_content=text, metadata={"source": source})


def test_query_chains_retrieve_rerank_generate(monkeypatch):
    calls = []

    def fake_retrieve(query):
        calls.append(("retrieve", query))
        return [_doc("d1"), _doc("d2")]

    def fake_rerank(query, docs):
        calls.append(("rerank", query, [d.page_content for d in docs]))
        return [docs[0]]

    def fake_generate(query, context):
        calls.append(("generate", query, context))
        return "最终答案"

    monkeypatch.setattr(milvus_rag, "_retrieve_span", fake_retrieve)
    monkeypatch.setattr(milvus_rag, "_rerank_span", fake_rerank)
    monkeypatch.setattr(milvus_rag, "_generate_span", fake_generate)
    monkeypatch.setattr(milvus_rag, "update_current_trace", lambda **kwargs: None)

    answer = milvus_rag.milvus_rag_query("怎么开发票", thread_id="t-1")

    assert answer == "最终答案"
    assert calls[0][0] == "retrieve"
    assert calls[1][0] == "rerank"
    assert calls[2][0] == "generate"
    # generate 收到的 context 来自 rerank 后的 doc
    assert "d1" in calls[2][2]


def test_query_passes_thread_id_to_trace(monkeypatch):
    captured = {}

    monkeypatch.setattr(milvus_rag, "_retrieve_span", lambda q: [_doc("d1")])
    monkeypatch.setattr(milvus_rag, "_rerank_span", lambda q, docs: docs)
    monkeypatch.setattr(milvus_rag, "_generate_span", lambda q, c: "ans")

    def fake_update_trace(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(milvus_rag, "update_current_trace", fake_update_trace)

    milvus_rag.milvus_rag_query("问题", thread_id="thread-abc")

    assert captured.get("thread_id") == "thread-abc"
    assert captured.get("input") == "问题"
    assert captured.get("output") == "ans"


def test_query_thread_id_optional(monkeypatch):
    captured = {}
    monkeypatch.setattr(milvus_rag, "_retrieve_span", lambda q: [_doc("d1")])
    monkeypatch.setattr(milvus_rag, "_rerank_span", lambda q, docs: docs)
    monkeypatch.setattr(milvus_rag, "_generate_span", lambda q, c: "ans")
    monkeypatch.setattr(milvus_rag, "update_current_trace", lambda **kwargs: captured.update(kwargs))

    answer = milvus_rag.milvus_rag_query("问题")

    assert answer == "ans"
    assert captured.get("thread_id") is None
```

- [ ] **Step 2: 写最小骨架（让测试可加载，确认 FAIL）**

Create `E:\wwwroot\BZ\BZ-RAG\api\milvus_rag.py`:

```python
"""Milvus 混合检索流水线，DeepEval tracing 覆盖 span/trace/thread。"""


def update_current_trace(**kwargs):
    raise NotImplementedError


def _retrieve_span(query):
    raise NotImplementedError


def _rerank_span(query, docs):
    raise NotImplementedError


def _generate_span(query, context):
    raise NotImplementedError


def milvus_rag_query(query, thread_id=None):
    raise NotImplementedError
```

- [ ] **Step 3: 运行测试确认 FAIL**

Run from `E:\wwwroot\BZ\BZ-RAG`: `python -m pytest api/test_milvus_rag.py -v`
Expected: 3 个测试 FAIL（`NotImplementedError`）。

- [ ] **Step 4: 实现 milvus_rag.py 完整内容**

Replace `E:\wwwroot\BZ\BZ-RAG\api\milvus_rag.py` 全部内容：

```python
"""Milvus 混合检索流水线，DeepEval tracing 覆盖 span/trace/thread。

- 顶层 milvus_rag_query 是 agent span（= trace），设 thread_id 做会话分组
- retrieve / rerank / generate 各一个 span
- 读 CONFIDENT_API_KEY 后台异步上报 Confident AI Observatory
"""
import os

from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from deepeval.tracing import observe, update_current_span, update_current_trace

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"
DENSE_LIMIT = 10
SPARSE_LIMIT = 10
RETRIEVE_TOP_K = 6
RERANK_TOP_K = 2
RRF_K = 60


@observe(type="retriever")
def _retrieve_span(query: str) -> list[Document]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    client = MilvusClient(uri=MILVUS_URI)
    query_vector = embeddings.embed_query(query)

    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=DENSE_LIMIT,
    )
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=SPARSE_LIMIT,
    )
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=RRF_K),
        limit=RETRIEVE_TOP_K,
        output_fields=["text", "source"],
    )
    docs = [
        Document(
            page_content=r["entity"].get("text", ""),
            metadata={"source": r["entity"].get("source", "")},
        )
        for r in results[0]
    ]
    update_current_span(input=query, retrieval_context=[d.page_content for d in docs])
    return docs


@observe(type="retriever")
def _rerank_span(query: str, docs: list[Document]) -> list[Document]:
    from common.zhipu_rerank import rerank

    reranked = rerank(query, docs, top_n=RERANK_TOP_K)
    update_current_span(input=query, output=[d.page_content for d in reranked])
    return reranked


@observe(type="llm")
def _generate_span(query: str, context: str) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{context}"
    )
    msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    )
    answer = msg.content or ""
    update_current_span(input=query, output=answer)
    return answer


@observe(type="agent")
def milvus_rag_query(query: str, thread_id: str | None = None) -> str:
    docs = _retrieve_span(query)
    reranked = _rerank_span(query, docs)
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', '')}\nContent: {d.page_content}"
        for d in reranked
    )
    answer = _generate_span(query, context)
    update_current_trace(
        thread_id=thread_id,
        input=query,
        output=answer,
        name="milvus-hybrid-rag",
    )
    return answer
```

- [ ] **Step 5: 运行测试确认 PASS**

Run: `python -m pytest api/test_milvus_rag.py -v`
Expected: 3 PASSED。

注意：测试 monkeypatch 了 `update_current_trace` 与三个 span 函数，所以 `milvus_rag_query` 上的 `@observe(type="agent")` 装饰器仍会真实执行（创建 span context）。deepeval observe 在无 CONFIDENT_API_KEY 或测试环境下不应抛错（仅不上报）。若 `@observe` 在测试中因缺少 trace context 报错，改为在测试模块顶部 `import os; os.environ.setdefault("CONFIDENT_API_KEY", "")` 并确认 observe 容错；如仍报错，记录为 DONE_WITH_CONCERNS 待 controller 决策。

- [ ] **Step 6: import 自检**

Run: `python -c "import api.milvus_rag; print('ok')"`
Expected: `ok`（pydantic v1 警告可忽略）。

- [ ] **Step 7: Commit**

```bash
git add api/milvus_rag.py api/test_milvus_rag.py
git commit -m "feat(api): milvus 混合检索 + DeepEval tracing 流水线"
```

---

## Task 2: api/main.py 加端点 + thread_id + NO_PROXY

**Files:**
- Modify: `E:\wwwroot\BZ\BZ-RAG\api\main.py`

- [ ] **Step 1: 顶部加 NO_PROXY 设置**

读 `api/main.py`。当前顶部是：
```python
"""BZ-RAG HTTP API: 部署到 Railway 后通过 Cloudflare Worker 做金丝雀分流。"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
```

改成（在 `import os` 之后、其它 import 之前插入 NO_PROXY）：
```python
"""BZ-RAG HTTP API: 部署到 Railway 后通过 Cloudflare Worker 做金丝雀分流。"""

import os

# 本地 Milvus 走 gRPC，若环境设了 HTTP(S)_PROXY 且 NO_PROXY 不含 localhost，
# pymilvus 连接会被代理拦截 hang。在任何 milvus import 前兜底设置。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
```

- [ ] **Step 2: QueryRequest 加 thread_id 字段**

把：
```python
class QueryRequest(BaseModel):
    query: str
```
改成：
```python
class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
```

- [ ] **Step 3: 新增 /api/milvus/query 端点**

在现有 `query` 函数（`/api/query`）之后、`if __name__ == "__main__":` 之前插入：

```python
@app.post("/api/milvus/query")
def milvus_query(req: QueryRequest) -> QueryResponse:
    """Milvus 混合检索 + DeepEval tracing（span/trace/thread → Confident AI）。"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    try:
        from api.milvus_rag import milvus_rag_query
    except KeyError as e:
        raise HTTPException(status_code=503, detail=f"环境变量缺失：{e}") from e

    try:
        answer = milvus_rag_query(req.query, thread_id=req.thread_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG 调用失败：{type(e).__name__}: {e}",
        ) from e

    return QueryResponse(answer=answer, version=APP_VERSION)
```

- [ ] **Step 4: import 自检**

Run from `E:\wwwroot\BZ\BZ-RAG`: `python -c "import api.main; print('ok')"`
Expected: `ok`。

- [ ] **Step 5: FastAPI 路由自检（不连外部服务）**

Run:
```
python -c "from api.main import app; paths = [r.path for r in app.routes]; print(paths); assert '/api/milvus/query' in paths; print('route ok')"
```
Expected: 路由列表含 `/api/milvus/query`，打印 `route ok`。

- [ ] **Step 6: Commit**

```bash
git add api/main.py
git commit -m "feat(api): 加 /api/milvus/query 端点 + thread_id + NO_PROXY"
```

---

## Task 3: 端到端冒烟（需 Milvus + .env 在线）

> 需要 Milvus 在线、`.env` 完整、CONFIDENT_API_KEY 有效。**仅在前置满足时执行**。

**Files:** 不修改

- [ ] **Step 1: 起 API（后台）**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1" PYTHONUNBUFFERED=1 python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
（后台运行）Expected: `Uvicorn running on http://127.0.0.1:8000`。

- [ ] **Step 2: health 自检**

Run: `curl -s http://127.0.0.1:8000/api/health`
Expected: `{"status":"ok","version":"dev"}`。

- [ ] **Step 3: 调 milvus 检索端点**

Run:
```
curl -s -X POST http://127.0.0.1:8000/api/milvus/query -H "Content-Type: application/json" -d "{\"query\":\"如何注销禾蛙账号\",\"thread_id\":\"smoke-t1\"}"
```
Expected: JSON `{"answer":"...","version":"dev"}`，answer 非空。

再发一条同 thread_id 的，构成会话：
```
curl -s -X POST http://127.0.0.1:8000/api/milvus/query -H "Content-Type: application/json" -d "{\"query\":\"佣金怎么结算\",\"thread_id\":\"smoke-t1\"}"
```

- [ ] **Step 4: 在 Confident AI Observatory 验证三类型**

浏览器开 Confident AI → Observatory：
- **Traces** 视图：看到 2 条 trace，每条展开有 agent → retriever(retrieve) / retriever(rerank) / llm(generate) 的 span 树
- **Threads** 视图：`smoke-t1` 下聚合这 2 条 trace

若 trace 没出现：等 1-2 分钟（后台异步上报）；确认 CONFIDENT_API_KEY 有效（`python -c "import os;print(bool(os.environ.get('CONFIDENT_API_KEY')))"` 应 True）。

- [ ] **Step 5: 停 API，无文件改动，跳过 commit**

---

## Task 4: 提交 plan 到 git

**Files:**
- 已存在: `docs/superpowers/plans/2026-06-03-milvus-tracing-api.md`

- [ ] **Step 1: 强加 plan（`/docs` 在 .gitignore）**

Run: `git add -f docs/superpowers/plans/2026-06-03-milvus-tracing-api.md`

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: Milvus tracing API implementation plan"
```

- [ ] **Step 3: 确认仓库干净**

Run: `git status`
Expected: `nothing to commit, working tree clean`（除预存 .gitignore / .deepeval 之类）。

---

## 完成判据

- `api/milvus_rag.py` 四函数（retrieve/rerank/generate span + agent 顶层），DeepEval `@observe` 三 span + `update_current_trace` thread_id
- `api/test_milvus_rag.py` 3 单测全 PASS
- `api/main.py` 含 `/api/milvus/query`、`thread_id` 字段、NO_PROXY 设置；`/api/query` 等老端点不变
- 路由自检含 `/api/milvus/query`
- （冒烟）Confident AI Observatory 能看到 trace + 3 span + thread 聚合
