# Milvus 混合检索 API + DeepEval Tracing 设计

**日期**：2026-06-03
**目标**：在 `api/main.py` 新增一个 Milvus 混合检索 HTTP 端点，用 DeepEval tracing 覆盖 span / trace / thread 三种数据类型，结果推送到 Confident AI Observatory 供查看。

## 范围

**做**：
- 新建 `api/milvus_rag.py`：DeepEval `@observe` 包裹的 milvus 混合检索流水线，拆 retrieve / rerank / generate 三个 span，顶层 agent span 作为 trace，并设 thread_id 分组。
- 改 `api/main.py`：新增 `POST /api/milvus/query` 端点 + `thread_id` 请求字段 + NO_PROXY 设置。
- 新建 `api/test_milvus_rag.py`：mock 外部依赖的单测。

**不做**（YAGNI）：
- 不做多轮对话（thread 仅用于 tracing 会话分组，检索单轮、不带历史）。
- 不加 online metric / 在线评分（纯 tracing，只看 span/trace/thread 的 input/output/耗时）。
- 不动现有 chroma `/api/query` 端点。
- 不涉及 Railway/Cloudflare 部署改动。

## DeepEval Tracing API（已实测 deepeval 4.0.3）

- `from deepeval.tracing import observe, update_current_trace, update_current_span`
- `@observe(type="agent"|"llm"|"retriever"|"tool"|<str>)` → 装饰函数即一个 **span**；顶层 observe 调用构成一个 **trace**。
- `update_current_trace(thread_id=..., input=..., output=..., name=...)` → 设 trace 级属性；**thread_id 在此设置**，Confident AI 的 Threads 视图按 thread_id 聚合 trace。
- `update_current_span(input=..., output=..., retrieval_context=...)` → 设 span 级属性。
- 读 `CONFIDENT_API_KEY` 自动后台异步上报，无需手动 flush。

**三类型映射**：span = 每个 `@observe` 函数；trace = 顶层 agent span 包住的整次请求；thread = trace 上的 thread_id 会话分组。

## 架构 / 数据流

```
POST /api/milvus/query  {query, thread_id?}
        │
        ▼  api/milvus_rag.py
┌────────────────────────────────────────────────────────┐
│ @observe(type="agent")  milvus_rag_query(query, tid)    │ ← TRACE
│   update_current_trace(thread_id=tid, input=query,      │ ← THREAD
│                        output=answer, name=...)         │
│   │                                                      │
│   ├─ @observe(type="retriever") _retrieve_span(query)   │ ← SPAN 1
│   │     dense(embedding-3) + sparse(BM25) + 服务端 RRF   │
│   │     → top RETRIEVE_TOP_K docs                        │
│   │     update_current_span(input=query,                │
│   │                         retrieval_context=[doc...])  │
│   │                                                      │
│   ├─ @observe(type="retriever") _rerank_span(query,docs)│ ← SPAN 2
│   │     common.zhipu_rerank.rerank → top RERANK_TOP_K    │
│   │     update_current_span(input=query,                │
│   │                         output=[reranked...])        │
│   │                                                      │
│   └─ @observe(type="llm") _generate_span(query, ctx)    │ ← SPAN 3
│         ChatOpenAI(MODEL_ID) 生成答案                     │
│         update_current_span(input=prompt, output=answer) │
└────────────────────────────────────────────────────────┘
        │  DeepEval tracing 后台异步推送
        ▼
   Confident AI Observatory（CONFIDENT_API_KEY）
   Traces 视图：trace + 3 span 树；Threads 视图：按 thread_id 聚合
```

## 组件

### 1. `api/milvus_rag.py`（new）

模块常量（对齐 `app/milvus/hybrid_search.py`）：
```python
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"
DENSE_LIMIT = 10
SPARSE_LIMIT = 10
RETRIEVE_TOP_K = 6
RERANK_TOP_K = 2
RRF_K = 60
```

四个函数：

**`_retrieve_span(query: str) -> list[Document]`** — `@observe(type="retriever")`
- `ZhipuAIEmbeddings(model="embedding-3").embed_query(query)` 拿 dense 向量
- `MilvusClient(uri=MILVUS_URI)` + `AnnSearchRequest`（dense field `vector` COSINE，sparse field `sparse_vector` BM25）+ `hybrid_search(ranker=RRFRanker(k=RRF_K), limit=RETRIEVE_TOP_K, output_fields=["text","source"])`
- 转 `langchain_core.documents.Document(page_content=text, metadata={"source":...})`
- `update_current_span(input=query, retrieval_context=[d.page_content for d in docs])`
- return docs

**`_rerank_span(query: str, docs: list[Document]) -> list[Document]`** — `@observe(type="retriever")`
- `from common.zhipu_rerank import rerank`；`reranked = rerank(query, docs, top_n=RERANK_TOP_K)`
- `update_current_span(input=query, output=[d.page_content for d in reranked])`
- return reranked

**`_generate_span(query: str, context: str) -> str`** — `@observe(type="llm")`
- `ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)`
- system prompt 同 hybrid_search.rag（仅依据检索结果回答，不编造）
- `update_current_span(input=query, output=answer)`
- return answer

**`milvus_rag_query(query: str, thread_id: str | None = None) -> str`** — `@observe(type="agent")`
- `docs = _retrieve_span(query)`
- `reranked = _rerank_span(query, docs)`
- `context = "\n\n".join(f"Source: {d.metadata.get('source','')}\nContent: {d.page_content}" for d in reranked)`
- `answer = _generate_span(query, context)`
- `update_current_trace(thread_id=thread_id, input=query, output=answer, name="milvus-hybrid-rag")`
- return answer

> 说明：本模块自己写 retrieve + rerank 两步（复用底层 embedding/MilvusClient/RRFRanker/zhipu rerank），不调 `hybrid_search._retrieve`（那个把 retrieve+rerank 揉成一个函数、且带 langfuse 装饰 + 历史改写），以便拆出独立 span。

### 2. `api/main.py`（modify）

顶部（在任何 import 触发 milvus 连接前）：
```python
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
```

`QueryRequest` 加字段：
```python
class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
```

新增端点：
```python
@app.post("/api/milvus/query")
def milvus_query(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    try:
        from api.milvus_rag import milvus_rag_query
    except KeyError as e:
        raise HTTPException(status_code=503, detail=f"环境变量缺失：{e}") from e
    try:
        answer = milvus_rag_query(req.query, thread_id=req.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 调用失败：{type(e).__name__}: {e}") from e
    return QueryResponse(answer=answer, version=APP_VERSION)
```

现有 `/api/query`、`/api/health`、`/` 不动。`thread_id` 字段对老端点无影响（可选）。

### 3. `api/test_milvus_rag.py`（new）

mock 外部依赖（monkeypatch `_retrieve_span` / `_rerank_span` / `_generate_span` 内部用到的 embedding/client/rerank/ChatOpenAI，或直接 monkeypatch 三个 span 函数），验证：
- `milvus_rag_query` 串起三步并返回生成答案
- `thread_id` 透传：monkeypatch `update_current_trace`，断言被调用且 `thread_id` 参数正确
- query→retrieve→rerank→generate 调用顺序

不真连 Milvus / 智谱 / GLM / Confident AI。

## 错误处理

| 场景 | 行为 |
|---|---|
| query 空 | 400 |
| `MODEL_ID` 等 env 缺失 | 503（import 时 KeyError） |
| 检索/生成异常 | 500；DeepEval trace 仍记录（带 error），可在 Confident 看失败 trace |
| `thread_id` 不传 | DeepEval 自动生成独立 trace，不归会话，不报错 |
| `CONFIDENT_API_KEY` 缺失 | tracing 静默不上报，API 仍正常返回答案 |
| Milvus 不可达 / 未设 NO_PROXY | 检索 hang/超时 → 500（NO_PROXY 已在 main.py 设，规避 Privoxy gRPC 拦截） |

## 测试

| 类型 | 文件 | 验证内容 |
|---|---|---|
| 单测 | `api/test_milvus_rag.py` | 三 span 串联 + thread_id 透传 + 返回结构（全 mock） |
| 冒烟 | 手动 | 起 API（带 NO_PROXY + Milvus 在线 + .env），`curl -XPOST /api/milvus/query -d '{"query":"...","thread_id":"t1"}'`，去 Confident AI Observatory 看 trace 树 + Threads 视图 |

## 风险

- **DeepEval tracing 与评测 `evaluate()` 共用 CONFIDENT_API_KEY**：tracing 进 Observatory，评测进 test-runs，互不冲突。
- **NO_PROXY 坑**：API 进程同样受 Privoxy 影响，main.py 顶部设置是必须项（参见既往 Milvus proxy 教训）。
- **deepeval 4.0.3 tracing API**：observe/update_current_trace/update_current_span 签名已实测确认，不凭记忆。
- **后台异步上报**：FastAPI 单次请求返回后 tracing 在后台推送，偶发网络慢可能延迟出现在 Confident AI，非 bug。
