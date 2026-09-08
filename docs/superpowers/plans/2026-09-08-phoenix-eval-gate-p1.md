# Phoenix 离线评估门禁（期 1）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 PR 上一个变坏的 prompt/检索改动被自动挡下来——真跑 Milvus RAG 管线、判官打分、声明式验收条件聚合、pytest 退出码当 PR check。

**Architecture:** 本地自托管 Phoenix（:6006）收 trace 与 experiment；`arize-phoenix-client[pytest]` 插件把测试套件映射成 dataset/experiment；判官是 `ClassificationEvaluator` 打 MiniMax-m3；声明式验收条件（Arize 只在 TS 侧提供，Python 侧自造）在 `pytest_sessionfinish` 聚合并决定退出码；跑在本机 self-hosted runner 上，因为真管线依赖 `localhost:19530` 的 Milvus。

**Tech Stack:** arize-phoenix 20.8.0 · arize-phoenix-client[pytest,evals] 3.4.0 · openinference-instrumentation-langchain 0.1.74 · pytest · FastAPI · Milvus · MiniMax-m3 @ Vercel AI Gateway

**Spec:** `docs/superpowers/specs/2026-09-08-phoenix-eval-cicd-design.md`

## Global Constraints

- Python 3.10（`.python-version`）。arize-phoenix 系列要求 `>=3.10,<3.15`，兼容。
- **平行新建，不动旧栈**：`evaluation/mlflow_*.py`、`evaluation/evaluate.py`、`api/milvus_rag.py`、`api/milvus_rag_mlflow.py` 一行不改。
- **`NO_PROXY` 必须包含 `localhost,127.0.0.1`**，且在任何 `phoenix`/`milvus` import 之前设置。本仓库已因 Privoxy 拦 localhost 踩过三次（见 `api/main.py`、`evaluation/mlflow_evaluate.py` 顶部注释）。
- 判官凭证只从环境变量读，**不得出现在任何被提交的文件里**：`JUDGE_OPENAI_API_KEY` / `JUDGE_OPENAI_BASE_URL` / `JUDGE_MODEL_ID`（值已在 `.env`，`.env` 已 gitignore）。
- Phoenix 两个环境变量都要：`PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006`（OTel 发送端）与 `PHOENIX_ENDPOINT=http://localhost:6006`（`phoenix.client.Client` 读的是这个）。
- `/docs` 在 `.gitignore` 里，提交 docs 下的文件一律 `git add -f`（与已有 spec/plan 一致）。
- commit 前缀用 conventional commits；`feat:`/`fix:` 会触发 semantic-release 发版，纯评估设施用 `feat(eval):`，脚本与 CI 用 `ci:`/`chore:`。
- 判官输出一律**分类标签**，不让模型直接吐浮点数（Phoenix 官方建议，且 `glm-4-flash`/`minimax-m3` 这类模型数值推理不稳）。
- 跑评估套件时必须绕开仓库默认 addopts（`--cov=common`）：`pytest evaluation/phoenix -o addopts=""`。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `scripts/phoenix-up.ps1` | 起本地 Phoenix，固化 NO_PROXY 与端口 |
| `api/milvus_rag_phoenix.py` | Phoenix 埋点版 RAG 管线；导出 `milvus_rag_phoenix_query`（只回答案）与 `milvus_rag_phoenix_query_with_context`（回答案 + 检索片段） |
| `api/main.py` | 只加一个端点 `POST /api/milvus/query-phoenix` |
| `evaluation/phoenix/cases.py` | 读 `test_dataset.json`，算 content-hash example_id，导出 `CASES` / `CASE_IDS` |
| `evaluation/phoenix/dataset.py` | 把 CASES 推进 Phoenix dataset（`example_id_key`） |
| `evaluation/phoenix/evaluators.py` | 5 个判官（template 层），每个既返回插件要的 dict，也向 acceptance 登记 |
| `evaluation/phoenix/acceptance.py` | 自造：结果累加器 + 声明式验收条件求值 |
| `evaluation/phoenix/criteria.yaml` | 阈值声明（期 1 只用 `offline` 段） |
| `evaluation/phoenix/conftest.py` | 挂 acceptance 到 `pytest_sessionfinish` |
| `evaluation/phoenix/test_rag_eval.py` | 单轮门禁套件 |
| `tests/test_phoenix_cases.py` | 单测：content-hash 稳定性与幂等 |
| `tests/test_phoenix_acceptance.py` | 单测：验收条件求值的四条取舍 |
| `.github/workflows/eval-gate.yml` | PR/push 门禁 workflow（self-hosted） |

---

### Task 1: 环境就位 + 判官结构化输出验证

装依赖、起 Phoenix、并用一次真实调用证明 `ClassificationEvaluator` 能驱动 MiniMax-m3 产出结构化分类结果。这一步先做，因为它是唯一可能整体推翻方案的风险点（网关上的模型不一定支持 Phoenix 用的 structured-output/tool-calling 路径）。

**Files:**
- Create: `scripts/phoenix-up.ps1`
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `.env` 里的 `JUDGE_OPENAI_API_KEY` / `JUDGE_OPENAI_BASE_URL` / `JUDGE_MODEL_ID`
- Produces: 本机 `http://localhost:6006` 上可用的 Phoenix；确认可用的判官 LLM 配置

- [ ] **Step 1: 追加依赖**

在 `requirements.txt` 末尾（`pyjwt>=2.12.0` 那段安全约束之前）追加：

```
# Phoenix 评估栈（与现有 mlflow / deepeval 平行，互不影响）
arize-phoenix
arize-phoenix-client[pytest,evals]
openinference-instrumentation-langchain
pyyaml
```

- [ ] **Step 2: 安装并确认版本**

```bash
pip install arize-phoenix "arize-phoenix-client[pytest,evals]" openinference-instrumentation-langchain pyyaml
python -c "import phoenix, phoenix.client, phoenix.evals; print(phoenix.__version__)"
```
Expected: 打印 20.x 或更高，无 ImportError。

- [ ] **Step 3: 写启动脚本**

创建 `scripts/phoenix-up.ps1`：

```powershell
# 起本地 Phoenix（评估 CI/CD 用）。
# localhost 必须进 NO_PROXY：本机 Privoxy 会拦 127.0.0.1，表现为莫名其妙的连接失败。
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = $env:NO_PROXY
$env:PHOENIX_WORKING_DIR = Join-Path $PSScriptRoot "..\.phoenix"
if (-not (Test-Path $env:PHOENIX_WORKING_DIR)) {
    New-Item -ItemType Directory -Force $env:PHOENIX_WORKING_DIR | Out-Null
}
Write-Host "Phoenix working dir: $env:PHOENIX_WORKING_DIR"
Write-Host "UI: http://localhost:6006"
phoenix serve
```

- [ ] **Step 4: 忽略 Phoenix 数据目录**

在 `.gitignore` 末尾追加一行：

```
/.phoenix
```

- [ ] **Step 5: 起服务并确认健康**

在一个单独终端跑 `powershell -File scripts/phoenix-up.ps1`，然后在另一个终端：

```bash
NO_PROXY=localhost,127.0.0.1 curl -s -o /dev/null -w "%{http_code}\n" http://localhost:6006
```
Expected: `200`

- [ ] **Step 6: 写判官冒烟脚本并运行**

创建临时文件 `/tmp/judge_smoke.py`（**不提交**）：

```python
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
from dotenv import load_dotenv
load_dotenv()

from phoenix.evals import ClassificationEvaluator
from phoenix.evals.llm import LLM

llm = LLM(
    provider="openai",
    model=os.environ["JUDGE_MODEL_ID"],
    base_url=os.environ["JUDGE_OPENAI_BASE_URL"],
    api_key=os.environ["JUDGE_OPENAI_API_KEY"],
)
ev = ClassificationEvaluator(
    name="faithfulness",
    prompt_template=(
        "判断回答是否忠实于检索上下文。\n"
        "[检索上下文]\n{reference}\n\n[回答]\n{output}\n\n"
        "correct = 回答中的事实陈述全部能在检索上下文中找到依据；"
        "incorrect = 完全无依据或与上下文冲突；partial = 部分有依据。"
        "请用中文写理由。"
    ),
    llm=llm,
    choices={"incorrect": 0.0, "partial": 0.5, "correct": 1.0},
    direction="maximize",
)
scores = ev.evaluate({
    "reference": "公司年假为每年10天，入职满3年增至15天。",
    "output": "公司年假是每年20天。",
})
for s in scores:
    print("label:", s.label, "| score:", s.score, "| explanation:", s.explanation)
```

Run: `python /tmp/judge_smoke.py`
Expected: 打印 `label: incorrect | score: 0.0 | explanation: <中文>`。

**如果这里报结构化输出/tool-calling 不支持**，说明网关上的 minimax-m3 走不通 Phoenix 的 structured-output 路径。回退方案：把 `LLM(provider="openai", ...)` 换成 `ClassificationEvaluator` 之外的自写 `LLMEvaluator` 子类，用 `response_format={"type":"json_object"}` 手工解析（已实测该模型支持 json_object）。**先按上面跑，跑通就不用回退。**

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore scripts/phoenix-up.ps1
git commit -m "feat(eval): Phoenix 评估栈依赖与本地启动脚本"
```

---

### Task 2: Phoenix 埋点版 RAG 管线

**Files:**
- Create: `api/milvus_rag_phoenix.py`
- Modify: `api/main.py`（在 `milvus_query_mlflow` 之后追加一个端点）
- Test: 手工 curl（此任务的产物是一条真实链路，单测覆盖不了外部依赖）

**Interfaces:**
- Consumes: Task 1 的 Phoenix 服务
- Produces:
  - `milvus_rag_phoenix_query(query: str, session_id: str | None = None) -> str`
  - `milvus_rag_phoenix_query_with_context(query: str, session_id: str | None = None) -> tuple[str, list[str]]`

- [ ] **Step 1: 读参照实现**

先通读 `api/milvus_rag_mlflow.py`，照它的结构写 Phoenix 版：同样的检索 + 生成逻辑，只换 tracing 承载。**不要改动那个文件。**

- [ ] **Step 2: 写 Phoenix 版管线**

创建 `api/milvus_rag_phoenix.py`（span 结构对齐 `milvus_rag_mlflow.py`，检索/重排/生成的实际逻辑与常量照抄，只换 tracing 承载）：

```python
"""Milvus 混合检索流水线，Phoenix / OpenInference tracing。

与 api/milvus_rag_mlflow.py 平行存在：检索、重排、生成逻辑与常量完全一致，
只把 tracing 承载从 MLflow 换成 Phoenix。ChatOpenAI 的 LLM span 由
openinference-instrumentation-langchain 自动产出，不用手写。

与 MLflow 版的一处有意差异：本模块把**重排后**的片段作为返回值交出去，
判官直接用它，不再回头去 trace 里刨 RETRIEVER span。好处有两个——
判官看到的上下文与真正喂给 LLM 的完全同一份（MLflow 版的 scorer 取的是
重排前的 6 条，与生成实际用的 2 条不一致），且少一层"提取失败"的失败模式。
"""

import os

# 本机 Milvus 走 gRPC、Phoenix collector 走 HTTP，本机 Privoxy 会拦 localhost。
# 必须在任何 milvus / phoenix import 之前设置。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from phoenix.otel import register
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from api.history_utils import build_chat_messages, build_rewrite_prompt

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"
DENSE_LIMIT = 10
SPARSE_LIMIT = 10
RETRIEVE_TOP_K = 6
RERANK_TOP_K = 2
RRF_K = 60

# project_name 由环境变量决定：CI 里 bz-rag-ci，影子 canary 上 bz-rag-canary。
_tracer_provider = register(
    project_name=os.environ.get("PHOENIX_PROJECT_NAME", "bz-rag-ci"),
    endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    + "/v1/traces",
    auto_instrument=True,  # 挂载已安装的 openinference instrumentor（LangChain）
)
_tracer = _tracer_provider.get_tracer(__name__)


@_tracer.chain
def _rewrite_query(query: str, history: list[tuple[str, str]]) -> str:
    # 用对话历史把指代追问改写成独立问题，供检索用。temperature=0 保证可复现，
    # 改写失败（空串）退回原问题兜底。
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.0, request_timeout=60)
    msg = llm.invoke([{"role": "user", "content": build_rewrite_prompt(query, history)}])
    return (msg.content or "").strip() or query


def _retrieve(query: str) -> list[Document]:
    with _tracer.start_as_current_span(
        "retrieve", openinference_span_kind="retriever"
    ) as span:
        span.set_input(query)
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
        span.set_output([{"page_content": d.page_content, "metadata": d.metadata} for d in docs])
        return docs


@_tracer.chain
def _rerank(query: str, docs: list[Document]) -> list[Document]:
    from common.zhipu_rerank import rerank

    return rerank(query, docs, top_n=RERANK_TOP_K)


@_tracer.chain
def _generate(query: str, context: str, history: list[tuple[str, str]] | None = None) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"

--- 检索结果 ---
{context}"
    )
    msg = llm.invoke(build_chat_messages(system_prompt, query, history))
    return msg.content or ""


def milvus_rag_phoenix_query_with_context(
    query: str,
    session_id: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """跑一次检索 + 生成，返回 (答案, 重排后喂给 LLM 的片段列表)。"""
    with _tracer.start_as_current_span(
        "milvus-hybrid-rag", openinference_span_kind="agent"
    ) as span:
        span.set_input(query)
        if session_id:
            # OpenInference 的会话分组约定属性，多轮评估按它分组。
            span.set_attribute("session.id", session_id)
        search_query = _rewrite_query(query, history) if history else query
        docs = _retrieve(search_query)
        reranked = _rerank(search_query, docs)
        context = "

".join(
            f"Source: {d.metadata.get('source', '')}
Content: {d.page_content}"
            for d in reranked
        )
        answer = _generate(query, context, history)
        span.set_output(answer)
    return answer, [d.page_content for d in reranked]


def milvus_rag_phoenix_query(
    query: str,
    session_id: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    answer, _ = milvus_rag_phoenix_query_with_context(query, session_id, history)
    return answer
```

- [ ] **Step 3: 加 API 端点**

在 `api/main.py` 的 `milvus_query_mlflow` 函数之后追加：

```python
@app.post("/api/milvus/query-phoenix")
def milvus_query_phoenix(req: QueryRequest) -> QueryResponse:
    """Milvus 混合检索 + Phoenix tracing（span/trace → 本地 Phoenix），供评估门禁与线上评估。"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    try:
        from api.milvus_rag_phoenix import milvus_rag_phoenix_query
    except KeyError as e:
        raise HTTPException(status_code=503, detail=f"环境变量缺失：{e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Phoenix 初始化失败：{type(e).__name__}: {e}",
        ) from e

    try:
        answer = milvus_rag_phoenix_query(req.query, session_id=req.thread_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG 调用失败：{type(e).__name__}: {e}",
        ) from e

    return QueryResponse(answer=answer, version=APP_VERSION)
```

- [ ] **Step 4: 起服务并打一次真实请求**

```bash
NO_PROXY=localhost,127.0.0.1 python -m uvicorn api.main:app --port 8000 &
sleep 5
NO_PROXY=localhost,127.0.0.1 curl -s -X POST http://localhost:8000/api/milvus/query-phoenix \
  -H "Content-Type: application/json" \
  -d '{"query":"禾蛙平台是什么类型的平台？"}'
```
Expected: 返回 `{"answer": "...人力资源供应链内容生态平台...", "version": "dev"}`

- [ ] **Step 5: 确认 trace 落到 Phoenix 且带 RETRIEVER span**

打开 `http://localhost:6006`，选 project `bz-rag-ci`，应能看到一条 trace，展开后有 LangChain 自动产出的 RETRIEVER span，其 output 里是检索到的文档。

Expected: RETRIEVER span 存在。若没有，检查 `openinference-instrumentation-langchain` 是否装上，以及 `register(auto_instrument=True)` 是否在 LangChain import 之前执行。

- [ ] **Step 6: Commit**

```bash
git add api/milvus_rag_phoenix.py api/main.py
git commit -m "feat(eval): Phoenix 埋点版 Milvus RAG 管线 + query-phoenix 端点"
```

---

### Task 3: golden 集与内容寻址 example_id

**Files:**
- Create: `evaluation/phoenix/__init__.py`（空文件）
- Create: `evaluation/phoenix/cases.py`
- Create: `evaluation/phoenix/dataset.py`
- Test: `tests/test_phoenix_cases.py`

**Interfaces:**
- Consumes: `evaluation/test_dataset.json`（24 条，字段 `question` / `ground_truth` / `expected_source`）
- Produces:
  - `example_id(question: str) -> str`（16 位十六进制）
  - `CASES: list[Case]`，`Case` 是 `NamedTuple(example_id, question, ground_truth, expected_source)`
  - `CASE_IDS: list[str]`
  - `SMOKE_IDS: frozenset[str]`（前 3 条的 example_id）

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_phoenix_cases.py`：

```python
from evaluation.phoenix.cases import CASES, CASE_IDS, SMOKE_IDS, example_id


def test_example_id_is_stable_and_16_hex():
    a = example_id("禾蛙平台是什么类型的平台？")
    b = example_id("禾蛙平台是什么类型的平台？")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_example_id_ignores_surrounding_whitespace():
    assert example_id(" 同一个问题 ") == example_id("同一个问题")


def test_example_id_differs_for_different_questions():
    assert example_id("问题一") != example_id("问题二")


def test_cases_loaded_and_ids_unique():
    assert len(CASES) >= 20
    assert len(CASE_IDS) == len(CASES)
    assert len(set(CASE_IDS)) == len(CASE_IDS)
    assert CASE_IDS == [c.example_id for c in CASES]


def test_smoke_subset_is_small_and_is_a_subset():
    assert 0 < len(SMOKE_IDS) < len(CASES)
    assert SMOKE_IDS <= set(CASE_IDS)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phoenix_cases.py -o addopts="" -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix'`

- [ ] **Step 3: 实现**

创建 `evaluation/phoenix/__init__.py`（空）。创建 `evaluation/phoenix/cases.py`：

```python
"""golden 集加载与内容寻址 ID。

example_id 用问题文本的 sha256 前 16 位。Phoenix 的 example_id_key 要求调用方自己提供
稳定 ID（官方举例"数据库主键或 content hash"），用内容哈希的好处是"同一批 case"
变成可计算的集合——改了问题文本就是一条新 case，不改就永远映射回同一个 example。
"""

import hashlib
import json
import pathlib
from typing import NamedTuple

DATASET_PATH = pathlib.Path(__file__).resolve().parents[1] / "test_dataset.json"
SMOKE_SIZE = 3


class Case(NamedTuple):
    example_id: str
    question: str
    ground_truth: str
    expected_source: str


def example_id(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]


def _load() -> list[Case]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        items = json.load(f)
    return [
        Case(
            example_id=example_id(item["question"]),
            question=item["question"],
            ground_truth=item["ground_truth"],
            expected_source=item["expected_source"],
        )
        for item in items
    ]


CASES: list[Case] = _load()
CASE_IDS: list[str] = [c.example_id for c in CASES]
# smoke 子集：PR 上只跑这几条，控制判官调用量。取前 N 条而不是随机，保证可比。
SMOKE_IDS: frozenset[str] = frozenset(CASE_IDS[:SMOKE_SIZE])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phoenix_cases.py -o addopts="" -v`
Expected: 5 passed

- [ ] **Step 5: 写 dataset 推送脚本**

创建 `evaluation/phoenix/dataset.py`：

```python
"""把 golden 集推进 Phoenix dataset。

用 create_dataset + example_id_key：Phoenix 会 diff 出增/改/删，让 dataset 精确等于
本地文件（上传里没有的 example 会被删）。回灌走 add_examples_to_dataset（只增改不删），
那是期 3 的事。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from phoenix.client import Client

from evaluation.phoenix.cases import CASES

DATASET_NAME = os.environ.get("PHOENIX_TEST_DATASET", "bz-rag-golden")


def main() -> None:
    df = pd.DataFrame(
        [
            {
                "example_id": c.example_id,
                "question": c.question,
                "expected_response": c.ground_truth,
                "expected_source": c.expected_source,
            }
            for c in CASES
        ]
    )
    dataset = Client().datasets.create_dataset(
        name=DATASET_NAME,
        dataframe=df,
        input_keys=["question"],
        output_keys=["expected_response"],
        metadata_keys=["expected_source"],
        example_id_key="example_id",
    )
    print(f"dataset {dataset.name!r} version {dataset.version_id} 共 {dataset.example_count} 条")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑两次验证幂等**

```bash
NO_PROXY=localhost,127.0.0.1 PHOENIX_ENDPOINT=http://localhost:6006 python -m evaluation.phoenix.dataset
NO_PROXY=localhost,127.0.0.1 PHOENIX_ENDPOINT=http://localhost:6006 python -m evaluation.phoenix.dataset
```
Expected: 两次都打印 24 条；Phoenix UI 上 dataset 只有一个，example 数没翻倍（版本号会变，example 内容不变）。

- [ ] **Step 7: Commit**

```bash
git add evaluation/phoenix/__init__.py evaluation/phoenix/cases.py evaluation/phoenix/dataset.py tests/test_phoenix_cases.py
git commit -m "feat(eval): golden 集内容寻址 example_id + 推送 Phoenix dataset"
```

---

### Task 4: 声明式验收条件（自造）

先做 acceptance 再做判官，因为判官要向 acceptance 登记结果，接口方向是 acceptance → 判官。

**Files:**
- Create: `evaluation/phoenix/acceptance.py`
- Create: `evaluation/phoenix/criteria.yaml`
- Test: `tests/test_phoenix_acceptance.py`

**Interfaces:**
- Produces:
  - `reset() -> None`
  - `record(name: str, score: float | None, label: str | None = None, error: str | None = None) -> None`
  - `Criterion`（dataclass：`annotation`、`metric`、`threshold`、`direction`、`pass_when`、`min_pass_rate`、`min_samples`）
  - `load_criteria(path: str, section: str) -> list[Criterion]`
  - `evaluate_all(criteria: list[Criterion]) -> list[Outcome]`
  - `Outcome`（dataclass：`criterion`、`passed`、`observed`、`required`、`samples`、`reason`）
  - `format_scoreboard(outcomes: list[Outcome]) -> str`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_phoenix_acceptance.py`：

```python
import pytest

from evaluation.phoenix import acceptance as acc


@pytest.fixture(autouse=True)
def _clean():
    acc.reset()
    yield
    acc.reset()


def _crit(**kw):
    base = dict(annotation="faithfulness", metric="average", threshold=0.8)
    base.update(kw)
    return acc.Criterion(**base)


def test_average_passes_when_mean_clears_threshold():
    for s in (1.0, 1.0, 0.5):
        acc.record("faithfulness", s)
    (out,) = acc.evaluate_all([_crit(threshold=0.8)])
    assert out.passed
    assert out.observed == pytest.approx(0.8333, abs=1e-3)
    assert out.samples == 3


def test_average_fails_when_mean_below_threshold():
    for s in (0.5, 0.5, 1.0):
        acc.record("faithfulness", s)
    (out,) = acc.evaluate_all([_crit(threshold=0.8)])
    assert not out.passed


def test_direction_minimize_inverts_comparison():
    for s in (100.0, 200.0):
        acc.record("latency_ms", s)
    (out,) = acc.evaluate_all(
        [acc.Criterion(annotation="latency_ms", metric="average",
                       threshold=800, direction="minimize")]
    )
    assert out.passed


def test_pass_rate_uses_pass_when_expression():
    for s in (1.0, 1.0, 1.0, 0.0):
        acc.record("faithfulness", s)
    (out,) = acc.evaluate_all(
        [acc.Criterion(annotation="faithfulness", metric="pass_rate",
                       pass_when="score >= 0.5", min_pass_rate=0.9)]
    )
    assert not out.passed
    assert out.observed == pytest.approx(0.75)


def test_pass_when_can_read_label():
    acc.record("recall", 0.5, label="partial")
    acc.record("recall", 0.0, label="incorrect")
    (out,) = acc.evaluate_all(
        [acc.Criterion(annotation="recall", metric="pass_rate",
                       pass_when="label != 'incorrect'", min_pass_rate=1.0)]
    )
    assert not out.passed


# ——— 四条取舍，逐条锁死 ———

def test_missing_annotation_fails_rather_than_passing_vacuously():
    acc.record("something_else", 1.0)
    (out,) = acc.evaluate_all([_crit()])
    assert not out.passed
    assert "no faithfulness" in out.reason


def test_errored_judgement_is_a_third_state_not_a_zero():
    acc.record("faithfulness", 1.0)
    acc.record("faithfulness", None, error="judge timeout")
    (out,) = acc.evaluate_all([_crit(threshold=0.9)])
    # errored 不参与均值（否则等于判 0 分），但要在 reason 里显式报出来
    assert out.observed == pytest.approx(1.0)
    assert out.samples == 1
    assert "1 errored" in out.reason


def test_all_criteria_are_evaluated_even_when_the_first_fails():
    acc.record("a", 0.0)
    acc.record("b", 0.0)
    outs = acc.evaluate_all(
        [acc.Criterion(annotation="a", metric="average", threshold=0.5),
         acc.Criterion(annotation="b", metric="average", threshold=0.5)]
    )
    assert len(outs) == 2
    assert not any(o.passed for o in outs)


def test_min_samples_yields_insufficient_not_pass():
    acc.record("faithfulness", 1.0)
    (out,) = acc.evaluate_all([_crit(min_samples=5)])
    assert not out.passed
    assert "insufficient samples" in out.reason


def test_load_criteria_reads_the_named_section(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "offline:\n"
        "  - annotation: faithfulness\n"
        "    metric: average\n"
        "    threshold: 0.8\n"
        "online:\n"
        "  - annotation: faithfulness\n"
        "    metric: pass_rate\n"
        "    pass_when: \"score >= 0.5\"\n"
        "    min_pass_rate: 0.85\n",
        encoding="utf-8",
    )
    offline = acc.load_criteria(str(p), "offline")
    online = acc.load_criteria(str(p), "online")
    assert len(offline) == 1 and offline[0].metric == "average"
    assert len(online) == 1 and online[0].min_pass_rate == 0.85


def test_scoreboard_reports_observed_required_and_samples():
    acc.record("faithfulness", 1.0)
    board = acc.format_scoreboard(acc.evaluate_all([_crit(threshold=0.8)]))
    assert "faithfulness" in board
    assert "1.000" in board
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_phoenix_acceptance.py -o addopts="" -v`
Expected: FAIL，`ModuleNotFoundError` 或 `AttributeError: module has no attribute 'Criterion'`

- [ ] **Step 3: 实现**

创建 `evaluation/phoenix/criteria.yaml`：

```yaml
# 离线门禁与线上 monitor 共用的阈值声明。
# metric: average   —— 均值必须过 threshold（direction: maximize 默认 / minimize）
# metric: pass_rate —— 每条按 pass_when 判通过，通过比例必须达到 min_pass_rate
offline:
  - annotation: faithfulness
    metric: average
    threshold: 0.8
  - annotation: faithfulness
    metric: pass_rate
    pass_when: "score >= 0.5"
    min_pass_rate: 0.9
  - annotation: answer_relevancy
    metric: average
    threshold: 0.8
  - annotation: contextual_recall
    metric: pass_rate
    pass_when: "label != 'incorrect'"
    min_pass_rate: 1.0
  - annotation: refusal_check
    metric: pass_rate
    pass_when: "label == 'ok'"
    min_pass_rate: 1.0

# 期 3 用；线上比离线松一档，且样本量不足要判 hold 而不是通过。
online:
  - annotation: faithfulness
    metric: pass_rate
    pass_when: "score >= 0.5"
    min_pass_rate: 0.85
    min_samples: 20
  - annotation: refusal_check
    metric: pass_rate
    pass_when: "label == 'ok'"
    min_pass_rate: 0.98
    min_samples: 20
```

创建 `evaluation/phoenix/acceptance.py`：

```python
"""声明式验收条件——Arize 只在 TypeScript 侧提供 acceptanceCriteria，这是 Python 版。

设计上刻意抄了 Arize 的四条取舍，它们才是价值所在：
  1. 所有 case 跑完之后才判，一次看到全部回归，而不是第一个失败就中断
  2. 缺失的指标判失败，不 vacuously pass
  3. 判官报错是第三态（errored），既不算 0 分也不算通过
  4. 结果落库失败只 warn，不影响门禁判定——所以本模块完全不读 Phoenix，只读进程内累加器
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field

import yaml

_ALLOWED_METRICS = ("average", "pass_rate")


@dataclass
class Record:
    score: float | None
    label: str | None = None
    error: str | None = None


@dataclass
class Criterion:
    annotation: str
    metric: str
    threshold: float | None = None
    direction: str = "maximize"
    pass_when: str | None = None
    min_pass_rate: float | None = None
    min_samples: int = 1

    def __post_init__(self) -> None:
        if self.metric not in _ALLOWED_METRICS:
            raise ValueError(f"metric 必须是 {_ALLOWED_METRICS} 之一，收到 {self.metric!r}")
        if self.metric == "average" and self.threshold is None:
            raise ValueError(f"{self.annotation}: metric=average 必须给 threshold")
        if self.metric == "pass_rate" and (self.pass_when is None or self.min_pass_rate is None):
            raise ValueError(f"{self.annotation}: metric=pass_rate 必须给 pass_when 与 min_pass_rate")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction 必须是 maximize/minimize，收到 {self.direction!r}")


@dataclass
class Outcome:
    criterion: Criterion
    passed: bool
    observed: float | None
    required: float
    samples: int
    reason: str


_RECORDS: dict[str, list[Record]] = {}


def reset() -> None:
    _RECORDS.clear()


def record(
    name: str, score: float | None, label: str | None = None, error: str | None = None
) -> None:
    _RECORDS.setdefault(name, []).append(Record(score=score, label=label, error=error))


# —— pass_when 求值：只允许比较运算与 score/label 两个名字，不用 eval ——

_CMP = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _eval_pass_when(expr: str, rec: Record) -> bool:
    tree = ast.parse(expr, mode="eval").body

    def val(node):
        if isinstance(node, ast.Name):
            if node.id == "score":
                return rec.score
            if node.id == "label":
                return rec.label
            raise ValueError(f"pass_when 只能引用 score / label，收到 {node.id!r}")
        if isinstance(node, ast.Constant):
            return node.value
        raise ValueError(f"pass_when 不支持的表达式节点：{type(node).__name__}")

    def run(node) -> bool:
        if isinstance(node, ast.BoolOp):
            results = [run(v) for v in node.values]
            return all(results) if isinstance(node.op, ast.And) else any(results)
        if isinstance(node, ast.Compare):
            left = val(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = val(comparator)
                if left is None or right is None:
                    return False
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        raise ValueError(f"pass_when 不支持的表达式节点：{type(node).__name__}")

    return run(tree)


def _evaluate_one(c: Criterion) -> Outcome:
    recs = _RECORDS.get(c.annotation, [])
    errored = [r for r in recs if r.error is not None]
    usable = [r for r in recs if r.error is None]
    err_note = f"，{len(errored)} errored" if errored else ""

    required = c.threshold if c.metric == "average" else c.min_pass_rate

    if not usable:
        return Outcome(c, False, None, required, 0,
                       f"no {c.annotation} scores found{err_note}")

    if c.metric == "average":
        numeric = [r.score for r in usable if r.score is not None]
        if not numeric:
            return Outcome(c, False, None, required, 0,
                           f"no {c.annotation} numeric scores found{err_note}")
        if len(numeric) < c.min_samples:
            return Outcome(c, False, sum(numeric) / len(numeric), required, len(numeric),
                           f"insufficient samples: {len(numeric)} < {c.min_samples}{err_note}")
        observed = sum(numeric) / len(numeric)
        passed = observed >= c.threshold if c.direction == "maximize" else observed <= c.threshold
        cmp = ">=" if c.direction == "maximize" else "<="
        return Outcome(c, passed, observed, required, len(numeric),
                       f"mean {observed:.3f} {cmp} {c.threshold}{err_note}")

    if len(usable) < c.min_samples:
        return Outcome(c, False, None, required, len(usable),
                       f"insufficient samples: {len(usable)} < {c.min_samples}{err_note}")
    passing = sum(1 for r in usable if _eval_pass_when(c.pass_when, r))
    observed = passing / len(usable)
    passed = observed >= c.min_pass_rate
    return Outcome(c, passed, observed, required, len(usable),
                   f"pass rate {observed:.3f} >= {c.min_pass_rate}{err_note}")


def evaluate_all(criteria: list[Criterion]) -> list[Outcome]:
    """全部求值后再返回——一次看到所有回归，而不是第一个失败就短路。"""
    return [_evaluate_one(c) for c in criteria]


def load_criteria(path: str, section: str) -> list[Criterion]:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if section not in doc:
        raise KeyError(f"{path} 里没有 {section!r} 段")
    return [Criterion(**item) for item in doc[section]]


def format_scoreboard(outcomes: list[Outcome]) -> str:
    lines = ["", "Acceptance Criteria", "-" * 78,
             f"{'annotation':<22}{'metric':<12}{'observed':>10}{'required':>10}{'n':>6}  verdict"]
    for o in outcomes:
        obs = "n/a" if o.observed is None else f"{o.observed:.3f}"
        lines.append(
            f"{o.criterion.annotation:<22}{o.criterion.metric:<12}{obs:>10}"
            f"{o.required:>10.3f}{o.samples:>6}  {'PASS' if o.passed else 'FAIL'}"
        )
        if not o.passed:
            lines.append(f"{'':<22}└─ {o.reason}")
    lines.append("-" * 78)
    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_phoenix_acceptance.py -o addopts="" -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/phoenix/acceptance.py evaluation/phoenix/criteria.yaml tests/test_phoenix_acceptance.py
git commit -m "feat(eval): 声明式验收条件（Arize acceptanceCriteria 的 Python 版）"
```

---

### Task 5: 判官（template 层）

**Files:**
- Create: `evaluation/phoenix/evaluators.py`

**Interfaces:**
- Consumes: `acceptance.record`（Task 4）
- Produces: `EVALUATORS: list[callable]`，含 `faithfulness` / `answer_relevancy` / `contextual_precision` / `contextual_recall` / `refusal_check`。每个都是签名 `(output, input=None, expected=None, **_) -> dict`，返回 `{"name": ..., "score": ..., "label": ..., "explanation": ...}`。

- [ ] **Step 1: 实现**

创建 `evaluation/phoenix/evaluators.py`：

```python
"""判官（Arize 说的 evaluator template 层）——离线门禁与线上评估共用同一份。

prompt 沿用 evaluation/mlflow_evaluate.py 里已经调过中文 rationale 的那几条，
但输出形态从"让模型吐 0-1 浮点"改成分类标签 + 分数映射：Phoenix 官方明确建议
分类优于数值评分（模型的数值推理不稳，且与人类判断相关性更差）。

每个判官除了返回插件要的 dict，还向 acceptance 登记一条记录——判官抛异常时
登记成 errored（第三态），不静默丢弃、也不当成 0 分。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from dotenv import load_dotenv

load_dotenv()

from phoenix.evals import ClassificationEvaluator
from phoenix.evals.llm import LLM

from evaluation.phoenix import acceptance

_CHOICES = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}

_llm = LLM(
    provider="openai",
    model=os.environ["JUDGE_MODEL_ID"],
    base_url=os.environ["JUDGE_OPENAI_BASE_URL"],
    api_key=os.environ["JUDGE_OPENAI_API_KEY"],
)

_FAITHFULNESS_T = """判断回答是否忠实于检索上下文。
[检索上下文]
{reference}

[回答]
{output}

逐条检查回答中的事实陈述能否在检索上下文中找到依据。
correct = 全部有依据；incorrect = 完全无依据或与上下文冲突；partial = 部分有依据。
请用中文撰写理由。"""

_RELEVANCY_T = """判断回答与问题的相关性。
[问题]
{input}

[回答]
{output}

correct = 直接完整地回应了问题；incorrect = 答非所问；partial = 部分切题或夹杂无关内容。
请用中文撰写理由。"""

_PRECISION_T = """判断检索结果的排序质量。
[问题]
{input}

[按检索排名排列的知识库片段]
{reference}

[预期答案]
{expected}

对得出预期答案有用的片段应排在无用片段前面。
correct = 有用片段全部靠前；incorrect = 全部靠后；partial = 混杂。
请用中文撰写理由。"""

_RECALL_T = """判断检索上下文对预期答案的覆盖度。
[检索上下文]
{reference}

[预期答案]
{expected}

逐条检查预期答案中的信息点能否在检索上下文中找到出处。
correct = 全部能找到；incorrect = 完全找不到；partial = 部分能找到。
请用中文撰写理由。"""

_REFUSAL_T = """判断回答是不是无效答复（护栏检查）。
[问题]
{input}

[回答]
{output}

ok = 给出了实质性回答；refused = 拒答或声称无法回答；empty = 空答、纯客套、无任何信息量。
请用中文撰写理由。"""


def _classifier(name: str, template: str, choices: dict[str, float]) -> ClassificationEvaluator:
    return ClassificationEvaluator(
        name=name, prompt_template=template, llm=_llm, choices=choices, direction="maximize"
    )


_JUDGES = {
    "faithfulness": _classifier("faithfulness", _FAITHFULNESS_T, _CHOICES),
    "answer_relevancy": _classifier("answer_relevancy", _RELEVANCY_T, _CHOICES),
    "contextual_precision": _classifier("contextual_precision", _PRECISION_T, _CHOICES),
    "contextual_recall": _classifier("contextual_recall", _RECALL_T, _CHOICES),
    # 护栏型：稀有失败，期 2 里它的 sampling_rate 固定 1.0
    "refusal_check": _classifier(
        "refusal_check", _REFUSAL_T, {"empty": 0.0, "refused": 0.0, "ok": 1.0}
    ),
}


def _run(name: str, payload: dict) -> dict:
    """跑一个判官并向 acceptance 登记。异常记成 errored（第三态），不吞、也不当 0 分。"""
    try:
        (score,) = _JUDGES[name].evaluate(payload)
    except Exception as e:  # noqa: BLE001 —— 判官失败必须成为可见的第三态
        acceptance.record(name, None, error=f"{type(e).__name__}: {e}")
        return {"name": name, "score": None, "label": "errored", "explanation": str(e)}
    acceptance.record(name, score.score, label=score.label)
    return {
        "name": name,
        "score": score.score,
        "label": score.label,
        "explanation": score.explanation,
    }


def _unpack(output) -> tuple[str, str]:
    """测试用 log_output 记的是 {"answer":..., "contexts":[...]}；拆成 (答案, 拼好的上下文)。"""
    answer = output["answer"]
    contexts = output.get("contexts") or []
    reference = "

".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    return answer, reference


# 参数名由 Phoenix 插件的「按参数名绑定」约定决定：
#   output   —— log_output 记进去的值
#   input    —— 该 case 的 parametrize 字段组成的 mapping，即 {"question": ..., "expected": ...}
#   expected —— 名为 expected 的 parametrize 字段的值（这里是 ground truth 字符串）
# 一律带 **_ 兜住不消费的字段（trace_id、metadata、example 等）。


def faithfulness(output, **_) -> dict:
    answer, reference = _unpack(output)
    return _run("faithfulness", {"output": answer, "reference": reference})


def answer_relevancy(output, input=None, **_) -> dict:  # noqa: A002 —— 参数名由 Phoenix 约定
    answer, _ref = _unpack(output)
    return _run("answer_relevancy", {"output": answer, "input": input["question"]})


def contextual_precision(output, input=None, expected=None, **_) -> dict:  # noqa: A002
    _answer, reference = _unpack(output)
    return _run(
        "contextual_precision",
        {"input": input["question"], "reference": reference, "expected": expected},
    )


def contextual_recall(output, expected=None, **_) -> dict:
    _answer, reference = _unpack(output)
    return _run("contextual_recall", {"reference": reference, "expected": expected})


def refusal_check(output, input=None, **_) -> dict:  # noqa: A002
    answer, _ref = _unpack(output)
    return _run("refusal_check", {"output": answer, "input": input["question"]})


EVALUATORS = [
    faithfulness,
    answer_relevancy,
    contextual_precision,
    contextual_recall,
    refusal_check,
]
```

- [ ] **Step 2: 手工验证一个判官端到端**

```bash
NO_PROXY=localhost,127.0.0.1 python -c "
from evaluation.phoenix.evaluators import faithfulness
from evaluation.phoenix import acceptance
out = {'answer': '公司年假是每年20天。', 'contexts': ['公司年假为每年10天，入职满3年增至15天。']}
print(faithfulness(output=out))
print(acceptance._RECORDS)
"
```
Expected: 打印 `{'name': 'faithfulness', 'score': 0.0, 'label': 'incorrect', 'explanation': '<中文>'}`，且 `_RECORDS` 里有一条对应记录。

- [ ] **Step 3: 验证判官失败被记成 errored 而非崩溃**

```bash
NO_PROXY=localhost,127.0.0.1 JUDGE_OPENAI_API_KEY=bad_key python -c "
from evaluation.phoenix.evaluators import faithfulness
from evaluation.phoenix import acceptance
out = {'answer': 'x', 'contexts': ['y']}
r = faithfulness(output=out)
print('returned:', r['label'])
print('recorded error:', acceptance._RECORDS['faithfulness'][0].error is not None)
"
```
Expected: 打印 `returned: errored` 和 `recorded error: True`，进程不抛异常退出。

- [ ] **Step 4: Commit**

```bash
git add evaluation/phoenix/evaluators.py
git commit -m "feat(eval): 五个分类式判官（MiniMax-m3），判官失败落 errored 第三态"
```

---

### Task 6: 门禁套件接起来

**Files:**
- Create: `evaluation/phoenix/conftest.py`
- Create: `evaluation/phoenix/test_rag_eval.py`

**Interfaces:**
- Consumes: Task 2 的 `milvus_rag_phoenix_query_with_context`、Task 3 的 `CASES`/`SMOKE_IDS`、Task 4 的 acceptance、Task 5 的 `EVALUATORS`
- Produces: 一条可跑的门禁命令，退出码即门禁

- [ ] **Step 1: 写 conftest**

创建 `evaluation/phoenix/conftest.py`：

```python
"""把声明式验收条件挂进 pytest 生命周期。

判定发生在 sessionfinish：所有 case 都跑完之后再算，这样一次能看到全部回归。
"""

import pathlib

import pytest

from evaluation.phoenix import acceptance

CRITERIA_PATH = str(pathlib.Path(__file__).parent / "criteria.yaml")

_EXIT_ACCEPTANCE_FAILED = 3


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: PR 上只跑的小子集")
    acceptance.reset()


def pytest_sessionfinish(session, exitstatus):
    criteria = acceptance.load_criteria(CRITERIA_PATH, "offline")
    outcomes = acceptance.evaluate_all(criteria)
    board = acceptance.format_scoreboard(outcomes)
    session.config.get_terminal_writer().write(board + "\n")
    if any(not o.passed for o in outcomes) and exitstatus == 0:
        session.exitstatus = _EXIT_ACCEPTANCE_FAILED
```

- [ ] **Step 2: 写门禁套件**

创建 `evaluation/phoenix/test_rag_eval.py`：

```python
"""单轮 RAG 离线门禁。

suite → Phoenix dataset，case → example，断言结果 → 保留的 pass annotation，
pytest 退出码即 CI 门禁。挂在 marker 上的 evaluator 失败只降级成 warning，
真正让测试红的是断言；聚合阈值由 conftest 的 acceptance 在 sessionfinish 判。

parametrize 的字段名是有讲究的：插件把它们整体作为 example 的 input 传给判官
（即 input == {"question": ..., "expected": ...}），并把名为 expected 的字段
额外单独绑到判官的 expected 形参。所以字段必须叫 question / expected，
不能图省事传一个 Case 对象——那样 input 会变成 {"case": Case(...)}。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import pytest
from phoenix.client.pytest import log_output

from api.milvus_rag_phoenix import milvus_rag_phoenix_query_with_context
from evaluation.phoenix.cases import CASES, SMOKE_IDS
from evaluation.phoenix.evaluators import EVALUATORS

_PARAMS = [
    pytest.param(
        c.question,
        c.ground_truth,
        id=c.example_id,
        marks=[pytest.mark.smoke] if c.example_id in SMOKE_IDS else [],
    )
    for c in CASES
]


@pytest.mark.phoenix(
    dataset=os.environ.get("PHOENIX_TEST_DATASET", "bz-rag-golden"),
    dataset_description="BZ-RAG 黄金集，来自 evaluation/test_dataset.json",
    experiment_description="Phoenix 离线门禁（单轮）",
    experiment_metadata={"judge": os.environ.get("JUDGE_MODEL_ID", "")},
    evaluators=EVALUATORS,
    repetitions=int(os.environ.get("EVAL_REPETITIONS", "1")),
)
@pytest.mark.parametrize("question,expected", _PARAMS)
def test_rag_single_turn(question, expected):
    answer, contexts = milvus_rag_phoenix_query_with_context(question)
    log_output({"answer": answer, "contexts": contexts})
    # 硬断言只管"管线有没有产出"，质量由 acceptance 的聚合阈值管。
    # 一条 case 答得差不会立刻红，但整体质量掉下去会红——正是 Arize 的取舍。
    assert answer and answer.strip(), "管线返回了空答案"
    assert contexts, "检索没有返回任何上下文"
```

- [ ] **Step 3: 先跑 smoke 子集，验证接线**

确认 Phoenix 在跑、Milvus 在跑，然后：

```bash
NO_PROXY=localhost,127.0.0.1 \
PHOENIX_ENDPOINT=http://localhost:6006 \
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 \
PHOENIX_TEST_DATASET=bz-rag-golden-smoke \
pytest evaluation/phoenix -o addopts="" -m smoke -v
```

Expected: 3 条 case 跑过；末尾打印 Acceptance Criteria 表格，每行有 observed / required / n / PASS。
若判官取到的 `input` 键名不对（KeyError），按 Step 2 的注意事项改 parametrize 形状再跑。

- [ ] **Step 4: 确认 Phoenix UI 上出现 experiment**

打开 `http://localhost:6006`，dataset `bz-rag-golden-smoke` 下应有一个 experiment，
3 条 run，每条挂着 5 个 evaluator 的 annotation 加一个 `pass` annotation，
experiment metadata 里有 `git_sha`。

- [ ] **Step 5: 人为把阈值调高，验证门禁真的会红**

临时把 `criteria.yaml` 里 `faithfulness` 的 `average.threshold` 改成 `0.99`，重跑 Step 3 的命令。

Expected: 测试项全绿但**进程退出码是 3**，scoreboard 里 faithfulness 那行是 FAIL 且 reason 写着 `mean 0.xxx >= 0.99`。改回 0.8。

```bash
# 确认退出码
NO_PROXY=localhost,127.0.0.1 PHOENIX_ENDPOINT=http://localhost:6006 \
PHOENIX_TEST_DATASET=bz-rag-golden-smoke pytest evaluation/phoenix -o addopts="" -m smoke -q
echo "exit=$?"
```
Expected: `exit=3`

- [ ] **Step 6: 跑一次全量**

```bash
NO_PROXY=localhost,127.0.0.1 \
PHOENIX_ENDPOINT=http://localhost:6006 \
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 \
PHOENIX_TEST_DATASET=bz-rag-golden \
EVAL_REPETITIONS=2 \
pytest evaluation/phoenix -o addopts="" -q
```
Expected: 24 条 × 2 repetitions = 48 个测试项；scoreboard 的 n 列是 48。耗时数分钟。

- [ ] **Step 7: Commit**

```bash
git add evaluation/phoenix/conftest.py evaluation/phoenix/test_rag_eval.py
git commit -m "feat(eval): Phoenix 离线门禁套件，退出码即门禁"
```

---

### Task 7: 接进 CI

**Files:**
- Create: `.github/workflows/eval-gate.yml`
- Modify: `docs/CD-pipeline.md`（在第 6 节文件速查与第 10 节遗留项里反映新增内容）

**Interfaces:**
- Consumes: Task 6 的门禁命令
- Produces: PR 上的 `eval-gate` check

- [ ] **Step 1: 在本机注册 self-hosted runner**

在 GitHub 仓库 Settings → Actions → Runners → New self-hosted runner (Windows)，
按页面给的命令下载并配置，标签加 `bz-rag-local`。装成服务：

```powershell
./config.cmd --url https://github.com/brucezhu9594/BZ-RAG --token <页面给的 token> --labels bz-rag-local --runasservice
```

验证：Settings → Runners 里该 runner 显示 Idle。

- [ ] **Step 2: 写 workflow**

创建 `.github/workflows/eval-gate.yml`：

```yaml
name: Eval Gate

on:
  pull_request:
    branches: [master]
  push:
    branches: [master]

permissions:
  contents: read

jobs:
  eval:
    # 真管线依赖本机 Milvus(19530) 与本机 Phoenix(6006)，云 runner 碰不到。
    runs-on: [self-hosted, bz-rag-local]
    env:
      NO_PROXY: localhost,127.0.0.1
      no_proxy: localhost,127.0.0.1
      PHOENIX_ENDPOINT: http://localhost:6006
      PHOENIX_COLLECTOR_ENDPOINT: http://localhost:6006
      PHOENIX_PROJECT_NAME: bz-rag-ci
      # 并发跑同名 dataset 会互相 prune 例子（Arize 官方明说），按分支隔离。
      PHOENIX_TEST_DATASET: bz-rag-golden-${{ github.ref_name }}
      # PR 只跑 smoke 子集控制判官调用量；master 跑全量并重复 2 次压方差。
      EVAL_REPETITIONS: ${{ github.event_name == 'pull_request' && '1' || '2' }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Check Phoenix is up
        run: |
          curl -s -o /dev/null -w "phoenix=%{http_code}\n" --fail http://localhost:6006

      - name: Run eval gate (PR = smoke subset)
        if: github.event_name == 'pull_request'
        run: pytest evaluation/phoenix -o addopts="" -m smoke -q

      - name: Run eval gate (master = full)
        if: github.event_name == 'push'
        run: pytest evaluation/phoenix -o addopts="" -q
```

> **workflow 里没有"同步 dataset"这一步，是刻意的。** pytest 插件自己拥有 dataset：
> 全量跑时它会把 dataset 对齐到本次收集到的 case（缺席的 example 会被 prune），
> 而 example 的稳定身份来自 `parametrize` 的 `ids`——也就是我们的 content hash。
> 所以内容寻址这层由 `ids=` 保证，跑 `dataset.py` 是多余且会造成版本噪声的。
>
> `dataset.py` 的用途是另外两个：手工发布/检视 golden 集，以及期 3 回灌的落点。
> **期 3 的一个约束由此产生**：`harvest.py` 不能只往 Phoenix dataset 里追加 example，
> 还必须把捞回来的 case 写进 `evaluation/test_dataset.json`——否则下一次全量跑
> 会因为没有对应的测试项而把它们 prune 掉。
>
> judge 凭证不进 GitHub Secrets：runner 跑在本机，`evaluators.py` 顶部的 `load_dotenv()`
> 会读到仓库根的 `.env`（已 gitignore）。这是有意的边界，spec 第 5 节有记录。

- [ ] **Step 3: 开一个改坏东西的 PR，验证门禁变红**

新建分支，把 `evaluation/phoenix/criteria.yaml` 里 `faithfulness` 的 threshold 临时改到 0.99，推分支开 PR。

Expected: PR 的 Checks 里 `Eval Gate / eval` 红；点进日志能看到 Acceptance Criteria 表格，
faithfulness 那行 FAIL、observed 与 required 都在。

- [ ] **Step 4: 改回来，验证变绿**

把 threshold 改回 0.8 推同一分支。

Expected: `Eval Gate / eval` 绿。

- [ ] **Step 5: 更新 CD 文档**

在 `docs/CD-pipeline.md` 第 4.4 节的 workflow 表格里加一行：

```
| `.github/workflows/eval-gate.yml` | PR / push master（self-hosted runner） | 跑 Phoenix 离线评估门禁，聚合阈值不达标则挡下 |
```

并在第 6 节"关键文件速查"的 `.github/workflows/` 块里加 `├── eval-gate.yml   # 评估门禁（期 1）`。

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/eval-gate.yml
git add -f docs/CD-pipeline.md
git commit -m "ci: 评估门禁 workflow（self-hosted runner，PR 跑 smoke、master 跑全量）"
```

---

## 期 1 验收

改坏一个 prompt 或检索参数推 PR，`Eval Gate` 变红且日志里能看到**是哪条 criterion 没过、实测值多少、样本量多少**；改回来变绿。Phoenix UI 上这两次 run 在同一批 example 上可对比，且 experiment metadata 带 `git_sha`。

## 已知缺口

spec 第 8 节风险表最后一行要求把网关解析出的 `resolvedProvider` 记进 `experiment_metadata`
（判官供应商漂移会影响跨 run 可比性）。`ClassificationEvaluator` 把 HTTP 响应封在内部，
拿不到 `provider_metadata.gateway.routing.finalProvider`，因此期 1 **不做**，
`experiment_metadata` 里只记 `judge` 模型名。若日后发现分数出现无法解释的漂移，
再考虑自写 `LLMEvaluator` 子类直接持有响应体。

## 不在期 1 范围内

- 多轮门禁套件（`test_rag_multiturn.py`）——单轮跑顺了再加，避免一次引入两个变量
- 影子 canary、线上评估 worker、monitor、回灌——期 2 / 期 3
- 任何对现有 MLflow / DeepEval 路径的改动
