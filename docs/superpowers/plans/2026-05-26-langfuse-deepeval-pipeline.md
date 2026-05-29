# Langfuse + DeepEval 链路 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Langfuse Cloud 接进 BZ-RAG 评测链路：`test_dataset.json` 同步成 Langfuse dataset；评测时通过 `@observe` 装饰器为 `hybrid_search._retrieve` 与 `generate_answer` 打 trace、和 dataset item 绑定；DeepEval 跑 RAG 四件套，结果上 Confident AI。

**Architecture:** `app/milvus/hybrid_search.py` 仅加一个 `@observe` 装饰器；`evaluation/build_dataset.py` 一次性把 15 条样本 push 到 Langfuse；`evaluation/evaluate.py` 重写为 "pull dataset → 在 `item.observe()` context 里回放 pipeline → DeepEval → Confident AI"。复用 T3 的 `MiniMaxJudge`。

**Tech Stack:** Python 3.10+、langfuse（Cloud 免费 tier）、deepeval、langchain_openai、pymilvus、python-dotenv。

**Spec:** `docs/superpowers/specs/2026-05-26-langfuse-deepeval-pipeline-design.md`

**前置已落地（来自上一个 spec/plan）**：
- `evaluation/deepeval_judge.py` 含 `MiniMaxJudge`（commit `dc963c9`）
- `evaluation/test_judge.py` 含 `_extract_json` 单测（commit `af38556`）
- `requirements.txt` 已加 `deepeval`（commit `8efb04a`）
- `evaluation/evaluate.py` 的 DeepEval-only 版本（commit `8f2c308`）—— 本 plan 会再次重写
- `evaluation/test_dataset.json` 不动

---

## File Structure

- **Modify**：`requirements.txt` — 加 `langfuse`
- **Modify**：`app/milvus/hybrid_search.py` — `_retrieve` 上加 `@observe`
- **Create**：`evaluation/build_dataset.py` — Langfuse dataset 同步脚本
- **Create**：`evaluation/test_build_dataset.py` — `build_dataset` 单元测试（mock Langfuse client）
- **Replace**：`evaluation/evaluate.py` — 重写为 Langfuse dataset 驱动
- **Modify**：`.env.example` — 加 Langfuse 三件套 + `CONFIDENT_API_KEY`

**gitignore note**：`docs/superpowers/**` 下文件需要 `git add -f`。代码文件正常 `git add`。

---

## Task 1: 加 langfuse 依赖并安装

**Files:**
- Modify: `E:\wwwroot\BZ\BZ-RAG\requirements.txt`

- [ ] **Step 1: 在 `deepeval` 行之后插入 `langfuse`**

读 `requirements.txt`，在 `deepeval` 行下面、`lxml` 之前插入 `langfuse`。修改后该段应是：

```
zai-sdk
deepeval
langfuse
lxml
```

- [ ] **Step 2: 安装 langfuse**

Run: `pip install langfuse`
Expected: 成功安装。末行类似 `Successfully installed langfuse-X.X.X ...`，记下版本号。

- [ ] **Step 3: 验证 import 与 SDK 版本**

Run:
```
python -c "import langfuse; print(langfuse.__version__); from langfuse import Langfuse, observe; print('ok')"
```
Expected: 打印版本号（应是 3.x）+ `ok`。
若 `from langfuse import observe` 报错（如版本是 2.x），尝试 `from langfuse.decorators import observe`，并把整个 plan 中所有 `from langfuse import observe` 改为该路径。

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat(eval): 加 langfuse 依赖"
```

---

## Task 2: 给 hybrid_search._retrieve 加 @observe

**Files:**
- Modify: `E:\wwwroot\BZ\BZ-RAG\app\milvus\hybrid_search.py`

- [ ] **Step 1: 在文件顶部 import langfuse.observe**

读 `app/milvus/hybrid_search.py`，把现有的 import 区块改成（仅加最后一行 `from langfuse import observe`，其它行不动）：

```python
import os

from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from common.contextual_rewriter import contextual_rewrite
from common.zhipu_rerank import rerank
from langfuse import observe
```

若 Task 1 Step 3 发现 SDK 是 v2，把 `from langfuse import observe` 替换为 `from langfuse.decorators import observe`。

- [ ] **Step 2: 在 `_retrieve` 函数定义前加装饰器**

找到 `def _retrieve(query: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:`，在它正上方加一行：

```python
@observe(as_type="retriever")
def _retrieve(query: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    ...  # 原有函数体不动
```

不修改函数体里任何一行。

- [ ] **Step 3: 跑现有交互式 CLI 验证不破坏功能（仅当 Milvus 启动时执行）**

如果 Milvus 服务正在跑（`docker compose -f app/milvus/docker-compose.yml ps` 看到 `running`），跑：
```
echo exit | python app/milvus/hybrid_search.py
```
Expected: 打印 `Chat with AI (type 'exit' to quit)` 并退出。不能抛 ImportError。

若 Milvus 未启动，跳过此步骤——后面 Task 6 会做端到端冒烟。

- [ ] **Step 4: import 自检（不依赖 Milvus）**

Run: `python -c "from app.milvus.hybrid_search import _retrieve; print('ok')"`
Expected: 输出 `ok`，无 ImportError。

- [ ] **Step 5: Commit**

```bash
git add app/milvus/hybrid_search.py
git commit -m "feat(eval): _retrieve 加 langfuse @observe 装饰器"
```

---

## Task 3: build_dataset.py 与其单元测试（TDD）

**Files:**
- Create: `E:\wwwroot\BZ\BZ-RAG\evaluation\test_build_dataset.py`
- Create: `E:\wwwroot\BZ\BZ-RAG\evaluation\build_dataset.py`

- [ ] **Step 1: 写失败的测试**

Create `evaluation/test_build_dataset.py`:

```python
"""evaluation/build_dataset.py 单元测试：验证幂等同步逻辑。"""
from unittest.mock import MagicMock

import pytest

from evaluation import build_dataset


def test_sync_creates_dataset_and_pushes_all_items(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(build_dataset, "Langfuse", lambda: fake_client)

    items = [
        {"question": "Q1", "ground_truth": "A1", "expected_source": "src1"},
        {"question": "Q2", "ground_truth": "A2", "expected_source": "src2"},
    ]
    build_dataset.sync_dataset(items)

    fake_client.create_dataset.assert_called_once_with(
        name=build_dataset.DATASET_NAME,
        description=build_dataset.DATASET_DESCRIPTION,
    )
    assert fake_client.create_dataset_item.call_count == 2

    first_call = fake_client.create_dataset_item.call_args_list[0]
    assert first_call.kwargs["dataset_name"] == build_dataset.DATASET_NAME
    assert first_call.kwargs["input"] == "Q1"
    assert first_call.kwargs["expected_output"] == "A1"
    assert first_call.kwargs["metadata"] == {"expected_source": "src1"}

    fake_client.flush.assert_called_once()


def test_sync_empty_items_still_creates_dataset(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(build_dataset, "Langfuse", lambda: fake_client)

    build_dataset.sync_dataset([])

    fake_client.create_dataset.assert_called_once()
    fake_client.create_dataset_item.assert_not_called()
    fake_client.flush.assert_called_once()


def test_sync_propagates_client_errors(monkeypatch):
    fake_client = MagicMock()
    fake_client.create_dataset_item.side_effect = RuntimeError("network down")
    monkeypatch.setattr(build_dataset, "Langfuse", lambda: fake_client)

    items = [{"question": "Q", "ground_truth": "A", "expected_source": "s"}]
    with pytest.raises(RuntimeError, match="network down"):
        build_dataset.sync_dataset(items)
```

- [ ] **Step 2: 写最小骨架使测试可加载（确认 FAIL）**

Create `evaluation/build_dataset.py`:

```python
"""把 evaluation/test_dataset.json 同步到 Langfuse Cloud 的 dataset。"""

DATASET_NAME = "bz-rag-eval"
DATASET_DESCRIPTION = "BZ-RAG 评测集，来自 evaluation/test_dataset.json 回放。"


class Langfuse:
    def __init__(self):
        raise NotImplementedError


def sync_dataset(items: list[dict]) -> None:
    raise NotImplementedError
```

- [ ] **Step 3: 运行测试确认 FAIL**

Run: `python -m pytest evaluation/test_build_dataset.py -v`
Expected: 3 个测试全部 FAIL。`test_sync_creates_dataset_and_pushes_all_items` 与 `test_sync_empty_items_still_creates_dataset` 报 `NotImplementedError`；`test_sync_propagates_client_errors` 也 FAIL（同上）。

- [ ] **Step 4: 实现 sync_dataset 让测试通过**

Replace `evaluation/build_dataset.py` 全部内容：

```python
"""把 evaluation/test_dataset.json 同步到 Langfuse Cloud 的 dataset。"""
import json
import os
import pathlib
import sys

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langfuse import Langfuse

DATASET_NAME = "bz-rag-eval"
DATASET_DESCRIPTION = "BZ-RAG 评测集，来自 evaluation/test_dataset.json 回放。"

DATASET_PATH = pathlib.Path(__file__).parent / "test_dataset.json"


def sync_dataset(items: list[dict]) -> None:
    client = Langfuse()
    client.create_dataset(name=DATASET_NAME, description=DATASET_DESCRIPTION)
    for item in items:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["question"],
            expected_output=item["ground_truth"],
            metadata={"expected_source": item["expected_source"]},
        )
    client.flush()


def main() -> None:
    load_dotenv()
    with open(DATASET_PATH, encoding="utf-8") as f:
        items = json.load(f)
    print(f"同步 {len(items)} 条样本到 Langfuse dataset {DATASET_NAME!r}...")
    sync_dataset(items)
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"完成。请到 {host} 查看 dataset。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认 PASS**

Run: `python -m pytest evaluation/test_build_dataset.py -v`
Expected: 3 PASSED。

- [ ] **Step 6: 同时跑 _extract_json 测试做回归**

Run: `python -m pytest evaluation/ -v`
Expected: `test_judge.py` 4 个 + `test_build_dataset.py` 3 个 = 7 PASSED。

- [ ] **Step 7: Commit**

```bash
git add evaluation/build_dataset.py evaluation/test_build_dataset.py
git commit -m "feat(eval): 加 build_dataset 同步脚本 + 单测"
```

---

## Task 4: 重写 evaluation/evaluate.py 走 Langfuse + DeepEval

**Files:**
- Replace: `E:\wwwroot\BZ\BZ-RAG\evaluation\evaluate.py`

- [ ] **Step 1: 用下述内容完整覆盖 evaluate.py**

```python
"""BZ-RAG 评测入口：拉 Langfuse dataset，回放 pipeline 产生 trace，DeepEval 评测后上 Confident AI。"""
import os
import pathlib
import sys
from datetime import datetime, timezone

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evaluation.build_dataset import DATASET_NAME
from evaluation.deepeval_judge import MiniMaxJudge

load_dotenv()


@observe(as_type="generation")
def generate_answer(query: str, context: str) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{context}"
    )
    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
    )
    return msg.content or ""


def build_test_cases(dataset, run_name: str) -> list[LLMTestCase]:
    from app.milvus.hybrid_search import _retrieve

    cases: list[LLMTestCase] = []
    for i, item in enumerate(dataset.items):
        question = item.input
        print(f"[{i + 1}/{len(dataset.items)}] 回放: {question}")
        try:
            with item.observe(run_name=run_name) as trace:
                _, docs = _retrieve(question)
                ctx = [d.page_content for d in docs]
                ans = generate_answer(question, "\n\n".join(ctx))
                trace.update(output=ans)
        except Exception as e:
            print(f"  pipeline 抛错，跳过: {e}", file=sys.stderr)
            continue
        cases.append(
            LLMTestCase(
                input=question,
                actual_output=ans,
                expected_output=item.expected_output,
                retrieval_context=ctx,
            )
        )
    return cases


def main() -> None:
    langfuse = Langfuse()
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception as e:
        print(
            f"无法获取 Langfuse dataset {DATASET_NAME!r}: {e}\n"
            f"请先跑: python evaluation/build_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    run_name = f"deepeval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"评测 run: {run_name}")

    judge = MiniMaxJudge()
    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
        ContextualPrecisionMetric(model=judge, threshold=0.7),
        ContextualRecallMetric(model=judge, threshold=0.7),
    ]

    if not os.environ.get("CONFIDENT_API_KEY"):
        print(
            "[提示] 未设置 CONFIDENT_API_KEY，仅本地评测。"
            "如需云端报告：deepeval login --confident-api-key=<key>"
        )

    cases = build_test_cases(dataset, run_name)
    langfuse.flush()

    if not cases:
        print("没有可评测的样本，退出。", file=sys.stderr)
        sys.exit(1)

    evaluate(test_cases=cases, metrics=metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: import 自检（不依赖 Milvus / Langfuse 在线）**

Run: `python -c "import evaluation.evaluate; print('ok')"`
Expected: 输出 `ok`，无 ImportError / NameError。

若 `from langfuse import Langfuse, observe` 失败：
- v2 SDK：把这一行拆成 `from langfuse import Langfuse` + `from langfuse.decorators import observe`
- 同步修改 Task 2 步骤 1 的 import

若 `item.observe(run_name=...)` 在实际调用时（Step 3 端到端）抛 AttributeError：先 `pip show langfuse` 确认版本；v3 SDK 文档接口是 `dataset_item.run(run_name=...)`，可能命名差异。修法：把 `with item.observe(run_name=run_name) as trace:` 改为 `with item.run(run_name=run_name) as trace:` 或对应的实际 SDK 方法名，并在 commit message 注明。

- [ ] **Step 3: Commit**

```bash
git add evaluation/evaluate.py
git commit -m "feat(eval): 重写 evaluate.py 走 Langfuse dataset + DeepEval"
```

---

## Task 5: 更新 .env.example

**Files:**
- Modify: `E:\wwwroot\BZ\BZ-RAG\.env.example`

- [ ] **Step 1: 读现状**

Run: `python -c "print(open('.env.example', encoding='utf-8').read())"`
Expected: 看到现有 `OPENAI_API_KEY` / `ZHIPUAI_API_KEY` 等条目。

- [ ] **Step 2: 在文件末尾追加 Langfuse + Confident AI 段**

在文件末尾插入：

```
# Langfuse Cloud（链路追踪与 dataset 平台）
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Confident AI（DeepEval 云端报告，可选）
# CONFIDENT_API_KEY=your_confident_ai_key
```

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs(eval): .env.example 加 Langfuse 与 Confident AI 变量"
```

---

## Task 6: 端到端冒烟跑

> 需要 `.env` 已填好 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST（用户私下配置），Milvus 服务在线，知识库已建。仅在前置满足时执行。

**Files:** 不修改文件，仅运行命令

- [ ] **Step 1: 确认 .env 配置**

Run: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(bool(os.environ.get('LANGFUSE_PUBLIC_KEY')), bool(os.environ.get('LANGFUSE_SECRET_KEY')), os.environ.get('LANGFUSE_HOST'))"`
Expected: `True True https://cloud.langfuse.com`（或自托管 URL）。

- [ ] **Step 2: 确认 Milvus 集合存在**

Run: `python -c "from pymilvus import MilvusClient; c = MilvusClient(uri='http://localhost:19530'); print(c.list_collections())"`
Expected: 输出含 `hewa_help_collection`。
若 `connection refused`：先 `docker compose -f app/milvus/docker-compose.yml up -d`，再 `python app/milvus/knowledge_build.py` 建库。

- [ ] **Step 3: 同步 dataset**

Run: `python evaluation/build_dataset.py`
Expected: 打印 `同步 15 条样本到 Langfuse dataset 'bz-rag-eval'...` 与 `完成。请到 ... 查看 dataset。`，无异常。
在浏览器 Langfuse UI → Datasets 看到 `bz-rag-eval` 含 15 项。

- [ ] **Step 4: 跑评测**

Run: `python evaluation/build_dataset.py` 已完成的前提下：
```
python evaluation/evaluate.py
```
Expected:
- 打印 `评测 run: deepeval-2026XXXXTXXXXXX`
- 15 条 `[i/15] 回放: ...`
- DeepEval per-case metric 进度
- 结尾 DeepEval 自带汇总表（4 个 metric 的 pass/fail）
- 若设置了 CONFIDENT_API_KEY：控制台带 Confident AI 链接

在 Langfuse UI → 该 dataset → Runs 看到 `deepeval-...` 含 15 条 trace。

- [ ] **Step 5: 常见失败排查（仅当 Step 3 或 Step 4 失败时）**

| 现象 | 处理 |
|---|---|
| `Langfuse credentials missing` | `.env` 三件套未加载 / 拼错；先回 Step 1 |
| `connection refused 19530` | Milvus 未启 |
| `Dataset not found` 在 Step 4 | 跳过了 Step 3，先跑 build_dataset.py |
| `AttributeError: ... no 'observe'` on dataset item | SDK 版本差异，按 Task 4 Step 2 备注改方法名 |
| MiniMax JSON 解析失败 > 30% | 去 `evaluation/deepeval_judge.py` 删 `model_kwargs={"response_format":...}` 行再试 |

- [ ] **Step 6: 无文件改动，跳过 commit**

---

## Task 7: 提交 spec + plan 到 git（gitignore 处理）

> `/docs` 在 `.gitignore`，强加单文件。spec 已 commit（`9c43013`）。

**Files:**
- 已存在: `docs/superpowers/plans/2026-05-26-langfuse-deepeval-pipeline.md`

- [ ] **Step 1: 强加 plan 文件**

Run: `git add -f docs/superpowers/plans/2026-05-26-langfuse-deepeval-pipeline.md`
Expected: 无输出（或 LF→CRLF warning）。

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: Langfuse + DeepEval 链路 implementation plan"
```

- [ ] **Step 3: 确认仓库干净**

Run: `git status`
Expected: `nothing to commit, working tree clean`（除已有的 `.gitignore` 之类预存 modified）。

---

## 完成判据

- `requirements.txt` 含 `langfuse`
- `app/milvus/hybrid_search.py` 中 `_retrieve` 装饰为 `@observe(as_type="retriever")`
- `evaluation/build_dataset.py` 可独立运行同步 dataset
- `evaluation/test_build_dataset.py` 与 `evaluation/test_judge.py` 共 7 个测试全 PASS
- `evaluation/evaluate.py` 从 Langfuse 拉 dataset、生成 trace、跑 DeepEval、可上报 Confident AI
- `.env.example` 含 Langfuse 三件套 + Confident AI 注释
- Langfuse UI 能看到 dataset `bz-rag-eval` 与至少一次 run；Confident AI 能看到对应 test run（若 key 已配置）
