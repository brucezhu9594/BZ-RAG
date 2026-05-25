# BZ-RAG 接入 deepeval 评测设计

**日期**：2026-05-25
**目标**：用 deepeval 框架替换现有自研评测脚本 `evaluation/evaluate.py`，作为 BZ-RAG 的正式评测入口。

## 背景

现状：`evaluation/evaluate.py` 用自研指标（hit_rate / MRR / 忠实度 / 相关性）评测 Chroma / Milvus 两套方案，忠实度和相关性靠手写 prompt + 正则解析评分，鲁棒性弱。

目标：换成 deepeval 标准化评测，复用其成熟的 RAG metric 实现 + Confident AI 云端报告。

## 范围

**评测对象**：仅 `app/milvus/hybrid_search.py`（多查询历史感知 + 服务端 RRF + 智谱 Rerank）。
Chroma、Qdrant、以及 Milvus vector-only 暂不评，后续可加。

**评测指标**：deepeval RAG 四件套
- `FaithfulnessMetric` —— 回答是否忠于检索内容
- `AnswerRelevancyMetric` —— 回答是否切题
- `ContextualPrecisionMetric` —— 检索结果中相关片段是否排在前面
- `ContextualRecallMetric` —— 检索结果是否覆盖了 expected_output 中的信息

**Judge LLM**：MiniMax（复用项目 `MODEL_ID` / `OPENAI_API_KEY` / `OPENAI_BASE_URL`）。
通过自定义 `DeepEvalBaseLLM` 包装 `langchain_openai.ChatOpenAI`，开启 `response_format={"type":"json_object"}`，并提供正则降级解析。

**数据集**：现有 `evaluation/test_dataset.json` 不动结构。`ground_truth` 字段映射为 `LLMTestCase.expected_output`，`expected_source` 仍保留供调试用但不参与 deepeval 指标。

**输出**：默认上报 Confident AI（需 `CONFIDENT_API_KEY`），同时本地 stdout 打印 metric 平均分摘要。`CONFIDENT_API_KEY` 缺失时仅本地运行并打印提示。

## 架构

```
test_dataset.json (15 条 question + ground_truth + expected_source)
        │
        ▼
   for each item:
   ┌───────────────────────────────────────────┐
   │ ① _retrieve(question)                     │  ← app.milvus.hybrid_search._retrieve
   │    返回 langchain Documents               │
   ├───────────────────────────────────────────┤
   │ ② generate_answer(question, context)      │  ← MiniMax ChatOpenAI
   │    与 evaluate.py 原有 generate 行为一致  │
   ├───────────────────────────────────────────┤
   │ ③ LLMTestCase(                            │
   │     input=question,                       │
   │     actual_output=answer,                 │
   │     expected_output=ground_truth,         │
   │     retrieval_context=[d.page_content..]) │
   └───────────────────────────────────────────┘
        │
        ▼
   deepeval.evaluate(
       test_cases=[...],
       metrics=[Faithfulness, AnswerRelevancy,
                ContextualPrecision, ContextualRecall],
   )   ← 全部使用 MiniMaxJudge 作为 LLM
        │
        ├─→ Confident AI 云端（若 CONFIDENT_API_KEY 已设置）
        └─→ stdout：四个指标平均分 + pass 数
```

## 组件

### 1. `evaluation/deepeval_judge.py`（新增）

封装 MiniMax 为 deepeval 可用的 Judge LLM。

```python
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
import os, json, re

class MiniMaxJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            model=os.environ["MODEL_ID"],
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            request_timeout=60,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        msg = self.model.invoke([{"role": "user", "content": prompt}])
        text = msg.content or ""
        # schema 路径：deepeval 期望可被 schema.parse_raw 消费的 JSON
        if schema is not None:
            return schema.model_validate_json(self._extract_json(text))
        return text

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"MiniMax({os.environ['MODEL_ID']})"

    @staticmethod
    def _extract_json(text: str) -> str:
        # 1. 直接整体解析；失败则正则抽取最外层 {...}
        try:
            json.loads(text); return text
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m: return m.group(0)
            raise ValueError(f"无法解析 JSON: {text[:200]}")
```

边界说明：
- MiniMax 不一定每次都返回纯 JSON，`_extract_json` 提供正则降级。
- deepeval ≥ 1.0 metric 调用使用 schema 路径，旧版用 raw 字符串；wrapper 同时兼容。

### 2. `evaluation/evaluate.py`（完全替换）

```python
"""BZ-RAG 评测入口：用 deepeval 评测 milvus_hybrid 方案。"""
import json, os, pathlib, sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from evaluation.deepeval_judge import MiniMaxJudge

load_dotenv()
DATASET_PATH = pathlib.Path(__file__).parent / "test_dataset.json"

def generate_answer(query: str, context: str) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system = (
        "你是一个知识库检索助手。下面「检索结果」来自知识库片段，"
        "请仅依据这些内容回答用户问题。如果检索结果不足以回答，"
        "请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{context}"
    )
    msg = llm.invoke([{"role": "system", "content": system},
                      {"role": "user", "content": query}])
    return msg.content or ""

def build_test_cases(dataset):
    from app.milvus.hybrid_search import _retrieve
    cases = []
    for item in dataset:
        q = item["question"]
        _, docs = _retrieve(q)
        ctx = [d.page_content for d in docs]
        ans = generate_answer(q, "\n\n".join(ctx))
        cases.append(LLMTestCase(
            input=q,
            actual_output=ans,
            expected_output=item["ground_truth"],
            retrieval_context=ctx,
        ))
    return cases

def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    judge = MiniMaxJudge()
    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
        ContextualPrecisionMetric(model=judge, threshold=0.7),
        ContextualRecallMetric(model=judge, threshold=0.7),
    ]
    test_cases = build_test_cases(dataset)

    if not os.environ.get("CONFIDENT_API_KEY"):
        print("[提示] 未设置 CONFIDENT_API_KEY，仅本地评测；"
              "如需云端报告：deepeval login --confident-api-key=<key>")

    evaluate(test_cases=test_cases, metrics=metrics)

if __name__ == "__main__":
    main()
```

### 3. `requirements.txt`

新增一行 `deepeval`，不固定版本（项目其它依赖也未锁版本）。

### 4. `.env.example`

补一行 `# CONFIDENT_API_KEY=your_confident_ai_key  # optional, 用于上传评测结果`。

## 数据流细节

- `_retrieve(question)` 返回 `(serialized, docs)`，只用 `docs`（List[Document]）；`docs[i].page_content` 即 chunk 文本，作为 `retrieval_context` 元素。
- 生成回答时把 retrieval_context 用 `\n\n` 拼接成完整 context 给 MiniMax（与原 evaluate.py 行为一致，不带 source 前缀以减少 noise）。

## 错误处理

- 单条样本 `_retrieve` 或 `generate_answer` 抛错：跳过该样本，stderr 打印错误并 continue。最终 evaluate 时样本数 < 15 不影响。
- Judge LLM JSON 解析失败：抛 `ValueError`，由 deepeval 自身机制处理为该指标 skip。
- 缺 `MODEL_ID` / `OPENAI_API_KEY`：脚本启动即崩，错误信息明确。

## 测试

无单测。验证方式：
1. 启动 Milvus（`docker compose up -d`），跑 `python app/milvus/knowledge_build.py` 建库。
2. `python evaluation/evaluate.py`，期望：
   - 15 条样本全部完成 retrieve + generate；
   - deepeval 打印 4 个 metric 的 per-case 与平均分；
   - 若 `CONFIDENT_API_KEY` 已设置，浏览器可在 Confident AI 看到 test run。

## 不包含（YAGNI）

- 多方案对比（Chroma / Qdrant / Milvus vector）—— 用户明确只评 milvus_hybrid。
- 自动 CI 集成 —— 后续需要再加。
- HTML 报告 / 本地 dashboard —— 走 Confident AI。
- 数据集扩展 / 人工标注 expected_contexts —— 用 ground_truth 即可。

## 风险

- **MiniMax JSON 输出稳定性**：M2.7 在长 prompt 下偶发返回 markdown 包裹。降级正则可吃掉大部分；若失败率 > 10%，回退方案是把 Judge 切到智谱 GLM。
- **deepeval 版本飘移**：metric 类名与签名在 1.x 内偶有变动。requirement 不锁版本会有兼容隐患，但项目其它依赖也未锁，与项目风格一致。
