# Langfuse + DeepEval 评测链路设计

**日期**：2026-05-26
**目标**：在 BZ-RAG 现有 DeepEval 评测之上叠加 Langfuse 链路追踪：把 `app/milvus/hybrid_search.py` 的检索调用上报到 Langfuse Cloud，从同一份 dataset 出发跑 DeepEval，结果上 Confident AI。

## 上游设计

本文档接力于 `docs/superpowers/specs/2026-05-25-deepeval-rag-eval-design.md`。该设计的 T1-T4 已落地（deepeval 依赖、`MiniMaxJudge` wrapper、`evaluate.py` 替换），本设计在此基础上扩展，T5-T7 已废弃。

## 范围

**保留**
- `evaluation/deepeval_judge.py`（T3 完成的 `MiniMaxJudge`）—— 完全复用。
- 评测对象：仍只评 `app/milvus/hybrid_search.py`。
- 指标：DeepEval RAG 四件套（Faithfulness / AnswerRelevancy / ContextualPrecision / ContextualRecall）。
- Judge LLM：MiniMax（复用 `MODEL_ID` / `OPENAI_BASE_URL` / `OPENAI_API_KEY`）。
- 数据集：`evaluation/test_dataset.json`（15 条 question + ground_truth + expected_source）—— 字段不动。

**新增**
- Langfuse Cloud（免费 tier）作为 trace + dataset 平台。
- 装饰器形态的链路追踪嵌入 `hybrid_search.py`。
- `evaluation/build_dataset.py`：同步 `test_dataset.json` 到 Langfuse dataset 的 one-shot 脚本。
- `evaluation/evaluate.py`：重写为 "拉 Langfuse dataset → 在 `item.observe()` 内回放 pipeline → DeepEval 评测 → Confident AI"。

**不做**（YAGNI）
- 不回写 metric score 到 Langfuse trace（用户决定，Confident AI 单平台收口分数）。
- 不接 LangChain `LangchainCallbackHandler`（`@observe` 装饰器已够细，少一层集成）。
- 不采集生产真实流量；trace 100% 来自回放。
- 不评 Chroma / Qdrant / Milvus vector-only。

## 架构

```
                  ┌───────────────────────────────┐
                  │  evaluation/test_dataset.json │
                  │  (15 条 question+ground_truth) │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │  evaluation/build_dataset.py  （一次性同步）        │
       │  Langfuse Cloud → dataset "bz-rag-eval"          │
       │  item: input=question, expected_output=truth      │
       │        metadata={expected_source}                 │
       └──────────────────────────────────────────────────┘
                                 │
                                 ▼ （每次评测）
       ┌──────────────────────────────────────────────────┐
       │  evaluation/evaluate.py                          │
       │  dataset = langfuse.get_dataset("bz-rag-eval")   │
       │  for item in dataset.items:                      │
       │      with item.observe(run_name=ts) as trace:    │
       │          docs = _retrieve(q)        ← @observe   │
       │          ans  = generate_answer(..) ← @observe   │
       │          trace.update(output=ans)                │
       │      cases.append(LLMTestCase(...))              │
       │  deepeval.evaluate(cases, metrics, MiniMaxJudge) │
       │                          ↓                       │
       │                  Confident AI test run           │
       └──────────────────────────────────────────────────┘
                                 │
                                 ▼
        Langfuse: 一次评测 = 一个 run，留下 15 条 trace 树
        Confident AI: 4 个 metric 的 per-case 分数 + 汇总
```

## 组件

### 1. `requirements.txt`（modify）
追加 `langfuse`，不锁版本（与项目其它依赖风格一致）。

### 2. `app/milvus/hybrid_search.py`（modify）
最小侵入：仅在 `_retrieve(query, history=None)` 函数上加 `@observe(as_type="retriever")`。其它函数（`rag`、`main`）不动。
- `rag()` 仍带 history，是交互式 CLI 入口，不参与评测。
- 评测专用的 `generate_answer` 放在 `evaluation/evaluate.py`，那里自己拿 `@observe`。

```python
from langfuse import observe   # 顶部加 import

@observe(as_type="retriever")
def _retrieve(query: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    ...  # 原有实现不动
```

边界：
- `@observe` 缺 Langfuse 环境变量时不报错，会变成 no-op（Langfuse SDK 行为）。所以平时跑 CLI 也不被影响。
- 当 Langfuse 环境变量齐全且评测脚本在 `item.observe()` context 中调用 `_retrieve`，trace 自动嵌套到对应 dataset item run。

### 3. `evaluation/build_dataset.py`（new）
Idempotent dataset 同步脚本。一次性跑（或 dataset 变更后再跑）。

逻辑：
1. `load_dotenv()` 取 Langfuse key。
2. 读 `test_dataset.json`。
3. `langfuse.create_dataset(name=DATASET_NAME, description=...)` —— Langfuse SDK 接口对已存在 dataset 是幂等的（返回现有），不需要自己 try/except。
4. 对每条记录调 `langfuse.create_dataset_item(dataset_name=..., input=item["question"], expected_output=item["ground_truth"], metadata={"expected_source": item["expected_source"]})`。
   - Langfuse 的 `create_dataset_item` 同样对 (dataset, input) 组合幂等：相同 input 不会重复，会就地更新 expected_output / metadata。
5. `langfuse.flush()` 等待写入完成。
6. stdout 打印 dataset URL（拼接 host + project + dataset name）。

固定值：`DATASET_NAME = "bz-rag-eval"`。

### 4. `evaluation/evaluate.py`（rewrite again）
入口脚本：

```python
def main():
    1. load_dotenv()
    2. judge = MiniMaxJudge()
    3. langfuse = Langfuse()
    4. dataset = langfuse.get_dataset(DATASET_NAME)
       # 若 dataset 不存在 → 抛错并提示先跑 build_dataset.py
    5. run_name = f"deepeval-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    6. cases = []
       for item in dataset.items:
           with item.observe(run_name=run_name) as trace:
               docs = _retrieve(item.input)   # 自动嵌入 trace
               ctx  = [d.page_content for d in docs]
               ans  = generate_answer(item.input, "\n\n".join(ctx))
               trace.update(output=ans)
           cases.append(LLMTestCase(
               input=item.input,
               actual_output=ans,
               expected_output=item.expected_output,
               retrieval_context=ctx,
           ))
       langfuse.flush()
    7. metrics = [4 个 DeepEval metric，threshold=0.7，model=judge]
    8. if not CONFIDENT_API_KEY: 打印提示
    9. evaluate(test_cases=cases, metrics=metrics)
```

`generate_answer` 函数与 T4 版本相同，加 `@observe(as_type="generation")` 装饰器。

错误处理：循环里 `_retrieve` / `generate_answer` 抛错 → 把异常记录到 trace 后 `continue`，不让一条样本拖垮整轮。

### 5. `evaluation/deepeval_judge.py`（unchanged）
T3 完成的 `MiniMaxJudge` 复用，不动。

### 6. `evaluation/test_build_dataset.py`（new）
单元测试 build_dataset 的幂等行为，mock Langfuse client：
- 给同一条 question 调两次 `create_dataset_item`，验证调用次数为 2 但不重复创建（依赖 mock 的调用记录）。
- 验证 metadata 中 `expected_source` 字段正确传入。
- 验证错误时不被吞掉。

### 7. `evaluation/test_judge.py`（unchanged）
T2 完成的 `_extract_json` 单测保留。

### 8. `.env.example`（modify）
追加：

```
# Langfuse Cloud（链路追踪与 dataset 平台）
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Confident AI（DeepEval 云端报告，可选）
# CONFIDENT_API_KEY=your_confident_ai_key
```

`CONFIDENT_API_KEY` 注释行从上一个 spec 的 T5 计划里来，本设计直接落地（替代 T5）。

## 数据流细节

- **trace 嵌套**：`item.observe()` 是顶层 trace；`_retrieve` 和 `generate_answer` 的 `@observe` 在该 context 内会自动成为 span。Langfuse v3 SDK 自动维持 context（基于 contextvars），无需手动传 trace_id。
- **dataset.items 字段映射**：Langfuse Dataset Item 对象的 `.input` = question 字符串、`.expected_output` = ground_truth、`.metadata` = dict。
- **dataset name 冲突**：固定 `bz-rag-eval`。后续如果要 A/B 不同 pipeline 版本，可在 `run_name` 上加版本后缀（`deepeval-milvus-hybrid-v2`），dataset 本身不分裂。

## 错误处理

| 场景 | 行为 |
|---|---|
| Langfuse 环境变量缺失 | `build_dataset.py` / `evaluate.py` 启动即 KeyError；`hybrid_search.py` 的 `@observe` 无操作（SDK 静默） |
| Langfuse 网络不可达 | SDK 内部缓冲；最终 `flush()` 抛错。脚本崩，stderr 明确 |
| Dataset 不存在 | `evaluate.py` 报清晰错误并提示 `python evaluation/build_dataset.py` |
| 单条样本 `_retrieve` 抛错 | catch + skip，trace 标记 error，evaluate 继续 |
| Judge LLM JSON 解析失败 | DeepEval metric skip（沿用 T3 wrapper 的 `_extract_json` 降级） |
| `CONFIDENT_API_KEY` 缺失 | DeepEval 仅本地输出，stdout 打印 `deepeval login` 提示 |

## 测试

| 类型 | 文件 | 验证内容 |
|---|---|---|
| 单测 | `evaluation/test_judge.py` | `_extract_json` 4 case |
| 单测 | `evaluation/test_build_dataset.py` | dataset 幂等同步（mock Langfuse client） |
| 冒烟 | 手动 | Milvus 服务在线 + 三个 Langfuse 环境变量 + 跑 `build_dataset.py` 一次 + 跑 `evaluate.py`，验证 Langfuse UI 看到 15 条 trace、Confident AI 看到 test run |

## 风险

- **Langfuse v3 SDK API 变动**：v2 用 `@observe` from `langfuse.decorators`，v3 直接 `from langfuse import observe`。安装时不锁版本，需要在实现阶段以 `pip show langfuse` 实际版本为准；如 SDK 报错就走相应导入路径。
- **MiniMax JSON 输出稳定性**：T3 的 wrapper 已有降级；不变。
- **`item.observe()` context manager 行为**：Langfuse 文档示例用法是 `with item.observe(...) as trace`，trace 对象上有 `.update(output=...)`。具体方法名以实现阶段实测为准；如果 SDK 暴露的接口名是 `dataset_item.run` 或类似，按实际调整。
- **大量 trace 上传成本**：每次评测 15 条 trace + 4 个指标 LLM 调用 = 数十 token 量级，Langfuse Hobby 50k observations/月完全够用。
