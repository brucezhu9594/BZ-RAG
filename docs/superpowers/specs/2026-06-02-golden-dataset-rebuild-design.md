# 黄金集（evaluation/test_dataset.json）重做设计

**日期**：2026-06-02
**目标**：用 LLM 基于真实帮助页内容重生一份高质量黄金集，替代现有 15 项手写、部分答非所问的 `evaluation/test_dataset.json`。生成产物质量由人工 spot-check 兜底。

## 背景

现状 `evaluation/test_dataset.json` 15 项中有明显答非所问的样本，例：

```json
{
  "question": "禾蛙平台的联系方式是什么",
  "ground_truth": "禾蛙平台域名hewa.cn，成立于2020年，属于深圳精聘云聘数据技术有限公司"
}
```

这种 ground_truth 不能作为忠实度评判依据，会污染 DeepEval 分数。需要重做。

## 范围

**做**：
- 新建 `evaluation/generate_dataset.py`，复用 `common.etl.fetch_help_docs()` 拿同源同语义的页面内容，逐页调 GLM-4-Flash 生成 1 个事实类 Q&A 对，输出 `evaluation/test_dataset.draft.json`
- 新建 `evaluation/test_generate_dataset.py`，3 个 mock LLM 的单测
- 不动 `evaluation/test_dataset.json`，由用户 spot-check 后手动改名替换

**不做**（YAGNI）：
- 不接 Langfuse trace（生成只是离线一次性工具）
- 不做 LLM 自批 / 反向验证（人工 spot-check 是兜底）
- 不缓存 LLM 输出（draft 文件本身就是结果文件，重跑也才 3 分钟）
- 不并发（串行 3 分钟内完成）
- 不自动备份旧 `test_dataset.json`（git 历史够用）
- 不复用现有 15 项里"答案对头"的 Q&A（全部重生成更干净）

## 决策

- **颗粒度**：每页 1 问，共 25 项。`expected_source = helpContent/{id}`，与 Milvus 检索单元 / 现有评测的命中判定一致
- **生成模型**：复用 `OPENAI_*` 三件套（`.env` 当前指向 `glm-4-flash`，免费 tier，足够事实类生成）
- **prompt 风格**：事实类问答，禁止套话/页外信息，强制 JSON 输出
- **产物路径**：`evaluation/test_dataset.draft.json`（独立文件，避免直接覆盖）
- **JSON 解析**：复用 `GLMJudge._extract_json` 的三层兜底（think-strip → 栈扫平衡 {} → json_repair）
- **页面正文截断**：3000 字符（极少数 OCR 后超长的页面才生效，绝大多数原样）
- **失败处理**：单页生成失败 → log + skip + 继续；最终 items 若为空 → exit 1

## 架构

```
                          ┌──────────────────────────────────────┐
                          │  common.etl.fetch_help_docs()        │
                          │  → list[Document]，一页一项           │
                          │  （重爬 25 个 helpContent/{id} 页面）  │
                          └─────────────────┬────────────────────┘
                                            │
                                            ▼
       ┌──────────────────────────────────────────────────────────┐
       │  evaluation/generate_dataset.py                          │
       │  for doc in docs:                                        │
       │      prompt = PROMPT.format(content=doc.page_content     │
       │                                       [:3000])           │
       │      msg = ChatOpenAI(glm-4-flash, json_object).invoke() │
       │      parsed = json.loads(                                │
       │          GLMJudge._extract_json(msg.content))            │
       │      item = {                                            │
       │          "question": parsed["question"],                 │
       │          "ground_truth": parsed["ground_truth"],         │
       │          "expected_source": "helpContent/{id}",          │
       │      }                                                   │
       │      items.append(item)                                  │
       └─────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
              evaluation/test_dataset.draft.json (25 项)
                                 │
                                 ▼ 人工 spot-check，可修改 / 删除明显错误项
                                 ▼ mv ... test_dataset.json
              evaluation/test_dataset.json (替换旧 15 项)
                                 │
                                 ▼
              python evaluation/evaluate.py 跑新黄金集
```

## 组件

### 1. `evaluation/generate_dataset.py`（new）

入口脚本。串行执行：

```python
"""离线工具：从帮助中心生成 (question, ground_truth) 对，写入 test_dataset.draft.json。"""
import json
import os
import pathlib
import re
import sys

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from common.etl import fetch_help_docs
from evaluation.deepeval_judge import GLMJudge

load_dotenv()

DRAFT_PATH = pathlib.Path(__file__).parent / "test_dataset.draft.json"
CONTENT_TRUNC = 3000

PROMPT = """你是禾蛙平台帮助中心的 QA 出题助手。下面是一页帮助文档。
请仅基于这页内容，生成 1 个用户视角的事实类问题，以及完整、忠于原文的答案。

要求：
- question 用自然问句，贴近真实用户提问口吻
- ground_truth 必须用原文表述或紧密改写，不允许加入页外信息
- 不要"根据文档"、"如上所述"这类套话
- 输出严格 JSON，键名固定为 question 和 ground_truth

页面内容：
{content}
"""


def _short_source(source_url: str) -> str:
    m = re.search(r"helpContent/\d+", source_url)
    if not m:
        raise ValueError(f"无法从 source 抽取 helpContent/{{id}}: {source_url}")
    return m.group(0)


def generate_one(doc, llm: ChatOpenAI) -> dict | None:
    prompt = PROMPT.format(content=(doc.page_content or "")[:CONTENT_TRUNC])
    try:
        msg = llm.invoke([{"role": "user", "content": prompt}])
        parsed = json.loads(GLMJudge._extract_json(msg.content or ""))
    except Exception as e:
        print(f"  生成/解析失败 ({doc.metadata.get('source', '?')}): {e}", file=sys.stderr)
        return None
    return {
        "question": parsed["question"],
        "ground_truth": parsed["ground_truth"],
        "expected_source": _short_source(doc.metadata["source"]),
    }


def main() -> None:
    llm = ChatOpenAI(
        model=os.environ["MODEL_ID"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.3,
        request_timeout=60,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    docs = fetch_help_docs()
    items: list[dict] = []
    for i, doc in enumerate(docs):
        print(f"[{i + 1}/{len(docs)}] {doc.metadata.get('source', '?')}")
        item = generate_one(doc, llm)
        if item:
            items.append(item)
    if not items:
        print("没有任何条目生成成功，退出。", file=sys.stderr)
        sys.exit(1)
    DRAFT_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完成 {len(items)}/{len(docs)} 项 → {DRAFT_PATH}")
    print("人工 spot-check 后：mv test_dataset.draft.json test_dataset.json")


if __name__ == "__main__":
    main()
```

### 2. `evaluation/test_generate_dataset.py`（new）

3 个 mock LLM 单测：

- `test_generate_one_returns_three_fields`：mock LLM 返回干净 JSON，验证产出 dict 三字段齐全且 `expected_source` 抽出 `helpContent/{id}` 格式
- `test_generate_one_handles_broken_json_via_extract`：mock LLM 返回 think-block + 未转义引号的脏 JSON，验证 `_extract_json` 三层兜底救回来
- `test_generate_one_returns_none_on_invalid_source_url`：构造 metadata.source 不含 `helpContent/{id}` 的 Document，验证 `_short_source` 抛错被外层 catch → 返回 None

不需要 mock `fetch_help_docs`；上面三个测试都只测 `generate_one`，传入 Document 实例即可。

### 3. `evaluation/test_dataset.json`（unchanged by the script）

脚本不动这个文件。用户 spot-check `test_dataset.draft.json` 后手动改名。旧版本由 git 历史保留。

## 数据流细节

- `fetch_help_docs()` 走 `WebBaseLoader`，已在生产路径用过 25 次知识库构建，稳定。返回 list[Document]，每个 Document 一整页（已经 OCR 过短页面）。
- `expected_source` 格式与现有 `test_dataset.json` 字段一致（`helpContent/10006` 这种），评测脚本 `evaluate.py` 不需要任何改动即可读新文件。
- 单页生成的 Q&A 长度大致 50-200 字，整 25 项 JSON 文件估计 < 10KB。

## 错误处理

| 场景 | 行为 |
|---|---|
| `fetch_help_docs()` 网络挂 | 抛错传到 main，stderr 明确，整脚本崩 |
| 单页 LLM 调用超时 / 拒绝 | catch + log + skip 该页，items 少一项 |
| 单页 JSON 解析失败 | `_extract_json` 三层兜底都救不回 → catch + log + skip |
| `helpContent/{id}` 抽取失败 | `_short_source` 抛 ValueError → catch + log + skip |
| 最终 items 为空 | exit 1，stderr 明确 |
| 现有 draft 文件已存在 | 覆盖（脚本是幂等的，没什么好保护的） |

## 测试

| 类型 | 文件 | 验证内容 |
|---|---|---|
| 单测 | `evaluation/test_generate_dataset.py` | `generate_one` 3 case：正常 / 脏 JSON 救回 / source 异常 |

不写端到端单测（要真调 GLM）。冒烟用 `python evaluation/generate_dataset.py` 实跑验证。

## 风险

- **GLM-4-flash 生成质量不稳**：人工 spot-check 兜底；如果某页生成的 Q&A 明显偏离原文，用户手改。重跑可以拿不同结果，但 temperature=0.3 已经偏稳。
- **OCR 后的页面文本质量差**：少数图片帮助页 OCR 出来的文字可能不连贯，LLM 生成的 Q&A 也会跟着差。人工 review 时这些项要特别关注。
- **`fetch_help_docs` 重爬开销**：约 12 秒。可接受，不引入缓存。
