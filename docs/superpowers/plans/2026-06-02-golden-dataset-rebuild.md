# Golden Dataset Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 GLM-4-Flash 基于真实帮助页内容重生 25 项事实类 Q&A 写入 `evaluation/test_dataset.draft.json`，作为高质量黄金集草稿；用户人工 spot-check 后改名替换 `test_dataset.json`。

**Architecture:** 新建 `evaluation/generate_dataset.py` 一次性离线工具 + `evaluation/test_generate_dataset.py` 单元测试。复用 `common.etl.fetch_help_docs()` 拿同源页面，复用 `evaluation.deepeval_judge.GLMJudge._extract_json` 做 JSON 兜底。

**Tech Stack:** Python 3.10+、langchain_openai (ChatOpenAI)、`common.etl`（langchain WebBaseLoader）、`evaluation.deepeval_judge`、pytest、python-dotenv。

**Spec:** `docs/superpowers/specs/2026-06-02-golden-dataset-rebuild-design.md`

**前置已落地**：
- `evaluation/deepeval_judge.py` 的 `GLMJudge._extract_json` 静态方法（commit `4dfe53f`）
- `common/etl.py` 的 `fetch_help_docs()`（项目原有）
- `.env` 已配 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_ID=glm-4-flash`

---

## File Structure

- **Create**：`evaluation/generate_dataset.py` — 一次性生成脚本入口
- **Create**：`evaluation/test_generate_dataset.py` — `generate_one` 的 3 个 mock LLM 单测
- **Output**（不入 git）：`evaluation/test_dataset.draft.json` — 脚本运行产物，用户审后改名替换
- **Untouched**：`evaluation/test_dataset.json`、`evaluation/build_dataset.py`、`evaluation/evaluate.py`、`evaluation/deepeval_judge.py`

---

## Task 1: `generate_one` 与 `_short_source` TDD

**Files:**
- Create: `E:\wwwroot\BZ\BZ-RAG\evaluation\test_generate_dataset.py`
- Create: `E:\wwwroot\BZ\BZ-RAG\evaluation\generate_dataset.py`（先填骨架，逻辑下个 step）

- [ ] **Step 1: 写失败的测试**

Create `E:\wwwroot\BZ\BZ-RAG\evaluation\test_generate_dataset.py`:

```python
"""evaluation/generate_dataset.py 单元测试。"""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from evaluation import generate_dataset


class TestShortSource:
    def test_extracts_help_content_id(self):
        url = "https://cms.hewa.cn/content/mian/helpContent/10006"
        assert generate_dataset._short_source(url) == "helpContent/10006"

    def test_extracts_with_trailing_slash(self):
        url = "https://cms.hewa.cn/content/mian/helpContent/10573/"
        assert generate_dataset._short_source(url) == "helpContent/10573"

    def test_raises_on_unrecognized_url(self):
        with pytest.raises(ValueError, match="无法从 source 抽取"):
            generate_dataset._short_source("https://example.com/about")


class TestGenerateOne:
    def _make_doc(self, source: str = "https://cms.hewa.cn/content/mian/helpContent/10006") -> Document:
        return Document(
            page_content="禾蛙是一个人力资源平台。域名 hewa.cn，成立于 2020 年。",
            metadata={"source": source},
        )

    def test_returns_three_fields_on_clean_json(self):
        doc = self._make_doc()
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = '{"question": "禾蛙是什么平台", "ground_truth": "禾蛙是人力资源平台"}'
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item == {
            "question": "禾蛙是什么平台",
            "ground_truth": "禾蛙是人力资源平台",
            "expected_source": "helpContent/10006",
        }

    def test_handles_broken_json_via_extract(self):
        """LLM 返回 <think> 包裹 + markdown fence 的脏输出，_extract_json 应当救回。"""
        doc = self._make_doc()
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = (
            '<think>用户问的是这页讲什么。</think>\n'
            '```json\n'
            '{"question": "什么是禾蛙", "ground_truth": "人力资源平台"}\n'
            '```'
        )
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item is not None
        assert item["question"] == "什么是禾蛙"
        assert item["ground_truth"] == "人力资源平台"
        assert item["expected_source"] == "helpContent/10006"

    def test_returns_none_on_unparseable_llm_output(self):
        doc = self._make_doc()
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = "完全的散文，没有任何 JSON 结构。"
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item is None

    def test_returns_none_on_invalid_source_url(self):
        doc = self._make_doc(source="https://example.com/random-page")
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = '{"question": "Q", "ground_truth": "A"}'
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item is None
```

- [ ] **Step 2: 写最小骨架使测试可加载（FAIL 预期）**

Create `E:\wwwroot\BZ\BZ-RAG\evaluation\generate_dataset.py`:

```python
"""离线工具：从帮助中心生成 (question, ground_truth) 对，写入 test_dataset.draft.json。"""


def _short_source(source_url: str) -> str:
    raise NotImplementedError


def generate_one(doc, llm) -> dict | None:
    raise NotImplementedError
```

- [ ] **Step 3: 跑测试确认 FAIL**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
python -m pytest evaluation/test_generate_dataset.py -v
```
Expected: 7 个 FAIL（3 个 `TestShortSource` + 4 个 `TestGenerateOne`），错误为 `NotImplementedError`。

- [ ] **Step 4: 实现 `_short_source` + `generate_one`**

Replace `E:\wwwroot\BZ\BZ-RAG\evaluation\generate_dataset.py` 全部内容：

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


def generate_one(doc, llm) -> dict | None:
    """对单页生成 1 个 Q&A item。失败返回 None。"""
    prompt = PROMPT.format(content=(doc.page_content or "")[:CONTENT_TRUNC])
    try:
        msg = llm.invoke([{"role": "user", "content": prompt}])
        parsed = json.loads(GLMJudge._extract_json(msg.content or ""))
        short_src = _short_source(doc.metadata["source"])
    except Exception as e:
        print(f"  生成/解析失败 ({doc.metadata.get('source', '?')}): {e}", file=sys.stderr)
        return None
    return {
        "question": parsed["question"],
        "ground_truth": parsed["ground_truth"],
        "expected_source": short_src,
    }


def main() -> None:
    from common.etl import fetch_help_docs

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

Notes:
- `from common.etl import fetch_help_docs` 放在 `main()` 内部而非模块顶层，让单测不必触发 langchain_community 导入链
- `_short_source` 的 ValueError 在 `generate_one` 的 try/except 内被 catch → 返回 None，匹配 `test_returns_none_on_invalid_source_url`

- [ ] **Step 5: 跑测试确认 PASS**

Run: `python -m pytest evaluation/test_generate_dataset.py -v`
Expected: 7 PASSED。

- [ ] **Step 6: 全量评测测试套件回归**

Run: `python -m pytest evaluation/ -v`
Expected: 现有 12 + 新 7 = **19 PASSED**。

- [ ] **Step 7: Commit**

```bash
git add evaluation/generate_dataset.py evaluation/test_generate_dataset.py
git commit -m "feat(eval): 加 generate_dataset 离线工具 + 单测"
```

---

## Task 2: 跑生成、人工 spot-check、改名替换

> 这步需要 `.env` 正确（已确认），并联网到 `cms.hewa.cn`。脚本运行约 1-3 分钟。**仅在前置满足时执行**。

**Files:**
- 产出: `E:\wwwroot\BZ\BZ-RAG\evaluation\test_dataset.draft.json`
- 替换: `E:\wwwroot\BZ\BZ-RAG\evaluation\test_dataset.json`

- [ ] **Step 1: 运行生成脚本**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
PYTHONIOENCODING=utf-8 python evaluation/generate_dataset.py
```

Expected:
- 先看到 25 行 `[i/25] https://cms.hewa.cn/content/mian/helpContent/{id}`
- 末尾打印 `完成 X/25 项 → evaluation\test_dataset.draft.json` + `人工 spot-check 后：mv ...` 提示
- 退出码 0

若 X < 25：失败的页面 stderr 有 `生成/解析失败 ...` 行；不致命，继续。
若 X = 0：脚本 exit 1，stderr 提示。要排查 `.env` 与网络。

- [ ] **Step 2: 用户 spot-check `test_dataset.draft.json`**

> **这是人工环节**。打开 `evaluation/test_dataset.draft.json` + 浏览器开 `https://cms.hewa.cn/content/mian/helpContent/{id}` 对照查。

逐项检查：
- question 是否自然、像真实用户会问的
- ground_truth 是否在该 `expected_source` 页能找到依据
- 明显答非所问 → 直接改 `ground_truth`，或重写整项
- 明显不在该页能回答的问题 → 改 `question` 让它聚焦该页能答的内容

至少 20/25 项过关再继续。低于 20 项过关：考虑重跑或挑那几条手写。

- [ ] **Step 3: 改名替换旧黄金集**

> **可选先备份旧文件**：

```bash
git mv evaluation/test_dataset.json evaluation/test_dataset.json.old
```
（旧版本其实 git 历史里已经有，但显式重命名让 working tree 更清晰；commit 时一并提交）

然后：
```bash
mv evaluation/test_dataset.draft.json evaluation/test_dataset.json
```

如果你不想留 `.old` 备份，跳过第一条 `git mv`，直接：
```bash
mv evaluation/test_dataset.draft.json evaluation/test_dataset.json
```

- [ ] **Step 4: 验证 JSON 合法**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
python -c "import json; data = json.load(open('evaluation/test_dataset.json', encoding='utf-8')); print(f'共 {len(data)} 项'); print('字段齐全:', all({'question','ground_truth','expected_source'} <= set(d.keys()) for d in data))"
```
Expected: `共 25 项` + `字段齐全: True`（或低于 25 但 > 0、字段全 True）。

- [ ] **Step 5: Commit 新黄金集**

```bash
git add evaluation/test_dataset.json
git commit -m "feat(eval): 重做黄金集 25 项（GLM-4-Flash 生成 + 人工审查）"
```

---

## Task 3: 跑一次评测看新黄金集对分数的影响（可选）

> 与 Task 2 一样需要 `.env`、Milvus、Langfuse、Confident AI 都在线。**可选**——主要看新分数是否更可信。

**Files:** 不修改

- [ ] **Step 1: 跑 evaluate**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
PYTHONIOENCODING=utf-8 python evaluation/evaluate.py
```

Expected：完整跑通，4 个 metric 出 aggregate 表，stdout 给出 Confident AI URL。和旧黄金集分数对比，看变化方向（Faithfulness/Recall 应稳定上升，因为 ground_truth 更贴原文）。

- [ ] **Step 2: 在 Langfuse + Confident AI 看新 run**

浏览器开两个 dashboard，对比上一次 run。

- [ ] **Step 3: 无文件改动，跳过 commit**

---

## Task 4: 提交 plan 文件到 git

**Files:**
- 已存在: `docs/superpowers/plans/2026-06-02-golden-dataset-rebuild.md`

- [ ] **Step 1: 强加 plan（`/docs` 在 .gitignore）**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
git add -f docs/superpowers/plans/2026-06-02-golden-dataset-rebuild.md
```

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: 黄金集重做 plan"
```

- [ ] **Step 3: 确认仓库干净**

Run: `git status`
Expected: `nothing to commit, working tree clean`（除已有 `.gitignore` 之类预存项）。

---

## 完成判据

- `evaluation/generate_dataset.py` 可独立运行、生成 draft 文件
- `evaluation/test_generate_dataset.py` 7 个 mock LLM 单测全 PASS
- 完整 `evaluation/` 测试套 19 PASS（12 旧 + 7 新）
- `evaluation/test_dataset.json` 被人工审过的新 25 项替换，schema 字段一致
- （可选）跑一次 evaluate 验证整链路对新黄金集仍工作
