# Phoenix 线上评估（期 2）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让部署后的版本被持续评分——影子 canary 产生真实形态的流量，worker 按采样率抽 span、跑判官、把分数写回**具体的 span**，在 Phoenix UI 上能逐段看到 `eval.<name>.*`。

**Architecture:** 三块。① `shadow/`：复刻 `cf-worker/src/index.js` 的按权重分流，把流量打到两个不同 `APP_VERSION`、同一 `PHOENIX_PROJECT_NAME=bz-rag-canary` 的本地 uvicorn；② `online_worker.py`：按 `online_tasks.yaml` 的声明拉 span → 抽样 → 复用期 1 那五个判官里的三个 → `log_span_annotations_dataframe` 写回；③ `canary-watch.yml`：cron 触发 worker 跑一轮。**期 2 不做任何决策**——不 rollback、不 promote、不阻断，那是期 3。

**Tech Stack:** arize-phoenix-client 3.5.0（`spans.get_spans` / `get_span_annotations` / `log_span_annotations_dataframe`）· FastAPI + uvicorn · httpx · pandas · pyyaml · pytest · 本机 self-hosted runner（Windows 服务）

**Spec:** `docs/superpowers/specs/2026-09-08-phoenix-eval-cicd-design.md`（§3 系统形状、§4.7 影子 canary、§4.8 online worker、§5 CI 装配）

---

## Global Constraints

- **Python 3.14.3**（runner 与本机一致，`C:\Python314`）。CI 跑在 `RUNNER_WORKSPACE\_venv` 隔离环境里。
- **`NO_PROXY` 必须包含 `localhost,127.0.0.1`，且在任何 `phoenix` / `milvus` / `httpx` import 之前设置。** 本仓库已因 Privoxy 拦 localhost 踩过三次（见 `api/main.py`、`api/milvus_rag_phoenix.py` 顶部注释）。
- **凭证一律从 GitHub Repository secrets 注入，不依赖 runner 本机 `.env`。** ⚠️ 这一条**推翻了 spec §5** 的原文（"judge key 走 runner 本机 `.env`，不进 GitHub Secrets"）——实测证伪：runner 每次 `actions/checkout` 出来的是全新工作目录，`.env` 在 `.gitignore` 里不会进 checkout，`find_dotenv()` 往上也找不到，干净 clone 后 `import evaluation.phoenix.evaluators` 直接抛 `RuntimeError: 判官配置缺失`。现有七个 secret：`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_ID` / `ZHIPUAI_API_KEY` / `JUDGE_OPENAI_API_KEY` / `JUDGE_OPENAI_BASE_URL` / `JUDGE_MODEL_ID`。
- **workflow 的 `run:` 块内容必须是纯 ASCII**（说明写进 YAML 注释，注释不进生成的脚本）。GitHub 把 run 块写成 UTF-8 无 BOM 的 `.ps1`，powershell 5.1 在代码页 936 下按 GBK 解码，中文的 UTF-8 字节含非法 GBK 序列，会吃掉字符串终止符并报出与真因无关的 `Missing closing '}'`。
- **workflow 的 `env:` 键名不能大小写重复**（GitHub 判 `NO_PROXY` 与 `no_proxy` 为同一个键，整个文件会被拒收，症状是 0 个 job、workflow 名回退成文件路径、无日志）。
- **PowerShell 的 `run:` 多命令块只有最后一条命令的退出码算数**，每条关键命令后必须显式 `if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`。
- **Windows 默认 shell 是 `powershell.exe` 5.1**（runner 未装 pwsh）：不用三元运算符 / `??`；`curl` 是 `Invoke-WebRequest` 的别名，要写 `curl.exe`；不要写 `shell: bash`（PATH 第一个 bash 是 WSL 启动器，WSL 无发行版）。
- 跑测试一律绕开仓库默认 addopts：`pytest tests/ -o addopts=""`；跑 `evaluation/phoenix` 还要加 `--import-mode=importlib`（目录名与已安装的 `phoenix` 包冲突）。
- **`/docs` 在 `.gitignore` 里**，提交 docs 下的文件一律 `git add -f`。
- commit 用 conventional commits。`feat:` / `fix:` 会触发 semantic-release 发版；期 2 的评估设施用 `feat(eval):`，workflow 用 `ci:`。
- **不动期 1 的任何产出**：`evaluation/phoenix/{cases,dataset,evaluators,acceptance,conftest,test_rag_eval}.py`、`criteria.yaml` 的 `offline` 段、`.github/workflows/eval-gate.yml` 一行不改。期 2 只**读** `evaluators.py` 与 `criteria.yaml` 的 `online` 段。

---

## 与 spec 的三处偏差（实测得出，计划按偏差后的做法写）

**① 评估对象从 LLM span 改成根 AGENT span。** spec §4.8 的 `query_filter: "span_kind == 'LLM'"` 在真实 trace 上会出错。实测一条 trace 的 span 树：

```
AGENT      milvus-hybrid-rag   (root)   input.value=问题   output.value=答案
  RETRIEVER  retrieve                   input.value=问题   output.value=[检索前的 6 条]
  RERANKER   _rerank                    output.value=[重排后喂给 LLM 的 4 条]
  CHAIN      _generate
    LLM      ChatOpenAI                 ← langchain instrumentor 产出
    LLM      ChatCompletion             ← openai instrumentor 产出（同一次调用被记了两遍）
```

按 `span_kind == 'LLM'` 抽会**每次生成重复计两次**，多轮时还会额外抓到 `_rewrite_query` 的 LLM 调用。改用根 AGENT span：一次请求一条、天然去重，且 `input.value` / `output.value` 直接就是问题与答案。

**② `query_filter` 不做表达式解析。** `client.spans.get_spans()` 原生支持 `span_kind=` 参数，声明里直接写 `span_kind: AGENT` 即可，不需要自造 `"span_kind == 'LLM'"` 这种字符串的解析器。少一层能出错的东西。

**③ 线上只有三个判官可用。** `contextual_precision` / `contextual_recall` 的 prompt 需要 `expected`（标准答案），线上没有 ground truth。可用的是 `faithfulness`（答案 vs 检索上下文）、`answer_relevancy`（答案 vs 问题）、`refusal_check`（护栏）。spec 给的 `online_tasks.yaml` 示例与 `criteria.yaml` 的 `online` 段本来就只用了这三个，自洽——但校验器必须显式拒绝另外两个，否则线上会静默产出全 errored 的 annotation。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `evaluation/phoenix/online_tasks.yaml` | 任务声明（project / span_kind / evaluators / sampling_rate / cadence / window_minutes） |
| `evaluation/phoenix/online_tasks.py` | 读 + **eager 校验**任务声明，导出 `OnlineTask` 与 `load_tasks()` |
| `evaluation/phoenix/span_extract.py` | 把一条根 span + 它的同 trace 兄弟 span 拆成判官要的 `{question, answer, contexts}` |
| `evaluation/phoenix/sampling.py` | 按 span_id 做**确定性**抽样；护栏型（rate=1.0）永不抽掉 |
| `evaluation/phoenix/online_worker.py` | 主循环：拉 span → 去重 → 抽样 → 跑判官 → 写回 annotation |
| `evaluation/phoenix/shadow/router.py` | 复刻 cf-worker：读权重、掷骰子、转发、回写 `x-bz-backend` |
| `evaluation/phoenix/shadow/weight.json` | 权重文件（0–100 整数，形状与 `cf-kv-update.sh` 一致） |
| `evaluation/phoenix/shadow/replay.py` | 从金标集之外的查询池按节奏打流量 |
| `evaluation/phoenix/shadow/query_pool.json` | 查询池（**不得与 `evaluation/test_dataset.json` 重合**） |
| `evaluation/phoenix/shadow/up.ps1` | 一键起 stable + canary 两个 uvicorn + router |
| `tests/test_online_tasks.py` | 校验规则的单测 |
| `tests/test_span_extract.py` | 提取逻辑单测（用录下来的真实 span 形状做 fixture） |
| `tests/test_sampling.py` | 抽样确定性与护栏豁免 |
| `tests/test_online_worker.py` | 主循环单测（假 client，不打网络） |
| `tests/test_shadow_router.py` | 分流决策单测 |
| `.github/workflows/canary-watch.yml` | cron 跑 worker 一轮 |

---

### Task 1: 任务声明与校验

先做这个，因为后面每个组件都要消费 `OnlineTask`，且"配置错了要立刻炸"的教训在期 1 已经踩过（判官配置缺失改成 import 期 eager 失败）。

**Files:**
- Create: `evaluation/phoenix/online_tasks.yaml`
- Create: `evaluation/phoenix/online_tasks.py`
- Test: `tests/test_online_tasks.py`

**Interfaces:**
- Consumes: 无（只读自己的 yaml）
- Produces:
  - `class OnlineTask(NamedTuple)`：`name: str`、`project: str`、`span_kind: str`、`evaluators: tuple[str, ...]`、`sampling_rate: float`、`cadence: str`、`window_minutes: int`
  - `ONLINE_EVALUATORS: frozenset[str]` —— 线上可用的三个判官名
  - `load_tasks(path: str) -> list[OnlineTask]` —— 校验失败抛 `ValueError`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_online_tasks.py
"""线上任务声明的校验规则。

这些用例钉住的是"配置错了必须立刻炸"——线上 worker 是 cron 跑的，
没人盯着，一个静默接受的坏配置会安静地产出几小时的垃圾 annotation。
"""

import pytest

from evaluation.phoenix.online_tasks import ONLINE_EVALUATORS, load_tasks

VALID = """
- name: canary-faithfulness
  project: bz-rag-canary
  span_kind: AGENT
  evaluators: [faithfulness, answer_relevancy]
  sampling_rate: 0.2
  cadence: continuous
  window_minutes: 30
"""


def _write(tmp_path, text):
    p = tmp_path / "online_tasks.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestValid:
    def test_loads_and_normalizes(self, tmp_path):
        (t,) = load_tasks(_write(tmp_path, VALID))
        assert t.name == "canary-faithfulness"
        assert t.project == "bz-rag-canary"
        assert t.span_kind == "AGENT"
        assert t.evaluators == ("faithfulness", "answer_relevancy")
        assert t.sampling_rate == 0.2
        assert t.cadence == "continuous"
        assert t.window_minutes == 30

    def test_online_evaluator_set_excludes_ground_truth_judges(self):
        """线上没有 ground truth，需要 expected 的两个判官不可用。"""
        assert ONLINE_EVALUATORS == frozenset(
            {"faithfulness", "answer_relevancy", "refusal_check"}
        )


class TestRejects:
    def test_unknown_evaluator(self, tmp_path):
        bad = VALID.replace("[faithfulness, answer_relevancy]", "[nonesuch]")
        with pytest.raises(ValueError, match="nonesuch"):
            load_tasks(_write(tmp_path, bad))

    def test_ground_truth_evaluator_rejected(self, tmp_path):
        """contextual_recall 需要 expected，线上拿不到——必须拒绝而不是跑出一堆 errored。"""
        bad = VALID.replace("[faithfulness, answer_relevancy]", "[contextual_recall]")
        with pytest.raises(ValueError, match="contextual_recall"):
            load_tasks(_write(tmp_path, bad))

    def test_sampling_rate_out_of_range(self, tmp_path):
        bad = VALID.replace("sampling_rate: 0.2", "sampling_rate: 1.5")
        with pytest.raises(ValueError, match="sampling_rate"):
            load_tasks(_write(tmp_path, bad))

    def test_unknown_cadence(self, tmp_path):
        bad = VALID.replace("cadence: continuous", "cadence: hourly")
        with pytest.raises(ValueError, match="cadence"):
            load_tasks(_write(tmp_path, bad))

    def test_duplicate_task_name(self, tmp_path):
        with pytest.raises(ValueError, match="重复"):
            load_tasks(_write(tmp_path, VALID + VALID))

    def test_missing_required_key(self, tmp_path):
        bad = VALID.replace("  project: bz-rag-canary\n", "")
        with pytest.raises(ValueError, match="project"):
            load_tasks(_write(tmp_path, bad))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_online_tasks.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix.online_tasks'`

- [ ] **Step 3: 写声明文件**

创建 `evaluation/phoenix/online_tasks.yaml`：

```yaml
# 线上评估任务声明。复刻 Arize 的 template / task 分离：
# template（判官）就是 evaluation/phoenix/evaluators.py 里那几个，离线线上共用一份；
# task（本文件）只描述"对哪个 project 的哪种 span、按什么频率、抽多少比例、用哪几个判官"。
#
# span_kind 用 AGENT 而不是 spec 原文的 LLM：真实 trace 里一次生成会产出两条 LLM span
# （langchain 与 openai 两个 instrumentor 各记一遍），多轮时还会多出 _rewrite_query 的
# LLM span。根 AGENT span 一次请求一条，且 input.value / output.value 直接就是问题与答案。
#
# 线上没有 ground truth，所以只有三个判官可用：faithfulness / answer_relevancy /
# refusal_check。contextual_precision 与 contextual_recall 的 prompt 要 expected，
# 放进来会被 online_tasks.py 拒绝。

- name: canary-quality
  project: bz-rag-canary
  span_kind: AGENT
  evaluators: [faithfulness, answer_relevancy]
  sampling_rate: 0.2          # 唯一的成本旋钮
  cadence: continuous         # continuous | historical
  window_minutes: 30

- name: canary-guardrail
  project: bz-rag-canary
  span_kind: AGENT
  evaluators: [refusal_check]
  # 护栏型不采样：拒答/空答本来就稀有，采样会把它们采没，采样就失去意义。
  # 这是从 Arize 抄来的口径，不是随手写的 1.0。
  sampling_rate: 1.0
  cadence: continuous
  window_minutes: 30
```

- [ ] **Step 4: 写实现**

创建 `evaluation/phoenix/online_tasks.py`：

```python
"""线上评估任务声明的加载与校验。

有意做成 eager 校验、失败即抛：worker 是 cron 跑的，没人盯着终端，
一个被静默接受的坏配置会安静地产出几小时垃圾 annotation，比直接崩难查得多。
这与期 1 把判官配置缺失做成 import 期失败是同一个取舍。
"""

import pathlib
from typing import Any, NamedTuple

import yaml

DEFAULT_PATH = str(pathlib.Path(__file__).parent / "online_tasks.yaml")

# 线上可用的判官。另外两个（contextual_precision / contextual_recall）的 prompt
# 需要 expected（标准答案），线上没有 ground truth，放进来只会产出全 errored。
ONLINE_EVALUATORS = frozenset({"faithfulness", "answer_relevancy", "refusal_check"})

VALID_CADENCE = frozenset({"continuous", "historical"})
REQUIRED_KEYS = ("name", "project", "span_kind", "evaluators", "sampling_rate", "cadence")


class OnlineTask(NamedTuple):
    name: str
    project: str
    span_kind: str
    evaluators: tuple[str, ...]
    sampling_rate: float
    cadence: str
    window_minutes: int


def _check_one(raw: Any, idx: int) -> OnlineTask:
    if not isinstance(raw, dict):
        raise ValueError(f"第 {idx} 条任务不是 mapping：{raw!r}")

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"第 {idx} 条任务缺少必填键：{', '.join(missing)}")

    evaluators = raw["evaluators"]
    if not isinstance(evaluators, list) or not evaluators:
        raise ValueError(f"任务 {raw['name']} 的 evaluators 必须是非空列表")
    unknown = [e for e in evaluators if e not in ONLINE_EVALUATORS]
    if unknown:
        raise ValueError(
            f"任务 {raw['name']} 引用了线上不可用的判官：{', '.join(unknown)}。"
            f"线上没有 ground truth，可用的只有 {', '.join(sorted(ONLINE_EVALUATORS))}"
        )

    rate = raw["sampling_rate"]
    if not isinstance(rate, (int, float)) or not 0.0 <= float(rate) <= 1.0:
        raise ValueError(f"任务 {raw['name']} 的 sampling_rate 必须在 [0, 1]，实际 {rate!r}")

    cadence = raw["cadence"]
    if cadence not in VALID_CADENCE:
        raise ValueError(
            f"任务 {raw['name']} 的 cadence 必须是 {'/'.join(sorted(VALID_CADENCE))}，"
            f"实际 {cadence!r}"
        )

    return OnlineTask(
        name=str(raw["name"]),
        project=str(raw["project"]),
        span_kind=str(raw["span_kind"]),
        evaluators=tuple(evaluators),
        sampling_rate=float(rate),
        cadence=cadence,
        window_minutes=int(raw.get("window_minutes", 30)),
    )


def load_tasks(path: str = DEFAULT_PATH) -> list[OnlineTask]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} 必须是非空的任务列表")

    tasks = [_check_one(item, i) for i, item in enumerate(raw)]

    seen: set[str] = set()
    for t in tasks:
        if t.name in seen:
            raise ValueError(f"任务名重复：{t.name}")
        seen.add(t.name)
    return tasks
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_online_tasks.py -o addopts="" -q`
Expected: 8 passed

- [ ] **Step 6: 确认真实声明文件能被加载**

Run: `python -c "from evaluation.phoenix.online_tasks import load_tasks; [print(t.name, t.evaluators, t.sampling_rate) for t in load_tasks()]"`
Expected:
```
canary-quality ('faithfulness', 'answer_relevancy') 0.2
canary-guardrail ('refusal_check',) 1.0
```

- [ ] **Step 7: 提交**

```bash
git add evaluation/phoenix/online_tasks.py evaluation/phoenix/online_tasks.yaml tests/test_online_tasks.py
git commit -m "feat(eval): 线上评估任务声明与 eager 校验"
```

---

### Task 2: 从 span 提取判官输入

**Files:**
- Create: `evaluation/phoenix/span_extract.py`
- Test: `tests/test_span_extract.py`

**Interfaces:**
- Consumes: 无
- Produces: `extract_eval_input(root_span: dict, trace_spans: list[dict]) -> dict | None` —— 返回 `{"question": str, "answer": str, "contexts": list[str]}`，缺关键字段时返回 `None`（调用方跳过该 span，不当成失败）

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_span_extract.py
"""从 Phoenix span 里拆出判官要的三样东西。

fixture 用的是真实 span 形状（2026-09-10 从本机 bz-rag-ci project 取出后简化）：
根 AGENT span 的 input.value / output.value 是纯字符串；
RERANKER span 的 output.value 是 JSON 字符串，形如
[{"id": null, "metadata": {...}, "page_content": "..."}]。
"""

import json

from evaluation.phoenix.span_extract import extract_eval_input


def _span(kind, name, attrs, span_id="s1"):
    return {
        "id": span_id,
        "name": name,
        "span_kind": kind,
        "attributes": attrs,
        "context": {"trace_id": "t1", "span_id": span_id},
    }


ROOT = _span(
    "AGENT",
    "milvus-hybrid-rag",
    {"input.value": "禾蛙平台的创始人是谁？", "output.value": "创始人是何洪锴。"},
    span_id="root",
)
RERANK = _span(
    "RERANKER",
    "_rerank",
    {
        "output.value": json.dumps(
            [
                {"id": None, "metadata": {"source": "u1"}, "page_content": "片段一"},
                {"id": None, "metadata": {"source": "u2"}, "page_content": "片段二"},
            ],
            ensure_ascii=False,
        )
    },
    span_id="rr",
)


class TestHappyPath:
    def test_extracts_all_three(self):
        got = extract_eval_input(ROOT, [ROOT, RERANK])
        assert got == {
            "question": "禾蛙平台的创始人是谁？",
            "answer": "创始人是何洪锴。",
            "contexts": ["片段一", "片段二"],
        }

    def test_prefers_reranker_over_retriever(self):
        """判官必须看到真正喂给 LLM 的那份，也就是重排后的——与期 1 的取舍一致。"""
        retriever = _span(
            "RETRIEVER",
            "retrieve",
            {
                "output.value": json.dumps(
                    [{"page_content": "重排前的六条之一"}], ensure_ascii=False
                )
            },
            span_id="rt",
        )
        got = extract_eval_input(ROOT, [ROOT, retriever, RERANK])
        assert got["contexts"] == ["片段一", "片段二"]


class TestSkips:
    def test_missing_answer_returns_none(self):
        root = _span("AGENT", "milvus-hybrid-rag", {"input.value": "问题"}, span_id="root")
        assert extract_eval_input(root, [root]) is None

    def test_missing_question_returns_none(self):
        root = _span("AGENT", "milvus-hybrid-rag", {"output.value": "答案"}, span_id="root")
        assert extract_eval_input(root, [root]) is None

    def test_no_reranker_yields_empty_contexts_not_none(self):
        """没有重排 span 不算提取失败：answer_relevancy 与 refusal_check 不需要上下文。"""
        got = extract_eval_input(ROOT, [ROOT])
        assert got is not None
        assert got["contexts"] == []

    def test_malformed_reranker_json_yields_empty_contexts(self):
        bad = _span("RERANKER", "_rerank", {"output.value": "not-json"}, span_id="rr")
        got = extract_eval_input(ROOT, [ROOT, bad])
        assert got is not None
        assert got["contexts"] == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_span_extract.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix.span_extract'`

- [ ] **Step 3: 写实现**

创建 `evaluation/phoenix/span_extract.py`：

```python
"""把 Phoenix span 拆成判官要的输入。

评估对象是**根 AGENT span**，不是 spec §4.8 原文写的 LLM span。实测一条 trace：

    AGENT      milvus-hybrid-rag   (root)   input.value=问题  output.value=答案
      RETRIEVER  retrieve                   output.value=重排前的 6 条
      RERANKER   _rerank                    output.value=重排后喂给 LLM 的 4 条
      CHAIN      _generate
        LLM      ChatOpenAI                 langchain instrumentor 记的
        LLM      ChatCompletion             openai instrumentor 记的（同一次调用第二遍）

按 LLM 抽会把同一次生成算两次，多轮时还会多抓 _rewrite_query 的 LLM span。
根 AGENT span 一次请求恰好一条，且两个字段直接可用，不用去 prompt 里做字符串切割。

上下文取 RERANKER 而不是 RETRIEVER：判官必须看到真正喂给 LLM 的那一份。
这与 api/milvus_rag_phoenix.py 顶部记的取舍一致（MLflow 版取重排前的 6 条，
与生成实际用的不一致，是那边的已知缺陷）。
"""

import json
from typing import Any

_CONTEXT_SPAN_KIND = "RERANKER"


def _contexts_from(spans: list[dict[str, Any]]) -> list[str]:
    for s in spans:
        if s.get("span_kind") != _CONTEXT_SPAN_KIND:
            continue
        raw = (s.get("attributes") or {}).get("output.value")
        if not raw:
            return []
        try:
            docs = json.loads(raw)
        except (TypeError, ValueError):
            # 形状变了就当没有上下文：faithfulness 会因此判低分，
            # 这比让整个 worker 崩掉、或静默丢样本都好。
            return []
        if not isinstance(docs, list):
            return []
        return [d.get("page_content", "") for d in docs if isinstance(d, dict)]
    return []


def extract_eval_input(
    root_span: dict[str, Any], trace_spans: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """返回 {"question", "answer", "contexts"}；缺问题或答案时返回 None。

    返回 None 表示"这条 span 不适合评"，调用方应跳过并计数，**不要**当成判官失败——
    两者的处置完全不同：前者是正常的形状过滤，后者要进 max_error_rate 闸门。
    """
    attrs = root_span.get("attributes") or {}
    question = attrs.get("input.value")
    answer = attrs.get("output.value")
    if not question or not answer:
        return None
    return {
        "question": str(question),
        "answer": str(answer),
        "contexts": _contexts_from(trace_spans),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_span_extract.py -o addopts="" -q`
Expected: 6 passed

- [ ] **Step 5: 拿真实 span 验一遍**

Run（需要本机 Phoenix 在跑，且 `bz-rag-ci` 里有期 1 留下的 trace）：

```bash
NO_PROXY=localhost,127.0.0.1 python -c "
import os; os.environ['NO_PROXY']='localhost,127.0.0.1'
from collections import defaultdict
from phoenix.client import Client
from evaluation.phoenix.span_extract import extract_eval_input
sp = Client(base_url='http://localhost:6006').spans.get_spans(project_identifier='bz-rag-ci', limit=60)
byt = defaultdict(list)
for s in sp: byt[s['context']['trace_id']].append(s)
n = 0
for tid, spans in byt.items():
    root = next((x for x in spans if x['span_kind']=='AGENT' and not x.get('parent_id')), None)
    if not root: continue
    got = extract_eval_input(root, spans)
    if got:
        n += 1
        if n == 1:
            print('question:', got['question'][:40])
            print('answer  :', got['answer'][:40])
            print('contexts:', len(got['contexts']), '条')
print('可评的根 span 数:', n)
"
```
Expected: 打印出真实的问题/答案，`contexts` 条数为 4（`RERANK_TOP_K=4`），`可评的根 span 数` > 0

- [ ] **Step 6: 提交**

```bash
git add evaluation/phoenix/span_extract.py tests/test_span_extract.py
git commit -m "feat(eval): 从根 AGENT span 提取判官输入，上下文取重排后的片段"
```

---

### Task 3: 确定性抽样与护栏豁免

**Files:**
- Create: `evaluation/phoenix/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: 无
- Produces: `should_sample(span_id: str, rate: float) -> bool`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_sampling.py
"""抽样必须是确定性的：同一条 span 在任何一轮都得到同样的抽中/不抽中判定。

理由是窗口会重叠。cadence=continuous + window_minutes=30 的任务每轮都会
把最近 30 分钟的 span 全拉回来，同一条 span 会被反复看到。若用
random.random()，它可能这轮没抽中、下轮抽中，去重逻辑就失去意义，
成本旋钮也不再线性——采样率 0.2 跑六轮实际会评掉远超 20% 的 span。
"""

from evaluation.phoenix.sampling import should_sample


class TestDeterminism:
    def test_same_span_same_verdict(self):
        ids = [f"span-{i}" for i in range(200)]
        first = [should_sample(i, 0.3) for i in ids]
        second = [should_sample(i, 0.3) for i in ids]
        assert first == second

    def test_different_ids_not_all_equal(self):
        """不能退化成全抽或全不抽。"""
        verdicts = {should_sample(f"span-{i}", 0.5) for i in range(50)}
        assert verdicts == {True, False}


class TestRateBoundaries:
    def test_rate_one_always_samples(self):
        """护栏型任务写 sampling_rate: 1.0，必须一条都不漏。"""
        assert all(should_sample(f"span-{i}", 1.0) for i in range(500))

    def test_rate_zero_never_samples(self):
        assert not any(should_sample(f"span-{i}", 0.0) for i in range(500))

    def test_rate_roughly_proportional(self):
        """0.2 在 2000 条上应落在 20% 附近；给足容差，这里验的是量级不是精度。"""
        n = 2000
        hit = sum(should_sample(f"span-{i}", 0.2) for i in range(n))
        assert 0.15 * n < hit < 0.25 * n
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_sampling.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix.sampling'`

- [ ] **Step 3: 写实现**

创建 `evaluation/phoenix/sampling.py`：

```python
"""按 span_id 做确定性抽样。

**不用 random.random()**：cadence=continuous 的任务窗口会重叠，同一条 span
会被连续几轮反复拉回来。随机抽样下它每轮都重新掷一次骰子，采样率 0.2 跑六轮
实际评掉的远超 20%，成本旋钮失去意义；确定性抽样则让"这条 span 要不要评"
成为它自身的属性，跑多少轮都一样。

用 sha256 而不是内置 hash()：Python 的 str.__hash__ 默认带 PYTHONHASHSEED
随机化，进程间不一致——worker 每轮由 cron 起一个新进程，那样就不确定了。
"""

import hashlib

_MAX = 0xFFFFFFFF


def should_sample(span_id: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(span_id.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) / _MAX) < rate
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_sampling.py -o addopts="" -q`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add evaluation/phoenix/sampling.py tests/test_sampling.py
git commit -m "feat(eval): 按 span_id 的确定性抽样，护栏型 rate=1.0 全评"
```

---

### Task 4: online_worker 主循环

这是期 2 的核心交付物。做完这一步就能对着已有的 `bz-rag-ci` project 跑一轮，在 Phoenix UI 上看到 span 挂上 annotation——**不依赖影子 canary**，所以先做它、把最难的部分先验证掉。

**Files:**
- Create: `evaluation/phoenix/online_worker.py`
- Test: `tests/test_online_worker.py`

**Interfaces:**
- Consumes: `online_tasks.load_tasks` / `OnlineTask`、`span_extract.extract_eval_input`、`sampling.should_sample`、`evaluators` 里的 `faithfulness` / `answer_relevancy` / `refusal_check`
- Produces:
  - `run_task(client, task, now=None) -> RoundStats`
  - `class RoundStats(NamedTuple)`：`task: str`、`pulled: int`、`skipped: int`、`deduped: int`、`sampled: int`、`annotated: int`、`errored: int`
  - `main(argv=None) -> int` —— CLI 入口，退出码 0=成功，1=有任务抛异常

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_online_worker.py
"""worker 一轮的行为。全部用假 client，不打网络、不调判官。"""

import json
from datetime import datetime, timezone

import pytest

from evaluation.phoenix.online_tasks import OnlineTask
from evaluation.phoenix.online_worker import RoundStats, run_task


def _span(span_id, kind="AGENT", parent=None, q="问题", a="答案"):
    return {
        "id": span_id,
        "name": "milvus-hybrid-rag",
        "span_kind": kind,
        "parent_id": parent,
        "attributes": {"input.value": q, "output.value": a},
        "context": {"trace_id": f"t-{span_id}", "span_id": span_id},
    }


class FakeSpans:
    def __init__(self, spans, existing_annotations=()):
        self._spans = spans
        self._existing = list(existing_annotations)
        self.logged = []

    def get_spans(self, **kw):
        self.kw = kw
        return list(self._spans)

    def get_span_annotations(self, **kw):
        return list(self._existing)

    def log_span_annotations_dataframe(self, *, dataframe, annotator_kind, annotation_name):
        self.logged.append((annotation_name, annotator_kind, dataframe))
        return []


class FakeClient:
    def __init__(self, spans, existing=()):
        self.spans = FakeSpans(spans, existing)


TASK = OnlineTask(
    name="t", project="bz-rag-canary", span_kind="AGENT",
    evaluators=("refusal_check",), sampling_rate=1.0,
    cadence="continuous", window_minutes=30,
)
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def stub_judge(monkeypatch):
    calls = []

    def fake(output, input=None, **_):  # noqa: A002
        calls.append((output, input))
        return {"name": "refusal_check", "score": 1.0, "label": "ok", "explanation": "行"}

    monkeypatch.setattr(
        "evaluation.phoenix.online_worker.JUDGES", {"refusal_check": fake}
    )
    return calls


class TestPullAndFilter:
    def test_queries_with_span_kind_and_window(self, stub_judge):
        c = FakeClient([_span("a")])
        run_task(c, TASK, now=NOW)
        assert c.spans.kw["project_identifier"] == "bz-rag-canary"
        assert c.spans.kw["span_kind"] == "AGENT"
        assert (NOW - c.spans.kw["start_time"]).total_seconds() == 30 * 60

    def test_non_root_spans_skipped(self, stub_judge):
        """只评根 span：带 parent_id 的是子 span，评它会重复计数。"""
        c = FakeClient([_span("root"), _span("child", parent="root")])
        stats = run_task(c, TASK, now=NOW)
        assert stats.pulled == 2
        assert stats.annotated == 1

    def test_span_without_answer_is_skipped_not_errored(self, stub_judge):
        bad = _span("x")
        bad["attributes"] = {"input.value": "只有问题"}
        c = FakeClient([bad])
        stats = run_task(c, TASK, now=NOW)
        assert stats.skipped == 1
        assert stats.errored == 0
        assert stats.annotated == 0


class TestDedup:
    def test_span_with_existing_annotation_is_not_re_evaluated(self, stub_judge):
        """窗口重叠时同一条 span 会被反复拉到，已评过的必须跳过，否则重复计费。"""
        c = FakeClient(
            [_span("a")],
            existing=[{"span_id": "a", "name": "refusal_check"}],
        )
        stats = run_task(c, TASK, now=NOW)
        assert stats.deduped == 1
        assert stats.annotated == 0
        assert c.spans.logged == []


class TestSampling:
    def test_guardrail_rate_one_evaluates_everything(self, stub_judge):
        c = FakeClient([_span(f"s{i}") for i in range(20)])
        stats = run_task(c, TASK, now=NOW)
        assert stats.sampled == 20
        assert stats.annotated == 20

    def test_rate_zero_evaluates_nothing(self, stub_judge):
        c = FakeClient([_span(f"s{i}") for i in range(20)])
        stats = run_task(c, TASK._replace(sampling_rate=0.0), now=NOW)
        assert stats.sampled == 0
        assert c.spans.logged == []


class TestWriteBack:
    def test_dataframe_shape(self, stub_judge):
        c = FakeClient([_span("a")])
        run_task(c, TASK, now=NOW)
        (name, kind, df) = c.spans.logged[0]
        assert name == "refusal_check"
        assert kind == "LLM"
        assert list(df["span_id"]) == ["a"]
        assert list(df["label"]) == ["ok"]
        assert list(df["score"]) == [1.0]


class TestJudgeFailure:
    def test_errored_judge_counted_not_raised(self, monkeypatch):
        def boom(output, input=None, **_):  # noqa: A002
            return {"name": "refusal_check", "score": None, "label": "errored",
                    "explanation": "APIConnectionError"}

        monkeypatch.setattr(
            "evaluation.phoenix.online_worker.JUDGES", {"refusal_check": boom}
        )
        c = FakeClient([_span("a")])
        stats = run_task(c, TASK, now=NOW)
        assert stats.errored == 1
        assert stats.annotated == 0
        assert c.spans.logged == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_online_worker.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix.online_worker'`

- [ ] **Step 3: 写实现**

创建 `evaluation/phoenix/online_worker.py`：

```python
"""线上评估 worker：拉 span → 去重 → 抽样 → 跑判官 → 把分数写回 span。

与期 1 离线门禁的关系：**判官只有一份**（evaluation/phoenix/evaluators.py），
离线线上共用。区别只在输入从哪来、结果写到哪去——
离线是 dataset example → experiment run annotation，
线上是 project span → span annotation。这正是 Arize template/task 分离的好处。

期 2 的 worker **不做任何决策**：不 rollback、不 promote、不阻断。
它只负责把分数写上去。聚合与决策是期 3 的 monitor.py。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

import pandas as pd
from phoenix.client import Client

from evaluation.phoenix.evaluators import answer_relevancy, faithfulness, refusal_check
from evaluation.phoenix.online_tasks import OnlineTask, load_tasks
from evaluation.phoenix.sampling import should_sample
from evaluation.phoenix.span_extract import extract_eval_input

logger = logging.getLogger(__name__)

# 名字 → 判官函数。这三个是线上可用的全部（另两个需要 ground truth），
# 与 online_tasks.ONLINE_EVALUATORS 必须保持一致，校验器会把别的名字挡在外面。
JUDGES = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "refusal_check": refusal_check,
}

PULL_LIMIT = 1000


class RoundStats(NamedTuple):
    task: str
    pulled: int      # 从 Phoenix 拉回来的 span 总数
    skipped: int     # 形状不合适（非根 span / 缺问题或答案）——正常过滤，不是失败
    deduped: int     # 已有同名 annotation，本轮跳过
    sampled: int     # 通过采样、真正送去评的
    annotated: int   # 成功写回的 annotation 条数
    errored: int     # 判官失败（label == "errored"）


def _existing_annotation_ids(
    client: Any, project: str, span_ids: list[str], names: tuple[str, ...]
) -> set[tuple[str, str]]:
    """返回 {(span_id, annotation_name)}，用于去重。

    去重是必须的：cadence=continuous 的任务窗口重叠，同一条 span 会被连续
    几轮反复拉到。不去重就会对同一条 span 重复调判官——重复计费，且 Phoenix
    上会堆出一串同名 annotation。
    """
    if not span_ids:
        return set()
    got = client.spans.get_span_annotations(
        span_ids=span_ids,
        project_identifier=project,
        include_annotation_names=list(names),
        limit=PULL_LIMIT,
    )
    out: set[tuple[str, str]] = set()
    for a in got:
        sid = a.get("span_id") if isinstance(a, dict) else getattr(a, "span_id", None)
        nm = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        if sid and nm:
            out.add((sid, nm))
    return out


def run_task(client: Any, task: OnlineTask, now: datetime | None = None) -> RoundStats:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(minutes=task.window_minutes)

    spans = client.spans.get_spans(
        project_identifier=task.project,
        span_kind=task.span_kind,
        start_time=start,
        end_time=now,
        limit=PULL_LIMIT,
    )
    pulled = len(spans)

    # 同 trace 的兄弟 span 用来取上下文（RERANKER 的 output）。
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_trace[(s.get("context") or {}).get("trace_id", "")].append(s)

    roots = [s for s in spans if not s.get("parent_id")]
    skipped = pulled - len(roots)

    already = _existing_annotation_ids(
        client, task.project, [s["id"] for s in roots], task.evaluators
    )

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    deduped = sampled = errored = 0

    for root in roots:
        span_id = root["id"]
        pending = [n for n in task.evaluators if (span_id, n) not in already]
        if not pending:
            deduped += 1
            continue
        if not should_sample(span_id, task.sampling_rate):
            continue

        payload = extract_eval_input(root, by_trace[(root.get("context") or {}).get("trace_id", "")])
        if payload is None:
            skipped += 1
            continue

        sampled += 1
        output = {"answer": payload["answer"], "contexts": payload["contexts"]}
        judge_input = {"question": payload["question"]}
        for name in pending:
            res = JUDGES[name](output=output, input=judge_input)
            if res.get("label") == "errored":
                # 判官失败不写 annotation：写一条 label="errored" 的记录会污染
                # 期 3 monitor 的聚合口径（它按 label 判通过）。计数即可，
                # 让 max_error_rate 那层在 monitor 侧从"本该有几条"的角度发现。
                errored += 1
                continue
            rows[name].append(
                {
                    "span_id": span_id,
                    "label": res.get("label"),
                    "score": res.get("score"),
                    "explanation": res.get("explanation"),
                }
            )

    annotated = 0
    for name, items in rows.items():
        if not items:
            continue
        client.spans.log_span_annotations_dataframe(
            dataframe=pd.DataFrame(items),
            annotator_kind="LLM",
            annotation_name=name,
        )
        annotated += len(items)

    return RoundStats(task.name, pulled, skipped, deduped, sampled, annotated, errored)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑一轮线上评估")
    parser.add_argument("--tasks", default=None, help="online_tasks.yaml 路径")
    parser.add_argument(
        "--only", default=None, help="只跑指定名字的任务（调试用）"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    endpoint = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
    client = Client(base_url=endpoint)

    tasks = load_tasks(args.tasks) if args.tasks else load_tasks()
    if args.only:
        tasks = [t for t in tasks if t.name == args.only]
        if not tasks:
            logger.error("没有名为 %s 的任务", args.only)
            return 1

    failed = False
    print(f"{'task':<20}{'pulled':>8}{'skipped':>9}{'deduped':>9}"
          f"{'sampled':>9}{'annotated':>11}{'errored':>9}")
    print("-" * 75)
    for t in tasks:
        try:
            s = run_task(client, t)
        except Exception:  # noqa: BLE001 —— 一个任务炸不该让整轮停摆
            logger.exception("任务 %s 抛异常", t.name)
            failed = True
            continue
        print(f"{s.task:<20}{s.pulled:>8}{s.skipped:>9}{s.deduped:>9}"
              f"{s.sampled:>9}{s.annotated:>11}{s.errored:>9}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_online_worker.py -o addopts="" -q`
Expected: 9 passed

- [ ] **Step 5: 对真实 project 跑一轮（此时还没有 canary，先打 bz-rag-ci）**

Run（需要 Phoenix 在跑；把窗口开大以覆盖期 1 留下的旧 trace）：

```bash
NO_PROXY=localhost,127.0.0.1 PYTHONIOENCODING=utf-8 python -c "
import os; os.environ['NO_PROXY']='localhost,127.0.0.1'
from phoenix.client import Client
from evaluation.phoenix.online_tasks import OnlineTask
from evaluation.phoenix.online_worker import run_task
t = OnlineTask('smoke','bz-rag-ci','AGENT',('refusal_check',),1.0,'continuous',60*24*7)
print(run_task(Client(base_url='http://localhost:6006'), t))
"
```
Expected: `RoundStats(task='smoke', pulled=N, ..., annotated=M, errored=0)`，M > 0

- [ ] **Step 6: 在 Phoenix UI 上确认 annotation 真挂上了**

打开 http://localhost:6006 → Tracing → `bz-rag-ci` → 点开任一 `milvus-hybrid-rag` span，
在 Annotations 面板应能看到 `refusal_check`，带 label / score / explanation。

再跑一次同样的命令，Expected: `deduped` 等于上一轮的 `annotated`，`annotated=0`——去重生效。

- [ ] **Step 7: 提交**

```bash
git add evaluation/phoenix/online_worker.py tests/test_online_worker.py
git commit -m "feat(eval): 线上评估 worker——拉 span、去重、抽样、判官打分写回 span"
```

---

### Task 5: 影子 canary 分流器

**Files:**
- Create: `evaluation/phoenix/shadow/__init__.py`
- Create: `evaluation/phoenix/shadow/router.py`
- Create: `evaluation/phoenix/shadow/weight.json`
- Create: `evaluation/phoenix/shadow/up.ps1`
- Test: `tests/test_shadow_router.py`

**Interfaces:**
- Consumes: 无
- Produces: `pick_backend(weight: int, roll: float) -> str` 返回 `"stable"` 或 `"canary"`；`read_weight(path) -> int`；FastAPI `app`（转发用）

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_shadow_router.py
"""影子分流器的决策逻辑。

复刻 cf-worker/src/index.js：读 canary_weight (0-100)、Math.random()*100 < weight
则走 canary。参数形状必须与 scripts/cf-kv-update.sh 一致（0-100 整数），
这样期 3 的 monitor 输出能同时驱动影子侧和真云侧，将来上云不用改 monitor。
"""

import json

import pytest

from evaluation.phoenix.shadow.router import pick_backend, read_weight


class TestPickBackend:
    def test_weight_zero_always_stable(self):
        assert all(pick_backend(0, r / 100) == "stable" for r in range(100))

    def test_weight_hundred_always_canary(self):
        assert all(pick_backend(100, r / 100) == "canary" for r in range(100))

    def test_boundary_is_strict_less_than(self):
        """roll*100 < weight 才走 canary，与 cf-worker 的 Math.random()*100 < weight 一致。"""
        assert pick_backend(50, 0.49) == "canary"
        assert pick_backend(50, 0.50) == "stable"


class TestReadWeight:
    def test_reads_int(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"canary_weight": 25}), encoding="utf-8")
        assert read_weight(str(p)) == 25

    def test_missing_file_defaults_to_zero(self, tmp_path):
        """读不到就当全量走 stable——失败方向要安全。"""
        assert read_weight(str(tmp_path / "nope.json")) == 0

    def test_out_of_range_rejected(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"canary_weight": 150}), encoding="utf-8")
        with pytest.raises(ValueError, match="canary_weight"):
            read_weight(str(p))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_shadow_router.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 写权重文件与实现**

创建 `evaluation/phoenix/shadow/__init__.py`（空文件）。

创建 `evaluation/phoenix/shadow/weight.json`：

```json
{"canary_weight": 50}
```

创建 `evaluation/phoenix/shadow/router.py`：

```python
"""影子 canary 分流器：复刻 cf-worker/src/index.js 的按权重分流。

Railway 上跑不了 Milvus 管线（CD-pipeline Q5 遗留），所以线上段在本机复刻：
两个不同 APP_VERSION、相同 PHOENIX_PROJECT_NAME=bz-rag-canary 的 uvicorn，
前面挂这个分流器。

权重参数形状与 scripts/cf-kv-update.sh 完全一致（0-100 整数），这样期 3 的
monitor 输出能同时驱动影子侧和真云侧——将来 Milvus 上云时 monitor 一行不用改。
scripts/cf-kv-update.sh 本身保持原样不动。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import json
import pathlib
import random

import httpx
from fastapi import FastAPI, Request, Response

WEIGHT_PATH = str(pathlib.Path(__file__).parent / "weight.json")
STABLE_URL = os.environ.get("SHADOW_STABLE_URL", "http://127.0.0.1:8101")
CANARY_URL = os.environ.get("SHADOW_CANARY_URL", "http://127.0.0.1:8102")

app = FastAPI(title="bz-rag shadow router")


def read_weight(path: str = WEIGHT_PATH) -> int:
    """读权重。文件不存在时返回 0——失败方向要安全（全量走 stable）。"""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return 0
    w = raw.get("canary_weight", 0)
    if not isinstance(w, int) or not 0 <= w <= 100:
        raise ValueError(f"canary_weight 必须是 0-100 的整数，实际 {w!r}")
    return w


def pick_backend(weight: int, roll: float) -> str:
    """roll 是 [0,1) 的随机数。与 cf-worker 的 Math.random()*100 < weight 同语义。"""
    return "canary" if roll * 100 < weight else "stable"


@app.post("/api/milvus/query-phoenix")
async def route(request: Request) -> Response:
    backend = pick_backend(read_weight(), random.random())
    target = CANARY_URL if backend == "canary" else STABLE_URL
    body = await request.body()
    async with httpx.AsyncClient(timeout=120.0) as client:
        upstream = await client.post(
            f"{target}/api/milvus/query-phoenix",
            content=body,
            headers={"content-type": "application/json"},
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers={"x-bz-backend": backend},
    )


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "canary_weight": read_weight()}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_shadow_router.py -o addopts="" -q`
Expected: 6 passed

- [ ] **Step 5: 写一键启动脚本**

创建 `evaluation/phoenix/shadow/up.ps1`：

```powershell
# 起影子 canary：stable(8101) + canary(8102) + 分流器(8100)。
# 两个实例用不同 APP_VERSION、相同 PHOENIX_PROJECT_NAME=bz-rag-canary，
# 这样 trace 落在同一个 project 里、靠 APP_VERSION 区分版本。
$ErrorActionPreference = "Stop"
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = $env:NO_PROXY
$env:PHOENIX_PROJECT_NAME = "bz-rag-canary"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $repo

Start-Process powershell -ArgumentList @(
  "-NoExit","-Command",
  "`$env:NO_PROXY='localhost,127.0.0.1'; `$env:PHOENIX_PROJECT_NAME='bz-rag-canary';" +
  "`$env:APP_VERSION='stable'; python -m uvicorn api.main:app --port 8101"
)
Start-Process powershell -ArgumentList @(
  "-NoExit","-Command",
  "`$env:NO_PROXY='localhost,127.0.0.1'; `$env:PHOENIX_PROJECT_NAME='bz-rag-canary';" +
  "`$env:APP_VERSION='canary'; python -m uvicorn api.main:app --port 8102"
)
Write-Host "stable -> :8101   canary -> :8102"
Write-Host "router -> :8100  (前台运行，Ctrl+C 停止)"
python -m uvicorn evaluation.phoenix.shadow.router:app --port 8100
```

- [ ] **Step 6: 起起来验一次分流**

在一个终端跑 `powershell -File evaluation/phoenix/shadow/up.ps1`，另一个终端：

```bash
NO_PROXY=localhost,127.0.0.1 bash -c 'for i in 1 2 3 4 5 6; do
  curl -s -D - -o /dev/null -X POST http://localhost:8100/api/milvus/query-phoenix \
    -H "content-type: application/json" -d "{\"query\":\"禾蛙是什么平台\"}" | grep -i x-bz-backend
done'
```
Expected: 6 行里 stable 与 canary 都出现过（weight=50）

- [ ] **Step 7: 提交**

```bash
git add evaluation/phoenix/shadow/ tests/test_shadow_router.py
git commit -m "feat(eval): 影子 canary 分流器，复刻 cf-worker 的按权重分流"
```

---

### Task 6: replay 打流量 + canary-watch.yml + 期 2 验收

**Files:**
- Create: `evaluation/phoenix/shadow/query_pool.json`
- Create: `evaluation/phoenix/shadow/replay.py`
- Create: `.github/workflows/canary-watch.yml`

**Interfaces:**
- Consumes: `shadow/router.py` 的 :8100、`online_worker.main`
- Produces: 无（终端交付物）

- [ ] **Step 1: 写查询池**

创建 `evaluation/phoenix/shadow/query_pool.json`。**这些问题必须不在 `evaluation/test_dataset.json` 里**——线上评估不能只测训练过的 case：

```json
[
  "禾蛙的蛙贝可以提现吗",
  "接单之后多久必须推荐第一个候选人",
  "发单方能看到接单顾问的联系方式吗",
  "候选人入职之后佣金什么时候结算",
  "同一个候选人被两个顾问推荐了怎么算",
  "禾蛙对猎企的资质有什么要求",
  "职位关闭之后已推荐的简历还算数吗",
  "平台抽成比例是多少",
  "怎么申诉一条不合理的差评",
  "禾蛙盒子和普通发单有什么区别"
]
```

- [ ] **Step 2: 写 replay**

创建 `evaluation/phoenix/shadow/replay.py`：

```python
"""按节奏向影子分流器打流量。

查询池刻意取自 golden 集**之外**：线上评估如果只测训练过的 case，
测出来的是"金标集上的表现"，不是线上表现。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import json
import pathlib
import random
import time

import httpx

POOL_PATH = str(pathlib.Path(__file__).parent / "query_pool.json")
ROUTER_URL = os.environ.get("SHADOW_ROUTER_URL", "http://127.0.0.1:8100")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="向影子分流器打流量")
    p.add_argument("-n", "--count", type=int, default=10)
    p.add_argument("--interval", type=float, default=1.0, help="每次请求之间的秒数")
    args = p.parse_args(argv)

    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)

    backends: dict[str, int] = {}
    with httpx.Client(timeout=180.0) as client:
        for i in range(args.count):
            q = random.choice(pool)
            try:
                r = client.post(
                    f"{ROUTER_URL}/api/milvus/query-phoenix", json={"query": q}
                )
                b = r.headers.get("x-bz-backend", "?")
                backends[b] = backends.get(b, 0) + 1
                print(f"[{i + 1}/{args.count}] {b:<7} {r.status_code}  {q[:24]}")
            except Exception as e:  # noqa: BLE001 —— 打流量不该因单次失败中断
                print(f"[{i + 1}/{args.count}] ERROR {type(e).__name__}: {e}")
            if i + 1 < args.count:
                time.sleep(args.interval)

    print("分流统计:", backends)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 打一批流量并确认 trace 落进 bz-rag-canary**

Run: `python -m evaluation.phoenix.shadow.replay -n 12 --interval 0.5`
Expected: 12 行输出，`分流统计` 里 stable 与 canary 都有；然后：

```bash
NO_PROXY=localhost,127.0.0.1 python -c "
import os; os.environ['NO_PROXY']='localhost,127.0.0.1'
from phoenix.client import Client
sp = Client(base_url='http://localhost:6006').spans.get_spans(
    project_identifier='bz-rag-canary', span_kind='AGENT', limit=100)
print('bz-rag-canary 里的根 AGENT span:', len([s for s in sp if not s.get('parent_id')]))
"
```
Expected: 12（或接近，取决于是否有失败请求）

- [ ] **Step 4: 对 canary project 跑一轮 worker**

Run: `python -m evaluation.phoenix.online_worker`
Expected: 两行统计，`canary-guardrail` 的 `sampled` 等于根 span 数（rate=1.0 全评），
`canary-quality` 的 `sampled` 约为 20%

- [ ] **Step 5: 写 cron workflow**

创建 `.github/workflows/canary-watch.yml`。⚠️ 注意 Global Constraints 里那几条 Windows/PowerShell 约束——**`run:` 块必须纯 ASCII**、每条命令后查 `$LASTEXITCODE`、`env` 键名不能大小写重复：

```yaml
name: Canary Watch

on:
  schedule:
    # 每 30 分钟一轮，与 online_tasks.yaml 的 window_minutes: 30 对齐。
    # cron 漏跑不影响正确性：worker 是幂等的（去重靠 span 上已有的同名 annotation），
    # 下一轮会把上一轮漏掉的窗口一并覆盖。
    - cron: "*/30 * * * *"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  watch:
    # 与 eval-gate.yml 同一台机器：worker 要访问 localhost:6006 的 Phoenix。
    runs-on: [self-hosted, bz-rag-local]
    timeout-minutes: 30
    env:
      NO_PROXY: localhost,127.0.0.1
      PHOENIX_ENDPOINT: http://localhost:6006
      PHOENIX_COLLECTOR_ENDPOINT: http://localhost:6006
      PYTHONIOENCODING: utf-8
      # 凭证走 Repository secrets，不依赖 runner 本机 .env——
      # checkout 出来的是全新工作目录，.env 在 .gitignore 里不会进去。
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      ZHIPUAI_API_KEY: ${{ secrets.ZHIPUAI_API_KEY }}
      JUDGE_OPENAI_API_KEY: ${{ secrets.JUDGE_OPENAI_API_KEY }}
      OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
      MODEL_ID: ${{ secrets.MODEL_ID }}
      JUDGE_OPENAI_BASE_URL: ${{ secrets.JUDGE_OPENAI_BASE_URL }}
      JUDGE_MODEL_ID: ${{ secrets.JUDGE_MODEL_ID }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      # 复用 eval-gate.yml 建的同一个 venv（在 RUNNER_WORKSPACE 下、checkout 目录之外）。
      # 不重建：两个 workflow 依赖完全一致，重建只是白白多花几分钟下载。
      - name: Reuse venv
        run: |
          $venv = Join-Path $env:RUNNER_WORKSPACE "_venv"
          if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
            Write-Host "::error::venv not found at $venv. Run the Eval Gate workflow once first."
            exit 1
          }
          Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $venv "Scripts") -Encoding utf8

      - name: Check Phoenix is up
        run: |
          $code = & curl.exe -s -o NUL -w "%{http_code}" --max-time 10 http://localhost:6006/readyz
          Write-Host "phoenix /readyz -> $code"
          if ($code -ne "200") {
            Write-Host "::error::Phoenix is not ready (/readyz returned $code)."
            exit 1
          }

      - name: Run online eval worker (one round)
        run: |
          python -m evaluation.phoenix.online_worker
          if ($LASTEXITCODE -ne 0) {
            Write-Host "::error::online worker round failed"
            exit $LASTEXITCODE
          }
```

- [ ] **Step 6: 本地把 workflow 验一遍再推**

期 1 在这里烧掉了六轮 CI，别重蹈覆辙。跑这段本地校验（大小写重复键 + run 块纯 ASCII + PowerShell 语法）：

```bash
PYTHONIOENCODING=utf-8 python - <<'PY'
import pathlib, subprocess, tempfile, yaml, collections
p = pathlib.Path(r"E:\wwwroot\BZ\BZ-RAG\.github\workflows\canary-watch.yml")
class L(yaml.SafeLoader): pass
dup = []
def cm(loader, node, deep=False):
    seen = {}
    for kn, _ in node.value:
        k = str(loader.construct_object(kn, deep=True)); cf = k.casefold()
        if cf in seen: dup.append(f"L{kn.start_mark.line+1} '{k}'")
        else: seen[cf] = k
    return yaml.SafeLoader.construct_mapping(loader, node, deep)
L.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, cm)
d = yaml.load(p.read_text(encoding="utf-8"), L)
print("大小写重复键:", dup or "无")
ok = not dup
for job in d["jobs"].values():
    for s in job["steps"]:
        run = s.get("run")
        if not run: continue
        na = [c for c in run if ord(c) > 127]
        f = pathlib.Path(tempfile.mkdtemp()) / "s.ps1"; f.write_bytes(run.encode("utf-8"))
        cmd = ("$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{f}',[ref]$null,[ref]$e); if($e.Count){{'FAIL: '+$e[0].Message}}else{{'OK'}}")
        r = subprocess.run(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-Command",cmd],
                           capture_output=True, text=True, timeout=90)
        res = (r.stdout or r.stderr).strip().splitlines()[0]
        if na or res != "OK": ok = False
        print(f"  {s['name']:<34} 非ASCII={len(na):<3} 解析={res[:40]}")
print("结论:", "通过" if ok else "有问题")
PY
```
Expected: `大小写重复键: 无`，每个 step `非ASCII=0` 且 `解析=OK`，`结论: 通过`

- [ ] **Step 7: 提交并推送**

```bash
git add evaluation/phoenix/shadow/query_pool.json evaluation/phoenix/shadow/replay.py .github/workflows/canary-watch.yml
git commit -m "feat(eval): replay 打流量 + canary-watch cron workflow"
git push origin master
```

- [ ] **Step 8: 手动触发一次确认 CI 上能跑**

Run: `gh workflow run canary-watch.yml --repo brucezhu9594/BZ-RAG`
然后 `gh run list --workflow=canary-watch.yml --limit 1`，等它 completed。
Expected: conclusion=success，日志里有两行任务统计

- [ ] **Step 9: 期 2 验收（spec §7 原文的三条）**

**① span 上挂着 `eval.<name>.*`**：Phoenix UI → Tracing → `bz-rag-canary` → 点开一条 `milvus-hybrid-rag` span → Annotations 面板有 `faithfulness` / `answer_relevancy` / `refusal_check`。

**② 采样率生效**：把 `online_tasks.yaml` 里 `canary-quality` 的 `sampling_rate` 从 `0.2` 改成 `1.0`，清掉已有 annotation 后重打一批流量再跑 worker，`sampled` 应从约 20% 变成 100%；改回 0.2 后新流量的 `sampled` 回落。

**③ 护栏不受采样率影响**：同一轮里 `canary-guardrail` 的 `sampled` 始终等于根 span 数，与 `canary-quality` 的采样率无关。

三条都成立后，把结果记进 `docs/CD-pipeline.md`（记得 `git add -f`）。

---

## Self-Review

**1. Spec coverage**

| spec 章节 | 覆盖它的 Task |
|---|---|
| §4.7 影子 canary（router / 双实例 / replay / 权重形状与 cf-kv-update.sh 一致） | Task 5、Task 6 |
| §4.8 online_worker + online_tasks.yaml（template/task 分离、filter、sampling_rate、cadence、window、写回、去重） | Task 1、Task 3、Task 4 |
| §4.8 两条 Arize 口径（护栏不采样；新判官先跑 historical） | 护栏不采样：Task 1 的 `canary-guardrail` + Task 3 的 `rate>=1.0` 分支 + Task 6 验收③。**historical 先验区分度：`cadence` 字段已实现并校验，但期 2 没有"跑一批 historical 再切 continuous"的操作流程** —— 这是有意留给期 3 的，因为它要配合 monitor 看区分度才有意义 |
| §4.9 monitor / §4.10 harvest | **不在期 2 范围**，属期 3 |
| §5 `canary-watch.yml` | Task 6（只跑 worker，不跑 monitor——monitor 是期 3） |
| §5 secrets 边界 | Global Constraints 已改写，Task 6 的 workflow 按改写后的做 |

**2. Placeholder scan** — 无 TBD/TODO；每个代码步骤都有完整可粘贴的实现；测试都有真实断言。

**3. Type consistency** — `OnlineTask` 七个字段在 Task 1 定义，Task 4 的 `run_task` 与测试按同名同序消费；`RoundStats` 七个字段在 Task 4 定义并被 Task 6 的验收步骤引用（`sampled`）；`extract_eval_input` 的返回键 `question/answer/contexts` 在 Task 2 定义，Task 4 按同名取用；`should_sample(span_id, rate)` 签名在 Task 3 定义，Task 4 同名调用；`pick_backend` / `read_weight` 在 Task 5 定义并只在该 Task 内使用。

**一处已知的范围外遗留**：期 1 留下的 `contextual_precision` 没有 criterion（有判官、有分数、不参与判定）——它是离线侧的问题，与期 2 无关，不在本计划内处理。
