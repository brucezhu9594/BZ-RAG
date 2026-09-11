# Phoenix 闭环（期 3）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让线上评估的结果**驱动部署决策** —— monitor 把 span annotation 聚合成 rollback / hold / promote 三态，rollback 自动执行、promote 需人确认，并把线上失败样本捞回来供人工补进金标集。

**Architecture:** monitor 复用期 1 的 `acceptance.py` 聚合引擎（同一套阈值语言、同一份实现），只是输入从 experiment run 换成 span annotation、读 `criteria.yaml` 的 `online` 段。三态里 **hold 优先于 rollback**：样本不足必须独立成态，否则刚部署完没几条 trace 就会被判"全绿"。决策与执行分离：`monitor.py` 只输出判定，动作由 `rollback.py` 执行。

**Tech Stack:** arize-phoenix-client 3.4.0 · `evaluation/phoenix/acceptance.py`（复用）· pytest · 本机 self-hosted runner

**Spec:** `docs/superpowers/specs/2026-09-08-phoenix-eval-cicd-design.md`（§4.9 monitor、§4.10 harvest、§5 CI 装配）

---

## Global Constraints

沿用期 2 计划的全部约束，逐条仍然有效：

- **Python 3.14.3**；CI 跑在 `RUNNER_WORKSPACE\_venv` 隔离环境里。
- **`NO_PROXY` 必须含 `localhost,127.0.0.1`，且在任何 `phoenix` import 之前设置。**
- **凭证一律从 GitHub Repository secrets 注入**，不依赖 runner 本机 `.env`（spec §5 原文已被实测推翻）。七个：`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_ID` / `ZHIPUAI_API_KEY` / `JUDGE_OPENAI_API_KEY` / `JUDGE_OPENAI_BASE_URL` / `JUDGE_MODEL_ID`。
- **workflow 的 `run:` 块必须纯 ASCII**（PowerShell 5.1 在代码页 936 下按 GBK 解码 UTF-8 无 BOM 脚本，中文含非法 GBK 序列会吃掉字符串终止符）。
- **workflow 的 `env:` 键名不能大小写重复**（GitHub 判 `NO_PROXY`/`no_proxy` 为同一键，整个文件被拒收）。
- **PowerShell 多命令 `run:` 块只有最后一条命令的退出码算数**，每条关键命令后显式 `if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`。
- **Windows 默认 shell 是 `powershell.exe` 5.1**：不用三元/`??`；`curl` 是 `Invoke-WebRequest` 别名要写 `curl.exe`；不要写 `shell: bash`。
- **Phoenix span 有两个 id**：`span["id"]` 是 Phoenix 节点 ID，`span["context"]["span_id"]` 才是 OTel 十六进制 id；**注解 API 只认后者**（用错直接 404）。复用 `online_worker.otel_span_id()`。
- 跑测试用 `pytest tests/ -o addopts=""`。
- **`/docs` 在 `.gitignore` 里**，提交 docs 下的文件一律 `git add -f`。
- commit 用 conventional commits；评估设施用 `feat(eval):`，workflow 用 `ci:`。
- **不动期 1/期 2 的既有产出**：`evaluators.py` / `acceptance.py` / `cases.py` / `online_worker.py` / `criteria.yaml` 的 `offline` 段 / `eval-gate.yml` 一行不改。期 3 只**读**它们。

---

## 与 spec 的两处偏差（都在写计划时查证得出）

**① harvest 不能直接写进 `bz-rag-golden`，要单独建 `bz-rag-harvested`。**

spec §4.10 写的是"追加进 golden 集"。但查证后这条行不通，两个独立的原因：

1. **会被下一次 `dataset.py` 抹掉。** 门禁的真实数据源是本地的
   `evaluation/test_dataset.json`（`cases.py` 从它读），Phoenix 上的
   `bz-rag-golden` 只是它的镜像。而 `dataset.py` 用的是
   `create_dataset(..., example_id_key=...)` —— 它是**全量替换**语义，
   "上传里没有的 example 会被删"（该文件顶部注释原话）。回灌进 Phoenix 的样本
   下一次推送就没了。
2. **回灌样本没有 ground truth。** 金标集的 `expected_response` 是
   `contextual_precision` / `contextual_recall` 两个判官的必需输入，线上 span
   上没有标准答案。混进金标集只会让这两个判官对这些样本全部 errored。

所以改成：harvest 写进**独立的 `bz-rag-harvested` dataset**，作为人工分诊队列。
人看过、补上标准答案之后，手工加进 `test_dataset.json`，再走正常的 `dataset.py` 推送。
**回灌这条环由人闭合，不自动闭合** —— 这符合"promote 需人确认"的同一条谨慎原则。

**② 真云侧 rollback 现在跑不通，只能走影子侧。**

`scripts/cf-kv-update.sh` 需要 `CF_API_TOKEN` / `CF_ACCOUNT_ID` / `KV_NAMESPACE_ID`，
而 `CF_API_TOKEN` 至今未配置（`docs/CD-pipeline.md` 的遗留项之一）。
所以 `rollback.py` 默认只改影子侧的 `weight.json`，真云侧走
`--target cloud` 显式开启，且在 token 缺失时**明确报错而不是静默跳过**。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `evaluation/phoenix/monitor.py` | 读 annotation → 喂 acceptance → 输出三态 + 记分卡。**只判定，不动作** |
| `evaluation/phoenix/rollback.py` | 执行权重归零：影子侧写 `weight.json`，云侧调 `cf-kv-update.sh` |
| `evaluation/phoenix/harvest.py` | 按 annotation 筛失败 span → 写进 `bz-rag-harvested` |
| `.github/workflows/canary-watch.yml` | 改：worker 之后加 monitor 步骤 |
| `.github/workflows/promote-stable.yml` | 改：加 `gate` 前置 job，不达标拒绝 promote |
| `tests/test_monitor.py` | 三态判定的单测（受控数据） |
| `tests/test_rollback.py` | 权重写入与 token 缺失的行为 |
| `tests/test_harvest.py` | 失败筛选与 example 形状 |

---

### Task 1: monitor —— 三态判定

**Files:**
- Create: `evaluation/phoenix/monitor.py`
- Test: `tests/test_monitor.py`

**Interfaces:**
- Consumes: `acceptance.load_criteria(path, "online")` / `acceptance.reset()` / `acceptance.record(name, score, label=None, error=None)` / `acceptance.evaluate_all(criteria) -> list[Outcome]` / `acceptance.format_scoreboard(outcomes) -> str`；`online_worker.otel_span_id(span) -> str`
- Produces:
  - `class Decision(NamedTuple)`：`state: str`（`"rollback"` / `"hold"` / `"promote"`）、`outcomes: list`、`scoreboard: str`、`reason: str`
  - `decide(outcomes) -> Decision`
  - `collect(client, project, window_minutes) -> dict[str, int]` —— 把 annotation 喂进 acceptance，返回每个 annotation 名字的记录数
  - `main(argv=None) -> int` —— 退出码 0=promote、1=rollback、2=hold

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_monitor.py
"""三态判定。

最要紧的一条：**hold 优先于 rollback**。样本不足必须独立成态，
否则刚部署完没几条 trace 就会被判"全绿"（spec §4.9 原话）。
判"样本不足"用 Outcome.samples < criterion.min_samples，
**不匹配 reason 字符串**——那是显示文本，改一个字就会把判定改坏。
"""

from evaluation.phoenix.acceptance import Criterion, Outcome
from evaluation.phoenix.monitor import decide


def _outcome(passed, samples, min_samples=20, annotation="faithfulness"):
    c = Criterion(
        annotation=annotation,
        metric="pass_rate",
        pass_when="score >= 0.5",
        min_pass_rate=0.85,
        min_samples=min_samples,
    )
    return Outcome(
        criterion=c,
        passed=passed,
        observed=0.9 if passed else 0.5,
        required=0.85,
        samples=samples,
        reason="",
    )


class TestThreeStates:
    def test_all_pass_with_enough_samples_is_promote(self):
        d = decide([_outcome(True, 30), _outcome(True, 25, annotation="refusal_check")])
        assert d.state == "promote"

    def test_quality_failure_with_enough_samples_is_rollback(self):
        d = decide([_outcome(True, 30), _outcome(False, 25, annotation="refusal_check")])
        assert d.state == "rollback"
        assert "refusal_check" in d.reason

    def test_insufficient_samples_is_hold(self):
        d = decide([_outcome(False, 3)])
        assert d.state == "hold"

    def test_hold_wins_over_rollback(self):
        """一个够样本且失败、一个样本不足——必须 hold，不能 rollback。

        理由：样本不足意味着'还判不了'，此时任何质量结论都不可信，
        贸然 rollback 会把好版本也打回去。
        """
        d = decide(
            [_outcome(False, 30), _outcome(False, 2, annotation="refusal_check")]
        )
        assert d.state == "hold"

    def test_zero_samples_is_hold_not_promote(self):
        """刚部署完一条 trace 都没有时，绝不能判 promote。"""
        d = decide([_outcome(False, 0)])
        assert d.state == "hold"

    def test_empty_outcomes_is_hold(self):
        assert decide([]).state == "hold"


class TestReasonText:
    def test_rollback_reason_names_the_failing_criterion(self):
        d = decide([_outcome(False, 30, annotation="faithfulness")])
        assert "faithfulness" in d.reason

    def test_hold_reason_says_how_many_short(self):
        d = decide([_outcome(False, 7, min_samples=20)])
        assert "7" in d.reason and "20" in d.reason
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_monitor.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix.monitor'`

- [ ] **Step 3: 写实现**

创建 `evaluation/phoenix/monitor.py`：

```python
"""线上监控与决策：把 span annotation 聚合成 rollback / hold / promote 三态。

**复用期 1 的聚合引擎**：criteria.yaml 的 online 段和 offline 段用的是同一套
阈值语言、同一份 acceptance.py 实现。区别只在输入来源——离线是 experiment run
的 annotation，线上是 project span 的 annotation。这是 spec 强调的一致性。

**本模块只判定，不动作。** 权重归零由 rollback.py 执行，开 issue 由 workflow 做。
决策与执行分开，才能在 CI 之外单独跑 monitor 看一眼当前状态而不产生副作用。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

from evaluation.phoenix import acceptance
from evaluation.phoenix.online_worker import otel_span_id

CRITERIA_PATH = str(pathlib.Path(__file__).parent / "criteria.yaml")
PULL_LIMIT = 1000

# 退出码：给 workflow 用。0/1 是惯例的成功/失败，hold 单独占 2，
# 这样 CI 步骤能区分"不达标"和"还判不了"——两者的处置完全不同。
EXIT_PROMOTE = 0
EXIT_ROLLBACK = 1
EXIT_HOLD = 2


class Decision(NamedTuple):
    state: str  # "rollback" | "hold" | "promote"
    outcomes: list[Any]
    scoreboard: str
    reason: str


def decide(outcomes: list[Any]) -> Decision:
    """三态判定。**hold 优先于 rollback。**

    判"样本不足"用 samples < criterion.min_samples，不匹配 reason 字符串——
    那是给人看的显示文本，改一个字就会把判定改坏。
    """
    scoreboard = acceptance.format_scoreboard(outcomes) if outcomes else "(无 criteria)"

    if not outcomes:
        return Decision("hold", outcomes, scoreboard, "没有任何 criterion 可判")

    short = [o for o in outcomes if o.samples < o.criterion.min_samples]
    if short:
        detail = "；".join(
            f"{o.criterion.annotation} 只有 {o.samples} 条，需要 {o.criterion.min_samples}"
            for o in short
        )
        # 样本不足意味着"还判不了"，此时任何质量结论都不可信——
        # 贸然 rollback 会把好版本也打回去，所以 hold 必须优先。
        return Decision("hold", outcomes, scoreboard, f"样本不足：{detail}")

    failed = [o for o in outcomes if not o.passed]
    if failed:
        detail = "；".join(
            f"{o.criterion.annotation}/{o.criterion.metric} "
            f"实测 {o.observed} 未达 {o.required}"
            for o in failed
        )
        return Decision("rollback", outcomes, scoreboard, f"未达标：{detail}")

    return Decision("promote", outcomes, scoreboard, "全部达标且样本量足够")


def collect(client: Any, project: str, window_minutes: int) -> dict[str, int]:
    """拉窗口内的 span annotation，喂进 acceptance 的记录表。

    返回 {annotation_name: 记录数}，供调用方打印/排查。
    注意先 reset()——acceptance 的记录表是模块级全局，不清会把上一轮的混进来。
    """
    acceptance.reset()
    now = datetime.now(timezone.utc)
    spans = client.spans.get_spans(
        project_identifier=project,
        start_time=now - timedelta(minutes=window_minutes),
        end_time=now,
        limit=PULL_LIMIT,
    )
    roots = [s for s in spans if not s.get("parent_id")]
    if not roots:
        return {}

    anns = client.spans.get_span_annotations(
        span_ids=[otel_span_id(s) for s in roots],
        project_identifier=project,
        limit=PULL_LIMIT,
    )
    counts: dict[str, int] = {}
    for a in anns:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        result = a.get("result") if isinstance(a, dict) else getattr(a, "result", None)
        if not name or not isinstance(result, dict):
            continue
        acceptance.record(name, result.get("score"), label=result.get("label"))
        counts[name] = counts.get(name, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="聚合线上评分并输出三态判定")
    p.add_argument("--project", default="bz-rag-canary")
    p.add_argument("--window-minutes", type=int, default=180)
    p.add_argument("--criteria", default=CRITERIA_PATH)
    args = p.parse_args(argv)

    from phoenix.client import Client

    client = Client(base_url=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006"))

    counts = collect(client, args.project, args.window_minutes)
    print(f"窗口 {args.window_minutes} 分钟，收集到 annotation：{counts or '无'}")

    criteria = acceptance.load_criteria(args.criteria, "online")
    outcomes = acceptance.evaluate_all(criteria)
    d = decide(outcomes)

    print(d.scoreboard)
    print(f">>> 判定：{d.state}  （{d.reason}）")

    return {"promote": EXIT_PROMOTE, "rollback": EXIT_ROLLBACK, "hold": EXIT_HOLD}[d.state]


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_monitor.py -o addopts="" -q`
Expected: 8 passed

- [ ] **Step 5: 对真实 canary 跑一次**

Run: `NO_PROXY=localhost,127.0.0.1 python -m evaluation.phoenix.monitor --window-minutes 4320; echo "EXIT=$?"`

Expected: **`判定：hold`，`EXIT=2`**。
当前 `bz-rag-canary` 只有 16 条 refusal_check、2 条 faithfulness，
而 `criteria.yaml` 的 online 段两条都要求 `min_samples: 20`——
所以真机上现在必然是 hold。**这是正确行为，不是 bug**：
spec §4.9 明确要求"样本量不足必须是独立的第三态，不能算通过"。
rollback / promote 两态由上面的单测用受控数据覆盖。

- [ ] **Step 6: 提交**

```bash
git add evaluation/phoenix/monitor.py tests/test_monitor.py
git commit -m "feat(eval): monitor 三态判定，复用离线那套聚合引擎"
```

---

### Task 2: rollback 执行

**Files:**
- Create: `evaluation/phoenix/rollback.py`
- Test: `tests/test_rollback.py`

**Interfaces:**
- Consumes: `evaluation/phoenix/shadow/weight.json`（期 2 建的）、`scripts/cf-kv-update.sh`
- Produces: `set_weight(weight: int, target: str = "shadow", weight_path: str | None = None) -> None`；`main(argv=None) -> int`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_rollback.py
"""权重归零的执行。

影子侧写 weight.json，云侧调 scripts/cf-kv-update.sh。
云侧现在跑不通（CF_API_TOKEN 未配置），所以默认 target=shadow，
且 token 缺失时必须**明确报错**而不是静默跳过——
"以为回滚了其实没回滚"比直接失败危险得多。
"""

import json

import pytest

from evaluation.phoenix.rollback import set_weight


class TestShadow:
    def test_writes_weight_file(self, tmp_path):
        p = tmp_path / "weight.json"
        p.write_text(json.dumps({"canary_weight": 50}), encoding="utf-8")
        set_weight(0, target="shadow", weight_path=str(p))
        assert json.loads(p.read_text(encoding="utf-8")) == {"canary_weight": 0}

    def test_creates_file_if_missing(self, tmp_path):
        p = tmp_path / "weight.json"
        set_weight(25, target="shadow", weight_path=str(p))
        assert json.loads(p.read_text(encoding="utf-8")) == {"canary_weight": 25}

    def test_rejects_out_of_range(self, tmp_path):
        with pytest.raises(ValueError, match="0-100"):
            set_weight(150, target="shadow", weight_path=str(tmp_path / "w.json"))


class TestCloud:
    def test_missing_token_raises_not_silently_skips(self, tmp_path, monkeypatch):
        """CF_API_TOKEN 缺失必须抛，不能当作'回滚成功'。"""
        monkeypatch.delenv("CF_API_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="CF_API_TOKEN"):
            set_weight(0, target="cloud")


class TestTargetValidation:
    def test_unknown_target_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="target"):
            set_weight(0, target="nonesuch", weight_path=str(tmp_path / "w.json"))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_rollback.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix.rollback'`

- [ ] **Step 3: 写实现**

创建 `evaluation/phoenix/rollback.py`：

```python
"""把 canary 权重归零。

参数形状与 scripts/cf-kv-update.sh 完全一致（0-100 整数）——期 2 的影子分流器
刻意保持了这个形状，就是为了让这里一份实现能同时驱动影子侧和真云侧，
将来 Milvus 上云时不用改。

**真云侧现在跑不通**：cf-kv-update.sh 需要 CF_API_TOKEN，而它至今未配置
（docs/CD-pipeline.md 的遗留项）。所以默认 target=shadow；
走 cloud 且 token 缺失时**明确抛错**，绝不静默跳过——
"以为回滚了其实没回滚"比直接失败危险得多。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import json
import pathlib
import subprocess

DEFAULT_WEIGHT_PATH = str(
    pathlib.Path(__file__).parent / "shadow" / "weight.json"
)
CF_SCRIPT = str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "cf-kv-update.sh")
VALID_TARGETS = ("shadow", "cloud")


def set_weight(weight: int, target: str = "shadow", weight_path: str | None = None) -> None:
    if target not in VALID_TARGETS:
        raise ValueError(f"target 必须是 {'/'.join(VALID_TARGETS)}，实际 {target!r}")
    if not isinstance(weight, int) or isinstance(weight, bool) or not 0 <= weight <= 100:
        raise ValueError(f"权重必须是 0-100 的整数，实际 {weight!r}")

    if target == "shadow":
        path = pathlib.Path(weight_path or DEFAULT_WEIGHT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"canary_weight": weight}) + "\n", encoding="utf-8")
        return

    if not os.environ.get("CF_API_TOKEN"):
        raise RuntimeError(
            "CF_API_TOKEN 未设置，无法改真云侧权重。"
            "这是 docs/CD-pipeline.md 记录的遗留项；"
            "在配好之前请用 target=shadow，或手工执行 scripts/cf-kv-update.sh。"
        )
    subprocess.run(["bash", CF_SCRIPT, str(weight)], check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="设置 canary 权重（回滚就是设 0）")
    p.add_argument("weight", type=int, help="0-100 整数")
    p.add_argument("--target", default="shadow", choices=VALID_TARGETS)
    args = p.parse_args(argv)
    set_weight(args.weight, target=args.target)
    print(f"canary_weight -> {args.weight} (target={args.target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_rollback.py -o addopts="" -q`
Expected: 5 passed

- [ ] **Step 5: 真机验一次往返**

```bash
cat evaluation/phoenix/shadow/weight.json
python -m evaluation.phoenix.rollback 0
cat evaluation/phoenix/shadow/weight.json
NO_PROXY=localhost,127.0.0.1 curl -s http://127.0.0.1:8100/api/health
python -m evaluation.phoenix.rollback 50
cat evaluation/phoenix/shadow/weight.json
```
Expected: 权重在 50 → 0 → 50 之间往返；中间那次 `/api/health` 报
`{"status":"ok","canary_weight":0}`（分流器每次请求都重读文件，不用重启）。

- [ ] **Step 6: 提交**

```bash
git add evaluation/phoenix/rollback.py tests/test_rollback.py
git commit -m "feat(eval): 权重归零执行，影子侧写文件、云侧调 cf-kv-update.sh"
```

---

### Task 3: canary-watch 接上 monitor

**Files:**
- Modify: `.github/workflows/canary-watch.yml`

**Interfaces:**
- Consumes: `evaluation.phoenix.online_worker.main`（期 2）、`evaluation.phoenix.monitor.main`、`evaluation.phoenix.rollback.set_weight`
- Produces: 无（workflow 终端交付物）

- [ ] **Step 1: 在 worker 步骤之后追加两步**

在 `.github/workflows/canary-watch.yml` 的 `Run online eval worker (one round)` 之后追加。⚠️ `run:` 块必须纯 ASCII，说明写进 YAML 注释：

```yaml
      # monitor 的退出码：0=promote 1=rollback 2=hold。
      # 三个都是"正常结果"，所以这一步用 continue-on-error 接住非零退出码，
      # 把判定放进 step output 交给下一步处置——让 job 因为 rollback 而变红
      # 是错的：rollback 说明机制正常工作了，不是 CI 故障。
      - name: Run monitor
        id: monitor
        continue-on-error: true
        run: |
          python -m evaluation.phoenix.monitor --project bz-rag-canary --window-minutes 180
          $code = $LASTEXITCODE
          $state = switch ($code) { 0 { "promote" } 1 { "rollback" } 2 { "hold" } default { "error" } }
          Write-Host "monitor state = $state (exit $code)"
          Add-Content -Path $env:GITHUB_OUTPUT -Value "state=$state" -Encoding utf8
          if ($state -eq "error") { exit $code }

      # rollback 是收敛动作（错了只是回到老版本），所以自动执行。
      # promote 是扩散动作，机器判"够好了"的置信度不足以省掉人点一下——
      # 这里只把判定打出来，开 issue / 点 promote 由人做（spec 的决策表）。
      - name: Apply rollback if needed
        if: steps.monitor.outputs.state == 'rollback'
        run: |
          python -m evaluation.phoenix.rollback 0 --target shadow
          if ($LASTEXITCODE -ne 0) {
            Write-Host "::error::rollback failed"
            exit $LASTEXITCODE
          }
          Write-Host "::warning::canary rolled back to weight 0"
```

- [ ] **Step 2: 本机校验 workflow（不靠 CI 试错）**

```bash
PYTHONIOENCODING=utf-8 python - <<'PY'
import pathlib, subprocess, tempfile, yaml
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
Expected: 无重复键、每步 `非ASCII=0` 且 `解析=OK`

- [ ] **Step 3: 提交并推送**

```bash
git add .github/workflows/canary-watch.yml
git commit -m "ci(canary-watch): worker 之后接上 monitor 与自动 rollback"
git push origin master
```

- [ ] **Step 4: 手动触发确认**

```bash
gh workflow run canary-watch.yml --repo brucezhu9594/BZ-RAG
```
等它 completed 后看日志。Expected: `monitor state = hold`（当前样本量不足），
`Apply rollback if needed` 步骤被 skip，job 整体 success。

---

### Task 4: promote-stable 加前置门

**Files:**
- Modify: `.github/workflows/promote-stable.yml`

**Interfaces:**
- Consumes: `evaluation.phoenix.monitor.main`
- Produces: 无

- [ ] **Step 1: 加 gate job 并让 promote 依赖它**

在 `.github/workflows/promote-stable.yml` 的 `jobs:` 下，**在 `promote:` 之前**插入：

```yaml
  # 质量前置门：monitor 说不达标就不许晋级。
  # 这一步是 docs/CD-pipeline.md §10 第 4 条遗留项（"观察期没有自动化告警，
  # promote 完全靠人主观判断观察够了没"）的正解——把判据从"人觉得可以了"
  # 换成"线上评分达标了"。
  #
  # 必须跑在 self-hosted 上：monitor 要访问 localhost:6006 的 Phoenix。
  # 而下面的 promote job 跑 ubuntu-latest（它只调 Railway / Cloudflare API）。
  gate:
    runs-on: [self-hosted, bz-rag-local]
    timeout-minutes: 15
    env:
      NO_PROXY: localhost,127.0.0.1
      PHOENIX_ENDPOINT: http://localhost:6006
      PYTHONIOENCODING: utf-8
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Reuse venv
        run: |
          $venv = Join-Path $env:RUNNER_WORKSPACE "_venv"
          if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
            Write-Host "::error::venv not found at $venv. Run the Eval Gate workflow once first."
            exit 1
          }
          Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $venv "Scripts") -Encoding utf8

      # 这里**不**用 continue-on-error：promote 前置门就是要在非 promote 时挡住。
      # hold（exit 2）同样挡住——样本不足意味着还判不了，此时晋级是赌博。
      - name: Quality gate
        run: |
          python -m evaluation.phoenix.monitor --project bz-rag-canary --window-minutes 180
          $code = $LASTEXITCODE
          if ($code -eq 0) {
            Write-Host "monitor says promote - gate passed"
          } elseif ($code -eq 2) {
            Write-Host "::error::monitor says HOLD (not enough samples). Let the canary run longer, then retry."
            exit 1
          } else {
            Write-Host "::error::monitor says ROLLBACK (criteria not met). Promotion refused."
            exit 1
          }
```

然后给原有的 `promote:` job 加一行依赖（`runs-on` 那行之前）：

```yaml
  promote:
    needs: gate
    runs-on: ubuntu-latest
```

- [ ] **Step 2: 本机校验**

用 Task 3 Step 2 里那段脚本，把路径换成 `promote-stable.yml` 再跑一遍。
Expected: 无重复键、新增两个 run 块 `非ASCII=0` 且 `解析=OK`。

另外确认 job 依赖关系正确：

```bash
PYTHONIOENCODING=utf-8 python -c "
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(r'E:\wwwroot\BZ\BZ-RAG\.github\workflows\promote-stable.yml').read_text(encoding='utf-8'))
for name, job in d['jobs'].items():
    print(f'{name}: runs-on={job.get(\"runs-on\")} needs={job.get(\"needs\")}')
"
```
Expected:
```
gate: runs-on=['self-hosted', 'bz-rag-local'] needs=None
promote: runs-on=ubuntu-latest needs=gate
```

- [ ] **Step 3: 提交并推送**

```bash
git add .github/workflows/promote-stable.yml
git commit -m "ci(promote-stable): 加质量前置门，monitor 不达标拒绝晋级"
git push origin master
```

- [ ] **Step 4: 不要真触发**

`promote-stable.yml` 是手动触发的真实部署流程（会把流量切到 canary、重新部署 stable）。
**本 task 不触发它**——改动的正确性靠 Step 2 的静态校验 + Task 3 里 monitor
在 canary-watch 上已经真跑过来保证。真正第一次跑它，应该由人在真的要晋级时执行。

---

### Task 5: harvest 回灌

**Files:**
- Create: `evaluation/phoenix/harvest.py`
- Test: `tests/test_harvest.py`

**Interfaces:**
- Consumes: `online_worker.otel_span_id`、`span_extract.extract_eval_input`、`monitor.PULL_LIMIT`
- Produces: `failed_spans(client, project, window_minutes, fail_when) -> list[dict]`；`harvest(client, project, window_minutes, dataset_name) -> int`（返回写入条数）

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_harvest.py
"""把线上失败样本捞进人工分诊队列。

**不写进 bz-rag-golden**，写进独立的 bz-rag-harvested。两个原因：
1. 会被抹掉——门禁的真实数据源是本地 evaluation/test_dataset.json，
   Phoenix 上的 bz-rag-golden 只是它的镜像，而 dataset.py 用的
   create_dataset 是全量替换语义（"上传里没有的 example 会被删"）。
2. 没有 ground truth——线上 span 上没有标准答案，而 expected_response 是
   contextual_precision / contextual_recall 两个判官的必需输入。

所以回灌这条环由人闭合：人看过、补上标准答案，再手工加进 test_dataset.json。
"""

import json

from evaluation.phoenix.harvest import failed_spans


def _span(sid, q="问题", a="答案"):
    return {
        "id": f"node-{sid}",
        "name": "milvus-hybrid-rag",
        "span_kind": "AGENT",
        "parent_id": None,
        "attributes": {"input.value": q, "output.value": a},
        "context": {"trace_id": f"t-{sid}", "span_id": sid},
    }


class FakeSpans:
    def __init__(self, spans, annotations):
        self._spans = spans
        self._anns = annotations
        self.added = []

    def get_spans(self, **kw):
        if kw.get("span_kind") == "RERANKER":
            return []
        return list(self._spans)

    def get_span_annotations(self, **kw):
        return list(self._anns)


class FakeDatasets:
    def __init__(self):
        self.calls = []

    def add_examples_to_dataset(self, **kw):
        self.calls.append(kw)
        return type("D", (), {"name": kw.get("dataset"), "example_count": 1})()


class FakeClient:
    def __init__(self, spans, annotations):
        self.spans = FakeSpans(spans, annotations)
        self.datasets = FakeDatasets()


def _ann(span_id, name, label):
    return {"span_id": span_id, "name": name, "result": {"label": label, "score": 0.0}}


class TestFailedSelection:
    def test_picks_refused(self):
        c = FakeClient(
            [_span("a"), _span("b")],
            [_ann("a", "refusal_check", "refused"), _ann("b", "refusal_check", "ok")],
        )
        got = failed_spans(c, "bz-rag-canary", 180)
        assert [s["context"]["span_id"] for s in got] == ["a"]

    def test_picks_incorrect_faithfulness(self):
        c = FakeClient(
            [_span("a"), _span("b")],
            [
                _ann("a", "faithfulness", "correct"),
                _ann("b", "faithfulness", "incorrect"),
            ],
        )
        got = failed_spans(c, "bz-rag-canary", 180)
        assert [s["context"]["span_id"] for s in got] == ["b"]

    def test_span_with_no_annotation_is_not_harvested(self):
        """没评过不等于失败——不能把未评估的样本当成失败捞回来。"""
        c = FakeClient([_span("a")], [])
        assert failed_spans(c, "bz-rag-canary", 180) == []

    def test_all_good_yields_nothing(self):
        c = FakeClient([_span("a")], [_ann("a", "refusal_check", "ok")])
        assert failed_spans(c, "bz-rag-canary", 180) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_harvest.py -o addopts="" -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation.phoenix.harvest'`

- [ ] **Step 3: 写实现**

创建 `evaluation/phoenix/harvest.py`：

```python
"""把线上失败样本捞进人工分诊队列。

**写进独立的 bz-rag-harvested，不写 bz-rag-golden。** spec §4.10 原文说
"追加进 golden 集"，但查证后行不通，两个独立原因：

1. 会被下一次推送抹掉。门禁的真实数据源是本地 evaluation/test_dataset.json
   （cases.py 从它读），Phoenix 上的 bz-rag-golden 只是镜像；而 dataset.py 用的
   create_dataset(example_id_key=...) 是**全量替换**语义——"上传里没有的
   example 会被删"（该文件顶部注释原话）。回灌进去的样本下一次推送就没了。
2. 回灌样本没有 ground truth。expected_response 是 contextual_precision /
   contextual_recall 两个判官的必需输入，线上 span 上没有标准答案，
   混进金标集只会让这两个判官对它们全部 errored。

所以：harvest 只负责"捞出来 + 带上回跳链接"，人看过、补上标准答案之后，
手工加进 test_dataset.json，再走正常的 dataset.py 推送。
**回灌这条环由人闭合，不自动闭合**——与 promote 需人确认是同一条谨慎原则。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any

from evaluation.phoenix.online_worker import otel_span_id

PULL_LIMIT = 1000
HARVEST_DATASET = "bz-rag-harvested"

# 什么算"失败"。按 annotation 的 label 判，与 criteria.yaml 的 pass_when 同口径：
#   refusal_check   pass_when: label == 'ok'        -> 非 ok 即失败
#   faithfulness    pass_when: score >= 0.5         -> incorrect(0.0) 即失败
# answer_relevancy 不进回灌：它判的是"答得切不切题"，切题但答错的样本更值得捞，
# 而不切题往往是问题本身模糊，补进金标集意义不大。
FAIL_LABELS = {
    "refusal_check": {"refused", "empty"},
    "faithfulness": {"incorrect"},
}


def failed_spans(
    client: Any, project: str, window_minutes: int
) -> list[dict[str, Any]]:
    """返回窗口内被判失败的根 span。

    没有 annotation 的 span **不算失败**——没评过不等于评差了，
    把未评估样本当失败捞回来会污染分诊队列。
    """
    now = datetime.now(timezone.utc)
    spans = client.spans.get_spans(
        project_identifier=project,
        span_kind="AGENT",
        start_time=now - timedelta(minutes=window_minutes),
        end_time=now,
        limit=PULL_LIMIT,
    )
    roots = [s for s in spans if not s.get("parent_id")]
    if not roots:
        return []

    anns = client.spans.get_span_annotations(
        span_ids=[otel_span_id(s) for s in roots],
        project_identifier=project,
        limit=PULL_LIMIT,
    )
    bad: set[str] = set()
    for a in anns:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        sid = a.get("span_id") if isinstance(a, dict) else getattr(a, "span_id", None)
        result = a.get("result") if isinstance(a, dict) else getattr(a, "result", None)
        if not name or not sid or not isinstance(result, dict):
            continue
        if result.get("label") in FAIL_LABELS.get(name, set()):
            bad.add(str(sid))
    return [s for s in roots if otel_span_id(s) in bad]


def harvest(
    client: Any,
    project: str = "bz-rag-canary",
    window_minutes: int = 1440,
    dataset_name: str = HARVEST_DATASET,
) -> int:
    spans = failed_spans(client, project, window_minutes)
    if not spans:
        return 0

    examples = []
    for s in spans:
        attrs = s.get("attributes") or {}
        examples.append(
            {
                "question": str(attrs.get("input.value", "")),
                # 线上没有标准答案，留空由人补。写成空串而不是省略，
                # 是为了让分诊的人一眼看到"这里要填"。
                "expected_response": "",
                "observed_answer": str(attrs.get("output.value", "")),
                "source_span_id": otel_span_id(s),
                "origin": "harvested",
            }
        )

    client.datasets.add_examples_to_dataset(
        dataset=dataset_name,
        examples=examples,
        input_keys=["question"],
        output_keys=["expected_response"],
        metadata_keys=["observed_answer", "source_span_id", "origin"],
    )
    return len(examples)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="把线上失败样本捞进人工分诊队列")
    p.add_argument("--project", default="bz-rag-canary")
    p.add_argument("--window-minutes", type=int, default=1440)
    p.add_argument("--dataset", default=HARVEST_DATASET)
    args = p.parse_args(argv)

    from phoenix.client import Client

    client = Client(base_url=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006"))
    n = harvest(client, args.project, args.window_minutes, args.dataset)
    print(f"捞回 {n} 条失败样本 -> dataset {args.dataset!r}")
    if n:
        print("下一步（人工）：补上 expected_response，加进 evaluation/test_dataset.json，")
        print("               再跑 python -m evaluation.phoenix.dataset 推送。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_harvest.py -o addopts="" -q`
Expected: 4 passed

- [ ] **Step 5: 对真实 canary 跑一次**

Run: `NO_PROXY=localhost,127.0.0.1 python -m evaluation.phoenix.harvest --window-minutes 4320`

Expected: 打印捞回条数。当前 `bz-rag-canary` 里 `refusal_check` 全是 `ok`、
`faithfulness` 有一条 `partial`（不在 FAIL_LABELS 里），所以很可能是 **0 条**。
0 条是正确结果，不是 bug——说明影子流量上没有失败样本。
要制造一条来验证链路，可以临时把 `router.py` 的权重打到 100、
用期 1 演示时那个"完全忽略检索结果"的坏 prompt 跑几条，再 harvest。
**本 step 不强制制造失败样本**，验证 0 条路径也算通过。

- [ ] **Step 6: 提交**

```bash
git add evaluation/phoenix/harvest.py tests/test_harvest.py
git commit -m "feat(eval): harvest 把线上失败样本捞进人工分诊队列"
git push origin master
```

---

## Self-Review

**1. Spec coverage**

| spec 章节 | 覆盖它的 Task | 备注 |
|---|---|---|
| §4.9 monitor 三态（rollback/hold/promote） | Task 1 | hold 优先于 rollback，按 `samples < min_samples` 判而非匹配字符串 |
| §4.9 rollback 自动执行（权重置 0） | Task 2 + Task 3 | 影子侧可用；真云侧因 CF_API_TOKEN 未配置只能显式报错 |
| §4.9 promote 开 issue 等人点 | **部分覆盖** | Task 3 只打印判定、不自动开 issue。理由：开 issue 需要 `issues: write` 权限且会产生噪声；spec 的核心诉求"promote 要人确认"由 Task 4 的前置门实现——不达标直接拒绝晋级，比开 issue 更硬。**若确实要自动开 issue，需另开一个 task。** |
| §4.10 harvest 回灌 | Task 5 | **偏差**：写进独立的 `bz-rag-harvested` 而非 `bz-rag-golden`，理由见上方"与 spec 的两处偏差" |
| §5 `canary-watch.yml` 跑 monitor | Task 3 | |
| §5 `promote-stable.yml` 加前置 job | Task 4 | |
| §7 期 3 完成标志（故意变差→权重归 0→追到 span；改好→开 promote issue） | **部分覆盖** | 权重归零与追溯可验；"开 promote issue"未实现（见上）。完整演示还需要先把样本量打到 `min_samples: 20` 以上，当前 faithfulness 只有 2 条 |

**2. Placeholder scan** —— 无 TBD/TODO；每个代码步骤都是可直接粘贴的完整实现；测试都有真实断言。

**3. Type consistency** —— `Decision` 四字段在 Task 1 定义，Task 3 的 workflow 只消费其退出码（0/1/2，在 Task 1 里以 `EXIT_PROMOTE/EXIT_ROLLBACK/EXIT_HOLD` 命名）；`set_weight(weight, target, weight_path)` 在 Task 2 定义，Task 3 的 workflow 按 `rollback 0 --target shadow` 的 CLI 形式调用，与 `main()` 的 argparse 一致；`failed_spans(client, project, window_minutes)` 与 `harvest(client, project, window_minutes, dataset_name)` 在 Task 5 内自洽；`otel_span_id` 来自期 2 的 `online_worker`，签名未变。

**4. 一个执行前必须知道的现状**

当前 `bz-rag-canary` 只有 16 条 `refusal_check` + 2 条 `faithfulness`，而
`criteria.yaml` 的 `online` 段两条 criteria 都要求 `min_samples: 20`。
**所以真机跑 monitor 必然输出 hold，`rollback` 与 `promote` 两态只能靠单测的受控数据覆盖。**
这是正确行为（spec §4.9：样本量不足必须独立成态）。

要在真机上看到另外两态，需要先把样本打够：`refusal_check` 采样率 1.0，
再打 4 条以上 replay 流量即可到 20；`faithfulness` 采样率 0.2，
要到 20 条需要约 100 条流量（约 35 分钟）。**是否要打这波流量，执行到 Task 3 时再定。**
