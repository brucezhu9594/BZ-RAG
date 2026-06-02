# MiniMaxJudge → GLMJudge Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `evaluation/` 下所有 `MiniMaxJudge` 类名、`get_model_name` 返回的 `MiniMax(...)` 字符串、以及 MiniMax 字眼注释统一改成 `GLMJudge` 与 `GLM(...)`，对齐当前实际使用的智谱 GLM-4-Flash judge。

**Architecture:** 纯 refactor。原子改动 3 个文件（`deepeval_judge.py`、`evaluate.py`、`test_judge.py`），无逻辑变化。`test_build_dataset.py` 不引用该类，不动。

**Tech Stack:** Python, pytest。无新依赖。

**Spec:** `docs/superpowers/specs/2026-05-29-minimaxjudge-rename-design.md`

---

## File Structure

- **Modify**: `E:\wwwroot\BZ\BZ-RAG\evaluation\deepeval_judge.py` — 类名、docstring、`get_model_name`、内部注释
- **Modify**: `E:\wwwroot\BZ\BZ-RAG\evaluation\evaluate.py` — import、实例化、注释
- **Modify**: `E:\wwwroot\BZ\BZ-RAG\evaluation\test_judge.py` — import + 所有引用
- **Unchanged**: `evaluation/test_build_dataset.py`、`evaluation/build_dataset.py`、`evaluation/test_dataset.json`

---

## Task 1: 原子重命名 + 回归测试

**Files:**
- Modify: `E:\wwwroot\BZ\BZ-RAG\evaluation\deepeval_judge.py`
- Modify: `E:\wwwroot\BZ\BZ-RAG\evaluation\evaluate.py`
- Modify: `E:\wwwroot\BZ\BZ-RAG\evaluation\test_judge.py`

- [ ] **Step 1: 修改 `deepeval_judge.py`**

执行以下 4 处替换（**使用 Edit 工具，逐处 replace**，避免 replace_all 误伤其他位置）：

1. 模块 docstring（行 1）：
   - old: `"""把 MiniMax 包装成 deepeval 可用的 Judge LLM。"""`
   - new: `"""把 OpenAI 兼容 LLM（默认智谱 GLM）包装成 deepeval 可用的 Judge LLM。"""`

2. 类名（行 31）：
   - old: `class MiniMaxJudge(DeepEvalBaseLLM):`
   - new: `class GLMJudge(DeepEvalBaseLLM):`

3. `get_model_name` 返回值（行 62）：
   - old: `return f"MiniMax({os.environ['MODEL_ID']})"`
   - new: `return f"GLM({os.environ['MODEL_ID']})"`

4. 内部注释（行 66）：
   - old: `        # MiniMax M2.7 等 thinking 模型会在 JSON 前夹 <think>...</think>，先剥掉。`
   - new: `        # 部分 thinking 模型会在 JSON 前夹 <think>...</think>，先剥掉。`

- [ ] **Step 2: 修改 `evaluate.py`**

执行以下 4 处替换：

1. timeout 注释（行 10）：
   - old: `# 调高 DeepEval per-task 超时（默认 180s），给 MiniMax thinking 模型 + 重试 + 节流留余量。`
   - new: `# 调高 DeepEval per-task 超时（默认 180s），给 LLM judge + 重试 + 节流留余量。`

2. import（行 28）：
   - old: `from evaluation.deepeval_judge import MiniMaxJudge`
   - new: `from evaluation.deepeval_judge import GLMJudge`

3. 实例化（行 66）：
   - old: `    judge = MiniMaxJudge()`
   - new: `    judge = GLMJudge()`

4. 注释行 110：
   - old: `    # max_concurrency=1 avoids rate-limit (429) bursts on MiniMax API.`
   - new: `    # max_concurrency=1 avoids rate-limit (429) bursts on LLM API.`

5. 注释行 142：
   - old: `    # max_concurrent=1 + throttle_value=1.0 prevent concurrent LLM calls that trigger MiniMax 429s.`
   - new: `    # max_concurrent=1 + throttle_value=1.0 prevent concurrent LLM calls that trigger 429s.`

注意：上面虽然写了"4 处替换"，实际是 5 处，按列出的全做。

- [ ] **Step 3: 修改 `test_judge.py`**

执行以下替换：

1. 模块级 import（行 4）：
   - old: `from evaluation.deepeval_judge import MiniMaxJudge`
   - new: `from evaluation.deepeval_judge import GLMJudge`

2. 注释行 41（在 `test_repairs_broken_json_with_unescaped_quotes` 中）：
   - old: `        # MiniMax M2.7 实际产出过这种 reason 里嵌未转义引号的 broken JSON。`
   - new: `        # 部分 thinking 模型实际产出过这种 reason 里嵌未转义引号的 broken JSON。`

3. 函数体内 import（行 69）：
   - old: `        from evaluation.deepeval_judge import MiniMaxJudge`
   - new: `        from evaluation.deepeval_judge import GLMJudge`

4. 函数体内 import（行 106）：
   - old: `        from evaluation.deepeval_judge import MiniMaxJudge`
   - new: `        from evaluation.deepeval_judge import GLMJudge`

5. 所有 `MiniMaxJudge` 符号 → `GLMJudge`（在该文件内共 8 处出现的实例化与静态方法调用）。

由于该文件内 `MiniMaxJudge` 是 unique 符号，可以用一次 `replace_all=true` 的 Edit 把剩余 8 处一次性改完：
   - old: `MiniMaxJudge`
   - new: `GLMJudge`
   - replace_all: true

执行完毕后该文件内不应再有 `MiniMaxJudge` 字样。

- [ ] **Step 4: 检查没有遗漏**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
python -c "import subprocess; r = subprocess.run(['rg', '-n', 'MiniMax', 'evaluation/'], capture_output=True, text=True); print('STDOUT:', r.stdout); print('STDERR:', r.stderr); print('RC:', r.returncode)"
```

或者直接 grep（Bash 工具或 PowerShell）：
```
rg -n MiniMax evaluation/
```

Expected: 没有任何输出（exit 1），整个 `evaluation/` 下 MiniMax 字眼全部清除。
若仍有：用 Edit 工具补改剩余位置，再次 grep 直至 clean。

- [ ] **Step 5: 跑全量测试**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
python -m pytest evaluation/ -v
```

Expected: **11 PASSED**（4 `_extract_json` 基础 + 2 think/balanced + 1 json_repair + 2 retry + 3 build_dataset = 12 — 实际看运行后真实数量，应该全部 PASS，无 ImportError 无 NameError）。

如有 `ImportError: cannot import name 'MiniMaxJudge'` → 说明哪个引用没替换；按错误信息修补。
如有 `NameError: GLMJudge is not defined` → 说明 import 还指向 MiniMaxJudge。

- [ ] **Step 6: 验证 display name**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
python -c "from dotenv import load_dotenv; load_dotenv(); from evaluation.deepeval_judge import GLMJudge; print(GLMJudge().get_model_name())"
```

Expected: `GLM(glm-4-flash)`
（前提：`.env` 里 `MODEL_ID=glm-4-flash`、`OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4` 已配置。如果 MODEL_ID 是别的值，输出格式仍是 `GLM(<模型名>)`。）

- [ ] **Step 7: Commit**

```bash
git add evaluation/deepeval_judge.py evaluation/evaluate.py evaluation/test_judge.py
git commit -m "refactor(eval): MiniMaxJudge → GLMJudge 重命名"
```

---

## Task 2: 端到端冒烟（可选）

> 验证 Confident AI dashboard 上 model 列显示 `GLM(glm-4-flash)`。前置：`.env` 完整、Milvus 在线、`bz-rag-eval` dataset 已 sync。这套以前都通过过。

**Files:** 不修改文件

- [ ] **Step 1: 跑评测**

Run: `PYTHONIOENCODING=utf-8 python evaluation/evaluate.py`

Expected:
- 15 条 `pipeline:` 进度
- DeepEval 4 个 metric 全跑完，stdout 出 aggregate 表
- 末尾 `Done 🎉! View results on https://app.confident-ai.com/project/.../test-runs/...`

如失败：与 Task 1 的代码改动无关（功能已经通过 11 次评测验证），多半是外部状态（Milvus 没起、Confident key 过期）。

- [ ] **Step 2: 浏览器查看 Confident AI 新 test run**

打开 stdout 给的 URL，找 Models / Hyperparameters 区域，确认 model 显示是 `GLM(glm-4-flash)` 而非 `MiniMax(...)`。

- [ ] **Step 3: 无文件改动，跳过 commit**

---

## Task 3: 提交 plan 到 git

**Files:**
- 已存在: `docs/superpowers/plans/2026-05-29-minimaxjudge-rename.md`

- [ ] **Step 1: 强加 plan（`/docs` 在 .gitignore）**

Run from `E:\wwwroot\BZ\BZ-RAG`:
```
git add -f docs/superpowers/plans/2026-05-29-minimaxjudge-rename.md
```

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: MiniMaxJudge 重命名 plan"
```

- [ ] **Step 3: 确认 working tree clean**

Run: `git status`
Expected: `nothing to commit, working tree clean`（或仅有 .gitignore 之类预存 modified）。

---

## 完成判据

- `evaluation/` 下 `rg MiniMax` 无任何结果
- `python -m pytest evaluation/ -v` 全 PASS
- `GLMJudge().get_model_name()` 输出 `GLM(<MODEL_ID>)`
- Confident AI 新 test run 显示 `GLM(glm-4-flash)`（可选验证）
