# MiniMaxJudge → GLMJudge 重命名设计

**日期**：2026-05-29
**目标**：评测链路实际 judge LLM 已切到智谱 GLM-4-Flash，但 `evaluation/` 目录下类名、注释、Confident AI 显示字符串仍是 MiniMax 字样，对外有误导。本次纯 refactor 全部对齐 GLM。

## 范围

**改**：`evaluation/` 下 4 个文件里的 `MiniMaxJudge` 类名 + 字符串 + 注释。
**不改**：任何运行时行为、retry/throttle 逻辑、`_extract_json` 的 think-strip / 栈扫 / json_repair 三层兜底。

## 决策

- 类名：`MiniMaxJudge` → **`GLMJudge`**（贴合当前实际 judge 模型家族 GLM-4）
- Confident AI dashboard 显示串：`MiniMax(<MODEL_ID>)` → **`GLM(<MODEL_ID>)`**（例如 `GLM(glm-4-flash)`）
- 类不与具体供应商绑死，仍读 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_ID` 三件套，换 GLM 别的型号无需改代码

## 文件改动

### 1. `evaluation/deepeval_judge.py`

- 模块 docstring：`把 MiniMax 包装成 deepeval 可用的 Judge LLM。` → `把 OpenAI 兼容 LLM（默认智谱 GLM）包装成 deepeval 可用的 Judge LLM。`
- 类名：`MiniMaxJudge` → `GLMJudge`
- `get_model_name`：返回 `f"GLM({os.environ['MODEL_ID']})"`
- `_extract_json` 内注释 `# MiniMax M2.7 等 thinking 模型...` → `# 部分 thinking 模型...`（不再点名）

### 2. `evaluation/evaluate.py`

- import：`from evaluation.deepeval_judge import MiniMaxJudge` → `GLMJudge`
- `judge = MiniMaxJudge()` → `judge = GLMJudge()`
- timeout override 注释 `# 调高 DeepEval per-task 超时（默认 180s），给 MiniMax thinking 模型 + 重试 + 节流留余量。` → `# 调高 DeepEval per-task 超时（默认 180s），给 LLM judge + 重试 + 节流留余量。`

### 3. `evaluation/test_judge.py`

- `from evaluation.deepeval_judge import MiniMaxJudge` → `GLMJudge`
- 所有 `MiniMaxJudge` 实例化和静态方法引用 → `GLMJudge`
- `TestRetryOnRateLimit` 类的 docstring 与 `test_repairs_broken_json_with_unescaped_quotes` 注释里的 "MiniMax M2.7" 提及 → 改为 "thinking 模型" / 通用描述

### 4. `evaluation/test_build_dataset.py`

不引用 `MiniMaxJudge`，**不改**。Refactor 时用 grep 确认。

## 验证

1. `python -m pytest evaluation/ -v` —— 11/11 PASS（4 + 2 + 2 + 3 包含的所有原测试）
2. `python -c "from dotenv import load_dotenv; load_dotenv(); from evaluation.deepeval_judge import GLMJudge; print(GLMJudge().get_model_name())"` 输出 `GLM(glm-4-flash)`
3. 端到端跑一次 `python evaluation/evaluate.py`，Confident AI 新 test run 上 model 列显示 `GLM(glm-4-flash)`

## 提交

单 commit：`refactor(eval): MiniMaxJudge → GLMJudge 重命名`

## 风险

- 几乎为零。无逻辑变化，仅符号 + 字符串重命名。最坏情况 grep 漏一处，pytest 会立刻报 NameError。
