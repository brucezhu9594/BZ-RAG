# Phoenix 评估 CI/CD 设计（对标 Arize Phoenix / AX）

> 2026-09-08
>
> 目标：把 Arize 那套"评估闭环 CI/CD"在 BZ-RAG 上完整实现一遍，全程本地，
> OSS Phoenix 有的直接用，AX 独有的（常驻线上评估、采样、monitor、自动决策）自己造。

---

## 1. 为什么做，以及为什么是这个仓库

`docs/CD-pipeline.md` 第 10 节"当前遗留 / 可优化项"第 4 条写着：

> **观察期没有自动化告警**：promote 完全靠人主观判断"观察够了没"。

BZ-RAG 已经有一条完整的金丝雀流水线（canary-deploy → promote-stable → rollback-canary，
Cloudflare Worker + KV 按权重分流），缺的恰好就是 Arize 三段环里的第二、三段：
**线上评估**和**评估驱动的放行/回滚决策**。所以这不是凭空造一套 CI/CD，
而是把评估接进已有流水线，顺带把 Arize 的机制吃透。

### Arize 的三段环（已按官方文档核实）

| 段 | Arize 怎么做 | 谁提供 |
|---|---|---|
| 离线门禁（PR 时） | `arize-phoenix-client[pytest]` 插件：suite→dataset、case→example、断言→`pass` annotation，pytest 退出码即门禁；TS 侧另有声明式 `acceptanceCriteria` | OSS Phoenix |
| 线上评估 | evaluator **template** / eval **task** 两层分离；task 带 filter、`sampling_rate`、cadence；结果写回 span 的 `eval.<name>.*` | **仅 AX** |
| 决策与回灌 | monitor 聚合线上分数 → 告警；annotation 筛 span → 建 dataset 回灌 | monitor 仅 AX；回灌 OSS 有 |

OSS Phoenix **没有**：按计划常驻的 online eval task、采样率、monitors/告警、Signal、
managed agents 提 PR。这些本设计自己实现。

---

## 2. 设计决策

| 决策 | 选了 | 没选什么 | 理由 |
|---|---|---|---|
| AX 那半边 | 自己造（worker + monitor + criteria） | 注册 AX Free 账号 | 要求全本地；且"自己实现一遍"才真正吃透机制，成果能直接搬去 mira-eval |
| 与现有评估栈的关系 | 平行新建 `evaluation/phoenix/` | 迁移 MLflow / DeepEval | 不弄丢已有 baseline 历史；随时能拿同一批 case 在两套栈上对照 |
| CI 在哪跑 | 本机 self-hosted runner | 云端 ubuntu-latest | 真管线依赖 `localhost:19530` 的 Milvus，云 runner 碰不到 |
| 线上段的流量来源 | 本地影子 canary | 打真 Railway canary | Railway 上没有 Milvus，`/api/query` 的 chroma db 也没上传（CD-pipeline Q5 遗留）。真上云是另一个工程任务 |
| 判官 | `minimax/minimax-m3` @ Vercel AI Gateway | 沿用 `glm-4-flash` | 实测 1.3s / $0.0000591 一次调用，`response_format: json_object` 被尊重；网关有 4 个 fallback provider |
| 判官输出形态 | 分类标签 → 映射分数 | 让判官直接吐 0–1 float | Phoenix 官方明确建议分类优于数值（"reasoning about scales introduces additional variability"）；现有 MLflow 那套是 float，这里借机改 |
| promote 自动化程度 | 自动 rollback、半自动 promote | 全自动 promote | rollback 是收敛动作（错了只是回到老版本），promote 是扩散动作，机器判"够好了"的置信度不足以省掉人点一下 |

---

## 3. 系统形状

```
                  ┌──────────────── Phoenix (本地自托管, :6006, sqlite) ────────────────┐
                  │  project: bz-rag-ci            project: bz-rag-canary               │
                  │  dataset: bz-rag-golden        span annotations (eval.<name>.*)     │
                  └──────▲──────────────────────────────▲───────────────┬───────────────┘
                         │ trace + experiment run       │ trace         │ 拉 span / 写 annotation
   ┌─────────────────────┴─────┐        ┌───────────────┴──────┐  ┌─────┴───────────────────┐
   │ pytest 门禁套件（期 1）   │        │ 影子 canary（期 2）  │  │ online_worker（期 2）   │
   │  真跑 Milvus RAG 管线     │        │  双 uvicorn + 分流器 │  │  filter + 采样 + cadence│
   │  → acceptance → exit code │        │  replay 喂流量       │  │ monitor（期 3）→ 三态   │
   └─────────────────────┬─────┘        └──────────────────────┘  └─────┬───────────────────┘
                         │ exit code                                     │ promote / hold / rollback
        ┌────────────────┴──────────── self-hosted runner（本机）────────┴────────────────┐
        │ eval-gate.yml (PR/push)   canary-watch.yml (cron)   promote-stable.yml (加前置)  │
        └────────────────────────────────────────────────────────────────────────────────┘
```

一个 Phoenix 实例，两个 project：CI 的与线上的分开，否则线上噪声会污染 experiment 历史。
**判官只有一份**，离线与线上共用同一组 evaluator —— 这是 AX 的 template/task 分离带来的直接好处。

### 数据流（环是闭的）

```
PR  → runner → pytest 套件 → 真跑 RAG → trace 落 bz-rag-ci → 判官打分
    → annotation 落 experiment run → acceptance 聚合 → exit code → PR check

push master → 影子 canary 分流 5% → trace 落 bz-rag-canary
    → cron worker 按采样率抽样评估 → annotation 写回 span
    → monitor 用同一份 criteria 聚合 → rollback 自动执行 / promote 开 issue 等人点

线上失败 span → harvest 筛出 → 带 source_span_id 回灌 golden 集 → 下个 PR 的门禁变严
```

---

## 4. 组件设计

### 4.1 Phoenix server（本地）

- `pip install arize-phoenix`（20.8.0，`requires_python >=3.10,<3.15`，本仓库 3.10 兼容）
- `scripts/phoenix-up.ps1` 起服务，固化两件事：
  - `NO_PROXY` 必须包含 `localhost,127.0.0.1`（本仓库已因 Privoxy 拦 localhost 踩过三次坑，
    见 `api/main.py`、`evaluation/mlflow_evaluate.py` 顶部的兜底注释）
  - sqlite 后端文件落 `phoenix.db`，加进 `.gitignore`
- 端口 6006（Phoenix 默认），与 MLflow 的 5000、Milvus 的 19530 不冲突

### 4.2 埋点：`api/milvus_rag_phoenix.py`

照 `api/milvus_rag_mlflow.py` 的结构复制一份，只换承载：

```python
from phoenix.otel import register
tracer_provider = register(
    project_name=os.environ.get("PHOENIX_PROJECT_NAME", "bz-rag-ci"),
    auto_instrument=True,   # 自动挂载已安装的 openinference instrumentor
)
```

装 `openinference-instrumentation-langchain`（0.1.74），LangChain 的检索链**自动**产出
RETRIEVER span，不需要像 MLflow 那版手写 span。这是相对现有实现的净减法。

project_name 走环境变量：CI 里是 `bz-rag-ci`，影子 canary 上是 `bz-rag-canary`，同一份代码。

新增端点 `POST /api/milvus/query-phoenix`，与现有两个端点并列，签名一致（`query` + `thread_id`）。

### 4.3 golden 集：`evaluation/phoenix/dataset.py`

沿用 `evaluation/test_dataset.json` 那批 case，推进 Phoenix：

```python
client.datasets.create_dataset(
    name="bz-rag-golden",
    dataframe=df,
    input_keys=["query"],
    output_keys=["expected_response"],
    example_id_key="example_id",     # ← 关键
)
```

`example_id` 取 **query 规范化后的 sha256 前 16 位**。Arize 不白送内容寻址
（官方只说"你自己给稳定 ID，可以是主键或 content hash"），但造完就有了
"同一批 case"的可计算定义 —— 这正是 mira-eval 那边 `dataset_version` 不可信问题的解法。

diff 语义要在文档里写清，因为两个方法行为不同：

| 方法 | 语义 |
|---|---|
| `create_dataset(..., example_id_key=)` | 全量替换：上传里没有的 example **会被删** |
| `add_examples_to_dataset(..., example_id_key=)` | 增改不删：缺席的 example 保留 |

`dataset.py` 用前者（golden 集以文件为准），`harvest.py` 用后者（回灌只追加）。

### 4.4 判官：`evaluation/phoenix/evaluators.py`（template 层）

```python
from phoenix.evals.llm import LLM
judge = LLM(
    provider="openai",
    model=os.environ["JUDGE_MODEL_ID"],            # minimax/minimax-m3
    base_url=os.environ["JUDGE_OPENAI_BASE_URL"],  # https://ai-gateway.vercel.sh/v1
    api_key=os.environ["JUDGE_OPENAI_API_KEY"],
)
```

四个指标沿用 `evaluation/mlflow_evaluate.py` 里已经调过的中文 prompt：
`faithfulness` / `answer_relevancy` / `contextual_precision` / `contextual_recall`，
多轮另加 `followup_resolution`（复用 `evaluation/multiturn_driver.py`）。

另新增一个**护栏型** evaluator `refusal_check`：检测答案是否为无依据的拒答/空答
（label `ok | refused | empty`）。它的存在是为了演示"护栏型指标不采样"这条口径 ——
拒答是稀有失败，采样会把它采掉，所以 4.8 里它的 `sampling_rate` 固定 1.0。

**输出形态改为分类**：judge 返回 `{"label": "correct|partial|incorrect", "rationale": "中文"}`，
Python 侧映射成 `1.0 / 0.5 / 0.0`。temperature 固定 0。

evaluator 签名按 Phoenix 的**按参数名绑定**约定写（`output`、`input`、`expected`、
`metadata`、`trace_id`），并一律带 `**_` 兜住不消费的字段 —— pytest 插件与
`run_experiment` 走同一个适配器，这样两边都能直接用。

### 4.5 离线门禁：`evaluation/phoenix/test_rag_eval.py`

```python
@pytest.mark.phoenix(
    dataset="bz-rag-golden",
    evaluators=[faithfulness, answer_relevancy, contextual_precision, contextual_recall],
    repetitions=int(os.environ.get("EVAL_REPETITIONS", 1)),   # PR smoke 用 1，master 全量用 2
)
@pytest.mark.parametrize("query,expected", CASES, ids=CASE_IDS)   # 均由 dataset.py 的 loader 提供
def test_rag(query, expected):
    answer = milvus_rag_phoenix_query(query)
    log_output(answer)
```

要点：

- `ids=` 给每条 case 稳定身份，重跑映射回同一个 example，历次 run 累积成同一批固定 case 上的实验序列
- 插件自动把 git commit 写进 `experiment_metadata.git_sha`
- 挂在 marker 上的 evaluator 失败只降级成 warning；只有断言 / inline `evaluate()` 失败才挂测试
- 上传到 Phoenix 是 best-effort，网络问题只 warn，不挂 CI
- 判官抛异常时插件记 **errored annotation**（存 error、无 score），不丢这条
- 两档规模：PR 跑 `-m smoke` 子集，push master 跑全量

**并发陷阱**（Arize 官方明说）：两个 run 同时写同名 dataset 会互相 prune 例子。
CI 里一律设 `PHOENIX_TEST_DATASET=bz-rag-golden-${GITHUB_REF_NAME}`。

### 4.6 声明式验收条件：`evaluation/phoenix/acceptance.py` + `criteria.yaml`（自造）

Arize 的 `acceptanceCriteria` **只在 TypeScript 侧有**，Python 侧官方仍是"自己算 mean 然后
`sys.exit`"。这里把它移植成 Python pytest 插件（`conftest.py` 里注册，
`pytest_sessionfinish` 钩子聚合）。

```yaml
# criteria.yaml —— 离线门禁与线上 monitor 共用同一份
offline:
  - annotation: faithfulness
    metric: average
    threshold: 0.8
    direction: maximize          # 默认；minimize 用于延迟/成本
  - annotation: faithfulness
    metric: pass_rate
    pass_when: "score >= 0.5"
    min_pass_rate: 0.9
  - annotation: contextual_recall
    metric: pass_rate
    pass_when: "label != 'incorrect'"
    min_pass_rate: 1.0           # 硬底线
online:
  - annotation: faithfulness
    metric: pass_rate
    pass_when: "score >= 0.5"
    min_pass_rate: 0.85          # 线上比离线松一档
    min_samples: 20                # 低于此样本量判 hold，不算通过
```

必须一起实现的四条取舍（价值全在这里，不在功能本身）：

1. **跑完全部 case 再判**，一次看到所有回归，而不是第一个失败就中断
2. **缺失指标判失败**：`average` 一个数值都没有、或 `pass_rate` 的 annotation 从没被记过，
   报 `no <name> found` 并失败，绝不 vacuously pass
3. **判官报错记 errored**，与"判 0 分"和"通过"都区分开，是第三态
4. **落库失败只 warn**：观测层不该拖垮门禁

报告输出：一张 scoreboard（每条 criterion 的实测值、要过的线、样本数），
`average` 报均值，`pass_rate` 报通过比例。失败时抛一个聚合异常。

### 4.7 影子 canary：`evaluation/phoenix/shadow/`

Railway 上跑不了 Milvus 管线，所以线上段在本机复刻：

- `router.py`：50 行，复刻 `cf-worker/src/index.js` 的逻辑 —— 读权重文件
  （相当于 KV 的 `canary_weight`）、掷骰子、转发到 stable/canary 两个本地 uvicorn，
  响应头回写 `x-bz-backend`
- 两个实例用不同 `APP_VERSION` 和相同 `PHOENIX_PROJECT_NAME=bz-rag-canary` 起
- `replay.py`：从 golden 集**之外**的 query 池按节奏打流量（线上评估不能只测训练过的 case）

`scripts/cf-kv-update.sh` 保持原样不动；shadow 侧的权重改写走 `router.py` 自己的接口，
但**参数形状与 `cf-kv-update.sh` 一致**（0–100 整数），这样期 3 的 monitor 输出能同时
驱动两边，将来真上云不用改 monitor。

### 4.8 线上评估 worker：`online_worker.py` + `online_tasks.yaml`（自造）

复刻 AX 的 template / task 分离。template 就是 4.4 的 evaluator，task 是：

```yaml
# online_tasks.yaml
- name: canary-faithfulness
  project: bz-rag-canary
  evaluators: [faithfulness, answer_relevancy]
  query_filter: "span_kind == 'LLM'"
  sampling_rate: 0.2          # 唯一的成本旋钮
  cadence: continuous          # continuous | historical
  window_minutes: 30
- name: canary-guardrail
  project: bz-rag-canary
  evaluators: [refusal_check]
  sampling_rate: 1.0           # 护栏型不采样：稀有失败会被采样掉
  cadence: continuous
```

worker 循环：按 cadence 唤醒 → 按 filter + 时间窗拉 span → 按 `sampling_rate` 抽样 →
跑判官 → `client.spans.log_span_annotations_dataframe(annotation_name=..., annotator_kind="LLM")`
写回。去重靠"该 span 是否已有同名 annotation"，避免窗口重叠重复计费。

抄进来的两条 Arize 口径：

- **护栏型 evaluator 不采样**（稀有失败会被采样掉，采样就失去意义）；code evaluator 近似免费也不必采
- **新判官先跑 historical 批、验完区分度再切 continuous**，用来抓"prompt 看着合理但
  把所有 case 都判 correct"的判官

### 4.9 monitor 与决策：`monitor.py`（自造）

读 `bz-rag-canary` project 最近 N 小时的 annotation，用 `criteria.yaml` 的 `online` 段
（**和离线门禁同一套阈值语言、同一份实现**）聚合，输出三态：

| 输出 | 触发 | 动作 |
|---|---|---|
| `rollback` | 任一 criterion 未达标 | 自动执行：权重置 0（shadow 侧改权重文件；真云侧调 `cf-kv-update.sh 0`） |
| `hold` | 样本量不足（低于 `min_samples`） | 什么都不做，下个 cron 再看 |
| `promote` | 全部达标且样本量够 | 开一个 GitHub issue，附 scoreboard 和 Phoenix 链接，等人点 promote |

样本量不足必须是独立的第三态，不能算通过 —— 否则刚部署完没几条 trace 就会被判"全绿"。

### 4.10 回灌：`harvest.py`

按 annotation 筛线上失败 span → 生成 example，`metadata.source_span_id` 存回跳链接 →
`add_examples_to_dataset(..., example_id_key=...)` 追加进 golden 集（用 add 不用 create，
因为 create 是全量替换会删掉没上传的）。回灌进来的 case 打 `metadata.origin: harvested`
标记，便于日后区分人工挑的和线上捞的。

---

## 5. CI 装配

self-hosted runner 装在本机（Windows 服务）。三处改动：

| workflow | 触发 | 干什么 |
|---|---|---|
| `eval-gate.yml`（新） | PR / push master | 起/复用 Phoenix → 跑 pytest 门禁套件 → acceptance → exit code 当 PR check；PR 用 smoke 子集，master 全量 |
| `canary-watch.yml`（新） | cron | 跑 `online_worker` 一轮 → 跑 `monitor` → rollback 自动执行 / promote 开 issue |
| `promote-stable.yml`（改） | 手动（不变） | 加一个前置 job 调 `monitor`，不达标直接拒绝 promote |

Secrets 边界：judge key 走 runner 本机 `.env`（已 gitignore），**不进 GitHub Secrets**。
这台机器上的 runner 能读到 `.env` 是有意为之，spec 里显式记录这个边界。

---

## 6. 目录结构

```
api/milvus_rag_phoenix.py                 # Phoenix 埋点版管线
evaluation/phoenix/
├── conftest.py                           # acceptance 插件挂载
├── acceptance.py                         # 自造：声明式验收条件（Python 版）
├── criteria.yaml                         # 阈值声明（离线 + 线上共用）
├── dataset.py                            # golden 集推送（example_id_key = content hash）
├── evaluators.py                         # 判官 template 层（MiniMax-m3）
├── test_rag_eval.py                      # 单轮门禁套件
├── test_rag_multiturn.py                 # 多轮门禁套件
├── online_tasks.yaml                     # 自造：task 层
├── online_worker.py                      # 自造：常驻线上评估
├── monitor.py                            # 自造：阈值 monitor → 三态决策
├── harvest.py                            # 回灌
└── shadow/
    ├── router.py                         # 本地分流器（复刻 cf-worker 逻辑）
    └── replay.py                         # 流量重放
.github/workflows/eval-gate.yml
.github/workflows/canary-watch.yml
scripts/phoenix-up.ps1
```

---

## 7. 分期与验收标准

### 期 1 — 离线门禁

**做完的标志**：改坏一个 prompt 推 PR，`eval-gate` 变红且 PR check 里能看到是哪条
criterion 没过、实测值多少；改回来变绿。Phoenix UI 里能看到这两次 run 在同一批 example
上的对比。

### 期 2 — 自造线上评估

**做完的标志**：影子 canary 起来、replay 打流量后，Phoenix 的 `bz-rag-canary` project
里能看到 span 上挂着 `eval.faithfulness.*`；把 `sampling_rate` 从 1.0 调到 0.2，
被评的 span 数量按比例下降；护栏那条任务不受采样率影响。

### 期 3 — 闭环

**做完的标志**：故意让 canary 版本变差，cron 跑完后权重自动归 0（rollback），
且 Phoenix 上能追到是哪几条 span 触发的；改好后下一轮 cron 开出 promote issue。
`harvest` 能把线上失败 span 变成 golden 集里的新 case，且新 case 带得回原 span 链接。

---

## 8. 风险与已知边界

| 风险 | 处理 |
|---|---|
| 判官调用耗时 | 单次 1.3s。master 全量 = 10 case × 5 指标 × 2 repetitions = 100 次 ≈ 2 分钟串行；PR smoke = 3 case × 5 × 1 = 15 次 ≈ 20 秒。成本可忽略（全量 ≈ $0.006） |
| 并发 prune | `PHOENIX_TEST_DATASET=bz-rag-golden-${branch}`，官方明确的解法 |
| Privoxy 拦 localhost | `phoenix-up.ps1` 与所有入口固化 `NO_PROXY=localhost,127.0.0.1`；本仓库已踩过三次 |
| self-hosted runner 要机器开着 | 接受。cron 漏跑不影响正确性（monitor 是幂等的，下一轮重算） |
| 影子 canary ≠ 真线上 | 明确记录。真上云需先解 Milvus 上云或换 Milvus Lite，属另一个工程任务 |
| `minimax/minimax-m3` 经网关路由 | 实测落 fireworks，另有 minimax/nebius/gmicloud/morph 四个 fallback。判官供应商漂移会影响可比性，`experiment_metadata` 里记 `resolvedProvider` |

### 明确不做

- Agent-as-a-Judge（AX 闭源 Enterprise closed beta）
- Signal / managed agents 自动提 PR（Enterprise）
- agent experiments（要把 agent 暴露成 HTTP endpoint + W3C traceparent 串 trace，
  本项目形态不匹配）
- 迁移或改动现有 MLflow / DeepEval 评估路径

---

## 9. 参考

- [CI/CD with experiments (AX)](https://arize.com/docs/ax/improve/ci-cd-for-automated-experiments)
- [Eval CI with pytest (Phoenix)](https://arize.com/docs/phoenix/datasets-and-experiments/how-to-experiments/eval-ci-with-pytest)
- [CI Eval Tests / acceptanceCriteria (TS)](https://arize.com/docs/phoenix/sdk-api-reference/typescript/packages/phoenix-client/ci-evals)
- [Filters, Scope, and Cadence (AX)](https://arize.com/docs/ax/concepts/evaluators/filters-scope-and-cadence)
- [Online vs Offline Evaluators (AX)](https://arize.com/docs/ax/concepts/evaluators/online-vs-offline)
- [Updating Datasets / example_id_key](https://arize.com/docs/phoenix/datasets-and-experiments/how-to-datasets/updating-datasets)
- [Customize Your LLM Endpoint](https://arize.com/docs/phoenix/evaluation/tutorials/customize-your-llm-endpoint)
- [LangChain Tracing](https://arize.com/docs/phoenix/integrations/python/langchain/langchain-tracing)
- 本仓库 `docs/CD-pipeline.md`（金丝雀流水线现状与遗留项）
