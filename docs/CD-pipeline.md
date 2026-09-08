# BZ-RAG 金丝雀 CD 流水线 · 完全指南

> 写给：第一次接触这个项目的人，以及未来某天回来需要复习一下的我自己。
>
> 假设读者只懂 git 和基本 HTTP，不懂 CI/CD、不懂 Cloudflare、不懂 Railway。

---

## 1. 这套东西解决什么问题？

### 痛点

写完代码想发到线上，最朴素的方法是 SSH 上服务器、`git pull`、重启服务。这种办法的问题：

1. **谁有权限发**？人手一份服务器密码 = 灾难
2. **新版本一上线就 100% 流量打过去**，万一有 bug，所有用户立刻被波及
3. **回滚靠人记得上一版本号**，凌晨三点 oncall 想回滚得现登服务器、找 commit、重建
4. **没有版本号语义**，"明天上线 V0.3" 这种全凭口头协议

### 解决方案：金丝雀部署 + 自动化流水线

借用矿井里的"金丝雀报警"比喻：把新版本先放到一个**小服务**（canary）上接收**少量流量**（5%），观察一段时间没事再扩大；有事就立刻把流量切回旧版本（stable）。整个过程**用机器代替人执行**。

具体到这个项目：

- **每次推 `feat:` 或 `fix:` commit 到 master**，自动发布一个新版本号 → 自动部署到 canary → 自动放 5% 流量
- **观察一段时间觉得没问题**，手动点一个按钮，触发 promote → stable 也升级到这个版本，流量切回 stable
- **觉得有问题**，手动点 rollback → 5% 流量瞬间收回 stable

---

## 2. 整体架构（一张图）

```
                     ┌─────────────────────────────────────────────────┐
                     │  外部用户 / API 调用方                          │
                     └─────────────────────────┬───────────────────────┘
                                               │ HTTPS + X-Edge-Auth 头
                                               ▼
                  ┌────────────────────────────────────────────────────┐
                  │  Cloudflare Worker（公网入口）                     │
                  │  bz-rag-router.brucezhu9594.workers.dev            │
                  │  ┌──────────────────────────────────────────────┐  │
                  │  │ 1. 验 X-Edge-Auth 头，错就返 401             │  │
                  │  │ 2. 读 KV: canary_weight (0~100)              │  │
                  │  │ 3. Math.random() < weight → canary, 否则 stable │  │
                  │  │ 4. 转发请求到选中的 Railway service          │  │
                  │  └──────────────────────────────────────────────┘  │
                  └──────────────┬─────────────────────┬───────────────┘
                                 │                     │
                       weight 概率 │                     │ (1-weight) 概率
                                 ▼                     ▼
                ┌────────────────────────┐    ┌────────────────────────┐
                │ Railway: bz-rag-canary │    │ Railway: bz-rag-stable │
                │ (新版本试跑)           │    │ (上一稳定版本)         │
                │ FastAPI: /api/health   │    │ FastAPI: /api/health   │
                │          /api/query    │    │          /api/query    │
                └────────────────────────┘    └────────────────────────┘
                          ▲                              ▲
                          │ railway up                   │ railway up
                          │ (CD 触发部署)                │ (CD 触发部署)
                          │                              │
                          ├──────────────────────────────┘
                          │
                  ┌───────┴────────────────────────────────────────────┐
                  │  GitHub Actions（CD 流水线大脑）                   │
                  │  ┌──────────────────────────────────────────────┐  │
                  │  │ canary-deploy.yml  — push master 时自动跑    │  │
                  │  │   ├─ semantic-release 决定版本号             │  │
                  │  │   ├─ railway up 部署到 canary                │  │
                  │  │   └─ 写 KV canary_weight=5                   │  │
                  │  ├──────────────────────────────────────────────┤  │
                  │  │ promote-stable.yml  — 手动触发，输 version   │  │
                  │  │   ├─ KV=100（流量先全切 canary）             │  │
                  │  │   ├─ railway up 部署到 stable                │  │
                  │  │   └─ KV=0（流量切回 stable）                 │  │
                  │  ├──────────────────────────────────────────────┤  │
                  │  │ rollback-canary.yml — 手动触发，无参         │  │
                  │  │   └─ KV=0（5% 流量立刻收回 stable）          │  │
                  │  └──────────────────────────────────────────────┘  │
                  └────────────────────────────────────────────────────┘
                                          ▲
                                          │ git push (开发者)
                                          │
                                  ┌───────┴────────┐
                                  │ master 分支    │
                                  │ commit 例如：  │
                                  │ feat: 加 X 功能│
                                  │ fix:  修 Y bug │
                                  └────────────────┘
```

---

## 3. 一次发版的完整流程（最常用、最重要）

假设你刚写完一个新功能"问答缓存"，想发到线上。

### Step 1：本地写代码 + commit

```bash
# 在 master 分支上
git add .
git commit -m "feat: 加问答结果缓存，重复问题命中缓存直接返回"
git push origin master
```

**关键：commit 信息要带前缀**。因为 semantic-release 看前缀决定要不要发版本：

| 前缀 | 触发版本号变化 | 例 |
|---|---|---|
| `feat:` | 加 minor: 1.2.0 → 1.3.0 | 加新功能 |
| `fix:` | 加 patch: 1.2.0 → 1.2.1 | 修 bug |
| `feat!:` 或 commit body 有 `BREAKING CHANGE:` | 加 major: 1.2.0 → 2.0.0 | 不兼容更新 |
| `chore:` / `docs:` / `refactor:` / `test:` / `style:` / `ci:` / `build:` | **不发版本** | 杂活 |

### Step 2：等 GitHub Actions 自动跑（5-10 分钟）

打开 https://github.com/brucezhu9594/BZ-RAG/actions 看：

1. **Lint / Test / Build / Security** 四个 workflow 跑过 → 代码基本健康
2. **Canary Deploy** workflow 启动：
   - Job 1 `release`: semantic-release 看到 `feat:`，决定切 v1.3.0，自动写 CHANGELOG.md，打 git tag `v1.3.0`，建 GitHub Release
   - Job 2 `deploy-canary`:
     - 拉 `v1.3.0` tag 的代码
     - `railway up` 把它部署到 `bz-rag-canary` service（4-8 分钟）
     - `wait-for-health.sh` 轮询 canary 的 `/api/health`，等到返回的 version 字段包含 `1.3.0` 才继续
     - `cf-kv-update.sh 5` 调 Cloudflare API，把 KV 里 `canary_weight` 改成 5

3. **此时状态**：
   - canary 跑 v1.3.0
   - stable 还跑老版本（比如 v1.2.0）
   - 5% 用户被随机分到 canary，95% 被分到 stable
   - 你能在 Cloudflare Worker 响应头 `x-bz-backend` 看到这次请求落到了 stable 还是 canary

### Step 3：观察一段时间（你的判断时间）

具体观察什么取决于你的业务和工具：
- Railway 各 service 的日志有没有报错
- 业务监控（你接 Grafana / Sentry / 自建的话）有没有异常上扬
- 自己用浏览器调几次，多走几次看 5% 那一边表现

可以观察 1 小时、可以观察 1 天。**没人逼你立刻 promote**，金丝雀的本意就是给人留观察窗口。

### Step 4a：观察觉得 OK，promote 到 stable

打开 https://github.com/brucezhu9594/BZ-RAG/actions → 左边选 **`Promote Stable`** → 右上角 **`Run workflow`** → version 输 **`1.3.0`** → 点绿按钮。

`promote-stable.yml` 会：
1. 把 KV 改 `canary_weight=100`（5% → 100%，把流量全部切到 canary）
2. `railway up` 把 v1.3.0 部署到 stable service（4-8 分钟，stable 这时候没流量进来）
3. 等 stable 的 `/api/health` 返回 v1.3.0
4. 把 KV 改 `canary_weight=0`（100% → 0%，流量回到 stable，stable 现在已经是新版本了）

之后 canary 是闲置状态（仍然是 v1.3.0），等下次有新 commit 才会被覆盖。

### Step 4b：观察觉得不行，rollback

打开 Actions → 选 **`Rollback Canary`** → **`Run workflow`** → 直接点绿按钮（没有参数）。

`rollback-canary.yml` 一步完事：把 KV 改 `canary_weight=0`，5% 流量瞬间收回 stable。stable 还是老版本，用户感受不到 bug。

之后你回去修 bug、再 commit 一个 `fix:`，整个流程重来一遍。

### 流程图速览

```
[push feat:] → [自动: 切版本 v1.3.0] → [自动: 部署 canary] → [自动: KV=5]
                                                                    │
                                          ┌─────────────────────────┤ (你观察)
                                          │                         │
                                          ▼                         ▼
                              [手动: Promote Stable]      [手动: Rollback Canary]
                              [自动: KV=100]              [自动: KV=0]
                              [自动: 部署 stable]
                              [自动: KV=0]
```

---

## 4. 四块组件详解

### 4.1 Railway（跑应用的地方）

[Railway](https://railway.com) 是个 PaaS，类似 Heroku。我们在它上面建了**一个项目，两个 service**：

- **`bz-rag-stable`**: 稳定版的 service，对外公开 URL `https://bz-rag-stable-production.up.railway.app`
- **`bz-rag-canary`**: 金丝雀 service，对外公开 URL `https://bz-rag-canary-production.up.railway.app`

两个 service 都关闭了 **Auto Deploy**（默认 Railway 监听 GitHub push 自动部署，被我们关掉），改由 GitHub Actions 用 `railway` CLI 主动触发。

**为什么两个 service 而不是一个**？因为金丝雀的本质就是"同时存在新旧两个版本"。一个 service 没法同时跑两个版本。

每个 service 跑的是同一份代码（FastAPI 在 `api/main.py`），区别只在环境变量 `APP_VERSION`：
- stable 的 APP_VERSION = 当前生产版本（如 v1.2.0）
- canary 的 APP_VERSION = 待验证版本（如 v1.3.0）

`/api/health` 把 APP_VERSION 返回出来，CD 流水线就靠这个判断"部署到位没"。

### 4.2 Cloudflare Worker（流量分流器）

代码在仓库 `cf-worker/src/index.js`，逻辑就 50 行，做四件事：

```
1. 检查请求头 X-Edge-Auth，跟环境变量 EDGE_AUTH_TOKEN 比，错了 401
2. 读 KV: env.BZ_RAG_CANARY.get('canary_weight') → 整数 0-100
3. 掷骰子: Math.random() * 100 < weight → 去 canary，否则 stable
4. 把请求转发到选中的 Railway URL，把响应原样返回，加两个调试头：
   x-bz-backend: stable | canary
   x-bz-canary-weight: <当前 KV 值>
```

**为什么用 Worker + KV 而不是 Cloudflare 自家的 Load Balancer**？因为 LB 要付费 + 要域名，本项目都没有。Worker 是免费的，KV 也是免费的，效果一样。

**为什么需要 X-Edge-Auth**？Worker URL 是公网可访问的（`*.workers.dev`），不加锁的话，任何人调用都会消耗你的 LLM API 配额（OPENAI_API_KEY、ZHIPUAI_API_KEY），账单一夜爆掉。

### 4.3 GitHub App（自动化执行身份）

`bz-rag-release-bot` 是为这个项目专门建的 GitHub App，APP_ID = 3627908。

**为什么需要 App 而不是用 PAT**：

- semantic-release 跑在 GitHub Actions 里，需要往 master 分支 **push** 一个 `chore(release): X.Y.Z` 的 commit（更新 CHANGELOG）和一个 git tag
- 默认的 `GITHUB_TOKEN` 推不动有分支保护的 master
- 用 PAT 也行，但 PAT 跟个人账号绑定，离职 / 改密码 / 重置 token 时全废
- GitHub App 是**服务级身份**，独立存在，权限可以收得很窄

权限只开了 4 个：Contents (R/W), Issues (R/W), Pull Requests (R/W), Metadata (R)。

App 私钥（`.pem` 文件）的内容存在 GitHub Secret `APP_PRIVATE_KEY`，CI 跑的时候用 `actions/create-github-app-token@v2` 这个 action 拿 App ID + 私钥换成短期 access token，传给 semantic-release 用。

### 4.4 GitHub Actions（CD 流水线大脑）

五个核心 workflow（除原有的 lint/test/build/security 之外）：

| 文件 | 触发方式 | 干啥 |
|---|---|---|
| `.github/workflows/canary-deploy.yml` | push 到 master 自动 | 切版本 + 部署 canary + KV=5 |
| `.github/workflows/promote-stable.yml` | 手动 + 输入 version | KV=100 → 部署 stable → KV=0 |
| `.github/workflows/rollback-canary.yml` | 手动 | KV=0 |
| `.github/workflows/eval-gate.yml` | PR / push master（self-hosted runner，注册步骤见 4.5） | 跑 Phoenix 离线评估门禁，聚合阈值不达标则挡下 |
| `.github/workflows/<lint/test/build/security>.yml` | push / PR | 代码质量门禁 |

**`eval-gate.yml` 几个不写清楚会踩坑的细节**（期 1 才接进来）：

- **必须带 `--import-mode=importlib`**：`evaluation/phoenix/` 这个目录名和已安装的 `phoenix`
  这个 PyPI 包重名，pytest 默认的 prepend import 模式会在**加载 conftest 阶段**就
  `ModuleNotFoundError: No module named 'phoenix.conftest'` 崩掉——比"测试跑起来但结果不对"
  更隐蔽，因为收集阶段就崩了，一条测试都没跑，报错信息也完全联想不到"目录名撞包名"这个真因。
  日后要是想脱离 workflow、在本地手动跑门禁命令，务必带上这个 flag，否则会重现同一个报错。
- **门禁的专属退出码是 6，不是 3**：`evaluation/phoenix/conftest.py` 的 `pytest_sessionfinish`
  在聚合阈值不达标时会把 `session.exitstatus` 改写掉。没选 3 是因为 pytest 自己保留了
  0-5 做内置退出码（0=OK、1=测试失败、2=被中断、3=`INTERNAL_ERROR`、4=用法错误、
  5=没收集到测试），3 正好是 `INTERNAL_ERROR`——如果下游 CI 有"exit 3 = 基础设施抖动，
  自动重跑"这类特判，会把一次真实的 acceptance 失败误判成 infra 问题而重试掉。所以改用
  6，落在 pytest 保留区间之外，专属这条门禁"管线跑完了，但至少一条 acceptance criterion
  没达标"的含义。看到 `eval-gate` 以 6 收尾，去 Actions 日志里找 `Acceptance Criteria`
  那张表定位是哪条 criterion、实测值多少。
- **判官凭证不在这个 workflow 里，也没有对应的 GitHub Secrets**：`JUDGE_OPENAI_API_KEY` /
  `JUDGE_OPENAI_BASE_URL` / `JUDGE_MODEL_ID` 走 runner 本机仓库根目录下的 `.env`
  （已 `.gitignore`），`evaluation/phoenix/evaluators.py` 顶部的 `load_dotenv()` 直接读到，
  不经过网络。这是有意的边界：runner 就是跑 Milvus/Phoenix 的这台开发机，没必要把凭证
  再绕道 Secrets 走一遍。代价是这台机器的 `.env` 一旦被清空或者 runner 迁移到别的机器，
  门禁会在 pytest **收集阶段**就报可读的 `RuntimeError`（写清缺哪个变量），这是设计好的
  快速失败，不是 workflow 的 bug。
- **不用 `actions/setup-python`**：其余 workflow（lint/test/build/security）用
  `actions/setup-python@v5` 钉 Python 3.10；`eval-gate.yml` 不用，直接吃 self-hosted
  runner 机器上已经装好的解释器（实测 `C:\Python314\python.exe`，3.14.3）。原因两个：
  这台 runner 未必有权限/网络去联网下载额外的 Python 版本；而且这条门禁本来就要跑在
  "装了 Milvus/Phoenix 的那台机器"上，直接用机器自带解释器比额外装一份更贴近真实场景。
  `.python-version` 里记的 3.10 仍然只服务云 runner 那几个 workflow，两者互不冲突，
  不需要因为这条门禁去改它。
- **runner 服务必须以装了依赖的那个 Windows 账户运行**：实测发现 `phoenix` / `pytest` /
  `pandas` / `dotenv` 等一大票依赖，实际只装在了 `ci24871` 这个用户的 per-user
  site-packages（`C:\Users\ci24871\AppData\Roaming\Python\Python314\site-packages`，
  665 个条目），而不是 `C:\Python314\Lib\site-packages`（只有 2 个条目）。Windows 上
  per-user site-packages 是否可见，取决于**运行进程的账户**，不取决于调用的是哪个
  `python.exe`。self-hosted runner 装成 Windows 服务时**默认不以交互式用户身份跑**，
  如果注册时没指定账户，`eval-gate.yml` 第一次真跑大概率在 `pip install` 之后的
  `pytest evaluation/phoenix ...` 这步炸 `ModuleNotFoundError`（甚至 `pytest` 命令本身
  在那个账户的 PATH 上都可能找不到），**症状和"忘了 pip install"完全一样，容易把排障
  方向带偏**。解决办法二选一：① 注册 runner 服务时指定 `--windowslogonaccount`，让服务
  以 `ci24871` 这个账户运行（这台机器现在的做法）；② 更彻底的替代方案是把依赖装到系统级
  `site-packages`，或者建一个固定路径的虚拟环境、workflow 里用它的绝对路径调用
  `python`/`pip`——这样门禁就不再依赖"runner 服务恰好以哪个账户运行"这个隐藏前提。
  这个前提如果不对，`Install dependencies` 那一步也不再是"廉价空操作"：它会从零重装
  torch/transformers/langchain/mlflow 等一整套依赖，可能与 30 分钟的 `timeout-minutes`
  抢时间。**账户名要写成 `CAREERINTLINC\ci24871`，不能写 `.\ci24871`**：这台机器加入了
  `careerintlinc.local` 域（`whoami` 输出 `careerintlinc\ci24871`，`Get-LocalUser -Name
  ci24871` 查不到本地同名账户），`ci24871` 是**域账户**而不是本地账户，而 `.\` 前缀在
  Windows 账户解析里明确限定"只查本机"、不会回落去查域——用 `.\ci24871` 会在
  `config.cmd` 注册这一步就直接账户解析失败。日后如果这台机器移出域、或改用本地账户
  跑 runner，写法要相应改回 `.\<账户名>`。
- **健康检查打的是 `/readyz` 而不是根路径 `/`**：Phoenix 前端是个 SPA，根路径乃至任意
  不存在的路径都会被前端的 catch-all 路由兜成 200，所以打根路径只能证明"6006 端口上有
  个 HTTP server 在听"，对"进程活着但连不上数据库、记不了 trace"这个真实失败模式
  （`phoenix.otel.register()` 不做连通性检查，这种情况下端点照样 200）零覆盖。`/readyz`
  是 Phoenix 自己暴露的就绪探针，服务端内部真的会执行一次 `select 1` 探数据库连通性，
  能覆盖到这个失败模式。
- **全量规模（n=48：24 条 case × 2 次 repetitions）在开发阶段没有被完整跑完过**（只验证到
  17/48 就手动停了），所以**第一次 push 到 master 触发 `eval-gate` 跑全量，实际上就是
  这条路径的首次真实验证**。预计要跑 240 次判官调用 + 48 次完整 RAG 管线，10-20 分钟，
  首次跑建议盯着 Actions 日志看到底，别把中途卡住当成"还在跑"。（这条提示是一次性的——
  首次全量跑验证通过、确认过程稳定之后，可以把这条从文档里删掉。）
- **已知局限：源分支名带斜杠时，`PHOENIX_TEST_DATASET` 也会带斜杠**。`PHOENIX_TEST_DATASET:
  bz-rag-golden-${{ github.head_ref || github.ref_name }}` 解决的是"PR 事件下
  `ref_name` 不是分支名"这一半问题；本仓库分支命名习惯本身带斜杠（如
  `feat/sourcing-component-eval`），所以 PR 触发时 dataset 名会形如
  `bz-rag-golden-feat/sourcing-component-eval`，字面带斜杠。这是**有意不修的观感问题，
  不是功能缺陷**：dataset 名走 HTTP 的 query 参数/JSON body 而非 URL path 段，Phoenix
  服务端没有对它做字符集校验，斜杠不会导致路由错乱或建错资源，纯粹是在 Phoenix UI 里看
  着别扭。没修的原因是 GitHub Actions 的表达式语法没有 `replace()` 函数，真要 sanitize
  得在 workflow 里专门加一步写 `$GITHUB_ENV` 的 shell 脚本，为一个纯观感问题给 workflow
  添复杂度不划算。日后如果真觉得碍眼，配方是加一步：
  `echo "PHOENIX_TEST_DATASET=bz-rag-golden-${BRANCH//\//-}" >> $GITHUB_ENV`
  （`BRANCH` 取 `github.head_ref || github.ref_name` 的值，`${VAR//\//-}` 是 bash 的
  批量字符替换，把所有 `/` 换成 `-`）。

### 4.5 评估门禁的一次性安装：注册 self-hosted runner

`eval-gate.yml`（见 4.4）依赖本机 Milvus 与本机 Phoenix，云端 runner 碰不到，所以它**不跑在
GitHub 托管的 runner 上**，得先在这台开发机上手动注册一个 self-hosted runner。这是**一次性
操作**——注册好装成服务之后，以后每次 PR / push master 都会自动被派到这台机器上跑，不需要
重复本节步骤。

#### 前提：账户必须是装了依赖的那个 Windows 账户

实测发现 `phoenix` / `pytest` / `pandas` / `python-dotenv` 等一大票依赖只装在了 `ci24871`
这个账户的 per-user site-packages 里，系统级的 `site-packages` 只有 2 个条目。runner
装成 Windows 服务后**默认不以交互式用户身份运行**，账户不对的话，门禁第一次真跑会在
`pytest evaluation/phoenix ...` 这步报 `ModuleNotFoundError`——**症状和"忘了 pip
install"一模一样**，容易把排障方向带偏（原理见 4.4 对应条目）。

而且这个账户名不能随手写成 `.\ci24871`：这台机器加入了 `careerintlinc.local` 域，`ci24871`
是**域账户**，没有同名本地账户，`.\` 前缀只查本机、不会回落去查域，写 `.\ci24871` 会让下面
第 2 步的 `config.cmd` 在注册阶段就直接账户解析失败。正确写法是 `CAREERINTLINC\ci24871`。

#### 注册步骤

1. 打开 `https://github.com/brucezhu9594/BZ-RAG/settings/actions/runners/new`，平台选
   **Windows**，架构按机器实际情况选（多数是 x64）。页面会给一段专属临时 token，**必须
   现取现用**——有效期短，过期了就回这个页面重新拿一个新的。
2. 按页面给的命令下载、解压 runner 包之后，在 runner 目录下执行（把
   `<页面给的 token>` 换成第 1 步页面上的真实 token）：

   ```powershell
   ./config.cmd --url https://github.com/brucezhu9594/BZ-RAG --token <页面给的 token> --labels bz-rag-local --runasservice --windowslogonaccount "CAREERINTLINC\ci24871"
   ```

   - `--labels bz-rag-local` 必须原样带上——`eval-gate.yml` 里 `runs-on: [self-hosted,
     bz-rag-local]` 靠这个标签才能把任务派到这台机器，标签打错或漏打，job 会一直排队
     找不到能跑的 runner。
   - `--windowslogonaccount "CAREERINTLINC\ci24871"` 让 runner 服务以装了依赖的这个
     账户运行（就是本机日常登录、跑 `pip install -r requirements.txt` 用的那个账户）。
   - **不要在命令里额外拼一个 `--windowslogonpassword <密码>`**：`config.cmd` 检测到
     指定了 `--windowslogonaccount` 但没给密码时，会**交互式**提示你输入，输入过程
     不回显、也不进 PowerShell 历史记录，比明文写进命令行安全。如果你的场景必须
     非交互执行、不得不用 `--windowslogonpassword`，要清楚这个值会明文出现在
     PowerShell 历史（`Get-History`）里，用完记得清理。
   - `--runasservice` 把 runner 装成 Windows 服务（开机自启、不需要留一个终端窗口）。
     装服务这步如果提示需要管理员权限，换一个管理员 PowerShell 重新执行。
3. 装完后确认服务在跑、且**运行账户对**：

   ```powershell
   Get-Service actions.runner.*
   Get-CimInstance Win32_Service -Filter "Name LIKE 'actions.runner%'" | Select-Object Name, StartName, State
   ```

   第一条的 `Status` 应为 `Running`；第二条的 `StartName` 必须是 `CAREERINTLINC\ci24871`
   （或等价显示为 `ci24871`）。如果显示成 `LocalSystem` / `NT AUTHORITY\...` 之类的
   内建账户，说明 `--windowslogonaccount` 没生效，按下面"重做"小节卸载后重新走一遍
   第 2 步。
4. 打开 `https://github.com/brucezhu9594/BZ-RAG/settings/actions/runners`，确认能看到
   一个标签为 `bz-rag-local` 的 runner，状态显示 **Idle**（不是 Offline）——这才算
   注册成功。

#### 需要重做时：先卸载服务，别直接 `config.cmd remove`

一个已经装成 Windows 服务的 runner，直接 `config.cmd remove` 通常会报"仍配置为服务"。
标准顺序（GitHub 官方 Windows self-hosted runner 移除文档）：

```powershell
.\svc.cmd stop
.\svc.cmd uninstall
./config.cmd remove --token <新 token>
```

`<新 token>` 要回第 1 步的页面重新取——旧 token 大概率已经过期。卸载干净之后，回到
"注册步骤"的第 2 步重新走一遍。

#### 注册完之后：验证门禁真的会拦不达标的改动（原计划 Step 3/4）

1. 新建一个分支，把 `evaluation/phoenix/criteria.yaml` 里 `faithfulness` /
   `metric: average` 那条的 `threshold: 0.8` 临时改成 `threshold: 0.99`，提交、推到
   远程，对 `master` 开一个 PR。
2. **预期变红**：PR 的 Checks 列表里 `Eval Gate / eval` 显示红叉。点进这次 run 的日志，
   能看到一张 `Acceptance Criteria` 记分卡，`faithfulness` / `average` 那一行
   `verdict` = `FAIL`，`observed`（实测均值）和 `required`（`0.990`）都在。
3. 把 `threshold` 改回 `0.8`，提交、推到同一分支。
4. **预期变绿**：同一个 PR 上 `Eval Gate / eval` 变绿。
5. 验证完删掉这个测试分支即可，不需要合并。

第一次跑（尤其是 push 到 master 触发的全量跑）建议全程盯着 Actions 日志——理由见 4.4
"全量规模首跑"那条。

辅助 shell 脚本（被 workflow 调用）：

| 文件 | 用途 |
|---|---|
| `scripts/cf-kv-update.sh` | 调 Cloudflare API 改 KV 里的 canary_weight |
| `scripts/wait-for-health.sh` | 轮询 `/api/health` 直到返回期望版本 |

---

## 5. 日常使用场景速查

### 场景：发一个新功能

```bash
git commit -m "feat: 短描述"
git push origin master
# 等 5-10 分钟，看 Actions 跑成功，canary 上是新版本
# 观察一段，没问题就：
# Actions → Promote Stable → version 输 X.Y.Z → Run workflow
```

### 场景：发一个 bug 修复

```bash
git commit -m "fix: 短描述"
git push origin master
# 后续同上
```

### 场景：发一个不兼容的大改动

```bash
git commit -m "feat!: 短描述

BREAKING CHANGE: 详细说明，比如 /api/query 的 response 字段从 result 改成 answer"
git push origin master
# 切 X.0.0，后续同上
```

### 场景：只是改了个文档 / 改了 CI / 重构

```bash
git commit -m "docs: 更新 README"      # 不会发版本
git commit -m "ci: 调整 lint 规则"      # 不会发版本
git commit -m "refactor: 抽 helper"    # 不会发版本
git push origin master
# Lint/Test/Build/Security 跑，但 Canary Deploy 的 deploy-canary job 会被 skip
```

### 场景：刚 push 完发现要紧急回滚

```
Actions → Rollback Canary → Run workflow
# 5% 流量瞬间收回 stable
```

### 场景：promote 之后发现 stable 有问题

这种情况复杂一点，因为 stable 已经是新版本了。两条路：

**路径 A：再 fix 一版**
```bash
git commit -m "fix: 紧急修 X"
git push
# 走 canary → 观察 → promote 流程
```

**路径 B：手动回滚到老版本**
```bash
# 找到上一个稳定版本号（比如 1.2.0）
# Actions → Promote Stable → version 输 1.2.0 → Run workflow
# stable 会回到 1.2.0
```

---

## 6. 关键文件速查

```
.releaserc.json               # semantic-release 规则（master 分支、tag 格式 v${version}）
railway.toml                  # Railway 部署配置（NIXPACKS、健康检查路径、启动命令）
.python-version               # Python 3.10
requirements.txt              # Python 依赖（注意：torch 用 CPU 版避免 OOM）

api/                          # FastAPI 应用
├── main.py                   # /, /api/health, /api/query 三个端点

cf-worker/                    # Cloudflare Worker（流量分流器）
├── src/index.js              # 50 行核心路由代码
├── wrangler.toml             # Worker 配置（KV 绑定、STABLE_URL/CANARY_URL）
├── package.json              # wrangler devDep
└── .gitignore

scripts/
├── cf-kv-update.sh           # 改 KV canary_weight
└── wait-for-health.sh        # 轮询 health 直到期望版本

.github/workflows/
├── canary-deploy.yml         # 主 CD 流水线
├── promote-stable.yml        # 手动 promote
├── rollback-canary.yml       # 手动 rollback
├── eval-gate.yml             # 评估门禁（期 1，self-hosted runner，见 4.4 / runner 注册见 4.5）
├── lint.yml / test.yml / build.yml / security.yml  # 代码质量
```

---

## 7. 配置一览（GitHub Secrets / 环境变量）

### GitHub Repo Secrets（11 项）

去 https://github.com/brucezhu9594/BZ-RAG/settings/secrets/actions 看：

| Secret | 用途 | 性质 |
|---|---|---|
| `APP_ID` | GitHub App 的 ID | 公开 ID，但用 secret 装方便统一管理 |
| `APP_PRIVATE_KEY` | GitHub App 私钥 .pem 内容 | **真 secret** |
| `RAILWAY_TOKEN` | Railway 部署用的 Project Token | **真 secret** |
| `RAILWAY_PROJECT_ID` | Railway 项目 ID | 公开 ID |
| `RAILWAY_ENVIRONMENT_ID` | Railway 环境 ID（production）| 公开 ID |
| `RAILWAY_SERVICE_ID_STABLE` | stable service ID | 公开 ID |
| `RAILWAY_SERVICE_ID_CANARY` | canary service ID | 公开 ID |
| `CF_ACCOUNT_ID` | Cloudflare 账号 ID | 公开 ID |
| `CF_API_TOKEN` | Cloudflare API token（写 KV）| **真 secret** |
| `KV_NAMESPACE_ID` | KV namespace ID | 公开 ID |
| `EDGE_AUTH_TOKEN` | Worker 鉴权令牌 | **真 secret** |

### Railway 各 service 的环境变量（在 Railway dashboard 设）

每个 service（stable、canary）都需要设：

```
MODEL_ID=glm-4-plus              # 或者你 .env 里那个值
OPENAI_API_KEY=sk-...            # 真 secret
ZHIPUAI_API_KEY=...              # 真 secret
USER_AGENT=Mozilla/5.0 ...
APP_VERSION=v0.0.0-init          # 初始值，CI 部署时会覆盖
```

### Cloudflare Worker 的环境变量 + secret

在 `cf-worker/wrangler.toml` 里：

```
STABLE_URL = "https://bz-rag-stable-production.up.railway.app"
CANARY_URL = "https://bz-rag-canary-production.up.railway.app"
```

通过 `wrangler secret put` 单独设：

```
EDGE_AUTH_TOKEN = <43 字节随机串>    # 真 secret，跟 GitHub Secret 同步
```

---

## 8. 客户端怎么调用线上 API

**所有外部访问都走 Worker URL，不走 Railway 直连**。

```python
import os, requests

resp = requests.post(
    "https://bz-rag-router.brucezhu9594.workers.dev/api/query",
    headers={
        "X-Edge-Auth": os.environ["BZ_RAG_EDGE_AUTH_TOKEN"],
        "Content-Type": "application/json",
    },
    json={"query": "你的问题"},
    timeout=30,
)
print(resp.json())
print("This time hit:", resp.headers.get("x-bz-backend"))   # stable / canary
```

健康检查（无需鉴权 body，但仍需鉴权 header）：

```bash
curl -i \
  -H "X-Edge-Auth: $EDGE_AUTH_TOKEN" \
  https://bz-rag-router.brucezhu9594.workers.dev/api/health
```

---

## 9. 常见问题（踩过的坑）

### Q1：Railway build 失败 OOM，下载 PyTorch 时被 kill

**原因**：默认 `pip install torch` 装的是 GPU 版本（含 CUDA 依赖，~600MB）。Railway 免费档构建容器内存有限，下载过程中 OOM。

**已解决**：`requirements.txt` 顶部加了 `--extra-index-url https://download.pytorch.org/whl/cpu`，pip 改装 CPU 版（~200MB）。

### Q2：`Invalid RAILWAY_TOKEN` 但 token 看起来没错

**原因**：Railway 有两种 token —— Personal Token (`/account/tokens`) 和 Project Token（项目里建）。`railway up -p PROJECT_ID -s SERVICE_ID` 这种写法**必须用 Project Token**，Personal Token 会被拒。

**已解决**：建了 Project Token 替换 GitHub Secret。

### Q3：Cloudflare dashboard 报 403 / "you have been blocked"

**原因**：CF WAF 把数据中心 IP 段（Vultr/DO/Linode 等）拉黑，在国内翻墙过去做高频写操作（创建 token、deploy worker）容易被识别为滥用。

**解决**：换 ISP 出口节点 / 装 Cloudflare WARP（CF 自家工具，不会被自家 WAF 拦）/ 用手机热点。

### Q4：GHA 里的 `wait-for-health.sh` 一直 mismatch

**原因**：Railway 那边 build 失败了（你只看到了警告但漏了真正的错误日志），service 没起新版本，所以 health 端点一直返回旧 APP_VERSION。

**怎么办**：去 Railway 看具体那次 deployment 的 Build Logs，往下滚找红字。常见原因 OOM、网络超时、依赖冲突。

### Q5：`/api/query` 在线上 500 / 503

**原因**：本地 chroma 向量库（`/db` 目录）被 `.gitignore` 排除，没传到 Railway。这是个**已知遗留**，CD 流水线本身没问题。

**怎么办**：单独的工程任务，要么挂 Railway Volume 做持久化，要么把建库脚本加到部署流程里。

### Q6：semantic-release 不发版本

可能原因：
- commit 信息没有 `feat:` / `fix:` / `BREAKING CHANGE:` 前缀
- 自上次发版以来没有可发版的 commit（全是 chore/docs/refactor）
- 这个是**正常行为**，看 release job 日志会写 `There are no relevant changes, so no new version is released`

### Q7：pip-audit 说 `torch+cpu not on PyPI`，Security workflow 红

**原因**：torch 的 CPU 版本只在 PyTorch 自家 index 上，PyPI 没有。pip-audit `--strict` 模式遇到查不到的包直接判失败。

**解决**：去掉 `--strict`（让 pip-audit 警告但不失败），或者完全切回 GPU 版（牺牲构建稳定性）。

---

## 10. 当前一些遗留 / 可优化项

1. **`/api/query` 在线上跑不通**：chroma db 缺失（见 Q5）
2. **Worker 端鉴权是单 token**：所有调用方共用一个 EDGE_AUTH_TOKEN，不能给单个调用方撤权。如果有多客户端场景，要升级成"按 client_id 独立 token"
3. **Railway 直连 URL 仍然公开**：理论上有人猜对 URL 能绕过 Worker 直接调 Railway。学习场景没事，生产环境要在 FastAPI 加一层 X-Internal-Auth 校验头
4. **观察期没有自动化告警**：promote 完全靠人主观判断"观察够了没"，理想情况是接 Sentry / Grafana 自动判断错误率

---

## 11. 架构决策记录（为啥这样选，不那样选）

| 决策 | 选了 | 没选什么 | 理由 |
|---|---|---|---|
| 流量分流方式 | Cloudflare Worker + KV 自实现 | Cloudflare Load Balancer | LB 要付费 + 要自有域名，本项目都没有 |
| 应用部署平台 | Railway | Heroku / Vercel / 自建 K8s | 免费档够用，DX 接近 Heroku，对 Python 友好 |
| CI/CD 编排 | GitHub Actions | Jenkins / GitLab CI / CircleCI | 跟仓库共生，免费额度足，Mira 也用这个 |
| 自动版本号 | semantic-release + conventionalcommits | 手动 git tag | 强制 commit 消息规范，倒逼规范化 |
| 鉴权 token 怎么管 | GitHub App | Personal Access Token | App 是服务身份，独立于个人账号，权限收得窄 |
| Worker 鉴权 | 单 X-Edge-Auth header | mTLS / OAuth | 简单够用，Worker 免费档不支持 mTLS |

---

## 12. 想深入了解的话

- [Conventional Commits 规范](https://www.conventionalcommits.org/zh-hans/)
- [semantic-release 文档](https://semantic-release.gitbook.io/)
- [Cloudflare Workers 文档](https://developers.cloudflare.com/workers/)
- [Railway CLI 文档](https://docs.railway.com/reference/cli-api)
- [GitHub Actions: 创建 GitHub App Token](https://github.com/actions/create-github-app-token)
- [PEP 440 版本号规范](https://peps.python.org/pep-0440/)（"为什么 +cpu 后缀的版本会被优先选"的依据）
