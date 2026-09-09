# [1.4.0](https://github.com/brucezhu9594/BZ-RAG/compare/v1.3.0...v1.4.0) (2026-09-09)


### Bug Fixes

* **eval:** _rerank 换成 RERANKER kind（R13） ([9e6e124](https://github.com/brucezhu9594/BZ-RAG/commit/9e6e124c959880741070c3f2cdbdf9ecd34e09b1))
* **eval-gate:** 门禁转入观测态，提高超时余量，拆分评估依赖，修复GBK乱码 ([168ea81](https://github.com/brucezhu9594/BZ-RAG/commit/168ea81c056c0bf07fe8279325771b894ca9ecbc))
* **eval:** phoenix-up.ps1 PATH 无关启动 + 显式守卫；.env.example 补 PHOENIX_ENDPOINT ([c83623c](https://github.com/brucezhu9594/BZ-RAG/commit/c83623c3764799e2ddf896e7c698440d22feb693))
* **eval:** phoenix-up.ps1 避免探测分支被 Stop+stderr 重定向炸出 NativeCommandError ([b15284b](https://github.com/brucezhu9594/BZ-RAG/commit/b15284b09452d4fe9b60a2f57bf4ee63201f39b9))
* **eval:** smoke 子集改为内容锚定（哈希排序取前 3），不再锚在文件行序上 ([c57343e](https://github.com/brucezhu9594/BZ-RAG/commit/c57343e6574d81df967f68d941b89c6329860942))
* **eval:** smoke 子集测试改为调用生产代码的 smoke_ids()，不再平行重实现选取规则 ([922d633](https://github.com/brucezhu9594/BZ-RAG/commit/922d633734dc4aaa997ceb2fe40c5abb84ed7223))
* **eval:** 判官大面积报错时门禁仍绿灯 + pass_when 前移校验 + 真containment ([378ccd2](https://github.com/brucezhu9594/BZ-RAG/commit/378ccd281e26d8f7e3339119184101343be91ca4))
* **eval:** 提取动作纳入 try 保护范围 + 配置缺失 eager 快速失败 ([07cd9ff](https://github.com/brucezhu9594/BZ-RAG/commit/07cd9ffe48863d9db0d8c1988268b67e0ffe30c2))
* **eval:** 错误率闸门改绝对下限+比例、记分卡标记改ASCII、Criterion可JSON化 ([90349d9](https://github.com/brucezhu9594/BZ-RAG/commit/90349d9eb643ff9670c49f499e1839c150f865f9))
* **phoenix-acceptance:** 检测判官落库失败的静默丢样本，criteria.yaml 提前校验 ([ba27625](https://github.com/brucezhu9594/BZ-RAG/commit/ba2762580084bfd3ed65dd2e3157ccac8486f0df))
* **retrieval:** 重排不再系统性挑中检索结果里最差的几条 ([94677fe](https://github.com/brucezhu9594/BZ-RAG/commit/94677fee620677b0f2561f03a36cb724e9f7a97e))


### Features

* **eval:** golden 集内容寻址 example_id + 推送 Phoenix dataset ([f5067fa](https://github.com/brucezhu9594/BZ-RAG/commit/f5067fa4b4b3156e42ab4d5021554b62bd7ed58e))
* **eval:** Phoenix 埋点版 Milvus RAG 管线 + query-phoenix 端点 ([12a6dd6](https://github.com/brucezhu9594/BZ-RAG/commit/12a6dd647abacdba96b8eece35fb20a04662b023))
* **eval:** Phoenix 离线门禁套件，退出码即门禁 ([bc9c4ca](https://github.com/brucezhu9594/BZ-RAG/commit/bc9c4cac92cbcbd5308b5d527a8b1053462916da))
* **eval:** Phoenix 评估栈依赖与本地启动脚本 ([c5e0ad9](https://github.com/brucezhu9594/BZ-RAG/commit/c5e0ad9b17af35927ba2f1f3b8f8a290b047cee9))
* **eval:** 五个分类式判官（MiniMax-m3），判官失败落 errored 第三态 ([ef9d818](https://github.com/brucezhu9594/BZ-RAG/commit/ef9d81829d4a76beb12cd49c54424b326b36b482))
* **eval:** 声明式验收条件（Arize acceptanceCriteria 的 Python 版） ([30c730d](https://github.com/brucezhu9594/BZ-RAG/commit/30c730d40d184bfb9b9f227ab012a699c8fb6af6))

# [1.3.0](https://github.com/brucezhu9594/BZ-RAG/compare/v1.2.0...v1.3.0) (2026-09-08)


### Bug Fixes

* **eval:** 多轮驱动同会话重复 query fail-fast + 测试/文档补强 ([3e2dcc2](https://github.com/brucezhu9594/BZ-RAG/commit/3e2dcc2ba0548d46aa9da7d623115fc663e504df))


### Features

* **api:** 多轮历史纯助手 build_rewrite_prompt / build_chat_messages ([9c5dabb](https://github.com/brucezhu9594/BZ-RAG/commit/9c5dabb394a9d5305e9033bca1b397a29c554e1d))
* **api:** 管线支持多轮——查询改写 + 生成带历史 ([29035f4](https://github.com/brucezhu9594/BZ-RAG/commit/29035f4ff520f3ac95cf01a63d4134f7b7989132))
* **eval:** MLflow 单轮+多轮评估 baseline(有状态升级前) ([769ef27](https://github.com/brucezhu9594/BZ-RAG/commit/769ef2726e313472097468e11e4e084a16881366))
* **eval:** MLflow 多轮(session 级)评估 + followup_resolution 指标 ([a3ed79c](https://github.com/brucezhu9594/BZ-RAG/commit/a3ed79c418858469b9fcfb2cc72c4be983f39007))
* **eval:** 有状态 predict_fn 包装器，按 session 穿线真实历史 ([15be2c2](https://github.com/brucezhu9594/BZ-RAG/commit/15be2c2f9f944708c43f017a21554ad130fa2c42))

# [1.2.0](https://github.com/brucezhu9594/BZ-RAG/compare/v1.1.0...v1.2.0) (2026-06-11)


### Bug Fixes

* **api:** main.py 启动即 load_dotenv，修 DeepEval trace 不上报 ([da134be](https://github.com/brucezhu9594/BZ-RAG/commit/da134bef87e725f56e987eebc469e0c0c33a7726))
* **api:** retriever span 补 embedder + rerank 改 tool 类型 ([47c3bc7](https://github.com/brucezhu9594/BZ-RAG/commit/47c3bc76c147434decee7640b5ad88c18a8ad76a))
* **eval:** _extract_json 兜底 json_repair 救 broken JSON ([89998da](https://github.com/brucezhu9594/BZ-RAG/commit/89998dae6a2a2a10575359a190fe03f12807a790))
* **eval:** _extract_json 剥离 <think> 块 + 兜底栈扫平衡 JSON ([b966435](https://github.com/brucezhu9594/BZ-RAG/commit/b966435dd13cbb553b3920087b5bc8f3cf94bc01))
* **eval:** cap run_experiment concurrency=1 and add task error handling ([eaf7e99](https://github.com/brucezhu9594/BZ-RAG/commit/eaf7e9942d8c765de8d6f70d15eb7fd68f639352))
* **eval:** dataset 换名 bz-rag-eval-v2 隔离旧黄金集 ([4d1c7d1](https://github.com/brucezhu9594/BZ-RAG/commit/4d1c7d19c055c57e49ad4775ab28582e8fd6d311))
* **eval:** DeepEval 强制串行 max_concurrent=1 + throttle 1s ([f22b730](https://github.com/brucezhu9594/BZ-RAG/commit/f22b730968e3a80c7ab4f5a7f591b2dcda1ce9f5))
* **eval:** GLMJudge 重试范围扩到超时/连接/5xx ([59f9b18](https://github.com/brucezhu9594/BZ-RAG/commit/59f9b18665c95efb6611de5ea7dae5c672717883))
* **eval:** make MiniMaxJudge.a_generate truly async via asyncio.to_thread ([48f567a](https://github.com/brucezhu9594/BZ-RAG/commit/48f567a9e721e5eccf21f9a4816c656153f0bb95))
* **eval:** MiniMaxJudge 加 tenacity 重试应对 429 ([81c1945](https://github.com/brucezhu9594/BZ-RAG/commit/81c19452b7d2cd730eead4894eca729abb000914))
* **eval:** per-task 超时抬到 600s 给 MiniMax 留余量 ([1d7d0e7](https://github.com/brucezhu9594/BZ-RAG/commit/1d7d0e723390fd63bf5ba223c54c67a958f3fc7c))
* **eval:** suppress deepeval rich emoji banner on Windows GBK terminal ([adb9f87](https://github.com/brucezhu9594/BZ-RAG/commit/adb9f8745ebb1e6a88cdbfea60cbe9ffcbbf20d4))
* **eval:** timeout 拉长到 150s 给慢请求出结果，retry 降到 2 ([dd3cc9c](https://github.com/brucezhu9594/BZ-RAG/commit/dd3cc9c6df22b376a304949ca0008caf20547399))
* **eval:** 协调超时层 — per-task 1200s + retry 收紧防撞墙 ([51e28e4](https://github.com/brucezhu9594/BZ-RAG/commit/51e28e4af925aaa4b8f73056a67ee68c1d330751))
* **eval:** 适配 Langfuse v4.7 run_experiment API ([cd44c1d](https://github.com/brucezhu9594/BZ-RAG/commit/cd44c1de13e5f7f76a8350577fd84230c0251f37))


### Features

* **api:** milvus 混合检索 + DeepEval tracing 流水线 ([65ebeaf](https://github.com/brucezhu9594/BZ-RAG/commit/65ebeaf6d4cb20062a4a525fbbee462c610595b9))
* **api:** 加 /api/milvus/query 端点 + thread_id + NO_PROXY ([70345f5](https://github.com/brucezhu9594/BZ-RAG/commit/70345f513549fee1743cf69dd8275dc7cf613b2b))
* **eval:** _retrieve 加 langfuse [@observe](https://github.com/observe) 装饰器 ([53001b8](https://github.com/brucezhu9594/BZ-RAG/commit/53001b88f11c843d1f43e94bdd3c952e49aa99f5))
* **eval:** MiniMaxJudge 包装成 DeepEvalBaseLLM ([dc963c9](https://github.com/brucezhu9594/BZ-RAG/commit/dc963c9ee705652127fef56b90f63538ac7a4bda))
* **eval:** 加 build_dataset 同步脚本 + 单测 ([d38e280](https://github.com/brucezhu9594/BZ-RAG/commit/d38e2803ee85d47d32cfdc7ad5ddd595d7fc9c81))
* **eval:** 加 deepeval 依赖 ([8efb04a](https://github.com/brucezhu9594/BZ-RAG/commit/8efb04aa3f3d973545f26aa355f01318628a5fa9))
* **eval:** 加 generate_dataset 离线工具 + 单测 ([854cd0e](https://github.com/brucezhu9594/BZ-RAG/commit/854cd0eb2901d56ee5e7fdd9d0b3e956ea6fc61c))
* **eval:** 加 langfuse 依赖 ([8353e64](https://github.com/brucezhu9594/BZ-RAG/commit/8353e6406c4527bce689079ba73f77da55d19671))
* **eval:** 用 deepeval 重写 evaluate.py（仅 milvus_hybrid） ([8f2c308](https://github.com/brucezhu9594/BZ-RAG/commit/8f2c30871478bda323627c50f9ca89b3e205ccf4))
* **eval:** 重做黄金集 24 项（GLM-4-Flash 生成 + 人工审查） ([43a09c4](https://github.com/brucezhu9594/BZ-RAG/commit/43a09c47a775a5a0065d8fbb3a094bda0ad1f024))
* **eval:** 重写 evaluate.py 走 Langfuse dataset + DeepEval ([02167c0](https://github.com/brucezhu9594/BZ-RAG/commit/02167c0512a33c3a0e3feb1cf948e02bde966d61))

# [1.1.0](https://github.com/brucezhu9594/BZ-RAG/compare/v1.0.1...v1.1.0) (2026-05-25)


### Features

* lancedb ([71cbeb3](https://github.com/brucezhu9594/BZ-RAG/commit/71cbeb3d23a130f45e7159795aa38826742f4feb))

## [1.0.1](https://github.com/brucezhu9594/BZ-RAG/compare/v1.0.0...v1.0.1) (2026-05-08)


### Bug Fixes

* **build:** 改用 CPU 版 torch 避免 Railway 构建 OOM ([dff58ab](https://github.com/brucezhu9594/BZ-RAG/commit/dff58abddfc39f40df393576ec1fe2acb21fd39e))

# 1.0.0 (2026-05-07)


### Bug Fixes

* chroma 实现rrf混合检索 ([b1d2535](https://github.com/brucezhu9594/BZ-RAG/commit/b1d2535e7033e26b0293d46ffcdeba58147e1a67))
* readme更新 ([f3f5e51](https://github.com/brucezhu9594/BZ-RAG/commit/f3f5e51d5e6e275289f8d25f5f1c2a7d84e5d4ee))
* rerank重排 ([14a3f8a](https://github.com/brucezhu9594/BZ-RAG/commit/14a3f8ab450213c481725868cb1add3baa224c0d))
* **security:** 约束 pyjwt>=2.12.0 修复 CVE-2026-32597 ([daf1eee](https://github.com/brucezhu9594/BZ-RAG/commit/daf1eee0fd31b10a472da447af282d7ecdffe7dc))


### Features

* chroma rag agent+qdrant混合检索+向量检索 ([00768d8](https://github.com/brucezhu9594/BZ-RAG/commit/00768d8861d5ceae8f474eaca9c9575acbb18f9b))
* milvus向量检索+混合检索 ([182d1d5](https://github.com/brucezhu9594/BZ-RAG/commit/182d1d583be21be70810f7ec3ae4989bc6c910b2))
* PR守门员四件套 ([c0e0510](https://github.com/brucezhu9594/BZ-RAG/commit/c0e0510358f6a07c4a4d306accd63563b8f99af0))
* PR守门员四件套-security ([625c275](https://github.com/brucezhu9594/BZ-RAG/commit/625c275a6db50248b0d363d32f4ad1a0a5d02a36))
* readme ([8a061e8](https://github.com/brucezhu9594/BZ-RAG/commit/8a061e82a29d6eefc95116f3f5e68d010dfbb213))
* 加 Cloudflare Worker 路由 + KV 切流 ([56abcb6](https://github.com/brucezhu9594/BZ-RAG/commit/56abcb618e05dc3312821875b15b97c71a7e3854))
* 加 FastAPI 入口 + Railway 配置 ([a527b4f](https://github.com/brucezhu9594/BZ-RAG/commit/a527b4fbbf11553a73635d92359b4b7dac718af0))
* 加金丝雀 CD 流水线（semantic-release + Railway + CF Worker KV） ([9fe6966](https://github.com/brucezhu9594/BZ-RAG/commit/9fe69665ace7d3e180eec00a8fc799c3ac4345e9))
* 多查询扩展+关键词扩展+滑动窗口记忆+历史会话改写 ([0a436c3](https://github.com/brucezhu9594/BZ-RAG/commit/0a436c3696e2e0f2eca81213deb926be2fa78803))
* 查询改写 ([7074eec](https://github.com/brucezhu9594/BZ-RAG/commit/7074eec43bc4b0d5d49b6e58d843aa94de0138df))
* 知识库构建：识别图片内容 ([650e6c2](https://github.com/brucezhu9594/BZ-RAG/commit/650e6c2bc2bad2ac54e1fcb0ca05876762c84541))
* 评测 ([f5a3613](https://github.com/brucezhu9594/BZ-RAG/commit/f5a36139b269fdcb050ee212f7950c6a8274f800))
