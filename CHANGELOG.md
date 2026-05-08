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
