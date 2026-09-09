"""智谱 embedding，走 zai-sdk。替代 langchain_community 的 ZhipuAIEmbeddings。

**为什么要换掉 langchain_community.ZhipuAIEmbeddings**：它依赖 `zhipuai` 这个旧 SDK，
而 `zhipuai` 2.x 全部 pin `pyjwt>=2.8.0,<2.9.0`，`zai-sdk`（同一家的新 SDK，
common/ocr.py 在用）pin 的是 `pyjwt>=2.9.0,<3.0.0`——两者**互相不可满足**。
requirements.txt 里两个 SDK 并存，pip 只能靠把 `zhipuai` 一路回溯降级到 1.0.7
来"解决"，而 1.0.7 的 API 完全不同，langchain 那边直接
`ImportError: Could not import zhipuai python package`。

这个坑之所以一直没暴露：本地环境早就装了 zhipuai 2.x，`pip install -r requirements.txt`
只说 "already satisfied"；直到评估门禁第一次真在 CI 上做全新解析才炸出来（48 项全挂），
而 Railway 用 NIXPACKS 全新构建生产镜像时装到的同样是坏版本。

换成 zai-sdk 之后只剩一个 SDK，pyjwt 的约束不再互斥，`pyjwt>=2.12.0` 那条
安全约束也保得住。

**向量等价性已实测**：同一段文本，zai-sdk 与 zhipuai 算出的向量余弦
= 1.0000000000，与 db/_kb_cache.parquet 里缓存的那一条 = 0.9999970463
（差值来自 parquet 的浮点存储精度）。两者默认端点都是
https://open.bigmodel.cn/api/paas/v4，同平台同模型，所以在同一向量空间里，
换 SDK 不需要重新灌库。
"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from zai import ZhipuAiClient

load_dotenv()

DEFAULT_MODEL = "embedding-3"
BATCH_SIZE = 64  # 智谱 embedding 单次上限 64


class ZhipuEmbeddings(Embeddings):
    """LangChain Embeddings 接口的智谱实现。

    继承 langchain_core 的 Embeddings 而不是只提供两个函数，是因为有几处调用点
    把它直接传给 LangChain 的 VectorStore（`embedding_function=` / `embedding=`），
    那些地方需要的是真正的 Embeddings 实例。
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        # **必须显式传 ZHIPUAI_API_KEY**：ZhipuAiClient() 不传 key 时读的是
        # ZAI_API_KEY，而本项目的 ZAI_API_KEY 与 ZHIPUAI_API_KEY 是两个不同的
        # key（实测长度都不一样），用前者调 embedding 会 401
        # （{"error":{"code":"1000","message":"身份验证失败。"}}）。
        key = api_key or os.environ.get("ZHIPUAI_API_KEY", "")
        if not key:
            raise ValueError("未设置 ZHIPUAI_API_KEY 环境变量")
        self.model = model
        self._client = ZhipuAiClient(api_key=key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批嵌入多段文本，顺序与输入一致。"""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            # 不假设服务端按输入顺序返回：按 index 排一遍再取。
            for item in sorted(resp.data, key=lambda d: d.index):
                vectors.append(item.embedding)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding
