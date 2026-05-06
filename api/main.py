"""BZ-RAG HTTP API: 部署到 Railway 后通过 Cloudflare Worker 做金丝雀分流。"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_VERSION = os.environ.get("APP_VERSION", "dev")

app = FastAPI(title="BZ-RAG", version=APP_VERSION)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    version: str


@app.get("/")
def root() -> dict:
    return {"service": "bz-rag", "version": APP_VERSION, "docs_url": "/docs"}


@app.get("/api/health")
def health() -> dict:
    """健康检查：不依赖外部服务，只返回进程版本号，供 CD 流水线轮询确认部署成功。"""
    return {"status": "ok", "version": APP_VERSION}


@app.post("/api/query")
def query(req: QueryRequest) -> QueryResponse:
    """调用 RAG 流水线返回 LLM 生成的答案。"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    try:
        # 延迟 import：避免启动阶段加载 LangChain 全套依赖
        from app.chroma.hybrid_search import rag
    except KeyError as e:
        raise HTTPException(
            status_code=503,
            detail=f"环境变量缺失：{e}",
        ) from e

    try:
        answer = rag(req.query)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG 调用失败：{type(e).__name__}: {e}",
        ) from e

    return QueryResponse(answer=answer, version=APP_VERSION)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
