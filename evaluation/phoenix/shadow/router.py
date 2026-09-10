"""影子 canary 分流器：复刻 cf-worker/src/index.js 的按权重分流。

Railway 上跑不了 Milvus 管线（CD-pipeline Q5 遗留），所以线上段在本机复刻：
两个不同 APP_VERSION、相同 PHOENIX_PROJECT_NAME=bz-rag-canary 的 uvicorn，
前面挂这个分流器。

权重参数形状与 scripts/cf-kv-update.sh 完全一致（0-100 整数），这样期 3 的
monitor 输出能同时驱动影子侧和真云侧——将来 Milvus 上云时 monitor 一行不用改。
scripts/cf-kv-update.sh 本身保持原样不动。

**这不是真线上**：流量来自 replay.py 的回放，不是真实用户请求。期 2 验证的是
机制正确性（采样、写回、去重、护栏豁免），不是"线上质量真的怎么样"。
设计文档 §8 把这个局限列为已知风险。
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
    """读权重。

    文件不存在或不是合法 JSON 时返回 0——失败方向要安全（全量走 stable）。
    但**越界或类型不对要抛**：那是配置写错了，静默当 0 会让人以为分流在跑、
    其实一直全量走 stable，这种"以为在测其实没测"比直接报错危险得多。
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return 0
    if not isinstance(raw, dict):
        return 0
    w = raw.get("canary_weight", 0)
    if not isinstance(w, int) or isinstance(w, bool) or not 0 <= w <= 100:
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
    async with httpx.AsyncClient(timeout=180.0) as client:
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
