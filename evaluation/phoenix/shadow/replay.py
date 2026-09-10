"""按节奏向影子分流器打流量。

查询池刻意取自 golden 集**之外**：线上评估如果只测训练过的 case，
测出来的是"金标集上的表现"，不是线上表现。设计文档 §4.7 明确要求这一点。
tests/test_shadow_replay.py 里有一条测试钉住"池子与 test_dataset.json 零重合"。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import json
import pathlib
import random
import time

import httpx

POOL_PATH = str(pathlib.Path(__file__).parent / "query_pool.json")
ROUTER_URL = os.environ.get("SHADOW_ROUTER_URL", "http://127.0.0.1:8100")


def load_pool(path: str = POOL_PATH) -> list[str]:
    with open(path, encoding="utf-8") as f:
        pool = json.load(f)
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"{path} 必须是非空的查询列表")
    return [str(q) for q in pool]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="向影子分流器打流量")
    p.add_argument("-n", "--count", type=int, default=10)
    p.add_argument("--interval", type=float, default=1.0, help="每次请求之间的秒数")
    args = p.parse_args(argv)

    pool = load_pool()
    backends: dict[str, int] = {}

    with httpx.Client(timeout=180.0) as client:
        for i in range(args.count):
            q = random.choice(pool)
            try:
                r = client.post(
                    f"{ROUTER_URL}/api/milvus/query-phoenix", json={"query": q}
                )
                b = r.headers.get("x-bz-backend", "?")
                backends[b] = backends.get(b, 0) + 1
                print(f"[{i + 1}/{args.count}] {b:<7} {r.status_code}  {q[:26]}")
            except Exception as e:  # noqa: BLE001 —— 打流量不该因单次失败中断
                print(f"[{i + 1}/{args.count}] ERROR {type(e).__name__}: {e}")
            if i + 1 < args.count:
                time.sleep(args.interval)

    print("分流统计:", backends)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
