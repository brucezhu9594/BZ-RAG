"""把 canary 权重归零（或设成任意 0-100 的值）。

参数形状与 scripts/cf-kv-update.sh 完全一致（0-100 整数）——期 2 的影子分流器
刻意保持了这个形状，就是为了让这里一份实现能同时驱动影子侧和真云侧，
将来 Milvus 上云时不用改。scripts/cf-kv-update.sh 本身一行不动。

**真云侧现在跑不通**：cf-kv-update.sh 需要 CF_API_TOKEN，而它至今未配置
（docs/CD-pipeline.md §10 的遗留项）。所以默认 target=shadow；
走 cloud 且 token 缺失时**明确抛错**，绝不静默跳过——
"以为回滚了其实没回滚"比直接失败危险得多。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import json
import pathlib
import subprocess

DEFAULT_WEIGHT_PATH = str(pathlib.Path(__file__).parent / "shadow" / "weight.json")
CF_SCRIPT = str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "cf-kv-update.sh")
VALID_TARGETS = ("shadow", "cloud")


def set_weight(weight: int, target: str = "shadow", weight_path: str | None = None) -> None:
    if target not in VALID_TARGETS:
        raise ValueError(f"target 必须是 {'/'.join(VALID_TARGETS)}，实际 {target!r}")
    # bool 是 int 的子类，True 会通过 isinstance 检查，必须单独挡掉。
    if isinstance(weight, bool) or not isinstance(weight, int) or not 0 <= weight <= 100:
        raise ValueError(f"权重必须是 0-100 的整数，实际 {weight!r}")

    if target == "shadow":
        path = pathlib.Path(weight_path or DEFAULT_WEIGHT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"canary_weight": weight}) + "\n", encoding="utf-8")
        return

    if not os.environ.get("CF_API_TOKEN"):
        raise RuntimeError(
            "CF_API_TOKEN 未设置，无法改真云侧权重。"
            "这是 docs/CD-pipeline.md 记录的遗留项；"
            "在配好之前请用 target=shadow，或手工执行 scripts/cf-kv-update.sh。"
        )
    subprocess.run(["bash", CF_SCRIPT, str(weight)], check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="设置 canary 权重（回滚就是设 0）")
    p.add_argument("weight", type=int, help="0-100 整数")
    p.add_argument("--target", default="shadow", choices=VALID_TARGETS)
    args = p.parse_args(argv)
    set_weight(args.weight, target=args.target)
    print(f"canary_weight -> {args.weight} (target={args.target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
