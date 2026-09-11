"""把线上失败样本捞进人工分诊队列。

**写进独立的 bz-rag-harvested，不写 bz-rag-golden。** spec §4.10 原文说
"追加进 golden 集"，但查证后行不通，两个独立原因：

1. 会被下一次推送抹掉。门禁的真实数据源是本地 evaluation/test_dataset.json
   （cases.py 从它读），Phoenix 上的 bz-rag-golden 只是镜像；而 dataset.py 用的
   create_dataset(example_id_key=...) 是**全量替换**语义——"上传里没有的
   example 会被删"（该文件顶部注释原话）。回灌进去的样本下一次推送就没了。
2. 回灌样本没有 ground truth。expected_response 是 contextual_precision /
   contextual_recall 两个判官的必需输入，线上 span 上没有标准答案，
   混进金标集只会让这两个判官对它们全部 errored。

所以 harvest 只负责"捞出来 + 带上回跳链接"。人看过、补上标准答案之后，
手工加进 test_dataset.json，再走正常的 dataset.py 推送。
**回灌这条环由人闭合，不自动闭合**——与 promote 需人确认是同一条谨慎原则。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any

from evaluation.phoenix.online_worker import otel_span_id

PULL_LIMIT = 1000
HARVEST_DATASET = "bz-rag-harvested"

# 什么算"失败"。按 annotation 的 label 判，与 criteria.yaml 的 pass_when 同口径：
#   refusal_check   pass_when: label == 'ok'   -> 非 ok 即失败
#   faithfulness    pass_when: score >= 0.5    -> 只有 incorrect(0.0) 算失败，
#                                                 partial(0.5) 按口径是通过
# answer_relevancy 不进回灌：它判的是"答得切不切题"，而不切题往往是问题本身
# 模糊，补进金标集意义不大；切题但答错的样本由 faithfulness 抓。
FAIL_LABELS = {
    "refusal_check": {"refused", "empty"},
    "faithfulness": {"incorrect"},
}


def failed_spans(client: Any, project: str, window_minutes: int) -> list[dict[str, Any]]:
    """返回窗口内被判失败的根 span。

    没有 annotation 的 span **不算失败**——没评过不等于评差了，
    把未评估样本当失败捞回来会污染分诊队列。
    """
    now = datetime.now(timezone.utc)
    spans = client.spans.get_spans(
        project_identifier=project,
        span_kind="AGENT",
        start_time=now - timedelta(minutes=window_minutes),
        end_time=now,
        limit=PULL_LIMIT,
    )
    roots = [s for s in spans if not s.get("parent_id")]
    if not roots:
        return []

    anns = client.spans.get_span_annotations(
        span_ids=[otel_span_id(s) for s in roots],
        project_identifier=project,
        limit=PULL_LIMIT,
    )
    bad: set[str] = set()
    for a in anns:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        sid = a.get("span_id") if isinstance(a, dict) else getattr(a, "span_id", None)
        result = a.get("result") if isinstance(a, dict) else getattr(a, "result", None)
        if not name or not sid or not isinstance(result, dict):
            continue
        if result.get("label") in FAIL_LABELS.get(str(name), set()):
            bad.add(str(sid))
    return [s for s in roots if otel_span_id(s) in bad]


def harvest(
    client: Any,
    project: str = "bz-rag-canary",
    window_minutes: int = 1440,
    dataset_name: str = HARVEST_DATASET,
) -> int:
    spans = failed_spans(client, project, window_minutes)
    if not spans:
        return 0

    # examples 路径要的是**嵌套**的 {input, output, metadata}；
    # input_keys / output_keys / metadata_keys 那套扁平写法只对 dataframe 路径生效
    # （dataset.py 走的是 dataframe，所以那边是扁平的）。用错会被真 API 拒：
    # "examples must be a single dictionary with required 'input' and 'output' keys"。
    examples = []
    for s in spans:
        attrs = s.get("attributes") or {}
        examples.append(
            {
                "input": {"question": str(attrs.get("input.value", ""))},
                # 线上没有标准答案，留空由人补。写成空串而不是省略这个键，
                # 是为了让分诊的人一眼看到"这里要填"。
                "output": {"expected_response": ""},
                "metadata": {
                    "observed_answer": str(attrs.get("output.value", "")),
                    "source_span_id": otel_span_id(s),
                    "origin": "harvested",
                },
            }
        )

    client.datasets.add_examples_to_dataset(dataset=dataset_name, examples=examples)
    return len(examples)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="把线上失败样本捞进人工分诊队列")
    p.add_argument("--project", default="bz-rag-canary")
    p.add_argument("--window-minutes", type=int, default=1440)
    p.add_argument("--dataset", default=HARVEST_DATASET)
    args = p.parse_args(argv)

    from phoenix.client import Client

    client = Client(base_url=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006"))
    n = harvest(client, args.project, args.window_minutes, args.dataset)
    print(f"捞回 {n} 条失败样本 -> dataset {args.dataset!r}")
    if n:
        print("下一步（人工）：补上 expected_response，加进 evaluation/test_dataset.json，")
        print("               再跑 python -m evaluation.phoenix.dataset 推送。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
