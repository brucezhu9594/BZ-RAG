"""golden 集加载与内容寻址 ID。

example_id 用问题文本的 sha256 前 16 位。Phoenix 的 example_id_key 要求调用方自己提供
稳定 ID（官方举例"数据库主键或 content hash"），用内容哈希的好处是"同一批 case"
变成可计算的集合——改了问题文本就是一条新 case，不改就永远映射回同一个 example。
"""

import hashlib
import json
import pathlib
from typing import NamedTuple

DATASET_PATH = pathlib.Path(__file__).resolve().parents[1] / "test_dataset.json"
SMOKE_SIZE = 3


class Case(NamedTuple):
    example_id: str
    question: str
    ground_truth: str
    expected_source: str


def example_id(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]


def _load() -> list[Case]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        items = json.load(f)
    return [
        Case(
            example_id=example_id(item["question"]),
            question=item["question"],
            ground_truth=item["ground_truth"],
            expected_source=item["expected_source"],
        )
        for item in items
    ]


CASES: list[Case] = _load()
CASE_IDS: list[str] = [c.example_id for c in CASES]
# smoke 子集：PR 上只跑这几条，控制判官调用量。取前 N 条而不是随机，保证可比。
SMOKE_IDS: frozenset[str] = frozenset(CASE_IDS[:SMOKE_SIZE])
