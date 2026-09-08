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


def smoke_ids(cases: list[Case]) -> frozenset[str]:
    """按 example_id（内容哈希）排序后取前 SMOKE_SIZE 条。

    PR 上只跑这几条，控制判官调用量；但选取方式必须和本模块的内容寻址原则保持一致——
    按哈希排序取前 N，而不是按 cases 列表的位置切片取前 N。位置切片会锚在
    test_dataset.json 的行序上：谁排在文件开头就被选中，和这条 case 的内容毫无关系，
    也会让"哪三条是 smoke"随文件改一次序就静默变化。哈希排序对输入顺序免疫，
    且这个排序结果本身就是在数据集上做的一次确定性铺开，不会像文件序那样
    系统性偏向开头那几条最简单的 case。
    """
    return frozenset(sorted(c.example_id for c in cases)[:SMOKE_SIZE])


CASES: list[Case] = _load()
CASE_IDS: list[str] = [c.example_id for c in CASES]
SMOKE_IDS: frozenset[str] = smoke_ids(CASES)
