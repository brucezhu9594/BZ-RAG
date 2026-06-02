"""把 evaluation/test_dataset.json 同步到 Langfuse Cloud 的 dataset。"""
import json
import os
import pathlib
import sys

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langfuse import Langfuse

# v2：黄金集从旧 15 项手写换成 24 项 GLM 生成 + 人工审。
# Langfuse create_dataset_item 是 upsert（按 input 幂等），换黄金集后 question 全变，
# 旧 item 不会被覆盖只会并存，因此换新 dataset 名以隔离，旧 bz-rag-eval 作废。
DATASET_NAME = "bz-rag-eval-v2"
DATASET_DESCRIPTION = "BZ-RAG 评测集 v2，24 项（GLM-4-Flash 生成 + 人工审），来自 evaluation/test_dataset.json。"

DATASET_PATH = pathlib.Path(__file__).parent / "test_dataset.json"


def sync_dataset(items: list[dict]) -> None:
    client = Langfuse()
    client.create_dataset(name=DATASET_NAME, description=DATASET_DESCRIPTION)
    for item in items:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["question"],
            expected_output=item["ground_truth"],
            metadata={"expected_source": item["expected_source"]},
        )
    client.flush()


def main() -> None:
    load_dotenv()
    with open(DATASET_PATH, encoding="utf-8") as f:
        items = json.load(f)
    print(f"同步 {len(items)} 条样本到 Langfuse dataset {DATASET_NAME!r}...")
    sync_dataset(items)
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"完成。请到 {host} 查看 dataset。")


if __name__ == "__main__":
    main()
