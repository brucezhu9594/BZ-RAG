"""把 evaluation/test_dataset.json 同步到 Langfuse Cloud 的 dataset。"""
import json
import os
import pathlib
import sys

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langfuse import Langfuse

DATASET_NAME = "bz-rag-eval"
DATASET_DESCRIPTION = "BZ-RAG 评测集，来自 evaluation/test_dataset.json 回放。"

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
