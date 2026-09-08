"""把 golden 集推进 Phoenix dataset。

用 create_dataset + example_id_key：Phoenix 会 diff 出增/改/删，让 dataset 精确等于
本地文件（上传里没有的 example 会被删）。回灌走 add_examples_to_dataset（只增改不删），
那是期 3 的事。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from evaluation.phoenix.cases import CASES
from phoenix.client import Client

DATASET_NAME = os.environ.get("PHOENIX_TEST_DATASET", "bz-rag-golden")


def main() -> None:
    df = pd.DataFrame(
        [
            {
                "example_id": c.example_id,
                "question": c.question,
                "expected_response": c.ground_truth,
                "expected_source": c.expected_source,
            }
            for c in CASES
        ]
    )
    dataset = Client().datasets.create_dataset(
        name=DATASET_NAME,
        dataframe=df,
        input_keys=["question"],
        output_keys=["expected_response"],
        metadata_keys=["expected_source"],
        example_id_key="example_id",
    )
    print(f"dataset {dataset.name!r} version {dataset.version_id} 共 {dataset.example_count} 条")


if __name__ == "__main__":
    main()
