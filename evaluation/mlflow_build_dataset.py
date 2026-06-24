"""把 evaluation/test_dataset.json 同步到 MLflow tracking server 的 EvaluationDataset。

仿照 build_dataset.py（Langfuse 版）：merge_records 按 inputs 哈希 upsert，重复执行幂等。
需要 SQL 后端的 tracking server（sqlite 即可）：
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
"""

import json
import os
import pathlib
import sys

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

# 本地 tracking server 走 HTTP，Privoxy 会拦 localhost，先兜底 NO_PROXY。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset, get_dataset

# 与 Langfuse 版同名同版本：黄金集 v2，24 项（GLM-4-Flash 生成 + 人工审）。
DATASET_NAME = "bz-rag-eval-v2"
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "bz-rag-milvus")

DATASET_PATH = pathlib.Path(__file__).parent / "test_dataset.json"


def sync_dataset(items: list[dict]) -> None:
    try:
        dataset = get_dataset(name=DATASET_NAME)
    except MlflowException:
        experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
        dataset = create_dataset(
            name=DATASET_NAME,
            experiment_id=experiment.experiment_id,
            tags={"description": "BZ-RAG 评测集 v2，来自 evaluation/test_dataset.json"},
        )
    # inputs 的 key 必须与 predict_fn（milvus_rag_mlflow_query）的形参名一致。
    dataset.merge_records(
        [
            {
                "inputs": {"query": item["question"]},
                "expectations": {
                    "expected_response": item["ground_truth"],
                    "expected_source": item["expected_source"],
                },
            }
            for item in items
        ]
    )


def main() -> None:
    load_dotenv()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with open(DATASET_PATH, encoding="utf-8") as f:
        items = json.load(f)
    print(f"同步 {len(items)} 条样本到 MLflow dataset {DATASET_NAME!r}...")
    sync_dataset(items)
    print(f"完成。请到 {mlflow.get_tracking_uri()} 查看 dataset。")


if __name__ == "__main__":
    main()
