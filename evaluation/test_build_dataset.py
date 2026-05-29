"""evaluation/build_dataset.py 单元测试：验证幂等同步逻辑。"""
from unittest.mock import MagicMock

import pytest

from evaluation import build_dataset


def test_sync_creates_dataset_and_pushes_all_items(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(build_dataset, "Langfuse", lambda: fake_client)

    items = [
        {"question": "Q1", "ground_truth": "A1", "expected_source": "src1"},
        {"question": "Q2", "ground_truth": "A2", "expected_source": "src2"},
    ]
    build_dataset.sync_dataset(items)

    fake_client.create_dataset.assert_called_once_with(
        name=build_dataset.DATASET_NAME,
        description=build_dataset.DATASET_DESCRIPTION,
    )
    assert fake_client.create_dataset_item.call_count == 2

    first_call = fake_client.create_dataset_item.call_args_list[0]
    assert first_call.kwargs["dataset_name"] == build_dataset.DATASET_NAME
    assert first_call.kwargs["input"] == "Q1"
    assert first_call.kwargs["expected_output"] == "A1"
    assert first_call.kwargs["metadata"] == {"expected_source": "src1"}

    fake_client.flush.assert_called_once()


def test_sync_empty_items_still_creates_dataset(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(build_dataset, "Langfuse", lambda: fake_client)

    build_dataset.sync_dataset([])

    fake_client.create_dataset.assert_called_once()
    fake_client.create_dataset_item.assert_not_called()
    fake_client.flush.assert_called_once()


def test_sync_propagates_client_errors(monkeypatch):
    fake_client = MagicMock()
    fake_client.create_dataset_item.side_effect = RuntimeError("network down")
    monkeypatch.setattr(build_dataset, "Langfuse", lambda: fake_client)

    items = [{"question": "Q", "ground_truth": "A", "expected_source": "s"}]
    with pytest.raises(RuntimeError, match="network down"):
        build_dataset.sync_dataset(items)
