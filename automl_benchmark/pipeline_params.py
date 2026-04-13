"""Map dataset manifest entries to pipeline argument dicts."""

from __future__ import annotations

from typing import Any

from automl_benchmark.settings import BenchmarkSettings


def build_pipeline_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    return {
        "train_data_secret_name": settings.train_data_secret_name,
        "train_data_bucket_name": settings.train_data_bucket_name,
        "train_data_file_key": str(dataset["train_data_file_key"]),
        "label_column": str(dataset["label_column"]),
        "task_type": str(dataset["task_type"]),
        "top_n": settings.top_n,
    }
