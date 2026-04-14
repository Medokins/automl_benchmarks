"""Resolved benchmark settings from raw config dict."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from automl_benchmark.paths import resolve_under


@dataclass(frozen=True)
class BenchmarkSettings:
    config_dir: Path
    pipeline_yaml: Path
    train_data_secret_name: str
    train_data_bucket_name: str
    top_n: int
    poll_interval_seconds: float
    timeout_seconds: float
    enable_caching: bool
    experiment_name: str
    run_name_prefix: str


def benchmark_settings_from_config(cfg: dict[str, Any], config_dir: Path) -> BenchmarkSettings:
    pipeline_cfg = cfg.get("pipeline") or {}
    storage_cfg = cfg.get("storage") or {}
    run_cfg = cfg.get("run") or {}
    kfp_cfg = cfg.get("kfp") or {}

    pipeline_yaml = resolve_under(config_dir, str(pipeline_cfg.get("package_path", "../pipelines/autogluon-tabular-training-pipeline.yaml")))
    secret = pipeline_cfg.get("train_data_secret_name")
    bucket = storage_cfg.get("train_data_bucket_name")
    if not secret or not bucket:
        raise ValueError(
            "pipeline.train_data_secret_name and storage.train_data_bucket_name are required "
            "(set in credentials.ini only, not in benchmark.yaml)"
        )

    return BenchmarkSettings(
        config_dir=config_dir,
        pipeline_yaml=pipeline_yaml,
        train_data_secret_name=str(secret),
        train_data_bucket_name=str(bucket),
        top_n=int(run_cfg.get("top_n", 3)),
        poll_interval_seconds=float(run_cfg.get("poll_interval_seconds", 30)),
        timeout_seconds=float(run_cfg.get("timeout_seconds", 86400)),
        enable_caching=bool(run_cfg.get("enable_caching", False)),
        experiment_name=str(kfp_cfg.get("experiment_name", "autogluon-benchmark")),
        run_name_prefix=str(run_cfg.get("run_name_prefix", "benchmark")),
    )
