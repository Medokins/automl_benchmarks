"""Coordinates manifest loading, pipeline submissions, waits, and CSV export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from automl_benchmark.config_loader import load_merged_benchmark_config
from automl_benchmark.kfp_client import create_kfp_client
from automl_benchmark.manifest import load_dataset_entries
from automl_benchmark.pipeline_params import build_pipeline_arguments
from automl_benchmark.pipeline_run import extract_run_id, submit_pipeline_package, wait_for_terminal_run
from automl_benchmark.result_rows import (
    base_row_for_dataset,
    completed_row,
    dry_run_row,
    run_name_for_dataset,
    submit_error_row,
    timeout_row,
)
from automl_benchmark.results_csv import write_results_csv
from automl_benchmark.run_state import is_success_state
from automl_benchmark.settings import benchmark_settings_from_config, BenchmarkSettings

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """High-level benchmark run: one pipeline run per dataset entry, then aggregate CSV."""

    def __init__(self, config_path: Path, credentials_ini_path: Path | None = None) -> None:
        self.config_path = config_path.resolve()
        self.credentials_ini_path = credentials_ini_path

    def load_config_and_datasets(self) -> tuple[dict[str, Any], BenchmarkSettings, list[dict[str, Any]]]:
        cfg, config_dir = load_merged_benchmark_config(self.config_path, self.credentials_ini_path)
        settings = benchmark_settings_from_config(cfg, config_dir)
        datasets = load_dataset_entries(cfg, config_dir)
        return cfg, settings, datasets

    def execute(
        self,
        *,
        output_csv: Path,
        dry_run: bool = False,
        fail_fast: bool = False,
    ) -> int:
        try:
            cfg, settings, datasets = self.load_config_and_datasets()
        except Exception as e:
            logger.error("%s", e)
            return 1

        if not settings.pipeline_yaml.is_file():
            logger.error("Pipeline package not found: %s", settings.pipeline_yaml)
            return 1

        rows: list[dict[str, Any]] = []
        client = None
        if not dry_run:
            try:
                client = create_kfp_client(cfg)
            except Exception as e:
                logger.error("KFP client failed: %s", e)
                return 1

        for i, ds in enumerate(datasets):
            ds_id = str(ds.get("id", ds.get("name", f"dataset_{i}")))
            file_key = ds.get("train_data_file_key")
            label_column = ds.get("label_column")
            task_type = ds.get("task_type")
            if not file_key or not label_column or not task_type:
                logger.error("Dataset %s missing train_data_file_key, label_column, or task_type", ds_id)
                if fail_fast:
                    return 1
                continue

            arguments = build_pipeline_arguments(ds, settings)
            run_name = run_name_for_dataset(settings.run_name_prefix, ds_id)
            base = base_row_for_dataset(ds, i, run_name, settings.top_n)

            if dry_run:
                rows.append(dry_run_row(base, arguments))
                logger.info("DRY_RUN %s -> %s", ds_id, arguments)
                continue

            assert client is not None
            try:
                run_result = submit_pipeline_package(
                    client,
                    pipeline_file=str(settings.pipeline_yaml),
                    arguments=arguments,
                    run_name=run_name,
                    experiment_name=settings.experiment_name,
                    enable_caching=settings.enable_caching,
                )
                rid = extract_run_id(run_result)
                logger.info("Started run_id=%s dataset=%s", rid, ds_id)

                detail, timed_out = wait_for_terminal_run(
                    client,
                    rid,
                    timeout_seconds=settings.timeout_seconds,
                    poll_interval_seconds=settings.poll_interval_seconds,
                )
                if timed_out:
                    rows.append(timeout_row(base, rid, settings.timeout_seconds))
                    logger.error("Timeout waiting for run %s", rid)
                    if fail_fast:
                        break
                    continue

                if detail is None:
                    detail = client.get_run(rid)
                rows.append(completed_row(base, rid, detail))

                state = rows[-1].get("state", "")
                if not is_success_state(str(state)) and fail_fast:
                    logger.error("Run %s ended with state=%s", rid, state)
                    break

            except Exception as e:
                logger.exception("Run failed for dataset %s", ds_id)
                rows.append(submit_error_row(base, str(e)))
                if fail_fast:
                    break

        write_results_csv(rows, output_csv)
        logger.info("Wrote %d row(s) to %s", len(rows), output_csv)
        return 0
