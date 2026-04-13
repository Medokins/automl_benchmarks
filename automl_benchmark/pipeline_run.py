"""Submit a compiled pipeline package and wait until the run reaches a terminal state."""

from __future__ import annotations

import time
from typing import Any

from automl_benchmark.run_state import is_terminal_state, read_run_state, unwrap_run_from_get_run


def submit_pipeline_package(
    client: Any,
    *,
    pipeline_file: str,
    arguments: dict[str, Any],
    run_name: str,
    experiment_name: str,
    enable_caching: bool,
) -> Any:
    try:
        return client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=run_name,
            experiment_name=experiment_name,
            enable_caching=enable_caching,
        )
    except TypeError:
        return client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=run_name,
            experiment_name=experiment_name,
        )


def extract_run_id(run_result: Any) -> str:
    rid = getattr(run_result, "run_id", None)
    if rid is None and isinstance(run_result, dict):
        rid = run_result.get("run_id")
    return str(rid) if rid is not None else ""


def wait_for_terminal_run(
    client: Any,
    run_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[Any | None, bool]:
    deadline = time.monotonic() + timeout_seconds
    detail = None
    while time.monotonic() < deadline:
        detail = client.get_run(run_id)
        run_obj = unwrap_run_from_get_run(detail)
        st = read_run_state(run_obj).upper()
        if is_terminal_state(st):
            return detail, False
        time.sleep(poll_interval_seconds)
    return detail, True
