#!/usr/bin/env python3
"""
Submit autogluon-tabular-training-pipeline runs (KFP / OpenShift Data Science Pipelines)
for each dataset in a manifest, wait for completion, and write a single CSV summary.

Implementation lives in the ``automl_benchmark`` package; this file is a thin CLI wrapper.

Configuration:
  - YAML ($BENCHMARK_CONFIG_PATH / config/benchmark.yaml): pipeline.package_path (default
    ``../pipelines/pipeline.yaml`` when this file is under ``config/``), run tuning, manifest only.
  - credentials.ini (required): kfp host/namespace/token, bucket, pipeline secret name, [s3] for your records.
    Use config/credentials.ini, $BENCHMARK_CREDENTIALS_PATH, or ``--credentials PATH``.

Usage:
  pip install -r requirements-benchmark.txt
  cp config/benchmark.example.yaml config/benchmark.yaml
  cp config/credentials.example.ini config/credentials.ini   # all cluster + storage identity here
  python scripts/benchmark_orchestrator.py --output results/benchmark_runs.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from automl_benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
