# AutoML benchmarks (KFP orchestrator)

This repository helps you run an AutoML tabular training **Kubeflow Pipelines** (KFP) workflow across many datasets, poll runs to completion, and write a single **CSV** of results. Configuration splits **non-secret layout** (YAML) from **cluster and storage identity** (INI).

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-benchmark.txt
```

Run scripts from the repo root so paths such as `config/benchmark.yaml` resolve correctly.

## Before running experiments

Complete the checklist below; placeholders in the template commands section mirror these items.

1. **`config/credentials.ini`** (copy from `templates/credentials.example.ini`)
   - **`[kfp]`**: `host` (Data Science Pipelines API URL), `namespace`, and authentication (`token`, or `token_file`, or `KFP_API_TOKEN` / `token_env`).
   - **`[storage]`**: `train_data_bucket_name` where training CSVs live (object keys must match your manifest).
   - **`[pipeline]`**: `train_data_secret_name` — Kubernetes secret in the project that pipeline pods use for S3 (or equivalent) access.
   - **`[s3]`**: Endpoint and keys for your own records / secret creation; values are not sent to the cluster by this tool.

2. **`config/benchmark.yaml`** (copy from `templates/benchmark.example.yaml`)
   - **`pipeline.package_path`**: compiled pipeline IR YAML (default when the file lives under `config/` is `../pipelines/pipeline.yaml`).
   - **`dataset_manifest_path`**: manifest of datasets (path relative to this YAML’s directory).
   - **`run`**: optional tuning (`top_n`, timeouts, caching, run name prefix).

3. **Dataset manifest** (YAML list of `id`, `name`, `train_data_file_key`, `label_column`, `task_type`). Start from `templates/dataset_manifest.example.yaml` or generate one (see below). Ensure objects exist in the bucket at the keys you declare.

4. **Pipeline package**: `pipelines/pipeline.yaml` (or the path you set in `pipeline.package_path`) must exist and match what your cluster expects.

## Template commands

Replace `FILL_*` values with your environment. Run from the repository root.

```bash
# One-time config from templates
cp templates/benchmark.example.yaml config/benchmark.yaml
cp templates/credentials.example.ini config/credentials.ini
cp templates/dataset_manifest.example.yaml config/dataset_manifest.example.yaml
# Edit benchmark.yaml and credentials.ini; adjust manifest paths or contents as needed.

# Optional: verbose logging, custom paths
export BENCHMARK_CONFIG_PATH="FILL_PATH_TO/benchmark.yaml"
export BENCHMARK_CREDENTIALS_PATH="FILL_PATH_TO/credentials.ini"

# Validate config wiring without calling KFP
python scripts/benchmark_orchestrator.py --dry-run -v

# Run the benchmark suite; write aggregated CSV
python scripts/benchmark_orchestrator.py --output results/benchmark_runs.csv

# Same entry point with explicit files
python scripts/benchmark_orchestrator.py \
  --config config/benchmark.yaml \
  --credentials config/credentials.ini \
  --output results/benchmark_runs.csv

# Stop on first pipeline failure
python scripts/benchmark_orchestrator.py --fail-fast --output results/benchmark_runs.csv
```

Optional: build a long-form summary from the runs CSV (see `scripts/summarize_benchmark_results.py --help`).

```bash
python scripts/summarize_benchmark_results.py results/benchmark_runs.csv \
  -o results/benchmark_summary.csv
```

## Optional: local datasets and manifest generation

Not required if you already upload CSVs and maintain a manifest.

```bash
pip install scikit-learn openml
python scripts/download_initial_datasets.py --out downloaded_datasets

pip install pandas pyyaml
python scripts/generate_dataset_manifest.py --root downloaded_datasets \
  --s3-key-prefix FILL_S3_PREFIX > config/dataset_manifest.generated.yaml
```

Then point `dataset_manifest_path` in `config/benchmark.yaml` at the generated file (or merge entries into your main manifest) and ensure the same keys exist in your training data bucket.
