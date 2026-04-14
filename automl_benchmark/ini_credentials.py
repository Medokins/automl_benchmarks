"""Load optional credentials and cluster settings from a .ini file (configparser)."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def load_credentials_ini(path: Path) -> dict[str, Any]:
    """
    Parse INI into a dict shaped like benchmark.yaml (nested sections).

    Recognized sections (case-insensitive): kfp, storage, pipeline, s3, run.
    """
    cp = configparser.ConfigParser(interpolation=None)
    if not path.is_file():
        raise FileNotFoundError(path)
    read = cp.read(path)
    if not read:
        raise OSError(f"Could not read INI: {path}")

    by_section: dict[str, dict[str, str]] = {}
    for sec in cp.sections():
        by_section[sec.lower()] = {k: v.strip() for k, v in cp.items(sec)}

    out: dict[str, Any] = {}

    if "kfp" in by_section:
        kfp: dict[str, Any] = {}
        raw = by_section["kfp"]
        for key in ("host", "namespace", "token", "token_file", "token_env", "experiment_name"):
            if key in raw and raw[key] != "":
                kfp[key] = raw[key]
        if kfp:
            out["kfp"] = kfp

    if "storage" in by_section:
        st: dict[str, Any] = {}
        raw = by_section["storage"]
        if raw.get("train_data_bucket_name"):
            st["train_data_bucket_name"] = raw["train_data_bucket_name"]
        if st:
            out["storage"] = st

    if "pipeline" in by_section:
        pl: dict[str, Any] = {}
        raw = by_section["pipeline"]
        for key in ("train_data_secret_name", "package_path", "timeseries_package_path"):
            if raw.get(key):
                pl[key] = raw[key]
        if pl:
            out["pipeline"] = pl

    if "s3" in by_section:
        s3: dict[str, Any] = {}
        raw = by_section["s3"]
        endpoint = raw.get("endpoint") or raw.get("aws_s3_endpoint")
        if endpoint:
            s3["endpoint"] = endpoint
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_default_region"):
            if raw.get(key):
                s3[key] = raw[key]
        if s3:
            out["s3"] = s3

    if "run" in by_section:
        rn: dict[str, Any] = {}
        raw = by_section["run"]
        if raw.get("top_n"):
            rn["top_n"] = int(raw["top_n"])
        if raw.get("poll_interval_seconds"):
            rn["poll_interval_seconds"] = float(raw["poll_interval_seconds"])
        if raw.get("timeout_seconds"):
            rn["timeout_seconds"] = float(raw["timeout_seconds"])
        if "enable_caching" in raw:
            rn["enable_caching"] = _truthy(raw["enable_caching"])
        if raw.get("run_name_prefix"):
            rn["run_name_prefix"] = raw["run_name_prefix"]
        if rn:
            out["run"] = rn

    return out
