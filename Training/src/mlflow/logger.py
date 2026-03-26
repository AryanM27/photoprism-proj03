import os
from pathlib import Path
from typing import Any, Dict

import mlflow


def configure_mlflow() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./Training/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value

    return items


def log_config_params(config: Dict[str, Any]) -> None:
    flat_config = flatten_dict(config)

    for key, value in flat_config.items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))


def log_metrics(metrics: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)


def log_artifact_if_exists(path: str) -> None:
    artifact_path = Path(path)
    if artifact_path.exists():
        mlflow.log_artifact(str(artifact_path))


def start_run(experiment_name: str, run_name: str = None):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)