from pathlib import Path
from typing import Any, Dict, List

from src.storage.resolver import resolve_storage_backend


def join_artifact_path(base: str, filename: str) -> str:
    if "://" in base:
        return f"{base.rstrip('/')}/{filename}"
    return str(Path(base) / filename)

def save_metrics_artifact(config: dict, metrics: Dict[str, Any], filename: str = "metrics.json") -> str:
    backend = resolve_storage_backend(config)
    artifact_dir = config["output"]["artifact_dir"]
    backend.makedirs(artifact_dir)

    output_path = join_artifact_path(artifact_dir,filename)
    backend.save_json(output_path, metrics)
    return output_path


def save_history_artifact(config: dict, history: List[Dict[str, Any]], filename: str = "history.json") -> str:
    backend = resolve_storage_backend(config)
    artifact_dir = config["output"]["artifact_dir"]
    backend.makedirs(artifact_dir)

    output_path = join_artifact_path(artifact_dir, filename)
    backend.save_json(output_path, {"history": history})
    return output_path


def save_summary_artifact(config: dict, summary: Dict[str, Any], filename: str = "training_summary.txt") -> str:
    backend = resolve_storage_backend(config)
    artifact_dir = config["output"]["artifact_dir"]
    backend.makedirs(artifact_dir)

    output_path = join_artifact_path(artifact_dir, filename)
    content = "\n".join([f"{key}: {value}" for key, value in summary.items()])
    backend.save_text(output_path, content)
    return output_path
