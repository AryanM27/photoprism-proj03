# from pathlib import Path
# import yaml


# def load_config(config_path: str) -> dict:
#     path= Path(config_path)
#     if not path.exists():
#         raise FileNotFoundError(f"Config file not found: {path}")

#     with path.open("r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)

#     if config is None:
#         raise ValueError(f"Config file is empty: {path}")

#     return config

from pathlib import Path
import os
import yaml

from src.common.paths import resolve_training_path


PATH_KEYS = {
    ("dataset", "manifest_path"),
    ("checkpoint", "root_dir"),
    ("output", "artifact_dir"),
}


def _apply_env_overrides(config: dict) -> dict:
    """
    Override selected runtime/storage values from environment variables.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    training_env = os.getenv("TRAINING_ENV")
    storage_backend = os.getenv("STORAGE_BACKEND")
    object_store_endpoint = os.getenv("OBJECT_STORE_ENDPOINT")
    object_store_bucket = os.getenv("OBJECT_STORE_BUCKET")
    object_store_prefix = os.getenv("OBJECT_STORE_PREFIX")
    object_store_access_key = os.getenv("OBJECT_STORE_ACCESS_KEY")
    object_store_secret_key = os.getenv("OBJECT_STORE_SECRET_KEY")

    if training_env:
        config.setdefault("runtime", {})
        config["runtime"]["training_env"] = training_env

    if mlflow_uri:
        config.setdefault("runtime", {})
        config["runtime"]["mlflow_tracking_uri"] = mlflow_uri

    if storage_backend:
        config.setdefault("storage", {})
        config["storage"]["backend"] = storage_backend

    if object_store_endpoint:
        config.setdefault("storage", {})
        config["storage"]["endpoint"] = object_store_endpoint

    if object_store_bucket:
        config.setdefault("storage", {})
        config["storage"]["bucket"] = object_store_bucket

    if object_store_prefix:
        config.setdefault("storage", {})
        config["storage"]["prefix"] = object_store_prefix

    if object_store_access_key:
        config.setdefault("storage", {})
        config["storage"]["access_key"] = object_store_access_key

    if object_store_secret_key:
        config.setdefault("storage", {})
        config["storage"]["secret_key"] = object_store_secret_key

    return config


def _resolve_known_paths(config: dict) -> dict:
    for section, key in PATH_KEYS:
        if section in config and key in config[section]:
            config[section][key] = resolve_training_path(config[section][key])

    return config


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.is_absolute():
        path = Path(resolve_training_path(config_path))

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    config = _apply_env_overrides(config)
    config = _resolve_known_paths(config)

    config["_resolved_config_path"] = str(path)
    return config
