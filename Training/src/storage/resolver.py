import os

from src.storage.local import LocalStorageBackend
from src.storage.object_store import ObjectStoreBackend


def resolve_storage_backend(config: dict):
    storage_config = config.get("storage", {})
    backend_type = os.getenv("STORAGE_BACKEND", storage_config.get("backend", "local"))

    if backend_type == "local":
        return LocalStorageBackend()

    if backend_type == "object_store":
        return ObjectStoreBackend()

    raise ValueError(f"Unsupported storage backend: {backend_type}")