# from src.storage.local import LocalStorageBackend
# from src.storage.object_store import ObjectStoreBackend


# def resolve_storage_backend(config: dict):
#     storage_config = config.get("storage", {})
#     backend_type = storage_config.get("backend", "local")

#     if backend_type == "local":
#         return LocalStorageBackend()

#     if backend_type == "object_store":
#         return ObjectStoreBackend(
#             endpoint=storage_config.get("endpoint", ""),
#             bucket=storage_config.get("bucket", ""),
#             prefix=storage_config.get("prefix", ""),
#         )

#     raise ValueError(f"Unsupported storage backend: {backend_type}")


# import os

# from src.storage.local import LocalStorageBackend
# from src.storage.object_store import ObjectStoreBackend


# def resolve_storage_backend(config: dict):
#     storage_config = config.get("storage", {})

#     backend_type = os.getenv("STORAGE_BACKEND", storage_config.get("backend", "local"))

#     if backend_type == "local":
#         return LocalStorageBackend()

#     if backend_type == "object_store":
#         endpoint = os.getenv("OBJECT_STORE_ENDPOINT", storage_config.get("endpoint", ""))
#         bucket = os.getenv("OBJECT_STORE_BUCKET", storage_config.get("bucket", ""))
#         prefix = os.getenv("OBJECT_STORE_PREFIX", storage_config.get("prefix", ""))

#         return ObjectStoreBackend(
#             endpoint=endpoint,
#             bucket=bucket,
#             prefix=prefix,
#         )

#     raise ValueError(f"Unsupported storage backend: {backend_type}")


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