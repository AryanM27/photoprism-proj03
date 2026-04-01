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
        return ObjectStoreBackend(
            auth_url=os.getenv("OS_AUTH_URL", storage_config.get("auth_url", "")),
            username=os.getenv("OS_USERNAME", storage_config.get("username", "")),
            auth_type=os.getenv("OS_AUTH_TYPE", storage_config.get("auth_type", "")),
            token=os.getenv("OS_TOKEN", storage_config.get("token", "")),
            storage_url=os.getenv("OS_STORAGE_URL", storage_config.get("storage_url", "")),
            password=os.getenv("OS_PASSWORD", storage_config.get("password", "")),
            project_name=os.getenv("OS_PROJECT_NAME", storage_config.get("project_name", "")),
            project_id=os.getenv("OS_PROJECT_ID", storage_config.get("project_id", "")),
            user_domain_name=os.getenv("OS_USER_DOMAIN_NAME", storage_config.get("user_domain_name", "Default")),
            project_domain_name=os.getenv("OS_PROJECT_DOMAIN_NAME", storage_config.get("project_domain_name", "Default")),
            region_name=os.getenv("OS_REGION_NAME", os.getenv("OS_REGION",storage_config.get("region_name", "CHI@TACC"))),
        )

    raise ValueError(f"Unsupported storage backend: {backend_type}")