from typing import Any, Dict, List

from src.storage.base import StorageBackend


class ObjectStoreBackend(StorageBackend):
    """
    Placeholder for future S3/MinIO-compatible implementation.
    This keeps the interface stable while local backend remains the active implementation.
    """

    def __init__(self, endpoint: str = "", bucket: str = "", prefix: str = ""):
        self.endpoint = endpoint
        self.bucket = bucket
        self.prefix = prefix

    def exists(self, path: str) -> bool:
        raise NotImplementedError("Object storage backend not implemented yet.")

    def makedirs(self, path: str) -> None:
        raise NotImplementedError("Object storage backend not implemented yet.")

    def save_text(self, path: str, content: str) -> None:
        raise NotImplementedError("Object storage backend not implemented yet.")

    def save_json(self, path: str, data: Dict[str, Any]) -> None:
        raise NotImplementedError("Object storage backend not implemented yet.")

    def load_json(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError("Object storage backend not implemented yet.")

    def list_files(self, path: str) -> List[str]:
        raise NotImplementedError("Object storage backend not implemented yet.")