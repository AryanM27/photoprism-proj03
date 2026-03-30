import json
from pathlib import Path
from typing import Any, Dict, List

from src.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def makedirs(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def save_text(self, path: str, content: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    def save_json(self, path: str, data: Dict[str, Any]) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str) -> Dict[str, Any]:
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_files(self, path: str) -> List[str]:
        target = Path(path)
        if not target.exists():
            return []
        return [str(p) for p in target.iterdir()]