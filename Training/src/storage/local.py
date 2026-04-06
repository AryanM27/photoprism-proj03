import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from src.storage.base import StorageBackend

class LocalStorageBackend(StorageBackend):
    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def makedirs(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def save_text(self, path: str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def save_json(self, path: str, data: Dict[str, Any]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_files(self, path: str) -> List[str]:
        p = Path(path)
        if not p.exists():
            return []
        return [str(x) for x in p.iterdir()]

    def upload_file(self, local_path: str, remote_path: str) -> None:
        target = Path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)

    def download_file(self, remote_path: str, local_path: str) -> None:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)