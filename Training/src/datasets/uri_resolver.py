import hashlib
from pathlib import Path
from urllib.parse import urlparse

from src.common.paths import MANIFEST_CACHE_DIR, IMAGE_CACHE_DIR
from src.storage.resolver import resolve_storage_backend

def _cache_name(uri: str) -> str:
    return hashlib.sha256(uri.encode("utf-8")).hexdigest()


def cache_manifest_from_uri(config: dict, manifest_uri: str) -> str:
    parsed = urlparse(manifest_uri)

    if parsed.scheme in ("", "file"):
        return parsed.path if parsed.scheme == "file" else manifest_uri

    if parsed.scheme == "swift":
        backend = resolve_storage_backend(config)
        suffix = Path(parsed.path).suffix or ".jsonl"
        local_path = MANIFEST_CACHE_DIR / f"{_cache_name(manifest_uri)}{suffix}"
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            backend.download_file(manifest_uri, str(local_path))
        return str(local_path)

    raise ValueError(f"Unsupported manifest URI: {manifest_uri}")


def cache_image_from_uri(config: dict, image_uri: str) -> str:
    parsed = urlparse(image_uri)

    if parsed.scheme in ("", "file"):
        return parsed.path if parsed.scheme == "file" else image_uri

    if parsed.scheme == "swift":
        backend = resolve_storage_backend(config)
        suffix = Path(parsed.path).suffix or ".jpg"
        local_path = IMAGE_CACHE_DIR / f"{_cache_name(image_uri)}{suffix}"
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            backend.download_file(image_uri, str(local_path))
        return str(local_path)

    raise ValueError(f"Unsupported image URI: {image_uri}")