from pathlib import Path

from src.storage.resolver import resolve_storage_backend


def sync_checkpoint_dir_to_remote(config: dict, local_dir: str, remote_prefix: str) -> None:
    backend = resolve_storage_backend(config)
    local_path = Path(local_dir)

    if not local_path.exists():
        return

    for file_path in local_path.glob("*"):
        if file_path.is_file():
            remote_path = f"{remote_prefix.rstrip('/')}/{file_path.name}"
            backend.upload_file(str(file_path), remote_path)


def sync_checkpoint_dir_from_remote(config: dict, remote_prefix: str, local_dir: str) -> None:
    backend = resolve_storage_backend(config)
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        remote_files = backend.list_files(remote_prefix)
    except Exception:
        return

    for remote_path in remote_files:
        filename = remote_path.rstrip("/").split("/")[-1]
        backend.download_file(remote_path, str(local_path / filename))