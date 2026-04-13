import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _atomic_save_torch(state: Dict[str, Any], final_path: Path) -> None:
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(state, temp_path)
    temp_path.replace(final_path)


def _atomic_save_json(data: Dict[str, Any], final_path: Path) -> None:
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_path.replace(final_path)


def build_checkpoint_dir(
    checkpoint_root: str,
    task: str,
    model_family: str,
    model_version: str,
) -> str:
    """
    sammple: Training/artifacts/checkpoints/aesthetic/linear_head/aesthetic-linear-v1
    """
    path = Path(checkpoint_root) / task / model_family / model_version
    print(path)
    return str(path)


def build_checkpoint_metadata(
    task: str,
    model_family: str,
    model_version: str,
    epoch: int,
    global_step: int,
    metric_name: Optional[str],
    metric_value: Optional[float],
    dataset_version: Optional[str],
    config_path: Optional[str],
    chunk_start_index: Optional[int] = None,
    chunk_max_records: Optional[int] = None,
    chunk_subset_seed: Optional[int] = None,
    next_start_index: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "saved_at": _timestamp(),
        "task": task,
        "model_family": model_family,
        "model_version": model_version,
        "epoch": epoch,
        "global_step": global_step,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "dataset_version": dataset_version,
        "config_path": config_path,
        "chunk_start_index": chunk_start_index,
        "chunk_max_records": chunk_max_records,
        "chunk_subset_seed": chunk_subset_seed,
        "next_start_index": next_start_index,
    }


def save_checkpoint(
    checkpoint_dir: str,
    state: Dict[str, Any],
    task: str,
    model_family: str,
    model_version: str,
    epoch: int,
    global_step: int,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
    dataset_version: Optional[str] = None,
    config_path: Optional[str] = None,
    is_best: bool = False,
    save_epoch_copy: bool = True,
    chunk_start_index: Optional[int] = None,
    chunk_max_records: Optional[int] = None,
    chunk_subset_seed: Optional[int] = None,
    next_start_index: Optional[int] = None,
) -> Dict[str, str]:
    checkpoint_root = Path(checkpoint_dir)
    _ensure_dir(checkpoint_root)

    latest_ckpt = checkpoint_root / "latest.pt"
    latest_meta = checkpoint_root / "latest_metadata.json"

    metadata = build_checkpoint_metadata(
        task=task,
        model_family=model_family,
        model_version=model_version,
        epoch=epoch,
        global_step=global_step,
        metric_name=metric_name,
        metric_value=metric_value,
        dataset_version=dataset_version,
        config_path=config_path,
        chunk_start_index=chunk_start_index,
        chunk_max_records=chunk_max_records,
        chunk_subset_seed=chunk_subset_seed,
        next_start_index=next_start_index,
    )

    _atomic_save_torch(state, latest_ckpt)
    _atomic_save_json(metadata, latest_meta)

    saved_paths = {
        "latest_checkpoint": str(latest_ckpt),
        "latest_metadata": str(latest_meta),
    }

    if save_epoch_copy:
        epoch_ckpt = checkpoint_root / f"epoch_{epoch}.pt"
        epoch_meta = checkpoint_root / f"epoch_{epoch}_metadata.json"

        _atomic_save_torch(state, epoch_ckpt)
        _atomic_save_json(metadata, epoch_meta)

        saved_paths["epoch_checkpoint"] = str(epoch_ckpt)
        saved_paths["epoch_metadata"] = str(epoch_meta)

    if is_best:
        best_ckpt = checkpoint_root / "best.pt"
        best_meta = checkpoint_root / "best_metadata.json"

        shutil.copy2(latest_ckpt, best_ckpt)
        shutil.copy2(latest_meta, best_meta)

        saved_paths["best_checkpoint"] = str(best_ckpt)
        saved_paths["best_metadata"] = str(best_meta)

    return saved_paths


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=map_location)


# def load_latest_checkpoint(checkpoint_dir: str, map_location: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     checkpoint_root = Path(checkpoint_dir)
#     latest_ckpt = checkpoint_root / "latest.pt"
#     latest_meta = checkpoint_root / "latest_metadata.json"

#     if not latest_ckpt.exists():
#         raise FileNotFoundError(f"Latest checkpoint not found: {latest_ckpt}")

#     if not latest_meta.exists():
#         raise FileNotFoundError(f"Latest checkpoint metadata not found: {latest_meta}")

#     state = torch.load(latest_ckpt, map_location=map_location)

#     with latest_meta.open("r", encoding="utf-8") as f:
#         metadata = json.load(f)

#     return state, metadata

def load_latest_checkpoint(checkpoint_dir: str, map_location: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint_root = Path(checkpoint_dir)
    latest_ckpt = checkpoint_root / "latest.pt"
    latest_meta = checkpoint_root / "latest_metadata.json"

    if not latest_ckpt.exists():
        raise FileNotFoundError(f"Latest checkpoint not found: {latest_ckpt}")

    if not latest_meta.exists():
        raise FileNotFoundError(f"Latest checkpoint metadata not found: {latest_meta}")

    # Always load checkpoint tensors onto CPU first.
    # This is safer on ROCm than restoring directly onto the GPU.
    state = torch.load(latest_ckpt, map_location="cpu")

    with latest_meta.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return state, metadata


def checkpoint_exists(checkpoint_dir: str) -> bool:
    checkpoint_root = Path(checkpoint_dir)
    return (checkpoint_root / "latest.pt").exists()