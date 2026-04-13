from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.aesthetic.model import build_aesthetic_model
from src.common.checkpointing import build_checkpoint_dir
from src.datasets.aesthetic_dataset import AestheticDataset
from src.datasets.uri_resolver import cache_manifest_from_uri

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def _resolve_manifest_ref_for_split(config: dict, split: str) -> str:
    dataset_cfg = config.get("dataset", {})
    eval_cfg = config.get("evaluation", {})

    base_manifest_ref = dataset_cfg.get("manifest_uri") or dataset_cfg.get("manifest_path")
    if not base_manifest_ref:
        raise ValueError("No dataset manifest_uri or manifest_path found in config")

    test_split_name = eval_cfg.get("test_split_name", "test")

    if split == test_split_name:
        explicit_test_ref = eval_cfg.get("test_manifest_uri") or eval_cfg.get("test_manifest_path")
        if explicit_test_ref:
            return explicit_test_ref

        test_manifest_filename = eval_cfg.get("test_manifest_filename", "train_test.jsonl")

        if str(base_manifest_ref).startswith("swift://"):
            return str(base_manifest_ref).rsplit("/", 1)[0] + f"/{test_manifest_filename}"

        return str(Path(base_manifest_ref).with_name(test_manifest_filename))

    return base_manifest_ref

def _resolve_checkpoint_path(config: dict) -> str:
    checkpoint_dir = build_checkpoint_dir(
        checkpoint_root=config["checkpoint"]["root_dir"],
        task=config["task"],
        model_family=config["model"]["family"],
        model_version=config["model"]["version"],
    )

    best_path = Path(checkpoint_dir) / "best.pt"
    latest_path = Path(checkpoint_dir) / "latest.pt"

    if best_path.exists():
        return str(best_path)
    if latest_path.exists():
        return str(latest_path)

    raise FileNotFoundError(f"No best.pt or latest.pt found in {checkpoint_dir}")

def evaluate_model(model, dataloader, device) -> Dict[str, float]:
    model.eval()

    total_mse_sum = 0.0
    total_abs_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image_tensor"].to(device)
            targets = batch["aesthetic_score"].to(device).float()

            preds = model(images).squeeze()

            mse_sum = F.mse_loss(preds, targets, reduction="sum")
            abs_sum = torch.abs(preds - targets).sum()

            total_mse_sum += mse_sum.item()
            total_abs_error += abs_sum.item()
            total_samples += targets.size(0)

    if total_samples == 0:
        return {
            "mse_loss": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "num_samples": 0,
        }

    mse = total_mse_sum / total_samples
    mae = total_abs_error / total_samples
    rmse = mse ** 0.5

    return {
        "mse_loss": mse,
        "mae": mae,
        "rmse": rmse,
        "num_samples": total_samples,
    }

def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}

def run_aesthetic_evaluation_for_split(
    config: dict,
    split: str = "val",
    checkpoint_path: Optional[str] = None,
) -> Dict:
    device = get_device(config["runtime"]["device"])

    manifest_ref = _resolve_manifest_ref_for_split(config, split)
    manifest_path = cache_manifest_from_uri(config, manifest_ref)

    eval_cfg = config.get("evaluation", {})
    dataset_cfg = config["dataset"]

    dataset = AestheticDataset(
        manifest_path=manifest_path,
        config=config,
        image_size=config["model"]["image_size"],
        split=split,
        start_index=dataset_cfg.get("start_index", 0),
        max_records=eval_cfg.get("max_records", 2500),
        subset_seed=dataset_cfg.get("subset_seed", None),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_cfg.get("batch_size", config["training"]["batch_size"]),
        shuffle=False,
        num_workers=config["runtime"].get("num_workers", 0),
    )

    model = build_aesthetic_model(config).to(device)

    if checkpoint_path is None:
        checkpoint_path = _resolve_checkpoint_path(config)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    model.eval()

    metrics = evaluate_model(model, dataloader, device)
    prefixed_metrics = _prefix_metrics(metrics, split)

    return {
        "candidate_name": config.get("candidate_name", "unknown_candidate"),
        "model_type": config["model"]["type"],
        "model_version": config["model"]["version"],
        "dataset_version": config["dataset"]["dataset_version"],
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "evaluation_split": split,
        "manifest_ref": manifest_ref,
        "manifest_path": manifest_path,
        **prefixed_metrics,
    }