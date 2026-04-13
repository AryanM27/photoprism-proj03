from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.datasets.semantic_dataset import SemanticRetrievalDataset
from src.datasets.uri_resolver import cache_manifest_from_uri
from src.semantic.model import build_semantic_model, build_text_features
from src.common.checkpointing import build_checkpoint_dir, load_latest_checkpoint
from src.storage.checkpoint_sync import sync_checkpoint_dir_from_remote

def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image_ids": [item["image_id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "image_tensors": torch.stack([item["image_tensor"] for item in batch]),
    }

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def _resolve_manifest_ref_for_split(config: dict, split: Optional[str]) -> str:
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

def _resolve_remote_checkpoint_prefix(config: dict) -> Optional[str]:
    artifact_dir = config["output"].get("artifact_dir")
    if artifact_dir is None or not artifact_dir.startswith("swift://"):
        return None

    task = config["task"]
    model_family = config["model"]["family"]
    model_version = config["model"]["version"]

    container = artifact_dir.replace("swift://", "", 1).split("/", 1)[0]

    return f"swift://{container}/checkpoints/{task}/{model_version}"

def _resolve_checkpoint_path(config: dict) -> Optional[str]:
    checkpoint_dir = build_checkpoint_dir(
        checkpoint_root=config["checkpoint"]["root_dir"],
        task=config["task"],
        model_family=config["model"]["family"],
        model_version=config["model"]["version"],
    )

    remote_checkpoint_prefix = _resolve_remote_checkpoint_prefix(config)
    if remote_checkpoint_prefix is not None:
        sync_checkpoint_dir_from_remote(config, remote_checkpoint_prefix, checkpoint_dir)

    try:
        state, metadata = load_latest_checkpoint(checkpoint_dir, map_location="cpu")
        latest_path = Path(checkpoint_dir) / "latest.pt"
        best_path = Path(checkpoint_dir) / "best.pt"

        if best_path.exists():
            print(f"Using best checkpoint for inference: {best_path}", flush=True)
            return str(best_path)

        if latest_path.exists():
            print(f"Using latest checkpoint for inference: {latest_path}", flush=True)
            return str(latest_path)

    except FileNotFoundError:
        pass

    print("No semantic checkpoint found for inference; using current/default weights", flush=True)
    return None

def _load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> Optional[str]:
    if checkpoint_path is None:
        print("No checkpoint path provided/resolved for inference", flush=True)
        return None

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Checkpoint path does not exist: {checkpoint_path}", flush=True)
        return None

    print(f"Loading semantic checkpoint into model: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    return str(ckpt_path)

@torch.no_grad()
def generate_semantic_embeddings(
    config: dict,
    split: Optional[str] = "val",
    checkpoint_path: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict:

    manifest_ref = _resolve_manifest_ref_for_split(config, split)
    print(f"Caching/loading semantic manifest for split='{split}' from: {manifest_ref}", flush=True)
    manifest_path = cache_manifest_from_uri(config, manifest_ref)
    print(f"Manifest cached at: {manifest_path}", flush=True)

    eval_cfg = config.get("evaluation", {})
    effective_batch_size = batch_size or eval_cfg.get("batch_size") or config["training"]["batch_size"]
    device = get_device(config["runtime"]["device"])

    if checkpoint_path is None:
        checkpoint_path = _resolve_checkpoint_path(config)

    print(f"Generating semantic embeddings on split='{split}' using device={device}", flush=True)

    dataset_cfg = config["dataset"]

    start_index = dataset_cfg.get("start_index", 0)
    max_records = dataset_cfg.get("max_records")
    subset_seed = dataset_cfg.get("subset_seed")

    print(
        f"Inference dataset config: split={split}, "
        f"start_index={start_index}, max_records={max_records}, subset_seed={subset_seed}",
        flush=True,
    )

    dataset = SemanticRetrievalDataset(
        manifest_path=manifest_path,
        config=config,
        image_size=config["model"]["image_size"],
        split=split,
        start_index=start_index,
        max_records=max_records,
        subset_seed=subset_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_semantic_model(config).to(device)
    loaded_checkpoint_path = _load_checkpoint_into_model(model, checkpoint_path, device)
    model.eval()

    all_image_embeddings = []
    all_text_embeddings = []
    all_image_ids = []
    all_texts = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print("Reached first inference batch", flush=True)

        images = batch["image_tensors"].to(device)
        text_features = build_text_features(batch["texts"]).to(device)

        image_emb = model.encode_image(images)
        text_emb = model.encode_text(text_features)

        all_image_embeddings.append(image_emb.cpu())
        all_text_embeddings.append(text_emb.cpu())
        all_image_ids.extend(batch["image_ids"])
        all_texts.extend(batch["texts"])

        if batch_idx % 50 == 0:
            print(
                f"[Infer] Batch {batch_idx}/{len(loader)} processed "
                f"(accumulated_images={len(all_image_ids)})",
                flush=True,
            )

    if not all_image_embeddings:
        raise ValueError("No samples were available for semantic embedding generation.")

    return {
        "image_embeddings": torch.cat(all_image_embeddings, dim=0),
        "text_embeddings": torch.cat(all_text_embeddings, dim=0),
        "image_ids": all_image_ids,
        "texts": all_texts,
        "device": str(device),
        "checkpoint_path": loaded_checkpoint_path,
        "candidate_name": config.get("candidate_name", "unknown_candidate"),
        "model_type": config["model"]["type"],
        "model_version": config["model"]["version"],
        "manifest_ref": manifest_ref,
        "manifest_path": manifest_path,
        "split": split,
    }

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate semantic embeddings for a config-driven candidate.")
    parser.add_argument("--config", required=True, help="Path to semantic config YAML")
    parser.add_argument("--split", default="val", help="Dataset split to embed")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = generate_semantic_embeddings(
        config=config,
        split=args.split,
        checkpoint_path=args.checkpoint_path,
    )

    summary = {
        "num_images": len(outputs["image_ids"]),
        "num_texts": len(outputs["texts"]),
        "device": outputs["device"],
        "checkpoint_path": outputs["checkpoint_path"],
        "candidate_name": outputs["candidate_name"],
        "model_type": outputs["model_type"],
        "model_version": outputs["model_version"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()