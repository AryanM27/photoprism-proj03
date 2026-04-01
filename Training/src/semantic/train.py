# import json
# from pathlib import Path

# import mlflow
# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.utils.data import DataLoader

# from src.common.checkpointing import (
#     build_checkpoint_dir,
#     checkpoint_exists,
#     load_latest_checkpoint,
#     save_checkpoint,
# )
# from src.common.config import load_config
# from src.common.seed import set_seed
# from src.datasets.semantic_dataset import SemanticRetrievalDataset
# from src.mlflow.logger import (
#     configure_mlflow,
#     log_artifact_if_exists,
#     log_config_params,
#     log_metrics,
#     start_run,
# )
# from src.semantic.model import TinySemanticModel, build_text_features


# def get_device(device_str: str) -> torch.device:
#     if device_str == "auto":
#         return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return torch.device(device_str)


# def collate_fn(batch):
#     return {
#         "image_ids": [item["image_id"] for item in batch],
#         "texts": [item["text"] for item in batch],
#         "image_tensors": torch.stack([item["image_tensor"] for item in batch]),
#     }


# def contrastive_loss(image_emb, text_emb, logit_scale):
#     logits = logit_scale.exp() * (image_emb @ text_emb.T)
#     labels = torch.arange(logits.size(0), device=logits.device)

#     loss_i = F.cross_entropy(logits, labels)
#     loss_t = F.cross_entropy(logits.T, labels)

#     return (loss_i + loss_t) / 2.0


# def run_one_epoch(model, loader, optimizer, device, train: bool):
#     if train:
#         model.train()
#     else:
#         model.eval()

#     total_loss = 0.0
#     total_samples = 0

#     for batch in loader:
#         images = batch["image_tensors"].to(device)
#         text_features = build_text_features(batch["texts"]).to(device)

#         if train:
#             optimizer.zero_grad()

#         with torch.set_grad_enabled(train):
#             image_emb = model.encode_image(images)
#             text_emb = model.encode_text(text_features)
#             loss = contrastive_loss(image_emb, text_emb, model.logit_scale)

#             if train:
#                 loss.backward()
#                 optimizer.step()

#         batch_size = images.size(0)
#         total_loss += loss.item() * batch_size
#         total_samples += batch_size

#     mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
#     return {"contrastive_loss": mean_loss}


# def save_history(history: list, artifact_dir: Path) -> Path:
#     artifact_dir.mkdir(parents=True, exist_ok=True)
#     history_file = artifact_dir / "history.json"

#     with history_file.open("w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2)

#     return history_file


# def train_semantic_baseline(config_path: str) -> dict:
#     config = load_config(config_path)
#     set_seed(config["runtime"]["seed"])

#     device = get_device(config["runtime"]["device"])

#     train_dataset = SemanticRetrievalDataset(
#         manifest_path=config["dataset"]["manifest_path"],
#         image_size=config["model"]["image_size"],
#         split="train",
#     )
#     val_dataset = SemanticRetrievalDataset(
#         manifest_path=config["dataset"]["manifest_path"],
#         image_size=config["model"]["image_size"],
#         split="val",
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=True,
#         collate_fn=collate_fn,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=False,
#         collate_fn=collate_fn,
#     )

#     model = TinySemanticModel(
#         embedding_dim=config["model"]["embedding_dim"]
#     ).to(device)

#     optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

#     checkpoint_dir = build_checkpoint_dir(
#         checkpoint_root=config["checkpoint"]["root_dir"],
#         task=config["task"],
#         model_family=config["model"]["family"],
#         model_version=config["model"]["version"],
#     )

#     start_epoch = 1
#     global_step = 0
#     best_val_loss = float("inf")
#     resumed_from_checkpoint = False
#     history = []

#     resume_mode = config["training"].get("resume", "auto")
#     if resume_mode == "always":
#         should_resume = True
#     elif resume_mode == "never":
#         should_resume = False
#     elif resume_mode == "auto":
#         should_resume = checkpoint_exists(checkpoint_dir)
#     else:
#         raise ValueError(f"Invalid resume mode: {resume_mode}")

#     if should_resume and checkpoint_exists(checkpoint_dir):
#         state, metadata = load_latest_checkpoint(checkpoint_dir, map_location=str(device))
#         model.load_state_dict(state["model_state_dict"])
#         optimizer.load_state_dict(state["optimizer_state_dict"])
#         start_epoch = state["epoch"] + 1
#         global_step = state["global_step"]
#         best_val_loss = metadata.get("metric_value", best_val_loss)
#         resumed_from_checkpoint = True

#     artifact_dir = Path(config["output"]["artifact_dir"])
#     artifact_dir.mkdir(parents=True, exist_ok=True)

#     tracking_uri = configure_mlflow()

#     with start_run(experiment_name=config["experiment_name"]):
#         log_config_params(config)
#         mlflow.log_param("resumed_from_checkpoint", resumed_from_checkpoint)

#         if start_epoch > config["training"]["epochs"]:
#             summary = {
#                 "best_val_contrastive_loss": best_val_loss,
#                 "device": str(device),
#                 "mlflow_tracking_uri": tracking_uri,
#                 "checkpoint_dir": checkpoint_dir,
#                 "resumed_from_checkpoint": resumed_from_checkpoint,
#                 "message": "No training run because checkpoint is already beyond configured epochs.",
#             }

#             summary_file = artifact_dir / "training_summary.txt"
#             with summary_file.open("w", encoding="utf-8") as f:
#                 for key, value in summary.items():
#                     f.write(f"{key}: {value}\n")

#             log_artifact_if_exists(str(summary_file))
#             return summary

#         for epoch in range(start_epoch, config["training"]["epochs"] + 1):
#             train_metrics = run_one_epoch(
#                 model=model,
#                 loader=train_loader,
#                 optimizer=optimizer,
#                 device=device,
#                 train=True,
#             )
#             val_metrics = run_one_epoch(
#                 model=model,
#                 loader=val_loader,
#                 optimizer=optimizer,
#                 device=device,
#                 train=False,
#             )

#             global_step += len(train_loader)

#             epoch_metrics = {
#                 "train_contrastive_loss": train_metrics["contrastive_loss"],
#                 "val_contrastive_loss": val_metrics["contrastive_loss"],
#                 "epoch": epoch,
#             }
#             log_metrics(epoch_metrics)

#             history.append(
#                 {
#                     "epoch": epoch,
#                     "train_contrastive_loss": train_metrics["contrastive_loss"],
#                     "val_contrastive_loss": val_metrics["contrastive_loss"],
#                 }
#             )

#             state = {
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "epoch": epoch,
#                 "global_step": global_step,
#             }

#             is_best = val_metrics["contrastive_loss"] < best_val_loss
#             if is_best:
#                 best_val_loss = val_metrics["contrastive_loss"]

#             save_checkpoint(
#                 checkpoint_dir=checkpoint_dir,
#                 state=state,
#                 task=config["task"],
#                 model_family=config["model"]["family"],
#                 model_version=config["model"]["version"],
#                 epoch=epoch,
#                 global_step=global_step,
#                 metric_name="val_contrastive_loss",
#                 metric_value=val_metrics["contrastive_loss"],
#                 dataset_version=config["dataset"]["dataset_version"],
#                 config_path=config_path,
#                 is_best=is_best,
#                 save_epoch_copy=True,
#             )

#         history_file = None
#         if config["output"].get("save_history", False):
#             history_file = save_history(history, artifact_dir)

#         summary = {
#             "best_val_contrastive_loss": best_val_loss,
#             "device": str(device),
#             "mlflow_tracking_uri": tracking_uri,
#             "checkpoint_dir": checkpoint_dir,
#             "resumed_from_checkpoint": resumed_from_checkpoint,
#         }

#         summary_file = artifact_dir / "training_summary.txt"
#         with summary_file.open("w", encoding="utf-8") as f:
#             for key, value in summary.items():
#                 f.write(f"{key}: {value}\n")

#         log_artifact_if_exists(str(summary_file))
#         if history_file is not None:
#             log_artifact_if_exists(str(history_file))

#     return summary


#NEW IMPL

import sys
from pathlib import Path
from src.datasets.uri_resolver import cache_manifest_from_uri
from src.storage.checkpoint_sync import (
    sync_checkpoint_dir_from_remote,
    sync_checkpoint_dir_to_remote,
)

# Ensure Training directory is in PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # points to Training/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from src.storage.artifact_io import save_history_artifact, save_summary_artifact

import mlflow
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.common.checkpointing import (
    build_checkpoint_dir,
    checkpoint_exists,
    load_latest_checkpoint,
    save_checkpoint,
)
from src.common.config import load_config
from src.common.seed import set_seed
from src.datasets.semantic_dataset import SemanticRetrievalDataset
from src.mlflow.logger import (
    configure_mlflow,
    log_artifact_if_exists,
    log_config_params,
    log_metrics,
    start_run,
)
from src.semantic.model import TinySemanticModel, build_text_features


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def collate_fn(batch):
    return {
        "image_ids": [item["image_id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "image_tensors": torch.stack([item["image_tensor"] for item in batch]),
    }


def contrastive_loss(image_emb, text_emb, logit_scale):
    logits = logit_scale.exp() * (image_emb @ text_emb.T)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2.0


def run_one_epoch(model, loader, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image_tensors"].to(device)
        text_features = build_text_features(batch["texts"]).to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            image_emb = model.encode_image(images)
            text_emb = model.encode_text(text_features)
            loss = contrastive_loss(image_emb, text_emb, model.logit_scale)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return {"contrastive_loss": mean_loss}


# def save_history(history: list, artifact_dir: Path) -> Path:
#     artifact_dir.mkdir(parents=True, exist_ok=True)
#     history_file = artifact_dir / "history.json"

#     with history_file.open("w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2)

#     return history_file


def train_semantic_baseline(config_path: str) -> dict:
    config = load_config(config_path)
    set_seed(config["runtime"]["seed"])

    device = get_device(config["runtime"]["device"])

    # train_dataset = SemanticRetrievalDataset(
    #     manifest_path=config["dataset"]["manifest_path"],
    #     image_size=config["model"]["image_size"],
    #     split="train",
    # )
    # val_dataset = SemanticRetrievalDataset(
    #     manifest_path=config["dataset"]["manifest_path"],
    #     image_size=config["model"]["image_size"],
    #     split="val",
    # )

    manifest_ref = config["dataset"].get("manifest_uri") or config["dataset"]["manifest_path"]
    manifest_path = cache_manifest_from_uri(config, manifest_ref)

    train_dataset = SemanticRetrievalDataset(
        manifest_path=manifest_path,
        config=config,
        image_size=config["model"]["image_size"],
        split="train",
    )
    val_dataset = SemanticRetrievalDataset(
        manifest_path=manifest_path,
        config=config,
        image_size=config["model"]["image_size"],
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = TinySemanticModel(
        embedding_dim=config["model"]["embedding_dim"]
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    checkpoint_dir = build_checkpoint_dir(
        checkpoint_root=config["checkpoint"]["root_dir"],
        task=config["task"],
        model_family=config["model"]["family"],
        model_version=config["model"]["version"],
    )

    local_checkpoint_dir = checkpoint_dir
    remote_checkpoint_prefix = None

    if config.get("storage", {}).get("backend") == "object_store":
        remote_checkpoint_prefix = (
            f"swift://training-module-proj03/checkpoints/"
            f"{config['task']}/{config['model']['version']}"
    )   
        sync_checkpoint_dir_from_remote(config, remote_checkpoint_prefix, local_checkpoint_dir)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    resumed_from_checkpoint = False
    history = []

    resume_mode = config["training"].get("resume", "auto")
    if resume_mode == "always":
        should_resume = True
    elif resume_mode == "never":
        should_resume = False
    elif resume_mode == "auto":
        should_resume = checkpoint_exists(checkpoint_dir)
    else:
        raise ValueError(f"Invalid resume mode: {resume_mode}")

    if should_resume and checkpoint_exists(checkpoint_dir):
        state, metadata = load_latest_checkpoint(checkpoint_dir, map_location=str(device))
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_val_loss = metadata.get("metric_value", best_val_loss)
        resumed_from_checkpoint = True

    artifact_dir = Path(config["output"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # tracking_uri = configure_mlflow()
    tracking_uri = configure_mlflow(config)


    with start_run(experiment_name=config["experiment_name"]):
        log_config_params(config)
        mlflow.log_param("resumed_from_checkpoint", resumed_from_checkpoint)

        if start_epoch > config["training"]["epochs"]:
            summary = {
                "best_val_contrastive_loss": best_val_loss,
                "device": str(device),
                "mlflow_tracking_uri": tracking_uri,
                "checkpoint_dir": checkpoint_dir,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "message": "No training run because checkpoint is already beyond configured epochs.",
            }

            # summary_file = artifact_dir / "training_summary.txt"
            # with summary_file.open("w", encoding="utf-8") as f:
            #     for key, value in summary.items():
            #         f.write(f"{key}: {value}\n")

            # log_artifact_if_exists(str(summary_file))
            summary_file = save_summary_artifact(config, summary)
            log_artifact_if_exists(str(summary_file))
            if history_file is not None:
                log_artifact_if_exists(str(history_file))
            return summary

        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                train=True,
            )
            val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                train=False,
            )

            global_step += len(train_loader)

            epoch_metrics = {
                "train_contrastive_loss": train_metrics["contrastive_loss"],
                "val_contrastive_loss": val_metrics["contrastive_loss"],
                "epoch": epoch,
            }
            log_metrics(epoch_metrics)

            history.append(
                {
                    "epoch": epoch,
                    "train_contrastive_loss": train_metrics["contrastive_loss"],
                    "val_contrastive_loss": val_metrics["contrastive_loss"],
                }
            )

            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }

            is_best = val_metrics["contrastive_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["contrastive_loss"]

            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                state=state,
                task=config["task"],
                model_family=config["model"]["family"],
                model_version=config["model"]["version"],
                epoch=epoch,
                global_step=global_step,
                metric_name="val_contrastive_loss",
                metric_value=val_metrics["contrastive_loss"],
                dataset_version=config["dataset"]["dataset_version"],
                config_path=config_path,
                is_best=is_best,
                save_epoch_copy=True,
            )

            if remote_checkpoint_prefix is not None:
                sync_checkpoint_dir_to_remote(config, local_checkpoint_dir, remote_checkpoint_prefix)

        history_file = None
        if config["output"].get("save_history", False):
            history_file = save_history_artifact(config, history)

        summary = {
            "best_val_contrastive_loss": best_val_loss,
            "device": str(device),
            "mlflow_tracking_uri": tracking_uri,
            "checkpoint_dir": checkpoint_dir,
            "resumed_from_checkpoint": resumed_from_checkpoint,
        }

        # summary_file = artifact_dir / "training_summary.txt"
        # with summary_file.open("w", encoding="utf-8") as f:
        #     for key, value in summary.items():
        #         f.write(f"{key}: {value}\n")

        # log_artifact_if_exists(str(summary_file))
        # if history_file is not None:
        #     log_artifact_if_exists(str(history_file))
        summary_file = save_summary_artifact(config, summary)
        log_artifact_if_exists(str(summary_file))
        if history_file is not None:
            log_artifact_if_exists(str(history_file))

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train the semantic baseline model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the semantic training YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = train_semantic_baseline(args.config)

    print("Semantic training summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
