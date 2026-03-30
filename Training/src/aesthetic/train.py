# import json
# from pathlib import Path

# import mlflow
# import torch
# from torch import nn
# from torch.optim import Adam
# from torch.utils.data import DataLoader

# from src.aesthetic.model import TinyAestheticRegressor
# from src.common.checkpointing import (
#     build_checkpoint_dir,
#     checkpoint_exists,
#     load_latest_checkpoint,
#     save_checkpoint,
# )
# from src.common.config import load_config
# from src.common.seed import set_seed
# from src.datasets.aesthetic_dataset import AestheticDataset
# from src.mlflow.logger import (
#     configure_mlflow,
#     log_artifact_if_exists,
#     log_config_params,
#     log_metrics,
#     start_run,
# )


# def get_device(device_str: str) -> torch.device:
#     if device_str == "auto":
#         return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return torch.device(device_str)


# def make_loader(dataset, batch_size: int, shuffle: bool):
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
#     if train:
#         model.train()
#     else:
#         model.eval()

#     total_loss = 0.0
#     total_abs_error = 0.0
#     total_samples = 0

#     for batch in loader:
#         images = batch["image_tensor"].to(device)
#         targets = batch["aesthetic_score"].to(device).float()

#         if train:
#             optimizer.zero_grad()

#         with torch.set_grad_enabled(train):
#             preds = model(images)
#             loss = criterion(preds, targets)

#             if train:
#                 loss.backward()
#                 optimizer.step()

#         batch_size = images.size(0)
#         abs_error = torch.abs(preds - targets).sum().item()

#         total_loss += loss.item() * batch_size
#         total_abs_error += abs_error
#         total_samples += batch_size

#     mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
#     mean_abs_error = total_abs_error / total_samples if total_samples > 0 else 0.0

#     return {
#         "mse_loss": mean_loss,
#         "mae": mean_abs_error,
#     }


# def save_history(history: list, artifact_dir: Path) -> Path:
#     artifact_dir.mkdir(parents=True, exist_ok=True)
#     history_file = artifact_dir / "history.json"

#     with history_file.open("w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2)

#     return history_file


# def train_aesthetic_baseline(config_path: str) -> dict:
#     config = load_config(config_path)
#     set_seed(config["runtime"]["seed"])

#     device = get_device(config["runtime"]["device"])

#     train_dataset = AestheticDataset(
#         manifest_path=config["dataset"]["manifest_path"],
#         split="train",
#         image_size=config["model"]["image_size"],
#     )
#     val_dataset = AestheticDataset(
#         manifest_path=config["dataset"]["manifest_path"],
#         split="val",
#         image_size=config["model"]["image_size"],
#     )

#     train_loader = make_loader(
#         train_dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=True,
#     )
#     val_loader = make_loader(
#         val_dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=False,
#     )

#     model = TinyAestheticRegressor().to(device)
#     criterion = nn.MSELoss()
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

#     should_resume = False

#     if resume_mode == "always":
#         should_resume = True
#     elif resume_mode == "never":
#         should_resume = False
#     elif resume_mode == "auto":
#         should_resume = checkpoint_exists(checkpoint_dir)
#     else:
#         raise ValueError(f"Invalid resume mode: {resume_mode}")

#     if should_resume:
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
#                 "best_val_loss": best_val_loss,
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
#                 criterion=criterion,
#                 optimizer=optimizer,
#                 device=device,
#                 train=True,
#             )
#             val_metrics = run_one_epoch(
#                 model=model,
#                 loader=val_loader,
#                 criterion=criterion,
#                 optimizer=optimizer,
#                 device=device,
#                 train=False,
#             )

#             global_step += len(train_loader)

#             epoch_metrics = {
#                 "train_mse_loss": train_metrics["mse_loss"],
#                 "train_mae": train_metrics["mae"],
#                 "val_mse_loss": val_metrics["mse_loss"],
#                 "val_mae": val_metrics["mae"],
#                 "epoch": epoch,
#             }
#             log_metrics(epoch_metrics)

#             history.append(
#                 {
#                     "epoch": epoch,
#                     "train_mse_loss": train_metrics["mse_loss"],
#                     "train_mae": train_metrics["mae"],
#                     "val_mse_loss": val_metrics["mse_loss"],
#                     "val_mae": val_metrics["mae"],
#                 }
#             )

#             state = {
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "epoch": epoch,
#                 "global_step": global_step,
#             }

#             is_best = val_metrics["mse_loss"] < best_val_loss
#             if is_best:
#                 best_val_loss = val_metrics["mse_loss"]

#             save_checkpoint(
#                 checkpoint_dir=checkpoint_dir,
#                 state=state,
#                 task=config["task"],
#                 model_family=config["model"]["family"],
#                 model_version=config["model"]["version"],
#                 epoch=epoch,
#                 global_step=global_step,
#                 metric_name="val_mse_loss",
#                 metric_value=val_metrics["mse_loss"],
#                 dataset_version=config["dataset"]["dataset_version"],
#                 config_path=config_path,
#                 is_best=is_best,
#                 save_epoch_copy=True,
#             )

#         history_file = None
#         if config["output"].get("save_history", False):
#             history_file = save_history(history, artifact_dir)

#         summary = {
#             "best_val_loss": best_val_loss,
#             "device": str(device),
#             "mlflow_tracking_uri": tracking_uri,
#             "checkpoint_dir": checkpoint_dir,
#             "resumed_from_checkpoint": resumed_from_checkpoint,
#             "epochs_completed_in_this_run": max(0, config["training"]["epochs"] - start_epoch + 1),
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
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.aesthetic.model import TinyAestheticRegressor
from src.common.checkpointing import (
    build_checkpoint_dir,
    checkpoint_exists,
    load_latest_checkpoint,
    save_checkpoint,
)
from src.common.config import load_config
from src.common.seed import set_seed
from src.datasets.aesthetic_dataset import AestheticDataset
from src.mlflow.logger import (
    configure_mlflow,
    log_artifact_if_exists,
    log_config_params,
    log_metrics,
    start_run,
)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def make_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image_tensor"].to(device)
        targets = batch["aesthetic_score"].to(device).float()

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds = model(images)
            loss = criterion(preds, targets)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        abs_error = torch.abs(preds - targets).sum().item()

        total_loss += loss.item() * batch_size
        total_abs_error += abs_error
        total_samples += batch_size

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    mean_abs_error = total_abs_error / total_samples if total_samples > 0 else 0.0

    return {
        "mse_loss": mean_loss,
        "mae": mean_abs_error,
    }


def save_history(history: list, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_file = artifact_dir / "history.json"

    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history_file


def train_aesthetic_baseline(config_path: str) -> dict:
    config = load_config(config_path)
    set_seed(config["runtime"]["seed"])

    device = get_device(config["runtime"]["device"])

    train_dataset = AestheticDataset(
        manifest_path=config["dataset"]["manifest_path"],
        split="train",
        image_size=config["model"]["image_size"],
    )
    val_dataset = AestheticDataset(
        manifest_path=config["dataset"]["manifest_path"],
        split="val",
        image_size=config["model"]["image_size"],
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    model = TinyAestheticRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    checkpoint_dir = build_checkpoint_dir(
        checkpoint_root=config["checkpoint"]["root_dir"],
        task=config["task"],
        model_family=config["model"]["family"],
        model_version=config["model"]["version"],
    )

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
                "best_val_loss": best_val_loss,
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
            summary_file = save_summary_artifact(config, summary)

            log_artifact_if_exists(str(summary_file))
            return summary

        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
            )
            val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=False,
            )

            global_step += len(train_loader)

            epoch_metrics = {
                "train_mse_loss": train_metrics["mse_loss"],
                "train_mae": train_metrics["mae"],
                "val_mse_loss": val_metrics["mse_loss"],
                "val_mae": val_metrics["mae"],
                "epoch": epoch,
            }
            log_metrics(epoch_metrics)

            history.append(
                {
                    "epoch": epoch,
                    "train_mse_loss": train_metrics["mse_loss"],
                    "train_mae": train_metrics["mae"],
                    "val_mse_loss": val_metrics["mse_loss"],
                    "val_mae": val_metrics["mae"],
                }
            )

            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }

            is_best = val_metrics["mse_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["mse_loss"]

            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                state=state,
                task=config["task"],
                model_family=config["model"]["family"],
                model_version=config["model"]["version"],
                epoch=epoch,
                global_step=global_step,
                metric_name="val_mse_loss",
                metric_value=val_metrics["mse_loss"],
                dataset_version=config["dataset"]["dataset_version"],
                config_path=config_path,
                is_best=is_best,
                save_epoch_copy=True,
            )

        history_file = None
        if config["output"].get("save_history", False):
            # history_file = save_history(history, artifact_dir)
            history_file = save_history_artifact(config, history)

        summary = {
            "best_val_loss": best_val_loss,
            "device": str(device),
            "mlflow_tracking_uri": tracking_uri,
            "checkpoint_dir": checkpoint_dir,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "epochs_completed_in_this_run": max(0, config["training"]["epochs"] - start_epoch + 1),
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
    parser = argparse.ArgumentParser(description="Train the aesthetic baseline model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the aesthetic training YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = train_aesthetic_baseline(args.config)

    print("Aesthetic training summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()