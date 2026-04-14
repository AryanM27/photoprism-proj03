import sys
from pathlib import Path
import time
from src.datasets.uri_resolver import cache_manifest_from_uri
from src.storage.checkpoint_sync import (
    sync_checkpoint_dir_from_remote,
    sync_checkpoint_dir_to_remote,
)

from torch.utils.data.dataloader import default_collate

from src.aesthetic.evaluate import (
    evaluate_model,
    run_aesthetic_evaluation_for_split,
)

#ensure Training directory is in PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  #points to Training folder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from src.storage.artifact_io import save_history_artifact, save_summary_artifact

import mlflow
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.aesthetic.model import build_aesthetic_model
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

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def make_loader(dataset, batch_size: int, shuffle: bool, safe_collate):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=safe_collate,)


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0

    for batch in loader:
        if batch is None:
            continue
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

def _resolve_best_checkpoint_path(config: dict) -> str:
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

def train_aesthetic_baseline(config_path: str) -> dict:
    config = load_config(config_path)

    set_seed(config["runtime"]["seed"])

    device = get_device(config["runtime"]["device"])

    train_cfg = config["training"]
    learning_rate = train_cfg["learning_rate"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    weight_decay = train_cfg.get("weight_decay", 0.0)

    print(f"Using aesthetic model type: {config['model']['type']}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    manifest_ref = config["dataset"].get("manifest_uri") or config["dataset"]["manifest_path"]
    manifest_path = cache_manifest_from_uri(config, manifest_ref)

    dataset_cfg = config["dataset"]

    start_index = dataset_cfg.get("start_index", 0)
    max_records = dataset_cfg.get("max_records", None)
    subset_seed = dataset_cfg.get("subset_seed", None)

    print(f"\n===== DATASET CONFIG =====")
    print(f"start_index: {start_index}")
    print(f"max_records: {max_records}")
    print(f"subset_seed: {subset_seed}")

    train_dataset = AestheticDataset(
        manifest_path=manifest_path,
        config=config,
        image_size=config["model"]["image_size"],
        split="train",
        start_index=start_index,
        max_records=max_records,
    )
    val_dataset = AestheticDataset(
        manifest_path=manifest_path,
        config=config,
        image_size=config["model"]["image_size"],
        split="val",
        start_index=15000,
        max_records=2500,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    model = build_aesthetic_model(config)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    # if should_resume and checkpoint_exists(checkpoint_dir):
    #     print("Will be resuming from checkpoint", flush=True)
    #     state, metadata = load_latest_checkpoint(checkpoint_dir, map_location=str(device))

    #     model.load_state_dict(state["model_state_dict"])
    #     resumed_from_checkpoint = True

    #     advance_chunk = config["training"].get("advance_chunk_on_resume", False)

    #     if advance_chunk:
    #         next_start = metadata.get("next_start_index")

    #         if next_start is not None:
    #             print(f"\n>>> ADVANCING TO NEXT CHUNK: {next_start}", flush=True)

    #             start_index = next_start

    #             #rebuild train dataset for next chunk
    #             train_dataset = AestheticDataset(
    #                 manifest_path=manifest_path,
    #                 split="train",
    #                 config=config,
    #                 image_size=config["model"]["image_size"],
    #                 start_index=start_index,
    #                 max_records=max_records,
    #                 subset_seed=subset_seed,
    #             )

    #             print(f"New Train samples: {len(train_dataset)}", flush=True)

    #             #reset epoch schedule for the new chunk
    #             start_epoch = 1
    #             global_step = 0
    #             best_val_loss = float("inf")

    #             #carry optimizer state forward
    #             if config["training"].get("carry_optimizer_to_next_chunk", True):
    #                 if "optimizer_state_dict" in state:
    #                     optimizer.load_state_dict(state["optimizer_state_dict"])
    #             else:
    #                 print("Not carrying optimizer state to next chunk", flush=True)

    #         else:
    #             print("advance_chunk_on_resume=True but no next_start_index found; resuming same chunk", flush=True)

    #             optimizer.load_state_dict(state["optimizer_state_dict"])
    #             start_epoch = state["epoch"] + 1
    #             global_step = state["global_step"]
    #             best_val_loss = metadata.get("metric_value", best_val_loss)

    #     else:
    #         optimizer.load_state_dict(state["optimizer_state_dict"])
    #         start_epoch = state["epoch"] + 1
    #         global_step = state["global_step"]
    #         best_val_loss = metadata.get("metric_value", best_val_loss)

    if should_resume and checkpoint_exists(checkpoint_dir):
        print("Will be resuming from checkpoint", flush=True)

        # Always load checkpoint to CPU first for ROCm safety
        state, metadata = load_latest_checkpoint(checkpoint_dir, map_location="cpu")

        # Restore model weights
        model.load_state_dict(state["model_state_dict"])

        ckpt_model_type = metadata.get("model_type")
        ckpt_model_version = metadata.get("model_version")

        current_model_type = config["model"]["type"]
        current_model_version = config["model"]["version"]

        if ckpt_model_type is not None and ckpt_model_type != current_model_type:
            raise ValueError(
                f"Checkpoint model_type mismatch: checkpoint={ckpt_model_type}, current={current_model_type}"
            )

        if ckpt_model_version is not None and ckpt_model_version != current_model_version:
            raise ValueError(
                f"Checkpoint model_version mismatch: checkpoint={ckpt_model_version}, current={current_model_version}"
            )

        resumed_from_checkpoint = True

        advance_chunk = config["training"].get("advance_chunk_on_resume", False)
        carry_optimizer = config["training"].get("carry_optimizer_to_next_chunk", False)

        if advance_chunk:
            next_start = metadata.get("next_start_index")

            if next_start is not None:
                print(f"\n>>> ADVANCING TO NEXT CHUNK: {next_start}", flush=True)

                start_index = next_start

                # rebuild train dataset for next chunk
                train_dataset = AestheticDataset(
                    manifest_path=manifest_path,
                    split="train",
                    config=config,
                    image_size=config["model"]["image_size"],
                    start_index=start_index,
                    max_records=max_records,
                    subset_seed=subset_seed,
                )

                print(f"New Train samples: {len(train_dataset)}", flush=True)

                # reset epoch schedule for the new chunk
                start_epoch = 1
                global_step = 0
                best_val_loss = float("inf")

                # only restore optimizer if explicitly allowed
                if carry_optimizer and "optimizer_state_dict" in state:
                    try:
                        optimizer.load_state_dict(state["optimizer_state_dict"])
                        print("Loaded optimizer state for next chunk.", flush=True)
                    except Exception as e:
                        print(
                            f"WARNING: Failed to load optimizer state for next chunk. "
                            f"Using fresh optimizer. Error: {e}",
                            flush=True,
                        )
                else:
                    print("Skipping optimizer restore for next chunk.", flush=True)

            else:
                print(
                    "advance_chunk_on_resume=True but no next_start_index found; resuming same chunk",
                    flush=True,
                )

                start_epoch = int(state.get("epoch", 0)) + 1
                global_step = int(state.get("global_step", 0))
                best_val_loss = metadata.get("metric_value", best_val_loss)

                if carry_optimizer and "optimizer_state_dict" in state:
                    try:
                        optimizer.load_state_dict(state["optimizer_state_dict"])
                        print("Loaded optimizer state for same chunk resume.", flush=True)
                    except Exception as e:
                        print(
                            f"WARNING: Failed to load optimizer state on resume. "
                            f"Using fresh optimizer. Error: {e}",
                            flush=True,
                        )
                else:
                    print("Skipping optimizer restore for same chunk resume.", flush=True)

        else:
            start_epoch = int(state.get("epoch", 0)) + 1
            global_step = int(state.get("global_step", 0))
            best_val_loss = metadata.get("metric_value", best_val_loss)

            if carry_optimizer and "optimizer_state_dict" in state:
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                    print("Loaded optimizer state on resume.", flush=True)
                except Exception as e:
                    print(
                        f"WARNING: Failed to load optimizer state on resume. "
                        f"Using fresh optimizer. Error: {e}",
                        flush=True,
                    )
            else:
                print("Skipping optimizer restore on resume.", flush=True)

        print(
            f"Resume state -> start_epoch: {start_epoch}, epochs: {epochs}, "
            f"global_step: {global_step}, best_val_loss: {best_val_loss}",
            flush=True,
        )

    # tracking_uri = configure_mlflow()
    tracking_uri = configure_mlflow(config)

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        safe_collate=safe_collate,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        safe_collate=safe_collate,
    )

    training_start_time = time.time()

    with start_run(experiment_name=config["experiment_name"]):
        log_config_params(config)

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("image_size", config["model"]["image_size"])
        mlflow.log_param("dataset_version", config["dataset"]["dataset_version"])
        mlflow.log_param("candidate_name", config.get("candidate_name", "unknown_candidate"))
        mlflow.log_param("experiment_name", config.get("experiment_name", "unknown_experiment"))
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("model_version", config["model"]["version"])
        mlflow.log_param("device", str(device))
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_param("resumed_from_checkpoint", resumed_from_checkpoint)

        if start_epoch > epochs:
            summary = {
                "best_val_loss": best_val_loss,
                "device": str(device),
                "mlflow_tracking_uri": tracking_uri,
                "checkpoint_dir": checkpoint_dir,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "message": "No training run because checkpoint is already beyond configured epochs.",
            }

            summary_file = save_summary_artifact(config, summary)

            log_artifact_if_exists(str(summary_file))
            return summary

        for epoch in range(start_epoch, epochs+1):
            epoch_start_time = time.time()
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
            
            next_start_index = start_index + max_records if max_records else None

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
                chunk_start_index=start_index,
                chunk_max_records=max_records,
                chunk_subset_seed=subset_seed,
                next_start_index=next_start_index,
            )

            if remote_checkpoint_prefix is not None:
                sync_checkpoint_dir_to_remote(config, local_checkpoint_dir, remote_checkpoint_prefix)
            
            epoch_duration_sec = time.time() - epoch_start_time
            print(f"Epoch {epoch} duration_sec: {epoch_duration_sec:.2f}")
            mlflow.log_metric("epoch_duration_sec", epoch_duration_sec, step=epoch)

        history_file = None
        if config["output"].get("save_history", False):
            history_file = save_history_artifact(config, history)

        mlflow.log_metric("best_val_loss", best_val_loss)
        
        total_training_time_sec = time.time() - training_start_time
        print(f"Total training time (sec): {total_training_time_sec:.2f}")
        mlflow.log_metric("total_training_time_sec", total_training_time_sec)

        evaluation_cfg = config.get("evaluation", {})
        run_test_after_training = evaluation_cfg.get("run_test_after_training", False)

        test_summary = {}

        if run_test_after_training:
            print("\n===== RUNNING FINAL TEST EVALUATION =====", flush=True)

            best_checkpoint_path = _resolve_best_checkpoint_path(config)
            test_split_name = evaluation_cfg.get("test_split_name", "test")

            test_summary = run_aesthetic_evaluation_for_split(
                config=config,
                split=test_split_name,
                checkpoint_path=best_checkpoint_path,
            )

            for key, value in test_summary.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

        summary = {
            "best_val_mse_loss": best_val_loss,
            "device": str(device),
            "mlflow_tracking_uri": tracking_uri,
            "checkpoint_dir": checkpoint_dir,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "total_training_time_sec": total_training_time_sec,
            "epochs_completed_in_this_run": max(0, epochs - start_epoch + 1),
            **test_summary,
        }

        summary_file = save_summary_artifact(config, summary)

        log_artifact_if_exists(str(summary_file))
        if history_file is not None:
            log_artifact_if_exists(str(history_file))

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train the aesthetic model candidate.")
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
